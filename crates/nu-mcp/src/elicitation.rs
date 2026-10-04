//! MCP elicitation support for the nu-mcp evaluator.
//!
//! Implements the Multi Round-Trip Request (MRTR) elicitation flow introduced in
//! the MCP `2026-07-28` spec revision (SEP-2322). Server-to-client back-channel
//! requests are gone: instead of holding a `tools/call` open while a human
//! answers, the `evaluate` tool *returns* an `InputRequiredResult` describing the
//! `elicitation/create` request, and the client retries the call with
//! `inputResponses` plus the echoed `requestState`.
//!
//! Because a running Nushell pipeline cannot be serialized/resumed, the
//! evaluation thread simply stays parked inside [`AskCommandPermission::run`]
//! between the two MCP rounds. The parked interpreter is bridged to the async
//! tool handler via the [`ElicitBridge`] installed for each evaluation (see
//! [`with_active_bridge`]):
//!
//! 1. `ask_command_permission` sends a [`PendingElicit`] over `request_tx` and
//!    blocks on `answer_rx.recv()`.
//! 2. The handler observes the request, parks the evaluation under an opaque
//!    `requestState` token, and answers the round with an `InputRequiredResult`.
//! 3. The client prompts the user and retries `tools/call` with `inputResponses`.
//! 4. The handler looks the token up, delivers the [`ElicitResult`] to the
//!    parked `ask_command_permission` call, and the pipeline resumes as if
//!    nothing happened.
//!
//! Waiting on a human is unbounded by design: there is no elicitation timeout.
//! The parked builtin still wakes cleanly (instead of hanging forever) because
//! its answer channel disconnects as soon as the parked session is dropped,
//! e.g. when the MCP session ends or the server shuts down.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self as sync_mpsc, RecvError};

use nu_engine::CallExt;
use nu_protocol::{
    Category, Example, PipelineData, ShellError, Signature, Span, SyntaxShape, Type, Value,
    engine::{Call, Command, EngineState, Stack},
    shell_error::generic::GenericError,
};
use rmcp::model::{
    ElicitRequest, ElicitRequestParams, ElicitResult, ElicitationAction, ElicitationSchema,
    MetaObject, RequestMetaObject,
};
use serde_json::Value as JsonValue;

/// A single elicitation request parked inside a running evaluation, together
/// with the channel its answer is delivered on.
pub(crate) struct PendingElicit {
    /// Server-assigned identifier used as the `inputRequests` map key; the
    /// client's `inputResponses` on the retry round are keyed the same way.
    pub(crate) id: String,
    /// The `elicitation/create` request to surface to the client.
    pub(crate) request: ElicitRequest,
    /// One-shot answer channel back into the blocked
    /// `ask_command_permission` command.
    pub(crate) answer_tx: sync_mpsc::Sender<ElicitResult>,
}

/// Per-evaluation bridge that gives the synchronous `ask_command_permission`
/// builtin access to
/// the asynchronous MCP request round-trip. Cloned into a thread-local by
/// [`with_active_bridge`] before evaluation starts.
#[derive(Clone)]
pub(crate) enum ElicitBridge {
    /// The client speaks MRTR elicitation (`2026-07-28` or newer).
    Mrtr {
        request_tx: futures::channel::mpsc::UnboundedSender<PendingElicit>,
        /// Set while the builtin is waiting on the user so the promote-to-job
        /// timer knows the evaluation is parked, not working.
        parked: Arc<AtomicBool>,
    },
    /// The client negotiated a pre-`2026-07-28` protocol with no MRTR support,
    /// so elicitation cannot be offered at all.
    Unsupported { version: String },
}

thread_local! {
    /// The bridge for the evaluation currently running on this thread.
    static ACTIVE_ELICIT_BRIDGE: RefCell<Option<ElicitBridge>> = const { RefCell::new(None) };
}

/// Runs `f` with `bridge` installed as the active elicitation bridge for this
/// thread, clearing it again (even on panic) before returning.
pub(crate) fn with_active_bridge<R>(bridge: Option<ElicitBridge>, f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE_ELICIT_BRIDGE.with(|slot| *slot.borrow_mut() = None);
        }
    }

    ACTIVE_ELICIT_BRIDGE.with(|slot| *slot.borrow_mut() = bridge);
    let _guard = Guard;
    f()
}

fn bridge_error(
    error: impl Into<std::borrow::Cow<'static, str>>,
    msg: impl Into<std::borrow::Cow<'static, str>>,
    span: Span,
) -> ShellError {
    ShellError::Generic(GenericError::new(error, msg, span))
}

/// `_meta` marker key that flags an elicitation as a command-execution
/// confirmation, letting clients render it distinctly. Arbitrary extension
/// keys in `_meta` are allowed by SEP-1319.
pub(crate) const COMMAND_EXECUTION_META_KEY: &str = "exidex/command_execution";

/// Builds the request `_meta` carrying [`COMMAND_EXECUTION_META_KEY`] with the
/// argv of the command the user is being asked to approve. The elicitation
/// itself stays a pure confirmation dialog: an empty schema, no title.
fn command_execution_meta(argv: Vec<String>) -> RequestMetaObject {
    let mut meta = MetaObject::new();
    meta.insert(
        COMMAND_EXECUTION_META_KEY.to_string(),
        JsonValue::Array(argv.into_iter().map(JsonValue::String).collect()),
    );
    RequestMetaObject(meta)
}

/// Maps the client's [`ElicitResult`] onto the Nushell pipeline: `accept`
/// yields `true`, `decline` yields `false`, while `cancel` aborts the pipeline
/// with a `ShellError`.
fn elicit_result_to_value(result: ElicitResult, span: Span) -> Result<Value, ShellError> {
    match result.action {
        ElicitationAction::Accept => Ok(Value::bool(true, span)),
        ElicitationAction::Decline => Ok(Value::bool(false, span)),
        ElicitationAction::Cancel => Err(bridge_error(
            "elicitation cancelled",
            "the user cancelled the operation through the elicitation request",
            span,
        )),
        other => Err(bridge_error(
            "unknown elicitation action",
            format!("the client returned an unsupported action: {other:?}"),
            span,
        )),
    }
}

/// `ask_command_permission <args: list<string>>`
///
/// Asks the MCP client (and through it the user) for permission to run a
/// single command, given as its argv, and blocks the pipeline until the client
/// answers. Only available while the code is running inside a live MCP
/// `evaluate` request.
#[derive(Clone)]
pub(crate) struct AskCommandPermission;

impl Command for AskCommandPermission {
    fn name(&self) -> &str {
        "ask_command_permission"
    }

    fn signature(&self) -> Signature {
        Signature::build("ask_command_permission")
            .input_output_types(vec![(Type::Nothing, Type::Bool)])
            .required(
                "args",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "The argv of the command that would run once the user grants permission, starting with the program name",
            )
            .category(Category::Misc)
    }

    fn description(&self) -> &str {
        "Ask the MCP client (and through it the user) for permission to run a single command, given as its argv, during an evaluate call."
    }

    fn extra_description(&self) -> &str {
        "Only available while running inside a live MCP `evaluate` request on a client that negotiated the 2026-07-28 protocol (multi round-trip elicitation).

The evaluated pipeline pauses at this command while the client prompts the user, then resumes with the answer:

* `accept` - the command outputs `true`
* `decline` - the command outputs `false`, so the pipeline can branch instead of aborting
* `cancel` - raises an error and aborts the pipeline

The command argv is sent to the client under the `_meta` marker key
`exidex/command_execution` so a supporting client can render a dedicated
permission dialog; clients without that support just see the generated
confirmation message. The elicitation form itself is a pure confirmation
dialog (empty schema).

This command is not registered in interactive Nushell sessions."
    }

    fn run(
        &self,
        engine_state: &EngineState,
        stack: &mut Stack,
        call: &Call,
        _input: PipelineData,
    ) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let args: Vec<String> = call.req(engine_state, stack, 0)?;

        // The human-facing message is generated from the argv; the
        // machine-readable payload travels in `_meta` under the marker key.
        let message = if args.is_empty() {
            "Allow Nushell to run the pending command?".to_string()
        } else {
            format!(
                "Allow Nushell to run the following command?\n\n{}",
                args.join(" ")
            )
        };

        // A pure confirmation dialog.
        let requested_schema = ElicitationSchema::builder()
            .build()
            .map_err(|err| bridge_error("invalid elicitation schema", err.to_string(), span))?;
        let request = ElicitRequest::new(ElicitRequestParams::FormElicitationParams {
            meta: Some(command_execution_meta(args)),
            message,
            requested_schema,
        });

        let bridge = ACTIVE_ELICIT_BRIDGE
            .with(|slot| slot.borrow().clone())
            .ok_or_else(|| {
                bridge_error(
                    "ask_command_permission outside an MCP evaluation",
                    "`ask_command_permission` can only run inside a live MCP `evaluate` request; outside the MCP server (for example in an interactive session or a background job) there is no client to prompt",
                    span,
                )
            })?;

        let result = match bridge {
            ElicitBridge::Unsupported { version } => Err(bridge_error(
                "MCP client does not support elicitation",
                format!(
                    "the client negotiated MCP protocol version '{version}', which predates multi round-trip elicitation (requires '2026-07-28' or newer)"
                ),
                span,
            )),
            ElicitBridge::Mrtr { request_tx, parked } => {
                let (answer_tx, answer_rx) = sync_mpsc::channel();
                let id = uuid::Uuid::new_v4().to_string();

                parked.store(true, Ordering::SeqCst);
                let sent = request_tx.unbounded_send(PendingElicit {
                    id,
                    request,
                    answer_tx,
                });
                if sent.is_err() {
                    parked.store(false, Ordering::SeqCst);
                    return Err(bridge_error(
                        "elicitation has no live MCP request",
                        "the evaluation is no longer attached to a live MCP request (for example it was promoted to a background job or the session closed), so no client can be prompted",
                        span,
                    ));
                }

                // No timeout: the pipeline waits as long as the user needs.
                // `recv` unblocks either when the client retries with an
                // answer, or with a clean error when the parked session is
                // dropped (MCP session ended, server shut down).
                match answer_rx.recv() {
                    Ok(result) => {
                        parked.store(false, Ordering::SeqCst);
                        Ok(result)
                    }
                    Err(RecvError) => {
                        parked.store(false, Ordering::SeqCst);
                        Err(bridge_error(
                            "elicitation abandoned",
                            "the elicitation was abandoned because the MCP session ended or the pending request was dropped",
                            span,
                        ))
                    }
                }
            }
        }?;

        let value = elicit_result_to_value(result, span)?;
        Ok(PipelineData::value(value, None))
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![
            Example {
                description: "Ask permission before running a command; accepting yields true",
                example: "if (ask_command_permission [\"rm\", \"-rf\", \"target\"]) { rm -rf target }",
                result: None,
            },
            Example {
                description: "Declining yields false, letting the pipeline branch instead of aborting",
                example: "let ok = ask_command_permission [\"systemctl\", \"restart\", \"myapp\"]; if not $ok { \"skipped\" }",
                result: None,
            },
        ]
    }

    fn search_terms(&self) -> Vec<&str> {
        vec![
            "mcp",
            "elicitation",
            "permission",
            "approve",
            "confirm",
            "ask",
        ]
    }
}
