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
//! evaluation thread simply stays parked inside [`RunExternalOnHost::run`]
//! between the two MCP rounds. The parked interpreter is bridged to the async
//! tool handler via the [`ElicitBridge`] installed for each evaluation (see
//! [`with_active_bridge`]):
//!
//! 1. `run-external-on-host` sends a [`PendingElicit`] over `request_tx` and
//!    blocks on `answer_rx.recv()`.
//! 2. The handler observes the request, parks the evaluation under an opaque
//!    `requestState` token, and answers the round with an `InputRequiredResult`.
//! 3. The client prompts the user, executes the command on the host (outside
//!    this Nushell process), and retries `tools/call` with `inputResponses`
//!    whose accepted entry carries the captured `{stdout_b64, stderr_b64,
//!    exit_code}` in its `content`.
//! 4. The handler looks the token up, delivers the [`ElicitResult`] to the
//!    parked `run-external-on-host` call, which rebuilds a byte stream from the
//!    result so the pipeline resumes as if an external command had run locally.
//!
//! The host execution result travels on the same `ElicitResult` as the approval
//! decision (the elicitation's requested schema declares the result fields), so
//! no extra back-channel is needed.
//!
//! Waiting on a human (and on the host command they approved) is unbounded by
//! design: there is no elicitation timeout. The parked builtin still wakes
//! cleanly (instead of hanging forever) because its answer channel disconnects
//! as soon as the parked session is dropped, e.g. when the MCP session ends or
//! the server shuts down.

use std::cell::RefCell;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self as sync_mpsc, RecvError};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use nu_engine::CallExt;
use nu_protocol::{
    ByteStream, ByteStreamType, Category, Example, PipelineData, PipelineMetadata, ShellError,
    Signature, Span, SyntaxShape, Type, Value,
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
    /// `run-external-on-host` command.
    pub(crate) answer_tx: sync_mpsc::Sender<ElicitResult>,
}

/// Per-evaluation bridge that gives the synchronous `run-external-on-host`
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
/// request, letting clients render it distinctly and knowing they must execute
/// the argv themselves. Arbitrary extension keys in `_meta` are allowed by
/// SEP-1319.
pub(crate) const COMMAND_EXECUTION_META_KEY: &str = "exidex/command_execution";

/// Elicitation content field carrying the command's captured stdout bytes,
/// base64-encoded (RFC 4648 standard alphabet).
const STDOUT_B64_FIELD: &str = "stdout_b64";

/// Elicitation content field carrying the command's captured stderr bytes,
/// base64-encoded (RFC 4648 standard alphabet).
const STDERR_B64_FIELD: &str = "stderr_b64";

/// Elicitation content field carrying the command's exit code.
const EXIT_CODE_FIELD: &str = "exit_code";

/// Elicitation content field a client may use on a *decline* to say why the request was
/// refused. A policy denial travels here: without it the model sees a bare "declined",
/// learns nothing about what stopped it, and retries the same argv.
const MESSAGE_FIELD: &str = "message";

/// Key under which the host exit code is attached to the resulting stream's
/// [`PipelineMetadata::custom`], following the namespaced-key convention of
/// that field.
pub(crate) const HOST_EXIT_CODE_METADATA_KEY: &str = "host_exit_code";

/// Builds the request `_meta` carrying [`COMMAND_EXECUTION_META_KEY`] with the
/// argv of the command the user is being asked to approve. The elicitation
/// itself stays a confirmation dialog whose (required) schema fields describe
/// the execution result the client must report when accepting.
fn command_execution_meta(argv: Vec<String>) -> RequestMetaObject {
    let mut meta = MetaObject::new();
    meta.insert(
        COMMAND_EXECUTION_META_KEY.to_string(),
        JsonValue::Array(argv.into_iter().map(JsonValue::String).collect()),
    );
    RequestMetaObject(meta)
}

/// The result contract the client fills when it accepts: the captured output
/// of the host-side execution, base64-encoded so binary streams survive the
/// JSON round trip (elicitation strings must be valid UTF-8).
fn host_command_schema() -> Result<ElicitationSchema, ShellError> {
    ElicitationSchema::builder()
        .required_string_property(STDOUT_B64_FIELD, |s| {
            s.title("stdout").description(
                "Captured standard output bytes, base64-encoded (RFC 4648, standard alphabet)",
            )
        })
        .required_string_property(STDERR_B64_FIELD, |s| {
            s.title("stderr").description(
                "Captured standard error bytes, base64-encoded (RFC 4648, standard alphabet)",
            )
        })
        .required_integer_property(EXIT_CODE_FIELD, |i| {
            i.title("exit code")
                .description("Exit code of the host command process")
        })
        .build()
        .map_err(|err| {
            bridge_error(
                "invalid elicitation schema",
                err.to_string(),
                Span::unknown(),
            )
        })
}

/// Decodes one required base64 string field of the accepted elicitation
/// content.
fn decode_b64_field(content: &JsonValue, field: &str, span: Span) -> Result<Vec<u8>, ShellError> {
    let encoded = content
        .get(field)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| missing_result_error(field, span))?;
    BASE64.decode(encoded).map_err(|err| {
        bridge_error(
            "invalid host command output",
            format!("elicitation content field '{field}' is not valid base64: {err}"),
            span,
        )
    })
}

fn missing_result_error(field: &str, span: Span) -> ShellError {
    bridge_error(
        "missing host command result",
        format!(
            "the client accepted the host execution but its elicitation content is missing the required '{field}' field; only clients implementing the '{COMMAND_EXECUTION_META_KEY}' contract can fulfill this request"
        ),
        span,
    )
}

/// Turns an accepted [`ElicitResult`] into the bytes and exit code of the host
/// command, merging stdout then stderr the way the evaluator's `capture_all`
/// stack merges a local external's output.
fn host_command_result(
    content: Option<JsonValue>,
    span: Span,
) -> Result<(Vec<u8>, i64), ShellError> {
    let content = content.ok_or_else(|| missing_result_error(STDOUT_B64_FIELD, span))?;
    if !content.is_object() {
        return Err(bridge_error(
            "invalid host command result",
            format!(
                "the elicitation content must be an object with '{STDOUT_B64_FIELD}', '{STDERR_B64_FIELD}' and '{EXIT_CODE_FIELD}' fields, got {content}"
            ),
            span,
        ));
    }
    let mut bytes = decode_b64_field(&content, STDOUT_B64_FIELD, span)?;
    let stderr = decode_b64_field(&content, STDERR_B64_FIELD, span)?;
    let exit_code = content
        .get(EXIT_CODE_FIELD)
        .and_then(JsonValue::as_i64)
        .ok_or_else(|| missing_result_error(EXIT_CODE_FIELD, span))?;
    bytes.extend_from_slice(&stderr);
    Ok((bytes, exit_code))
}

/// `run-external-on-host <args: list<string>>`
///
/// Asks the MCP client (and through it the user) to run a single command,
/// given as its argv, on the host machine outside this sandboxed Nushell
/// process, and blocks the pipeline until the client reports the execution
/// result. The captured stdout and stderr emerge as a byte stream, transparently
/// like a local external command's output. Only available while the code is
/// running inside a live MCP `evaluate` request.
#[derive(Clone)]
pub(crate) struct RunExternalOnHost;

impl Command for RunExternalOnHost {
    fn name(&self) -> &str {
        "run-external-on-host"
    }

    fn signature(&self) -> Signature {
        Signature::build("run-external-on-host")
            .input_output_types(vec![(Type::Nothing, Type::Any)])
            .required(
                "args",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "The argv of the command to run on the host, starting with the program name",
            )
            .category(Category::System)
    }

    fn description(&self) -> &str {
        "Ask the MCP client (and through it the user) to run a single command, given as its argv, on the host during an evaluate call, returning its captured output."
    }

    fn extra_description(&self) -> &str {
        "Only available while running inside a live MCP `evaluate` request on a client that negotiated the 2026-07-28 protocol (multi round-trip elicitation).

The evaluated pipeline pauses at this command while the client prompts the user and, once approved, executes the command outside the sandbox. The command's output emerges as a byte stream, just like a local external command, so `| lines`, `| collect` and string interpolation all work on it. The result is handled as follows:

* `accept` - the command must answer with the execution result in the elicitation content: `stdout_b64`/`stderr_b64` (captured output bytes, base64-encoded per RFC 4648 standard alphabet) and `exit_code`. stdout is reported first, then stderr, matching how a local external's merged capture reads.
* `decline` - raises an error and aborts the pipeline: the command never ran. A client may explain
  the refusal in the elicitation content under the key `message`; that text becomes the error
  the caller sees, so a policy denial reads as a reason rather than a bare no.
* `cancel` - raises an error and aborts the pipeline.

A non-zero `exit_code` does not abort the pipeline on its own, mirroring how externals behave in this evaluator (its exit code is attached to the stream metadata under the custom key `host_exit_code` and visible via `describe --detailed`); branch on the output explicitly if a failure must stop the pipeline.

The command argv is sent to the client under the `_meta` marker key
`exidex/command_execution` so a supporting client can render a dedicated
approval dialog and knows it must execute the argv itself; clients without
that support cannot fulfill the request (accepting without reporting the
result content is an error).

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

        if args.is_empty() {
            return Err(bridge_error(
                "empty command argv",
                "`run-external-on-host` needs at least the program name in the argv list",
                span,
            ));
        }

        // The human-facing message is generated from the argv; the
        // machine-readable payload travels in `_meta` under the marker key.
        let message = format!(
            "Allow Nushell to run the following command on the host?\n\n{}",
            args.join(" ")
        );

        let requested_schema = host_command_schema()?;
        let request = ElicitRequest::new(ElicitRequestParams::FormElicitationParams {
            meta: Some(command_execution_meta(args)),
            message,
            requested_schema,
        });

        let bridge = ACTIVE_ELICIT_BRIDGE
            .with(|slot| slot.borrow().clone())
            .ok_or_else(|| {
                bridge_error(
                    "run-external-on-host outside an MCP evaluation",
                    "`run-external-on-host` can only run inside a live MCP `evaluate` request; outside the MCP server (for example in an interactive session or a background job) there is no client to prompt",
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

                // No timeout: the pipeline waits as long as the user needs to
                // decide and as long as the approved host command takes to run.
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

        match result.action {
            ElicitationAction::Accept => {
                let (bytes, exit_code) = host_command_result(result.content, span)?;

                // Surface the exit code as stream metadata rather than an
                // error: like local externals in this evaluator, a non-zero
                // status never aborts the pipeline on its own.
                let mut metadata = PipelineMetadata::default();
                metadata.custom.push(
                    HOST_EXIT_CODE_METADATA_KEY.to_string(),
                    Value::int(exit_code, span),
                );

                let stream = ByteStream::read(
                    Cursor::new(bytes),
                    span,
                    engine_state.signals().clone(),
                    ByteStreamType::Unknown,
                );
                Ok(PipelineData::byte_stream(stream, Some(metadata)))
            }
            ElicitationAction::Decline => {
                let reason = result
                    .content
                    .as_ref()
                    .and_then(|content| content.get(MESSAGE_FIELD))
                    .and_then(JsonValue::as_str);
                Err(match reason {
                    // The client's own words become the error text: the pipeline aborts either
                    // way, but a reason is what makes a refusal actionable instead of a retry.
                    Some(text) => bridge_error("host command execution refused", text.to_string(), span),
                    None => bridge_error(
                        "host command execution declined",
                        "the user declined to run the command on the host, so the evaluation was interrupted",
                        span,
                    ),
                })
            }
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

    fn examples(&self) -> Vec<Example<'_>> {
        vec![
            Example {
                description: "Run a command on the host and process its output in the pipeline",
                example: "run-external-on-host [\"git\", \"status\"] | lines | first 3",
                result: None,
            },
            Example {
                description: "Capture the merged output as a string",
                example: "let out = run-external-on-host [\"uname\", \"-a\"] | collect; $out",
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
            "host",
            "external",
            "exec",
        ]
    }
}
