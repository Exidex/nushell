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
//! evaluation thread simply stays parked inside [`Elicit::run`] between the two
//! MCP rounds. The parked interpreter is bridged to the async tool handler via
//! the [`ElicitBridge`] installed for each evaluation (see
//! [`with_active_bridge`]):
//!
//! 1. `elicit` sends a [`PendingElicit`] over `request_tx` and blocks on
//!    `answer_rx.recv()`.
//! 2. The handler observes the request, parks the evaluation under an opaque
//!    `requestState` token, and answers the round with an `InputRequiredResult`.
//! 3. The client prompts the user and retries `tools/call` with `inputResponses`.
//! 4. The handler looks the token up, delivers the [`ElicitResult`] to the parked
//!    `elicit` call, and the pipeline resumes as if nothing happened.
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
    Category, Example, FromValue, IntoValue, PipelineData, ShellError, Signature, Span,
    SyntaxShape, Type, Value,
    engine::{Call, Command, EngineState, Stack},
    shell_error::generic::GenericError,
};
use rmcp::model::{
    ElicitRequest, ElicitRequestParams, ElicitResult, ElicitationAction, ElicitationSchema,
    EnumSchema, PrimitiveSchemaDefinition, StringFormat,
};
use serde_json::{Map as JsonMap, Value as JsonValue};

/// A single elicitation request parked inside a running evaluation, together
/// with the channel its answer is delivered on.
pub(crate) struct PendingElicit {
    /// Server-assigned identifier used as the `inputRequests` map key; the
    /// client's `inputResponses` on the retry round are keyed the same way.
    pub(crate) id: String,
    /// The `elicitation/create` request to surface to the client.
    pub(crate) request: ElicitRequest,
    /// One-shot answer channel back into the blocked `elicit` command.
    pub(crate) answer_tx: sync_mpsc::Sender<ElicitResult>,
}

/// Per-evaluation bridge that gives the synchronous `elicit` builtin access to
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

/// Converts a `serde_json` value into a Nushell `Value`.
fn json_to_nu_value(value: JsonValue, span: Span) -> Result<Value, ShellError> {
    let text = serde_json::to_string(&value).map_err(|err| {
        bridge_error(
            "invalid elicitation response",
            format!("could not encode the elicitation content: {err}"),
            span,
        )
    })?;
    let parsed: nu_json::Value = nu_json::from_str(&text).map_err(|err| {
        bridge_error(
            "invalid elicitation response",
            format!("could not decode the elicitation content: {err}"),
            span,
        )
    })?;
    Ok(parsed.into_value(span))
}

/// Maps the client's [`ElicitResult`] onto the Nushell pipeline: `accept`
/// yields the content record, while `decline`/`cancel` abort the pipeline with
/// a `ShellError`.
fn elicit_result_to_value(result: ElicitResult, span: Span) -> Result<Value, ShellError> {
    match result.action {
        ElicitationAction::Accept => {
            let content = result
                .content
                .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));
            json_to_nu_value(content, span)
        }
        ElicitationAction::Decline => Err(bridge_error(
            "elicitation declined",
            "the user declined the elicitation request",
            span,
        )),
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

/// Builds an [`ElicitationSchema`] from the `--schema` record. Each entry maps
/// a field name to one of:
///
/// * a primitive type name (`"string"`, `"int"`, `"number"`, `"bool"`,
///   `"email"`, `"uri"`, `"date"`, `"datetime"`),
/// * a list of choices (single-select enum), or
/// * a record holding a raw JSON Schema fragment for advanced constraints.
///
/// Every field is required: elicitation forms show all declared fields anyway,
/// and optional semantics add little value inside a tool call. Without
/// `--schema`, an empty object schema turns the prompt into a pure
/// confirmation dialog.
fn build_requested_schema(
    schema: Option<Value>,
    title: Option<String>,
    span: Span,
) -> Result<ElicitationSchema, ShellError> {
    let mut builder = ElicitationSchema::builder();
    if let Some(title) = title {
        builder = builder.title(title);
    }

    let Some(schema) = schema else {
        return builder
            .build()
            .map_err(|err| bridge_error("invalid elicitation schema", err.to_string(), span));
    };

    let Value::Record { val: record, .. } = &schema else {
        return Err(bridge_error(
            "invalid elicitation schema",
            "the --schema value must be a record of field name to type",
            schema.span(),
        ));
    };

    for (name, spec) in record.iter() {
        builder = match spec {
            Value::String { val, .. } => match val.as_str() {
                "string" | "str" => builder.required_string(name.clone()),
                "email" => builder.required_email(name.clone()),
                "int" | "integer" => builder.required_integer_property(name.clone(), |s| s),
                "number" | "float" => builder.required_number_property(name.clone(), |s| s),
                "bool" | "boolean" => builder.required_bool_property(name.clone(), |s| s),
                "uri" | "url" => builder.required_string_property(name.clone(), |mut s| {
                    s.format = Some(StringFormat::Uri);
                    s
                }),
                "date" => builder.required_string_property(name.clone(), |mut s| {
                    s.format = Some(StringFormat::Date);
                    s
                }),
                "datetime" | "date-time" => {
                    builder.required_string_property(name.clone(), |mut s| {
                        s.format = Some(StringFormat::DateTime);
                        s
                    })
                }
                other => {
                    return Err(bridge_error(
                        "unknown elicitation field type",
                        format!(
                            "field '{name}' has type '{other}'; expected one of string, int, \
                             number, bool, email, uri, date, datetime, a list of choices, or a \
                             JSON Schema record"
                        ),
                        spec.span(),
                    ));
                }
            },
            Value::List { vals, .. } => {
                let mut choices = Vec::with_capacity(vals.len());
                for item in vals {
                    let choice = item.coerce_str().map_err(|_| {
                        bridge_error(
                            "invalid elicitation enum",
                            format!(
                                "enum choices for field '{name}' must be strings, got {}",
                                item.get_type()
                            ),
                            item.span(),
                        )
                    })?;
                    choices.push(choice.into_owned());
                }
                if choices.is_empty() {
                    return Err(bridge_error(
                        "invalid elicitation enum",
                        format!("field '{name}' has an empty list of choices"),
                        spec.span(),
                    ));
                }
                let enum_schema = EnumSchema::builder(choices).build();
                builder.required_enum_schema(name.clone(), enum_schema)
            }
            Value::Record { .. } => {
                // Advanced case: pass a raw JSON Schema fragment through.
                let as_nu_json = nu_json::Value::from_value(spec.clone()).map_err(|err| {
                    bridge_error(
                        "invalid elicitation schema",
                        format!("field '{name}' could not be converted to JSON: {err}"),
                        spec.span(),
                    )
                })?;
                let as_json = serde_json::to_value(as_nu_json).map_err(|err| {
                    bridge_error(
                        "invalid elicitation schema",
                        format!("field '{name}' could not be serialized: {err}"),
                        spec.span(),
                    )
                })?;
                let definition: PrimitiveSchemaDefinition = serde_json::from_value(as_json)
                    .map_err(|err| {
                        bridge_error(
                            "invalid elicitation schema",
                            format!(
                                "field '{name}' is not a valid elicitation primitive schema: {err}"
                            ),
                            spec.span(),
                        )
                    })?;
                builder.required_property(name.clone(), definition)
            }
            other => {
                return Err(bridge_error(
                    "invalid elicitation schema",
                    format!(
                        "field '{name}' must be a type name string, a list of choices, or a \
                         JSON Schema record, got {}",
                        other.get_type()
                    ),
                    spec.span(),
                ));
            }
        };
    }

    builder
        .build()
        .map_err(|err| bridge_error("invalid elicitation schema", err.to_string(), span))
}

/// `elicit <message> [--title <string>] [--schema <record>]`
///
/// Prompts the MCP client for structured user input and blocks the pipeline
/// until the client answers (or the elicitation times out). Only available
/// while the code is running inside a live MCP `evaluate` request.
#[derive(Clone)]
pub(crate) struct Elicit;

impl Command for Elicit {
    fn name(&self) -> &str {
        "elicit"
    }

    fn signature(&self) -> Signature {
        Signature::build("elicit")
            .input_output_types(vec![(Type::Nothing, Type::record())])
            .required(
                "message",
                SyntaxShape::String,
                "Human-readable message explaining what input is needed",
            )
            .named(
                "title",
                SyntaxShape::String,
                "Optional title displayed above the elicitation form",
                None,
            )
            .named(
                "schema",
                SyntaxShape::record(),
                "Record of field name to primitive type ('string', 'int', 'number', 'bool', 'email', 'uri', 'date', 'datetime'), a list of enum choices, or a raw JSON Schema record; omit for a pure confirm dialog",
                None,
            )
            .category(Category::Misc)
    }

    fn description(&self) -> &str {
        "Ask the MCP client (and through it the user) for structured input during an evaluate call."
    }

    fn extra_description(&self) -> &str {
        "Only available while running inside a live MCP `evaluate` request on a client that negotiated the 2026-07-28 protocol (multi round-trip elicitation).

The evaluated pipeline pauses at this command while the client prompts the user, then resumes with the answer:

* `accept` - the provided content record becomes the command's output
* `decline` - raises an error and aborts the pipeline
* `cancel` - raises an error and aborts the pipeline

This command is not registered in interactive Nushell sessions."
    }

    fn search_terms(&self) -> Vec<&str> {
        vec![
            "mcp",
            "elicitation",
            "prompt",
            "user input",
            "confirm",
            "ask",
        ]
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![
            Example {
                description: "Ask a yes/no confirmation; declining aborts the pipeline",
                example: "elicit \"Delete this file? This cannot be undone.\" --schema {confirmed: bool}",
                result: None,
            },
            Example {
                description: "Collect a small form of named values",
                example: "let form = elicit \"Project details\" --schema {name: string, stars: int, license: ['MIT', 'Apache-2.0']}",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        engine_state: &EngineState,
        stack: &mut Stack,
        call: &Call,
        _input: PipelineData,
    ) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let message: String = call.req(engine_state, stack, 0)?;
        let title: Option<String> = call.get_flag(engine_state, stack, "title")?;
        let schema: Option<Value> = call.get_flag(engine_state, stack, "schema")?;

        let requested_schema = build_requested_schema(schema, title, span)?;
        let request = ElicitRequest::new(ElicitRequestParams::FormElicitationParams {
            meta: None,
            message,
            requested_schema,
        });

        let bridge = ACTIVE_ELICIT_BRIDGE
            .with(|slot| slot.borrow().clone())
            .ok_or_else(|| {
                bridge_error(
                    "elicit outside an MCP evaluation",
                    "`elicit` can only run inside a live MCP `evaluate` request; outside the MCP server (for example in an interactive session or a background job) there is no client to prompt",
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
}
