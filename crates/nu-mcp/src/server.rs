use crate::evaluation::Evaluator;
use nu_protocol::{UseAnsiColoring, engine::EngineState};
use rmcp::{
    RoleServer, ServerHandler,
    handler::server::{
        tool::{InputResponses as InputResponsesPart, RequestState, ToolRouter},
        wrapper::Parameters,
    },
    model::{CallToolResponse, Implementation, ServerCapabilities, ServerConfig},
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub struct NushellMcpServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    evaluator: Evaluator,
}

#[tool_router]
impl NushellMcpServer {
    pub fn new(mut engine_state: EngineState) -> Self {
        // Configure the engine state for MCP
        let mut config = engine_state.get_config().as_ref().clone();
        config.use_ansi_coloring = UseAnsiColoring::False;
        config.color_config.clear();
        engine_state.set_config(config);
        NushellMcpServer {
            tool_router: Self::tool_router(),
            evaluator: Evaluator::new(engine_state),
        }
    }

    #[tool(description = r#"List available Nushell native commands.
By default all available commands will be returned. To find a specific command by searching command names, descriptions and search terms, use the find parameter."#)]
    async fn list_commands(
        &self,
        _ctx: RequestContext<RoleServer>,
        Parameters(ListCommandsRequest { find }): Parameters<ListCommandsRequest>,
    ) -> Result<String, String> {
        self.evaluator.list_available_commands(find).await
    }

    #[tool(
        description = "Get help for a specific Nushell command. This will only work on commands that are native to nushell. To find out if a command is native to nushell you can use the list_commands tool."
    )]
    async fn command_help(
        &self,
        _ctx: RequestContext<RoleServer>,
        Parameters(CommandNameRequest { name }): Parameters<CommandNameRequest>,
    ) -> Result<String, String> {
        self.evaluator.command_help(&name).await
    }

    #[doc = include_str!("evaluate_tool.md")]
    #[tool]
    async fn evaluate(
        &self,
        ctx: RequestContext<RoleServer>,
        Parameters(NuSourceRequest { input }): Parameters<NuSourceRequest>,
        RequestState(request_state): RequestState,
        InputResponsesPart(input_responses): InputResponsesPart,
    ) -> CallToolResponse {
        self.evaluator
            .eval_tool(&input, ctx, request_state, input_responses)
            .await
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
struct ListCommandsRequest {
    #[schemars(description = "string to find in command names, descriptions, and search term")]
    find: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
struct CommandNameRequest {
    #[schemars(description = "The name of the command")]
    name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
struct NuSourceRequest {
    #[schemars(description = "The Nushell source code to evaluate")]
    input: String,
}

#[tool_handler]
impl ServerHandler for NushellMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("nushell-mcp-server", env!("CARGO_PKG_VERSION"))
                    .with_title("Nushell MCP Server")
                    .with_website_url("https://www.nushell.sh"),
            )
            .with_instructions(include_str!("instructions.md"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::channel::mpsc;
    use nu_cmd_lang::create_default_context;
    use rmcp::model::{CallToolResult, RequestId};
    use rmcp::service::{RxJsonRpcMessage, TxJsonRpcMessage, serve_directly};
    use serde_json::Value as JsonValue;
    use tokio_util::sync::CancellationToken;

    fn make_request_context(request_id: i64) -> RequestContext<RoleServer> {
        let engine_state = create_default_context();
        let server = NushellMcpServer::new(engine_state);

        let (tx, _rx_sink) = mpsc::unbounded::<TxJsonRpcMessage<RoleServer>>();
        let (_tx_stream, rx_stream) = mpsc::unbounded::<RxJsonRpcMessage<RoleServer>>();
        let transport = (tx, rx_stream);

        let running = serve_directly(server, transport, None);
        let peer = running.peer().clone();
        drop(running);

        RequestContext::new(RequestId::Number(request_id), peer)
    }

    /// Unwraps a completed tool response; panics when an elicitation round
    /// was expected but a plain result came back (or vice versa).
    fn completed(response: CallToolResponse) -> CallToolResult {
        match response {
            CallToolResponse::Complete(result) => result,
            other => panic!("expected a complete tool result, got {other:?}"),
        }
    }

    #[test]
    fn server_info_serializes_expected_mcp_metadata() {
        let engine_state = create_default_context();
        let server = NushellMcpServer::new(engine_state);
        let info = server.get_info();
        let json = serde_json::to_value(&info).expect("ServerInfo should serialize to JSON");

        let capabilities = json
            .get("capabilities")
            .expect("ServerInfo JSON should include capabilities");
        let tools = capabilities
            .get("tools")
            .and_then(JsonValue::as_object)
            .expect("Server capabilities should include tools");
        assert!(
            tools.is_empty(),
            "tools capability should be present as an empty object"
        );

        let instructions = json
            .get("instructions")
            .expect("ServerInfo JSON should include instructions")
            .as_str()
            .expect("instructions should be a string");
        assert!(
            instructions.contains("list_commands") && instructions.contains("command_help"),
            "instructions should point at the command discovery tools (list_commands / command_help)"
        );

        let server_info = json
            .get("serverInfo")
            .or_else(|| json.get("server_info"))
            .expect("ServerInfo JSON should include serverInfo");
        assert_eq!(
            server_info.get("name").and_then(JsonValue::as_str),
            Some("nushell-mcp-server")
        );
        assert_eq!(
            server_info.get("title").and_then(JsonValue::as_str),
            Some("Nushell MCP Server")
        );
        assert_eq!(
            server_info.get("version").and_then(JsonValue::as_str),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(
            server_info
                .get("websiteUrl")
                .or_else(|| server_info.get("website_url"))
                .and_then(JsonValue::as_str),
            Some("https://www.nushell.sh")
        );
    }

    #[test]
    fn tool_router_exposes_expected_mcp_tools() {
        let router = NushellMcpServer::tool_router();
        let tool_names: Vec<_> = router
            .list_all()
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        assert_eq!(
            tool_names,
            vec![
                "command_help".to_string(),
                "evaluate".to_string(),
                "list_commands".to_string()
            ]
        );

        let list_commands_tool = router
            .get("list_commands")
            .expect("list_commands tool should be registered");
        assert!(
            list_commands_tool
                .description
                .as_deref()
                .unwrap_or("")
                .contains("List available Nushell native commands"),
            "list_commands tool description should mention native command discovery"
        );
    }

    #[test]
    fn evaluate_tool_input_schema_exposes_only_input_property() {
        let router = NushellMcpServer::tool_router();
        let evaluate_tool = router
            .get("evaluate")
            .expect("evaluate tool should be registered");

        let properties = evaluate_tool
            .input_schema
            .get("properties")
            .and_then(JsonValue::as_object)
            .expect("evaluate input schema should expose a properties object");

        let mut property_names: Vec<_> = properties.keys().cloned().collect();
        property_names.sort();
        assert_eq!(
            property_names,
            vec!["input".to_string()],
            "evaluate tool should accept only `input`; per-call timeout must be \
             controlled exclusively via the NU_MCP_PROMOTE_AFTER env var"
        );
    }

    fn create_mcp_server() -> NushellMcpServer {
        let engine_state = create_default_context();
        NushellMcpServer::new(engine_state)
    }

    fn result_text(result: &CallToolResult) -> &str {
        result
            .content
            .first()
            .and_then(|content| content.as_text())
            .map(|text| text.text.as_str())
            .expect("tool result should include text content")
    }

    #[tokio::test]
    async fn list_commands_tool_returns_non_empty_help() {
        let server = create_mcp_server();
        let ctx = make_request_context(0);

        let result = server
            .list_commands(
                ctx,
                Parameters(ListCommandsRequest {
                    find: Some("version".to_string()),
                }),
            )
            .await
            .expect("list_commands should succeed");

        assert!(
            result.len() > 20,
            "list_commands output should not be empty"
        );
        assert!(
            result.to_lowercase().contains("version"),
            "list_commands output should mention version"
        );
    }

    #[tokio::test]
    async fn evaluate_tool_computes_basic_expression() {
        let server = create_mcp_server();
        let ctx = make_request_context(1);

        let result = completed(
            server
                .evaluate(
                    ctx,
                    Parameters(NuSourceRequest {
                        input: "5 + 2".to_string(),
                    }),
                    RequestState(None),
                    InputResponsesPart(None),
                )
                .await,
        );
        let text = result_text(&result);

        assert!(
            text.contains("output"),
            "evaluate output should include output metadata"
        );
        assert!(
            text.contains('7'),
            "evaluate output should include the computed result"
        );
        assert_eq!(
            result
                .structured_content
                .as_ref()
                .and_then(|value| value.get("output"))
                .and_then(JsonValue::as_i64),
            Some(7),
            "evaluate structuredContent should include the computed result"
        );
    }

    #[tokio::test]
    async fn evaluate_tool_history_index_increments() {
        let engine_state = nu_cmd_lang::create_default_context();
        let server = NushellMcpServer::new(engine_state);

        let result1 = completed(
            server
                .evaluate(
                    make_request_context(3),
                    Parameters(NuSourceRequest {
                        input: "1".to_string(),
                    }),
                    RequestState(None),
                    InputResponsesPart(None),
                )
                .await,
        );
        let result1 = result_text(&result1);
        assert!(
            result1.contains("history_index:0") || result1.contains("history_index: 0"),
            "first evaluation should have history_index 0"
        );

        let result2 = completed(
            server
                .evaluate(
                    make_request_context(4),
                    Parameters(NuSourceRequest {
                        input: "2".to_string(),
                    }),
                    RequestState(None),
                    InputResponsesPart(None),
                )
                .await,
        );
        let result2 = result_text(&result2);
        assert!(
            result2.contains("history_index:1") || result2.contains("history_index: 1"),
            "second evaluation should have history_index 1"
        );
    }

    #[tokio::test]
    async fn command_help_tool_returns_help_for_version_command() {
        let server = create_mcp_server();
        let ctx = make_request_context(2);

        let result = server
            .command_help(
                ctx,
                Parameters(CommandNameRequest {
                    name: "version".to_string(),
                }),
            )
            .await
            .expect("command_help should succeed");

        assert!(
            result.to_lowercase().contains("version"),
            "command_help output should mention the command name"
        );
        assert!(
            result.to_lowercase().contains("usage") || result.contains("Usage"),
            "command_help output should include usage information"
        );
    }

    /// The MRTR round trip: `run-external-on-host` parks the evaluation and the first
    /// `evaluate` call answers with `InputRequiredResult`; the retry with
    /// matching `requestState` + `inputResponses` resumes the pipeline with the
    /// host command's captured output.
    #[tokio::test]
    async fn run_external_on_host_parks_evaluation_until_client_answers() {
        use rmcp::model::{InputRequest, InputResponses};

        let server = create_mcp_server();
        let source = "let out = run-external-on-host [\"ls\", \"-l\"] | collect; $\"got: ($out)\"";

        let first = server
            .evaluate(
                make_request_context(30),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(None),
                InputResponsesPart(None),
            )
            .await;
        let CallToolResponse::InputRequired(input_required) = first else {
            panic!("run-external-on-host should park into an InputRequiredResult, got {first:?}");
        };
        let token = input_required
            .request_state
            .expect("parked elicitation should carry a requestState token");
        let requests = input_required
            .input_requests
            .expect("parked elicitation should carry inputRequests");
        assert_eq!(
            requests.len(),
            1,
            "one run-external-on-host call should park one request"
        );
        let (id, request) = requests.iter().next().expect("one request");
        match request {
            InputRequest::Elicitation(permission) => {
                let params = &permission.params;
                let rmcp::model::ElicitRequestParams::FormElicitationParams {
                    meta,
                    message,
                    requested_schema,
                } = params
                else {
                    panic!("expected form-mode elicitation params");
                };
                assert_eq!(
                    message,
                    "Allow Nushell to run the following command on the host?\n\nls -l"
                );
                assert_eq!(
                    requested_schema.properties.len(),
                    3,
                    "the schema should declare the host execution result fields"
                );
                let required = requested_schema.required.clone().unwrap_or_default();
                assert!(
                    required.contains(&"stdout_b64".to_string())
                        && required.contains(&"stderr_b64".to_string())
                        && required.contains(&"exit_code".to_string()),
                    "all result fields should be required"
                );
                let marker = meta
                    .as_ref()
                    .and_then(|meta| meta.get(crate::elicitation::COMMAND_EXECUTION_META_KEY))
                    .expect("elicitation should carry the command execution marker in _meta");
                assert_eq!(
                    marker,
                    &serde_json::json!(["ls", "-l"]),
                    "the marker should carry the argv of the command awaiting approval"
                );
            }
            other => panic!("expected an elicitation input request, got {other:?}"),
        }

        // The client executed `ls -l` on the host and reports the result:
        // "hello" and "!" base64-encoded, exit code 0.
        let mut responses = InputResponses::new();
        responses.insert(
            id.clone(),
            serde_json::json!({
                "action": "accept",
                "content": {
                    "stdout_b64": "aGVsbG8=",
                    "stderr_b64": "IQ==",
                    "exit_code": 0
                }
            }),
        );

        let second = server
            .evaluate(
                make_request_context(31),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(Some(token)),
                InputResponsesPart(Some(responses)),
            )
            .await;
        let result = completed(second);
        let text = result_text(&result);
        assert!(
            text.contains("got: hello!"),
            "resumed pipeline should see the merged host command output, got: {text}"
        );
    }

    /// A declined host execution aborts the evaluated pipeline with an error.
    #[tokio::test]
    async fn run_external_on_host_decline_aborts_pipeline() {
        use rmcp::model::InputResponses;

        let server = create_mcp_server();
        let source = "run-external-on-host [\"rm\", \"-rf\", \"backup\"]; \"should not get here\"";

        let first = server
            .evaluate(
                make_request_context(32),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(None),
                InputResponsesPart(None),
            )
            .await;
        let CallToolResponse::InputRequired(input_required) = first else {
            panic!("expected an InputRequiredResult, got {first:?}");
        };
        let token = input_required
            .request_state
            .expect("requestState should be present");
        let requests = input_required
            .input_requests
            .expect("inputRequests should be present");
        let id = requests
            .keys()
            .next()
            .cloned()
            .expect("one elicitation request");

        let mut responses = InputResponses::new();
        responses.insert(id, serde_json::json!({ "action": "decline" }));

        let second = server
            .evaluate(
                make_request_context(33),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(Some(token.clone())),
                InputResponsesPart(Some(responses)),
            )
            .await;
        let result = completed(second);
        assert_eq!(result.is_error, Some(true), "decline should error");
        let text = result_text(&result);
        assert!(
            text.contains("declined"),
            "error should mention the decline, got: {text}"
        );
    }

    /// An accept whose content lacks the required execution result fields is
    /// an error, never a silently empty success.
    #[tokio::test]
    async fn run_external_on_host_accept_without_result_content_errors() {
        use rmcp::model::InputResponses;

        let server = create_mcp_server();
        let source = "run-external-on-host [\"touch\"]";

        let first = server
            .evaluate(
                make_request_context(42),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(None),
                InputResponsesPart(None),
            )
            .await;
        let CallToolResponse::InputRequired(input_required) = first else {
            panic!("expected an InputRequiredResult, got {first:?}");
        };
        let token = input_required
            .request_state
            .expect("requestState should be present");
        let id = input_required
            .input_requests
            .expect("inputRequests")
            .keys()
            .next()
            .cloned()
            .expect("one elicitation request");

        let mut responses = InputResponses::new();
        responses.insert(id, serde_json::json!({ "action": "accept" }));

        let second = server
            .evaluate(
                make_request_context(43),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(Some(token)),
                InputResponsesPart(Some(responses)),
            )
            .await;
        let result = completed(second);
        assert_eq!(
            result.is_error,
            Some(true),
            "an accept without result content must error, got: {}",
            result_text(&result)
        );
        let text = result_text(&result);
        assert!(
            text.contains("missing") && text.contains("stdout_b64"),
            "error should explain the missing result fields, got: {text}"
        );
    }

    /// A cancelled elicitation aborts the evaluated pipeline with an error.
    #[tokio::test]
    async fn run_external_on_host_cancel_aborts_pipeline() {
        use rmcp::model::InputResponses;

        let server = create_mcp_server();
        let source = "run-external-on-host [\"rm\", \"-rf\", \"backup\"]; \"should not get here\"";

        let first = server
            .evaluate(
                make_request_context(40),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(None),
                InputResponsesPart(None),
            )
            .await;
        let CallToolResponse::InputRequired(input_required) = first else {
            panic!("expected an InputRequiredResult, got {first:?}");
        };
        let token = input_required
            .request_state
            .expect("requestState should be present");
        let requests = input_required
            .input_requests
            .expect("inputRequests should be present");
        let id = requests
            .keys()
            .next()
            .cloned()
            .expect("one elicitation request");

        let mut responses = InputResponses::new();
        responses.insert(id, serde_json::json!({ "action": "cancel" }));

        let second = server
            .evaluate(
                make_request_context(41),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(Some(token.clone())),
                InputResponsesPart(Some(responses)),
            )
            .await;
        let result = completed(second);
        assert_eq!(result.is_error, Some(true), "cancel should error");
        let text = result_text(&result);
        assert!(
            text.contains("cancelled"),
            "error should mention the cancellation, got: {text}"
        );
    }

    /// A second `run-external-on-host` in the same pipeline produces a second round under
    /// the same `requestState` token.
    #[tokio::test]
    async fn run_external_on_host_supports_consecutive_rounds() {
        use rmcp::model::InputResponses;

        let server = create_mcp_server();
        let source = "let a = run-external-on-host [\"first\"] | collect; let b = run-external-on-host [\"second\"] | collect; $\"($a)-($b)\"";

        let first = server
            .evaluate(
                make_request_context(34),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(None),
                InputResponsesPart(None),
            )
            .await;
        let CallToolResponse::InputRequired(round1) = first else {
            panic!("expected an InputRequiredResult, got {first:?}");
        };
        let token = round1.request_state.clone().expect("requestState");
        let id1 = round1
            .input_requests
            .expect("inputRequests")
            .keys()
            .next()
            .cloned()
            .expect("one request");

        let mut responses1 = InputResponses::new();
        responses1.insert(
            id1,
            serde_json::json!({
                "action": "accept",
                "content": { "stdout_b64": "QQ==", "stderr_b64": "", "exit_code": 0 }
            }),
        );

        let second = server
            .evaluate(
                make_request_context(35),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(Some(token.clone())),
                InputResponsesPart(Some(responses1)),
            )
            .await;
        let CallToolResponse::InputRequired(round2) = second else {
            panic!("second run-external-on-host should park again, got {second:?}");
        };
        assert_eq!(
            round2.request_state.as_deref(),
            Some(token.as_str()),
            "the same evaluation should keep its requestState token"
        );
        let id2 = round2
            .input_requests
            .expect("inputRequests")
            .keys()
            .next()
            .cloned()
            .expect("one request");

        let mut responses2 = InputResponses::new();
        responses2.insert(
            id2,
            serde_json::json!({
                "action": "accept",
                "content": { "stdout_b64": "Qg==", "stderr_b64": "", "exit_code": 3 }
            }),
        );

        let third = server
            .evaluate(
                make_request_context(36),
                Parameters(NuSourceRequest {
                    input: source.to_string(),
                }),
                RequestState(Some(token)),
                InputResponsesPart(Some(responses2)),
            )
            .await;
        let result = completed(third);
        let text = result_text(&result);
        assert!(
            text.contains("A-B"),
            "pipeline should finish with both host command outputs, got: {text}"
        );
    }

    /// Retrying with an unknown or already-consumed token is a clean error.
    #[tokio::test]
    async fn resume_with_unknown_request_state_errors() {
        let server = create_mcp_server();
        let result = completed(
            server
                .evaluate(
                    make_request_context(37),
                    Parameters(NuSourceRequest {
                        input: "'unreachable'".to_string(),
                    }),
                    RequestState(Some("not-a-real-token".to_string())),
                    InputResponsesPart(None),
                )
                .await,
        );
        assert_eq!(result.is_error, Some(true), "unknown token should error");
        let text = result_text(&result);
        assert!(
            text.contains("requestState"),
            "error should explain the requestState problem, got: {text}"
        );
    }

    /// Only one elicitation round stays parked at a time: when a newer
    /// evaluation parks, a previously abandoned `requestState` stops working.
    /// Waiting itself has no timeout.
    #[tokio::test]
    async fn newer_park_supersedes_abandoned_request_state() {
        let server = create_mcp_server();

        let response = server
            .evaluator
            .eval_tool(
                "run-external-on-host [\"abandoned\"]",
                make_request_context(380),
                None,
                None,
            )
            .await;
        let CallToolResponse::InputRequired(round1) = response else {
            panic!("first run-external-on-host should park, got {response:?}");
        };
        let stale_token = round1.request_state.expect("requestState token");

        // A second evaluation parks too, superseding the abandoned round.
        let response2 = server
            .evaluator
            .eval_tool(
                "run-external-on-host [\"new\"]",
                make_request_context(381),
                None,
                None,
            )
            .await;
        let CallToolResponse::InputRequired(round2) = response2 else {
            panic!("second run-external-on-host should park, got {response2:?}");
        };
        let fresh_token = round2.request_state.expect("requestState token");
        assert_ne!(stale_token, fresh_token, "rounds get distinct tokens");

        // The stale token is now unknown.
        let resume = server
            .evaluator
            .eval_tool(
                "run-external-on-host [\"abandoned\"]",
                make_request_context(382),
                Some(stale_token),
                None,
            )
            .await;
        let result = completed(resume);
        assert_eq!(
            result.is_error,
            Some(true),
            "stale token should be rejected"
        );
        assert!(
            result_text(&result).contains("requestState"),
            "error should explain the superseded token"
        );

        // The fresh round still resumes normally.
        let requests = round2
            .input_requests
            .expect("fresh round should carry its request");
        let rid = requests.keys().next().cloned().expect("one request id");
        let mut responses = rmcp::model::InputResponses::new();
        responses.insert(
            rid,
            serde_json::json!({
                "action": "accept",
                "content": { "stdout_b64": "", "stderr_b64": "", "exit_code": 0 }
            }),
        );
        let done = server
            .evaluator
            .eval_tool(
                "run-external-on-host [\"new\"]",
                make_request_context(383),
                Some(fresh_token),
                Some(responses),
            )
            .await;
        let result = completed(done);
        assert_ne!(
            result.is_error,
            Some(true),
            "fresh round should complete: {}",
            result_text(&result)
        );
    }

    /// `run-external-on-host` has no meaning outside a live MCP evaluation round.
    #[tokio::test]
    async fn run_external_on_host_without_active_mcp_request_errors() {
        let server = create_mcp_server();
        let result = server
            .evaluator
            .eval_async(
                "run-external-on-host [\"anywhere\"]",
                CancellationToken::new(),
            )
            .await;
        assert_eq!(
            result.is_error,
            Some(true),
            "run-external-on-host outside eval_tool should error"
        );
        let text = result_text(&result);
        assert!(
            text.contains("only run inside a live MCP"),
            "error should explain the missing MCP context, got: {text}"
        );
    }
}
