# Codex subscription brain protocol probe

Verified locally on 2026-09-11 and 2026-09-13 with `codex-cli 0.153.4`. The main-brain
integration uses a separate transport from the existing pinned subscription
voice client. Codex reads and refreshes its own existing ChatGPT login; Jarvis
does not read, copy, print, or supply authentication tokens.

## Evidence

The generated schema and reproducible probe live outside production sources:

- `.runtime/protocol-probe/schema/` at the workspace root: generated with
  `codex app-server generate-json-schema --experimental --out <directory>`.
- `.runtime/protocol-probe/echo_probe.py`: bounded echo-only JSONL client.
- `echo-summary.json`, `echo-wire.json`: successful real dynamic tool call and
  final `ECHO_RESULT:jarvis-probe-20260911` through ChatGPT authentication.
- `image-summary.json`, `image-wire.json`: successful `gpt-5.5` image round trip.
  The user supplied a generated red image; the tool supplied a generated blue
  image. The model called echo with `text: "red"` and answered
  `USER:red;TOOL:blue`. No expected colors were named in the prompt.
- `adapter_probe.py`, `adapter-summary.json`: the actual production transport
  and brain adapter passed the same two-phase tool/image round trip while
  preserving a synthetic verification word supplied only in `req.system`.
  The tool input was `red:PINEAPPLE_731`; the final answer was
  `USER:red;TOOL:blue;MEMORY:PINEAPPLE_731`. Readiness reported ChatGPT,
  `gpt-5.5`, CLI `0.153.4`; cleanup completed.

`gpt-5.5` remains the adapter's compatibility default. The Russian profile
selects **gpt-5.6-sol**. On September 13 the production adapter passed the same
tool/image/system-context probe with Sol:
`USER:red;TOOL:blue;MEMORY:PINEAPPLE_731`. Its model and child lifecycle were
verified, and the current main-model choice is persisted by the composer and
provider card through the same configuration writer. Background structured
memory curation was separately verified on **gpt-5.6-luna**.

With `gpt-6-astra` in the earlier
CLI build, user-image recognition succeeded but dynamic-tool image recognition
returned `TOOL:unknown` twice. Those calls used the code-mode dispatch path
(`exec-*` call IDs), while the successful `gpt-5.5` call used `call_*`.
Accordingly, vision capability is enabled only for the verified GPT-5.5 and
GPT-5.6-Sol models. Other catalog entries do not imply verified vision support.

**Tool images on code-mode models:** returning `inputImage` only in the dynamic
tool result did not expose the pixels to the code-mode model. The production
adapter sends those images with `turn/steer` and the exact `expectedTurnId`
before resolving the pending tool RPC. The steering text labels them as
untrusted tool output. A red user image followed by a blue tool image now
produces the correct independent colors on Sol; no expected color is included
in the request. Regression tests cover this ordering and turn correlation.

**Memory correction:** a real Luna curator request on copied fictional QA
pages updated `entities/aleksey-astronomer.md` and the related concept, retained
the separate baker entity, and created no alternate transliterations. Existing
page aliases, titles, complete bodies and source dates now reach the curator.
Oversize pages are never truncated into replacement inputs. The check took
46 seconds; generated proposals remained outside the live user vault.

## Wire contract

Transport is newline-delimited JSON objects over `codex app-server --stdio`.
Requests have `id`, `method`, and `params`; responses match the same opaque
`id`. Notifications have no `id`.

1. `initialize` takes `clientInfo: {name, version}` and
   `capabilities: {experimentalApi: true}`. Follow with `initialized`.
2. `account/read` takes `{refreshToken: false}`. Require
   `result.account.type === "chatgpt"`; do not expose the account's other
   fields. This establishes login mode, not available quota or model access.
3. `thread/start` takes `model: "gpt-5.5"`, `modelProvider: "openai"`,
   `ephemeral: true`, `approvalPolicy: "never"`, `sandbox: "read-only"`,
   `baseInstructions`, `developerInstructions`, and `dynamicTools`.
   `environments`, `runtimeWorkspaceRoots`, and `selectedCapabilityRoots` are
   explicitly empty. The result contains `thread.id`.
4. Each dynamic function requires `{type: "function", name, description,
   inputSchema}`. `deferLoading` is optional. This installed schema also
   supports namespace groups; the Jarvis adapter uses ordinary functions.
5. `turn/start` takes `{threadId, input, effort?}` and returns `turn.id`.
   Text input is `{type: "text", text}`. Image input is
   `{type: "image", url: "data:image/png;base64,..."}`; `localImage` with a
   `path` also exists in the schema but is unnecessary for Jarvis images.
6. `item/tool/call` is a **server request**. Its params are
   `{threadId, turnId, callId, tool, arguments, namespace?}`. The tested
   `arguments` is already an object. Reply to the outer request `id`, not to
   `callId`, with `{result: {contentItems, success: true}}`.
   Text output is `{type: "inputText", text}`; image output is
   `{type: "inputImage", imageUrl: "data:image/png;base64,..."}`.
7. Assistant text arrives via `item/agentMessage/delta` with `params.delta`.
   `item/completed` carries the final item; `turn/completed` carries
   `params.turn.status` and optional error information.
8. Cancellation is `turn/interrupt` with the exact `{threadId, turnId}`.
   `thread/unsubscribe` takes `{threadId}` after completion or cancellation.

Jarvis's ToolUseLoop executes a tool only after the provider's stream ends.
Therefore the adapter yields the dynamic call and ends that stream, retaining
the pending RPC. On the next `complete()` with the same `tool_loop_id`, it
responds using the matching tool result and continues the same Codex turn.
Waiting for ToolExecutor inside the first stream would deadlock.

## Native capability boundary

The process disables shell/unified exec, shell snapshots, browser/computer use,
apps/plugins, hooks, memories, skills, image generation, and related native
features. Web search is disabled. Project documentation and bundled skill
instructions are disabled; request system context is supplied by Jarvis.
Native approvals are declined and unexpected server requests are rejected.

`features.code_mode_host` must stay enabled: disabling it also breaks dynamic
tool dispatch, even with `code_mode` disabled. Keeping the dispatch host enabled
does not grant shell or other disabled native capabilities.

**MCP configuration merges:** `-c mcp_servers={}` does not remove inherited
servers. The transport first performs only initialize/config-read, obtains
server names in memory, stops that child, and starts a child with one
`-c 'mcp_servers.<name>.enabled=false'` per inherited server, after validating
names as letters, digits, underscores or hyphens. Quoted path segments do not
work in the CLI override parser. It verifies all
servers are disabled before any thread or model turn. It never calls the MCP
inventory endpoint. A probe attempt to query that endpoint was rejected by
automatic approval review because inventory may initialize external servers;
the implementation avoids that operation.

Child API-key environment variables are removed, `forced_login_method` is
`chatgpt`, and there is no API-key fallback. HOME/CODEX_HOME remain unchanged.
The CLI retains its own configured subscription endpoint: overriding
`chatgpt_base_url` with a presumed public default caused HTTP 401 in the live
adapter check, while preserving the native endpoint passed immediately.
Ephemeral threads, disabled history, and private temporary log/SQLite paths
keep the app's own transport artifacts under its configured runtime data
directory. A cross-platform ProcessTree contains the child; disconnection,
request timeout, cancellation, and shutdown resolve or terminate pending work.

Official background: [App Server](https://learn.chatgpt.com/docs/app-server)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
The generated schema from the installed binary is the authority for the exact
experimental field shapes above.
