# muse-code-exec

Muse Spark in Hermes, billed to your **Muse Code monthly subscription**.

Meta accepts the subscription key only from the Muse Code client (HTTP 403
`client_attestation` otherwise), so this provider never touches the Model API
directly. It drives the real `muse` CLI (`muse exec`, the attested client) as
its model backend — the same external-process pattern as the bundled
`copilot-acp` provider.

## Requirements

- Hermes Agent >= 0.21.3
- `muse` on PATH (or `HERMES_MUSE_CODE_EXEC_COMMAND` / `MUSE_CLI_PATH`)
- A working Muse Code login (`muse usage` succeeds)

## Usage

```powershell
hermes chat -m muse-spark-1.3 --provider muse-code-exec
```

Or pick provider `muse-code-exec` from `hermes model`. Serves the Muse
Spark family (`muse-spark-1.1` / `1.2` / `1.3`), standards first, then the
`-contributor` tiers. Contributor bills to the sub too, but it permits
training on your data — your choice, in that order.

## How it works

Each completion starts a fresh `muse serve` process and one MSP turn.
If the transport fails before turn submission, it can fall back to one
self-contained `muse exec --json` run:

```
muse exec --json --disable-request-user-input --approval-mode never \
  --workspace <session cwd> [--model <id>] --prompt-file <file>
```

Sandbox stays ON, approval prompts are off (non-interactive), and
`request_user_input` is withheld so a run can never hang waiting on Hermes.
The run uses its own tools internally; Hermes receives the final text.

MSP upgrade (v0.2.0): each completion is one MSP turn on a fresh
`muse serve` process (`initialize → initialized → session/start` →
`turn/start` → live `item/delta` → `turn/completed` → `session/leave`).
Text deltas stream as they arrive and reasoning/tool items surface in the
reasoning trace. Once a turn has been submitted, a failure is surfaced
without automatic replay: tools may already have executed.

Harness control (v0.3.0): sandbox and approval posture are harness
decisions, never the model's. The provider reads the profile's own
`approvals.mode`: `off` starts MSP sessions in `allowAll`, anything else
in `denyUnmatched` (unreadable config falls back to omitting the field).
`muse exec` runs `--approval-mode never` because an exec turn cannot
answer an interactive prompt. Sandbox stays ON; only an explicit
`HERMES_MUSE_CODE_EXEC_NO_SANDBOX=1` in the profile env adds
`--disable-sandbox`. Explicit overrides:
`HERMES_MUSE_CODE_EXEC_APPROVAL_MODE` (MSP session mode),
`HERMES_MUSE_CODE_EXEC_EXEC_APPROVAL` (exec flag),
`HERMES_MUSE_CODE_EXEC_COMMAND` | `muse` binary override (or `MUSE_CLI_PATH`) | `muse` on PATH |

Policy reliability (v0.3.1): streaming preserves the caller's context-local
Hermes profile. The host must confirm a requested approval mode before the
turn starts; a rejected or unconfirmed policy is reported explicitly, never
retried without that policy. The explicit sandbox override applies to both
MSP and exec. Traces record the host-confirmed approval mode, sandbox launch
setting, tool name/status and failure class, without raw tool arguments or
failure text. Past transcript denials do not establish the new turn's access.

Native Hermes tools (v0.5.0): the provider forwards the exact function schemas
Hermes supplies using its maintained `agent.acp_openai_bridge` contract. This
includes injected `message_agent` for internal Bot Chat, configured `a2a_*`
protocol tools, session/delegation/background-process tools, memory and plugin
tools. New enabled tools need no provider allowlist change. Structured calls
return to the actual Hermes AIAgent for native validation and dispatch; the
Muse subprocess is not handed a general board-mutation service.

Protocol A2A, internal `message_agent`, platform `messages_send`, and native
Kanban terminal returns are distinct routes. Telegram delivery proves only
platform messaging. Queued replies retain their native task/run/delivery IDs;
they are not completion evidence.

A native Hermes call (marked through the provider kwargs hook) or an explicitly
supplied tool list, including an empty list, is authoritative:
the provider does not attach a second, broader CLI MCP toolset. If no host tool
surface is supplied, `hermes mcp-serve --tools` remains the standalone curated
fallback, with `HERMES_MUSE_CODE_EXEC_NO_MCP=1` as its existing opt-out.

Host-tool responses are buffered and emitted as structured OpenAI completion
chunks so XML function envelopes are never exposed as user-facing streaming
text. Plain tool-free turns retain streaming and final-only responses. Tool
call IDs, names, arguments and result IDs survive the next prompt. Each child
uses Hermes's profile/session-aware environment builder instead of copying the
multiplexer's default profile environment.

## Limits

- Each completion starts a fresh Muse session; Hermes supplies history.
- Hermes owns tool visibility, role and permission policy. The provider neither
  enables disabled tools nor changes providers, credentials or memory spaces.
- No automatic replay after a turn may have executed effects: the provider
  classifies those errors through Hermes's native retry/fallback hook.
- Native cancellation terminates tracked MSP and exec-fallback children; EOF
  wakes the waiting turn, and late post-cancel processes are refused.
- No vision payloads in the chat envelope; available vision tools remain callable.
- Usage token counters are currently unavailable; zero counters are not measured
  zero usage or proof of token savings.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `HERMES_MUSE_CODE_EXEC_COMMAND` | `muse` binary override (or `MUSE_CLI_PATH`) | `muse` on PATH |
| `HERMES_MUSE_CODE_EXEC_NO_MCP` | `1` disables the Hermes MCP tool bridge | bridge on |

## Troubleshooting

- **Provider shows as unconfigured** — `muse` missing from PATH or `muse
  usage` fails: run `muse login`.
- **Empty prompt refusal** — Hermes sent no text; nothing was billed.
- **Non-zero exit surfaced** — the tail of the CLI stderr is raised, never
  the key (the key never leaves the CLI process).
- **`deny_unmatched`** — inspect the current turn's `policy:` and `toolCall:`
  trace entries. An old denial in conversation history is not a fresh tool
  result. A Hermes Bot Chat does not need replacing to apply a new Muse turn.
- **Approval policy mismatch** — the adapter stopped before starting tools
  because the host did not confirm the harness-selected policy. Inspect the
  active profile's `approvals.mode` and any explicit adapter override.

## License

MIT


### Native Hermes tool continuation (0.5.1)

Hermes completions register the exact current function schemas as a required
per-turn native MCP server. A tool call pauses Muse while the real parent
Hermes agent records and executes it, then returns that exact call ID's result
to the same Muse session. No text-only tool instructions, detached fake agent,
second tool allowlist, or board mutation service is needed. Tools added by
plugins are forwarded when Hermes offers them. Existing session visibility,
credential boundaries and approval policy remain authoritative. Removed tools
are refused on continuation; Hermes's tool discovery/call tools cover deferred
tools. A new provider turn rebuilds the native tool catalog.

The finite loopback MCP endpoint exists only during its completion, rejects
browser origins and unoffered tool names, and closes on completion or cancel.
Native MCP startup failure refuses tool-less fallback. The host execution wait
does not consume the model's response budget. Pending calls cannot be replayed
with a different or missing result. Native parent-loop cancellation remains
supported. Separate shell execution still follows the configured Muse policy;
this bridge does not claim to sandbox arbitrary model behavior.

Verification distinguishes small real MSP round trips from adoption by actual
Bot Chat sessions and from product PR merges.
