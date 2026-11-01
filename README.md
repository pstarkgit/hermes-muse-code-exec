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
`HERMES_MUSE_CODE_EXEC_COMMAND` / `MUSE_CLI_PATH` (binary).

Policy reliability (v0.3.1): streaming preserves the caller's context-local
Hermes profile. The host must confirm a requested approval mode before the
turn starts; a rejected or unconfirmed policy is reported explicitly, never
retried without that policy. The explicit sandbox override applies to both
MSP and exec. Traces record the host-confirmed approval mode, sandbox launch
setting, tool name/status and failure class, without raw tool arguments or
failure text. Past transcript denials do not establish the new turn's access.
On Hermes 0.22.x, PluginManager reload also refreshes the bound home's cached
provider registration, so an updated adapter can take effect on subsequent
turns without restarting a gateway that serves other active profiles.

## v1 limits

- No Hermes-side tool loop: Hermes tool schemas are ignored (logged) and the
  completion ends `stop` with final text. Each turn has a new Muse session;
  Hermes supplies the conversation history.
- One subprocess per completion (same cost profile as `copilot-acp`).
- No vision payloads in v1.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `HERMES_MUSE_CODE_EXEC_COMMAND` | `muse` binary override (or `MUSE_CLI_PATH`) | `muse` on PATH |

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
