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

Each completion is one self-contained `muse exec --json` run:

```
muse exec --json --disable-request-user-input --approval-mode never \
  --workspace <session cwd> [--model <id>] --prompt-file <file>
```

Sandbox stays ON, approval prompts are off (non-interactive), and
`request_user_input` is withheld so a run can never hang waiting on Hermes.
The run uses its own tools internally; Hermes receives the final text.

MSP upgrade (v0.2.0): each completion is one MSP turn on a fresh
`muse serve` process (`initialize → initialized → session/start` in
denyUnmatched mode → `turn/start` → live `item/delta` → `turn/completed`
→ `session/leave`). Text deltas stream as they arrive and reasoning/tool
items surface in the reasoning trace. If the MSP turn fails, the provider
falls back to `muse exec` for the same prompt.

## v1 limits

- No Hermes-side tool loop: Hermes tool schemas are ignored (logged) and the
  completion ends `stop` with final text. Upgrade path: speak MSP over
  `muse serve` stdio for session continuity.
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

## License

MIT
