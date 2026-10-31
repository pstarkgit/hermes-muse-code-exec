"""Muse Code Exec provider — Muse Spark billed to the monthly sub.

Drives the real `muse` CLI (the attested Muse Code client) as the model
backend — the external-process pattern of the bundled `copilot-acp`
provider. Meta accepts the subscription key only from the Muse Code client
(HTTP 403 otherwise), so this provider never touches the Model API.

Transport: MSP over `muse serve` stdio first (live streaming, one process
per turn, session in denyUnmatched approval mode, request_user_input
withheld). Any MSP failure falls back to one `muse exec --json` run for the
same turn, so a protocol regression never breaks the provider.
"""

import contextlib
import json
import logging
import os
import queue
import random
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

ENV_COMMAND_VARS = ("HERMES_MUSE_CODE_EXEC_COMMAND", "MUSE_CLI_PATH")
ENV_APPROVAL_MODE = "HERMES_MUSE_CODE_EXEC_APPROVAL_MODE"
ENV_NO_SANDBOX = "HERMES_MUSE_CODE_EXEC_NO_SANDBOX"
ENV_EXEC_APPROVAL = "HERMES_MUSE_CODE_EXEC_EXEC_APPROVAL"
PROFILE_NAME = "muse-code-exec"
EXEC_MARKER_BASE_URL = "exec://muse-code"

# Standards first, then contributor tiers. Contributor bills to the sub too
# (the attested CLI is allowed to use it), but it permits training on your
# data — the order reflects the recommendation, and the choice is yours.
CURATED_MODELS = ("muse-spark-1.3", "muse-spark-1.2", "muse-spark-1.1",
                  "muse-spark-1.3-contributor", "muse-spark-1.2-contributor")

try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError:
    _HERMES_AVAILABLE = False
else:
    _HERMES_AVAILABLE = True


def uuid7() -> str:
    ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    return "%08x-%04x-%04x-%04x-%012x" % (
        (ms >> 16) & 0xFFFFFFFF, ms & 0xFFFF,
        0x7000 | random.getrandbits(12),
        0x8000 | random.getrandbits(14),
        random.getrandbits(48))


def resolve_command() -> str:
    for var in ENV_COMMAND_VARS:
        override = os.getenv(var, "").strip()
        if override:
            return override
    return shutil.which("muse") or "muse"


def _harness_home() -> Path:
    """The Hermes home whose config owns this turn's policy."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        override = os.getenv("HERMES_HOME", "").strip()
        if override:
            return Path(override)
        return Path(os.path.expanduser("~")) / ".hermes"


def _read_harness_approvals_mode() -> str | None:
    """The harness's own approvals posture (`approvals.mode`), or None when
    unreadable. The provider never invents policy; it only carries this."""
    try:
        from hermes_cli.config_effective import load_user_config_effective
        cfg = load_user_config_effective(
            _harness_home() / "config.yaml") or {}
    except Exception:
        return None
    approvals = cfg.get("approvals")
    mode = approvals.get("mode") if isinstance(approvals, dict) else None
    return str(mode).strip().lower() or None


def _msp_approval_mode() -> str | None:
    """MSP session mode dictated by the harness. Explicit env wins; else the
    profile's approvals.mode (off → allowAll, anything else → denyUnmatched);
    None only when nothing is readable (caller then omits the field and the
    host default applies)."""
    override = os.getenv(ENV_APPROVAL_MODE, "").strip()
    if override:
        return override
    harness = _read_harness_approvals_mode()
    if harness is None:
        return None
    return "allowAll" if harness == "off" else "denyUnmatched"


def _exec_approval_flag() -> list[str]:
    """`muse exec` approval flag dictated by the harness. Explicit env wins;
    else off → never; any stricter harness posture also resolves to never
    because an exec turn cannot answer an interactive prompt (it denies
    instead of hanging)."""
    override = os.getenv(ENV_EXEC_APPROVAL, "").strip()
    if override:
        return ["--approval-mode", override]
    return ["--approval-mode", "never"]


def _serve_extra_flags() -> list[str]:
    """Host flags dictated by the harness. Sandbox stays ON unless the
    harness explicitly opts out via HERMES_MUSE_CODE_EXEC_NO_SANDBOX=1:
    sandbox posture is a harness decision, never the model's."""
    if os.getenv(ENV_NO_SANDBOX, "").strip() in ("1", "true", "yes"):
        return ["--disable-sandbox"]
    return []


def _agent_cwd(exec_cwd: str | None) -> str:
    try:
        from agent.runtime_cwd import resolve_agent_cwd
        return str(Path(exec_cwd or str(resolve_agent_cwd())).resolve())
    except Exception:
        return str(Path(exec_cwd or os.path.expanduser("~")).resolve())


def login_healthy(timeout: float = 20.0) -> bool:
    """True when the Muse subscription login works (no inference spent)."""
    try:
        proc = subprocess.run(
            [resolve_command(), "usage"],
            capture_output=True, text=True, timeout=timeout,
            env=os.environ.copy())
    except Exception:
        return False
    return proc.returncode == 0


TURN_PREAMBLE = (
    "You are answering through Hermes via the muse-code-exec provider. "
    "This turn has no Hermes-side tools: use your own tools only. "
    "Never advise the user to disable sandboxes, approvals, or permission "
    "profiles, and never claim a Hermes gateway setting changes this "
    "session. If a task needs tools or permissions you lack, say so "
    "plainly and stop."
)


def _render_messages(messages: list[dict]) -> str:
    """Flatten Hermes chat messages into one prompt for a fresh turn."""
    parts: list[str] = []
    for msg in messages or []:
        role = str(msg.get("role") or "user").strip().lower()
        content = msg.get("content")
        if isinstance(content, list):
            texts = [str(p.get("text") or "") for p in content
                     if isinstance(p, dict) and p.get("type") in (
                         "text", "input_text")]
            text = "\n".join(t for t in texts if t)
        else:
            text = str(content or "")
        if not text.strip():
            continue
        label = {"system": "System", "user": "User", "assistant": "Assistant",
                 "tool": "Tool result"}.get(role, "User")
        parts.append(f"{label}:\n{text}")
    return "\n\n".join(parts)


def run_exec_completion(prompt_text: str, *, model: str | None, cwd: str,
                        timeout_seconds: float) -> str:
    """Fallback: one `muse exec --json` turn; return its final text."""
    args = ([resolve_command(), "exec", "--json",
             "--disable-request-user-input"] + _exec_approval_flag() +
            ["--workspace", cwd])
    requested = str(model or "").strip()
    if requested and requested != PROFILE_NAME:
        args += ["--model", requested]
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".prompt.md",
                                     delete=False, encoding="utf-8") as fh:
        fh.write(prompt_text)
        prompt_path = fh.name
    args += ["--prompt-file", prompt_path]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True,
            timeout=max(float(timeout_seconds), 1.0),
            env=os.environ.copy(), cwd=cwd)
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"muse exec failed (exit {proc.returncode}): {tail}")
    deltas: list[str] = []
    final_text: str | None = None
    for line in (proc.stdout or "").splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        payload = event.get("payload") or {}
        kind = str(payload.get("kind") or "")
        if kind == "run_output_delta" and payload.get("text"):
            deltas.append(str(payload["text"]))
        elif kind == "run_terminal":
            if payload.get("terminal") != "completed":
                raise RuntimeError(
                    f"muse exec ended terminal={payload.get('terminal')!r}: "
                    f"{str(payload.get('text') or '')[:500]}")
            final_text = str(payload.get("text") or "")
    text = final_text if final_text is not None else "".join(deltas)
    if not text.strip():
        raise RuntimeError("muse exec returned no text.")
    return text


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id,
            "error": {"code": code, "message": message}}


def _collect_model_ids(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        mid = node.get("modelId", node.get("model_id", node.get("id")))
        if isinstance(mid, str) and mid and mid not in out:
            out.append(mid)
        for value in node.values():
            _collect_model_ids(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_model_ids(value, out)


def _answer_client_request(msg: dict[str, Any], *,
                           process: subprocess.Popen[str]) -> bool:
    """Answer host→client requests; True when fully handled.

    Approval traffic is never answered: denyUnmatched resolves it
    server-side (observed), and an error reply stalls the turn. Those
    messages flow to the turn's private queue and are ignored there.
    """
    method = msg.get("method")
    if not isinstance(method, str) or "id" not in msg:
        return False
    if method.startswith("approval/"):
        return False
    try:
        process.stdin.write(json.dumps(_jsonrpc_error(
            msg.get("id"), -32601,
            f"muse-code-exec is non-interactive and denies {method!r}; "
            "request_user_input is withheld and policy comes from the "
            "harness approvals posture, never this session.")) + "\n")
        process.stdin.flush()
    except Exception:
        pass
    return True


class _ServeSession:
    """One `muse serve` process with multiplexed requests and a private
    notification queue (concurrent turns never see each other's events)."""

    def __init__(self, proc: subprocess.Popen[str], deadline: float,
                 on_release: Callable[[], None]):
        self._proc = proc
        self._deadline = deadline
        self._on_release = on_release
        self.inbox: queue.Queue[str] = queue.Queue()
        self.notes: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=40)
        self._waiters: dict[str, queue.Queue] = {}
        self._waiters_lock = threading.Lock()
        threading.Thread(target=self._pump, args=(proc.stdout, self.inbox.put),
                         daemon=True).start()
        self._stderr_pump = threading.Thread(
            target=self._pump,
            args=(proc.stderr, lambda line: self.stderr_tail.append(
                line.rstrip("\n"))),
            daemon=True)
        self._stderr_pump.start()
        threading.Thread(target=self._dispatch, daemon=True).start()

    @staticmethod
    def _pump(stream, sink) -> None:
        for line in stream or ():
            sink(line)

    def _dispatch(self) -> None:
        """Continuously route raw lines: responses to their waiter,
        notifications to the notes queue. Runs for the proc lifetime so
        events are never stranded between requests."""
        while True:
            line = self.inbox.get()
            if line is None:
                return
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if _answer_client_request(msg, process=self._proc):
                continue
            if isinstance(msg, dict) and "method" in msg:
                self.notes.put(msg)
                continue
            if isinstance(msg, dict) and msg.get("id") is not None:
                with self._waiters_lock:
                    waiter = self._waiters.pop(msg.get("id"), None)
                if waiter is not None:
                    waiter.put(msg)
                    continue
            self.notes.put(msg)

    def request(self, method: str, params: dict[str, Any],
                timeout: float | None = None) -> Any:
        proc = self._proc
        request_id = uuid.uuid4().hex
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._waiters_lock:
            self._waiters[request_id] = waiter
        try:
            proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "id": request_id,
                 "method": method, "params": params}) + "\n")
            proc.stdin.flush()
        except Exception:
            with self._waiters_lock:
                self._waiters.pop(request_id, None)
            raise
        end = min(time.monotonic() + (timeout or 60.0), self._deadline)
        while time.monotonic() < end:
            if proc.poll() is not None:
                break
            try:
                msg = waiter.get(timeout=0.2)
            except queue.Empty:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise RuntimeError(
                    f"muse MSP {method} failed: {err.get('message') or err}")
            return msg.get("result")
        with self._waiters_lock:
            self._waiters.pop(request_id, None)
        if proc.poll() is not None:
            self._stderr_pump.join(timeout=1.0)
            stderr_text = "\n".join(self.stderr_tail).strip()
            raise RuntimeError(
                "muse serve exited early: "
                f"{stderr_text or f'exit code {proc.returncode}'}")
        raise TimeoutError(
            f"Timed out waiting for muse MSP response to {method}.")

    def poll_note(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self.notes.get(timeout=max(timeout, 0.05))
        except queue.Empty:
            return None

    def release(self) -> None:
        self._on_release()


def _msp_turn(prompt_text: str, *, model_id: str, cwd: str,
              timeout_seconds: float,
              on_delta: Callable[[str], None] | None,
              track: Callable[[subprocess.Popen[str]], None]) -> tuple[str,
                                                                      list[str]]:
    """One MSP turn on a fresh `muse serve` process. Raises on any failure
    so the caller can fall back to `muse exec`."""
    try:
        proc = subprocess.Popen(
            [resolve_command(), "serve", "--disable-request-user-input",
             "--no-session-log"] + _serve_extra_flags(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1, cwd=cwd, env=os.environ.copy())
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not start Muse serve command. Install Muse Code or set "
            "HERMES_MUSE_CODE_EXEC_COMMAND/MUSE_CLI_PATH.") from exc
    if proc.stdin is None or proc.stdout is None:
        proc.kill()
        raise RuntimeError("Muse serve process did not expose stdio pipes.")
    track(proc)
    deadline = time.monotonic() + max(float(timeout_seconds), 1.0)
    session = _ServeSession(proc, deadline, on_release=lambda: None)
    try:
        session.request("initialize", {"clientInfo": {
            "name": "hermes_muse_code_exec", "version": "0.3.0"}},
            timeout=30.0)
        proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "initialized"}) + "\n")
        proc.stdin.flush()
        params: dict[str, Any] = {"commandId": uuid7(), "modelId": model_id,
                                  "workspaceRoot": cwd}
        harnessed = _msp_approval_mode()
        if harnessed:
            params["approvalMode"] = harnessed
        try:
            started = session.request("session/start", params)
        except RuntimeError as exc:
            if ("commandRejected" in str(exc) or "-32030" in str(exc)) \
                    and "approvalMode" in params:
                params = dict(params, commandId=uuid7())
                params.pop("approvalMode", None)
                started = session.request("session/start", params)
            else:
                raise
        session_id = started["session"]["sessionId"]
        accepted = session.request("turn/start", {
            "commandId": uuid7(), "sessionId": session_id,
            "displayText": prompt_text[:200],
            "input": [{"type": "text", "text": prompt_text}]})
        turn_id = str((accepted or {}).get("turnId") or "")
        deltas: dict[str, list[str]] = {}
        final_text: str | None = None
        activity: list[str] = []
        ok = False
        while time.monotonic() < deadline:
            note = session.poll_note(
                timeout=min(deadline - time.monotonic(), 30.0))
            if note is None:
                continue
            method = note.get("method")
            p = note.get("params") or {}
            if p.get("sessionId") != session_id:
                continue
            if method and method.startswith("approval/"):
                continue
            if method == "item/delta":
                if p.get("field", "text") == "text":
                    text = str(p.get("delta") or "")
                    if text:
                        deltas.setdefault(str(p.get("itemId")), []).append(text)
                        if on_delta:
                            on_delta(text)
                elif str(p.get("field") or "").startswith("summary"):
                    text = str(p.get("delta") or "")
                    if text:
                        activity.append(f"reasoning: {text[:200]}")
                        if on_delta:
                            on_delta(text)
            elif method == "item/completed":
                item = p.get("item") or {}
                kind = str(item.get("kind") or "")
                if kind == "agentMessage" and \
                        str(item.get("turnId") or "") in (turn_id, ""):
                    final_text = str(item.get("text") or "") or None
                    if final_text is None:
                        joined = "".join(
                            deltas.get(str(item.get("itemId")), []))
                        final_text = joined or None
                elif kind and kind not in ("userMessage", "agentMessage"):
                    detail = str(item.get("status") or item.get("title")
                                  or "")[:120]
                    activity.append(f"{kind}: {detail}".rstrip(": "))
            elif method == "turn/completed":
                if str(p.get("turnId") or "") not in (turn_id, ""):
                    continue
                if p.get("terminal") != "completed":
                    raise RuntimeError(
                        f"muse turn ended terminal={p.get('terminal')!r}")
                ok = True
                break
            elif method == "turn/failed":
                raise RuntimeError(
                    f"muse turn failed: {json.dumps(p)[:300]}")
        if not ok:
            with contextlib.suppress(Exception):
                session.request("turn/cancel", {
                    "commandId": uuid7(), "sessionId": session_id,
                    "turnId": turn_id}, timeout=10.0)
            raise TimeoutError("Timed out waiting for muse turn/completed.")
        text = final_text or "".join(
            part for parts in deltas.values() for part in parts)
        if not text.strip():
            raise RuntimeError("muse MSP turn returned no text.")
        with contextlib.suppress(Exception):
            session.request("session/leave", {
                "commandId": uuid7(), "sessionId": session_id}, timeout=10.0)
        return text, [a for a in activity if a]
    finally:
        track(None, proc)


if _HERMES_AVAILABLE:
    from agent.acp_openai_bridge import (
        completion_to_stream_chunks as _completion_to_stream_chunks,
    )

    class MuseCodeExecProfile(ProviderProfile):
        """Muse Spark via the attested Muse Code CLI (external process)."""

        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_SKIP_ASYNC_WRAP = True

        def create_client(self, **client_kwargs: Any) -> Any:
            return MuseCodeExecClient(**client_kwargs)

        def fetch_models(self, *, api_key: str | None = None,
                         base_url: str | None = None,
                         timeout: float = 15.0) -> list[str] | None:
            if not shutil.which(resolve_command()) and not os.getenv(
                    ENV_COMMAND_VARS[0], "").strip() and not os.getenv(
                    ENV_COMMAND_VARS[1], "").strip():
                return None
            try:
                if login_healthy(timeout=timeout):
                    return list(CURATED_MODELS)
            except Exception:
                pass
            return None

    class MuseCodeExecClient:
        """OpenAI-client-compatible facade: MSP first, exec fallback."""

        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_SKIP_ASYNC_WRAP = True

        def __init__(self, *, api_key: str | None = None,
                     base_url: str | None = None,
                     default_headers: dict[str, str] | None = None,
                     exec_cwd: str | None = None, **_: Any):
            self.api_key = api_key or PROFILE_NAME
            self.base_url = base_url or EXEC_MARKER_BASE_URL
            self._default_headers = dict(default_headers or {})
            self._cwd = _agent_cwd(exec_cwd)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create_chat))
            self.is_closed = False
            self._active_processes: set[subprocess.Popen[str]] = set()
            self._active_process_lock = threading.Lock()

        @staticmethod
        def _terminate_process(proc: subprocess.Popen[str]) -> None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                with contextlib.suppress(Exception):
                    proc.kill()

        def _track(self, proc: subprocess.Popen[str] | None,
                   old: subprocess.Popen[str] | None = None) -> None:
            with self._active_process_lock:
                if old is not None:
                    self._active_processes.discard(old)
                if proc is not None:
                    self._active_processes.add(proc)
                    self.is_closed = False
                if not self._active_processes:
                    self.is_closed = True
            if old is not None:
                self._terminate_process(old)

        def close(self) -> None:
            with self._active_process_lock:
                procs, self._active_processes = tuple(
                    self._active_processes), set()
            self.is_closed = True
            for proc in procs:
                self._terminate_process(proc)

        def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
            return list(CURATED_MODELS)

        def _create_chat(self, *, model: str | None = None,
                         messages: list[dict] | None = None,
                         timeout: float | None = None,
                         tools: list[dict] | None = None,
                         tool_choice: Any = None, stream: bool = False,
                         **_: Any) -> Any:
            if tools:
                logger.debug("muse-code-exec ignores %d Hermes tool "
                             "schema(s): the turn uses its own tools.",
                             len(tools))
            prompt_text = _render_messages(list(messages or []))
            if not prompt_text.strip():
                raise RuntimeError("muse-code-exec received an empty prompt.")
            prompt_text = TURN_PREAMBLE + "\n\n" + prompt_text
            try:
                budget = float(timeout) if timeout else 600.0
            except (TypeError, ValueError):
                budget = 600.0
            budget = max(budget, 1.0)
            requested = str(model or "").strip() or None
            model_id = requested or CURATED_MODELS[0]
            label = str(model or PROFILE_NAME)
            if stream:
                return self._stream_turn(
                    prompt_text, model_id=model_id, model_label=label,
                    timeout_seconds=budget)
            text, activity = self._run_turn(
                prompt_text, model_id=model_id,
                timeout_seconds=budget, on_delta=None)
            message = SimpleNamespace(
                content=text, tool_calls=[],
                reasoning="\n".join(activity) or None,
                reasoning_content="\n".join(activity) or None,
                reasoning_details=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message,
                                         finish_reason="stop")],
                usage=SimpleNamespace(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
                model=label)

        def _completion_namespace(self, text: str, activity: list[str],
                                  label: str) -> SimpleNamespace:
            message = SimpleNamespace(
                content=text, tool_calls=[],
                reasoning="\n".join(activity) or None,
                reasoning_content="\n".join(activity) or None,
                reasoning_details=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message,
                                         finish_reason="stop")],
                usage=SimpleNamespace(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
                model=label)

        def _run_turn(self, prompt_text: str, *, model_id: str,
                      timeout_seconds: float,
                      on_delta: Callable[[str], None] | None
                      ) -> tuple[str, list[str]]:
            """MSP first; exec fallback on any transport failure."""
            try:
                return _msp_turn(
                    prompt_text, model_id=model_id, cwd=self._cwd,
                    timeout_seconds=timeout_seconds, on_delta=on_delta,
                    track=self._track_msp)
            except Exception as exc:  # noqa: BLE001
                logger.warning("muse-code-exec MSP turn failed (%s); "
                               "falling back to muse exec.", exc)

            def _collect(text: str) -> None:
                if on_delta and text:
                    on_delta(text)

            text = run_exec_completion(
                prompt_text, model=model_id if model_id != PROFILE_NAME
                else None, cwd=self._cwd, timeout_seconds=timeout_seconds)
            _collect(text)
            return text, ["transport: exec-fallback"]

        def _track_msp(self, proc: subprocess.Popen[str] | None,
                       old: subprocess.Popen[str] | None = None) -> None:
            self._track(proc, old)

        def _stream_turn(self, prompt_text: str, *, model_id: str,
                         model_label: str, timeout_seconds: float):
            chunks: queue.Queue = queue.Queue()
            outcome: dict[str, Any] = {}

            def _runner():
                try:
                    self._run_turn(
                        prompt_text, model_id=model_id,
                        timeout_seconds=timeout_seconds,
                        on_delta=lambda t: chunks.put(("delta", t)))
                    chunks.put(("done", "stop"))
                except Exception as exc:  # noqa: BLE001
                    outcome["error"] = exc
                    chunks.put(("done", "error"))

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            while True:
                kind, payload = chunks.get()
                if kind == "delta":
                    yield SimpleNamespace(choices=[SimpleNamespace(
                        index=0, delta=SimpleNamespace(
                            role="assistant", content=payload, tool_calls=None,
                            reasoning_content=None, reasoning=None),
                        finish_reason=None)], model=model_label, usage=None)
                else:
                    break
            thread.join(timeout=10)
            if outcome.get("error") is not None:
                raise outcome["error"]
            yield SimpleNamespace(choices=[SimpleNamespace(
                index=0, delta=SimpleNamespace(
                    role="assistant", content=None, tool_calls=None,
                    reasoning_content=None, reasoning=None),
                finish_reason="stop")], model=model_label, usage=None)

    muse_code_exec = MuseCodeExecProfile(
        name=PROFILE_NAME,
        display_name="Muse Code (subscription)",
        description="Muse Spark billed to the Muse Code monthly login "
                    "via the attested CLI (MSP session, exec fallback)",
        signup_url="https://github.com/TheStreamCode/hermes-muse-code",
        env_vars=(),
        base_url=EXEC_MARKER_BASE_URL,
        auth_type="external_process",
        api_mode="chat_completions",
        supports_vision=False,
        default_max_tokens=16384,
        process_command="muse",
        process_args=("serve",),
        process_command_env_vars=ENV_COMMAND_VARS,
        fallback_models=("muse-spark-1.3",),
    )

    register_provider(muse_code_exec)


def register(ctx) -> None:
    """PluginManager entry point (also required by `hermes plugins validate`).

    Provider registration already happened at import; nothing cached to
    refresh (the CLI owns auth per spawned process).
    """
