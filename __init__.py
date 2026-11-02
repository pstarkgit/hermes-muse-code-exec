"""Muse Code Exec provider — Muse Spark billed to the monthly sub.

Drives the real `muse` CLI (the attested Muse Code client) as the model
backend — the external-process pattern of the bundled `copilot-acp`
provider. Meta accepts the subscription key only from the Muse Code client
(HTTP 403 otherwise), so this provider never touches the Model API.

Transport: MSP over `muse serve` stdio first (live streaming, one process
per turn, harness-selected approval mode, request_user_input withheld).
Hermes supplies its exact authorized tools through the maintained host-tool
bridge; structured calls execute in the real parent AIAgent, not in this
subprocess. When no host tool surface is supplied, the curated session MCP
bridge remains available. Only pre-submission transport failures may fall
back to exec; the same host-tool contract survives that fallback.
"""

import contextlib
import contextvars
import hashlib
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
ENV_NO_MCP = "HERMES_MUSE_CODE_EXEC_NO_MCP"
PROFILE_NAME = "muse-code-exec"
EXEC_MARKER_BASE_URL = "exec://muse-code"


class MusePolicyError(RuntimeError):
    """The host could not establish the harness-selected approval policy."""
    muse_no_automatic_replay = True


class MuseTurnError(RuntimeError):
    """A turn may have executed tools and must not be automatically replayed."""
    muse_no_automatic_replay = True


def _classify_muse_error(error, **_):
    # A plain RuntimeError is retryable in Hermes by default. Carry the
    # no-replay decision through the REAL parent retry/fallback boundary too.
    if getattr(error, "muse_no_automatic_replay", False) is True:
        return {"reason": "format_error", "retryable": False,
                "should_fallback": False, "should_rotate_credential": False,
                "should_compress": False,
                "error_context": {"muse_no_automatic_replay": True}}
    return None

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


def _mcp_enabled() -> bool:
    """Hermes MCP bridge on/off. On unless the harness explicitly opts out
    via HERMES_MUSE_CODE_EXEC_NO_MCP=1."""
    return os.getenv(ENV_NO_MCP, "").strip() not in ("1", "true", "yes")


def _msp_mcp_servers() -> dict[str, Any]:
    """Native Hermes tools for this MSP turn, served over MCP.

    Attaches `hermes mcp-serve --tools` (conversation bridge + skills +
    web/vision tools) as a session MCP server so the model calls real
    Hermes tools — messages_send, channels_list, skills_list, skill_view
    and the rest — instead of working tool-less. The server runs with the
    ambient profile home, so the turn sees the calling profile's
    conversations and skills. Requires the mcp-serve plugin installed in
    that profile. Failures refuse loudly (mode required) so a turn never
    silently runs without the bridge; only transport failures before turn
    submission may fall back to exec, which stays tool-less.
    """
    if not _mcp_enabled():
        return {}
    return {"hermes": {
        "transport": "stdio",
        "command": shutil.which("hermes") or "hermes",
        "args": ["mcp-serve", "--tools"],
        "mode": "required"}}


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


def _build_subprocess_env() -> dict[str, str]:
    """Pin the real calling profile and session; never inherit a sibling's env."""
    from tools.environments.local import served_profile_child_env
    return served_profile_child_env(
        target_home=_harness_home(), inherit_credentials=False)


def _host_tools(tools: list[dict] | None) -> list[dict]:
    """Carry the caller's exact tool surface, including injected/plugin tools.

    The host dispatches these calls in its actual agent context. No new tool is
    granted by the provider. Hermes, not a second adapter allowlist, owns role
    visibility and dispatch. The Muse subprocess never executes these calls.
    """
    selected = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        selected.append(tool)
    return selected


def _native_context_digest(messages, system_text):
    """Pin the already-submitted prefix without retaining another history copy."""
    try:
        encoded = json.dumps([system_text, messages], ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise MusePolicyError('Native context is not JSON; no context replay submitted.') from exc
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


class _HostToolMCP:
    """Finite per-completion MCP admission endpoint; never executes a tool.

    Native MCP calls become normal Hermes function calls. Their HTTP requests
    stay pending while Hermes persists and executes them in the real agent.
    The next completion returns the actual matching result to the same Muse
    session, preserving native call state without a replay or fabricated result.
    No agent reference, credential or board mutation is exposed to this server.
    """

    def __init__(self, tools):
        self.tools = {t['function']['name']: t['function'] for t in tools}
        self.calls = queue.Queue()
        self.results = queue.Queue()
        self.call_serial = threading.Lock()
        self.requests = {}
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.wait_started = None
        self.wait_total = 0.0
        self.steer_ready = threading.Event()
        self.steer = None
        self.path = '/mcp/' + uuid.uuid4().hex
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Never log the ephemeral capability URL or tool payload.

            def reply(self, status, body=None):
                data = json.dumps(body).encode() if body is not None else b''
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(data)

            def authorized(self):
                # Native local clients have no browser Origin. Reject browser
                # and rebinding requests even if they discover the random path.
                return self.path == owner.path and not self.headers.get('Origin')

            def do_GET(self):
                self.reply(405 if self.authorized() else 403)

            def do_DELETE(self):
                self.reply(405 if self.authorized() else 403)

            def do_POST(self):
                if not self.authorized():
                    self.reply(403); return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 4 * 1024 * 1024:
                        self.reply(413); return
                    self.connection.settimeout(10)
                    msg = json.loads(self.rfile.read(size))
                    if not isinstance(msg, dict):
                        self.reply(400); return
                    method = msg.get('method')
                    ident = msg.get('id')
                    if ident is None:
                        self.reply(202); return
                    params = msg.get('params') or {}
                    if method == 'initialize':
                        result = {'protocolVersion': params.get('protocolVersion', '2025-03-26'),
                                  'capabilities': {'tools': {}},
                                  'serverInfo': {'name': 'hermes-host', 'version': '0.5.1'}}
                    elif method == 'ping':
                        result = {}
                    elif method == 'tools/list':
                        result = {'tools': [{'name': name,
                            'description': fn.get('description', name),
                            'inputSchema': fn.get('parameters') or {'type': 'object', 'properties': {}}}
                            for name, fn in owner.tools.items()]}
                    elif method == 'tools/call':
                        name = params.get('name'); arguments = params.get('arguments', {})
                        if name not in owner.tools or not isinstance(arguments, dict):
                            self.reply(200, _jsonrpc_error(ident, -32602, 'Tool outside current Hermes surface or invalid arguments'))
                            return
                        if not isinstance(ident, (str, int)) or isinstance(ident, bool):
                            self.reply(400); return
                        request_key = json.dumps(ident)
                        immutable = json.dumps(params, sort_keys=True, separators=(',', ':'))
                        with owner.lock:
                            record = owner.requests.get(request_key)
                            first = record is None
                            if first:
                                record = {'immutable': immutable, 'ready': threading.Event(), 'response': None}
                                owner.requests[request_key] = record
                        if record['immutable'] != immutable:
                            self.reply(200, _jsonrpc_error(ident, -32602, 'MCP request identity reused with changed payload')); return
                        if not first:
                            while not record['ready'].wait(0.1) and not owner.closed.is_set():
                                pass
                            self.reply(200, record['response'] or _jsonrpc_error(ident, -32800, 'Hermes turn closed'))
                            return

                        def settle(response):
                            record['response'] = response
                            record['ready'].set()
                            self.reply(200, response)

                        with owner.call_serial:
                            if name not in owner.tools or not isinstance(arguments, dict):
                                settle(_jsonrpc_error(ident, -32602, 'Tool no longer offered by Hermes')); return
                            if owner.closed.is_set():
                                settle(_jsonrpc_error(ident, -32800, 'Hermes turn closed')); return
                            call = {'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                                    'function': {'name': name, 'arguments': json.dumps(arguments)}}
                            owner.wait_started = time.monotonic()
                            owner.calls.put(('call', call))
                            while not owner.closed.is_set():
                                try:
                                    returned = owner.results.get(timeout=0.1)
                                except queue.Empty:
                                    continue
                                if returned['id'] != call['id']:
                                    settle(_jsonrpc_error(ident, -32603, 'Host result identity mismatch')); return
                                owner.wait_total += time.monotonic() - owner.wait_started
                                owner.wait_started = None
                                settle({'jsonrpc': '2.0', 'id': ident, 'result': {
                                    'content': [{'type': 'text', 'text': returned['content']}]}})
                                return
                        settle(_jsonrpc_error(ident, -32800, 'Hermes turn closed'))
                        return
                    else:
                        self.reply(200, _jsonrpc_error(ident, -32601, 'Method unavailable')); return
                    self.reply(200, {'jsonrpc': '2.0', 'id': ident, 'result': result})
                except (ValueError, OSError, TypeError):
                    self.reply(400)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.05}, daemon=True)
        self.thread.start()

    def paused_seconds(self):
        started = self.wait_started
        return self.wait_total + (time.monotonic() - started if started is not None else 0.0)

    def config(self):
        return {'hermes_host': {'transport': 'streamableHttp',
                'url': f'http://127.0.0.1:{self.server.server_port}{self.path}', 'mode': 'required'}}

    def take(self):
        try:
            kind, call = self.calls.get_nowait()
        except queue.Empty:
            return None
        return '<tool_call>' + json.dumps(call) + '</tool_call>'

    def close(self):
        with self.lock:
            if self.closed.is_set():
                return
            self.closed.set()
            self.steer_ready.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


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
    if mode is None:
        return None
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


def _native_scope_cwd() -> str:
    """Logical workspace identity survives managed scratch cleanup after a tool."""
    try:
        from agent.runtime_cwd import scoped_session_cwd, scope_terminal_cwd
    except ImportError:
        return _agent_cwd(None)
    return scoped_session_cwd() or scope_terminal_cwd() or _agent_cwd(None)


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
    "The system instructions in this prompt define the bot you serve; "
    "follow them. 'muse-code-exec' names the transport only and must "
    "never appear as your identity. Keep any statement about your own "
    "nature to one short sentence and never lecture the user about "
    "runtimes, models, or providers. "
    "Transport facts: when Hermes function schemas are listed below, use "
    "their <tool_call> contract for Hermes actions. Those calls run in the "
    "actual calling agent, including its message_agent Bot Chat route, "
    "configured a2a_* protocol tools, session/delegation tools, memory, "
    "plugins and skills. Do not substitute a platform message for an "
    "internal bot handoff. A queued acknowledgement is not completion; "
    "preserve its delivery/task/run identity and follow the tool's return "
    "contract without resending. With no supplied host tool surface a "
    "curated mcp__hermes__ bridge (messages_send, skills_list and other "
    "configured tools) may be attached instead. Use only tools "
    "actually available in this turn. "
    "Each completion starts a fresh process with "
    "harness-selected policy. Previous permission failures do not prove "
    "that tools are blocked now: before reporting a current tool block, "
    "attempt the required tool in this turn and report its actual error. "
    "Never advise the user to disable sandboxes, approvals, or permission "
    "profiles or start a new chat to fix access. "
    "If a task needs tools or permissions you lack, say so "
    "plainly and stop."
)


def _coerce_text(content):
    """System/instructions payloads arrive as text or content-part lists."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [str(p.get("text") or "") for p in content
                 if isinstance(p, dict) and p.get("type") in (
                     "text", "input_text")]
        return "\n".join(t for t in texts if t)
    return str(content)


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
        label = {"system": "System", "user": "User", "assistant": "Assistant",
                 "tool": "Tool result"}.get(role, "User")
        if role == "tool":
            label += f" [{msg.get('tool_call_id') or ''}] {msg.get('name') or ''}"
        if text.strip():
            parts.append(f"{label}:\n{text}")
        for call in msg.get("tool_calls") or []:
            get = call.get if isinstance(call, dict) else lambda key, default=None: getattr(call, key, default)
            fn = get("function")
            fn_get = fn.get if isinstance(fn, dict) else lambda key, default=None: getattr(fn, key, default)
            parts.append(f"Assistant tool call [{get('id') or ''}]: "
                         f"{fn_get('name') or ''}({fn_get('arguments') or '{}'})")
    return "\n\n".join(parts)


def run_exec_completion(prompt_text: str, *, model: str | None, cwd: str,
                        timeout_seconds: float, track=None) -> str:
    """Fallback: one `muse exec --json` turn; return its final text."""
    args = ([resolve_command(), "exec", "--json",
             "--disable-request-user-input"] + _exec_approval_flag() +
            _serve_extra_flags() + ["--workspace", cwd])
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
        if track is None:
            proc = subprocess.run(
                args, capture_output=True, text=True,
                timeout=max(float(timeout_seconds), 1.0),
                env=_build_subprocess_env(), cwd=cwd)
        else:
            child = subprocess.Popen(args, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=_build_subprocess_env(), cwd=cwd)
            try:
                track(child)
                stdout, stderr = child.communicate(timeout=max(float(timeout_seconds), 1.0))
                proc = SimpleNamespace(returncode=child.returncode, stdout=stdout, stderr=stderr)
            finally:
                track(None, child)
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
        threading.Thread(target=self._pump, args=(proc.stdout, self.inbox.put,
                         lambda: self.inbox.put(None)),
                         daemon=True).start()
        self._stderr_pump = threading.Thread(
            target=self._pump,
            args=(proc.stderr, lambda line: self.stderr_tail.append(
                line.rstrip("\n"))),
            daemon=True)
        self._stderr_pump.start()
        self._dispatch_thread = threading.Thread(target=self._dispatch, daemon=True)
        self._dispatch_thread.start()

    @staticmethod
    def _pump(stream, sink, on_end=None) -> None:
        try:
            for line in stream or ():
                sink(line)
        finally:
            if on_end:
                on_end()

    def _dispatch(self) -> None:
        """Continuously route raw lines: responses to their waiter,
        notifications to the notes queue. Runs for the proc lifetime so
        events are never stranded between requests."""
        while True:
            line = self.inbox.get()
            if line is None:
                # Wake an active turn immediately on exit/cancellation. Ending
                # only this dispatcher leaves poll_note asleep until its budget.
                self.notes.put({"method": "_transport/closed", "params": {}})
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
                timeout: float | None = None, *, host_pause: float = 0.0) -> Any:
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
        end = min(time.monotonic() + (timeout or 60.0), self._deadline + max(host_pause, 0.0))
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


def _initialize_params() -> dict[str, Any]:
    """MSP handshake: identity plus the sessionMcp capability that admits
    `config.mcpServers` on session/start (stable surface)."""
    return {"clientInfo": {
        "name": "hermes_muse_code_exec", "version": "0.5.4"},
        "capabilities": {"requestedCapabilities": ["sessionMcp"]}}


def _session_start_params(*, model_id: str, cwd: str,
                          attach_mcp: bool = True) -> dict[str, Any]:
    """session/start params: identity, workspace, harness approval mode and
    the Hermes MCP bridge (omitted only when explicitly disabled)."""
    params: dict[str, Any] = {"commandId": uuid7(), "modelId": model_id,
                              "workspaceRoot": cwd}
    mcp_servers = _msp_mcp_servers() if attach_mcp else {}
    if mcp_servers:
        params["config"] = {"mcpServers": mcp_servers}
    harnessed = _msp_approval_mode()
    if harnessed:
        params["approvalMode"] = harnessed
    return params


def _msp_turn(prompt_text: str, *, model_id: str, cwd: str,
              timeout_seconds: float,
              on_delta: Callable[[str], None] | None,
              track: Callable[[subprocess.Popen[str]], None],
              attach_mcp: bool = True, host_tools=None, host_mcp=None) -> tuple[str,
                                                                      list[str]]:
    """One MSP turn on a fresh `muse serve` process. Raises on any failure
    so the caller can fall back to `muse exec`."""
    try:
        proc = subprocess.Popen(
            [resolve_command(), "serve", "--disable-request-user-input",
             "--no-session-log"] + _serve_extra_flags(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1, cwd=cwd, env=_build_subprocess_env())
    except Exception as exc:
        if host_mcp:
            host_mcp.close()
        raise RuntimeError(
            "Could not start Muse serve command. Install Muse Code or set "
            "HERMES_MUSE_CODE_EXEC_COMMAND/MUSE_CLI_PATH.") from exc
    if proc.stdin is None or proc.stdout is None:
        proc.kill()
        if host_mcp:
            host_mcp.close()
        raise RuntimeError("Muse serve process did not expose stdio pipes.")
    try:
        track(proc)
    except Exception:
        if host_mcp:
            host_mcp.close()
        raise
    deadline = time.monotonic() + max(float(timeout_seconds), 1.0)
    session = _ServeSession(proc, deadline, on_release=lambda: None)
    turn_requested = False
    try:
        session.request("initialize", _initialize_params(), timeout=30.0)
        proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "initialized"}) + "\n")
        proc.stdin.flush()
        params = _session_start_params(model_id=model_id, cwd=cwd,
                                       attach_mcp=attach_mcp)
        if host_mcp:
            params["config"] = {"mcpServers": host_mcp.config()}
        harnessed = params.get("approvalMode")
        try:
            started = session.request("session/start", params)
        except RuntimeError as exc:
            raise MusePolicyError(
                f"Muse could not start with approval mode {harnessed!r}: {exc}") from exc
        state = started["session"]
        effective = (state.get("approvalMode") or {}).get("mode")
        if harnessed and effective != harnessed:
            raise MusePolicyError(
                f"Muse approval policy mismatch: requested {harnessed!r}, "
                f"host confirmed {effective!r}; no tools were started.")
        logger.info("Muse MSP policy: profile=%s requested=%s effective=%s sandbox=%s",
                    _harness_home().name, harnessed, effective,
                    "disabled" if _serve_extra_flags() else "enabled")
        session_id = state["sessionId"]
        # Even a lost turn/start reply is ambiguous: the host may have admitted it.
        turn_requested = True
        accepted = session.request("turn/start", {
            "commandId": uuid7(), "sessionId": session_id,
            "displayText": prompt_text[:200],
            "input": [{"type": "text", "text": prompt_text}]})
        turn_id = str((accepted or {}).get("turnId") or "")
        if host_mcp:
            def steer(request):
                packet = {**request, "sessionId": session_id, "expectedTurnId": turn_id}
                # Keep the exact admitted/uncertain wire identity on the pending
                # request. A lost ack never permits a new command or turn.
                request.update(packet)
                ack = session.request("turn/steer", packet, timeout=10.0,
                                      host_pause=host_mcp.paused_seconds())
                if (not isinstance(ack, dict) or ack.get("commandId") != packet["commandId"]
                        or ack.get("turnId") != turn_id or ack.get("status") != "accepted"):
                    raise MuseTurnError("Native steer acknowledgement identity was not accepted.")
                return ack
            host_mcp.steer = steer
            host_mcp.steer_ready.set()
        deltas: dict[str, list[str]] = {}
        final_text: str | None = None
        activity: list[str] = [f"policy: approval={effective or 'host-default'} "
                               f"sandbox={'disabled' if _serve_extra_flags() else 'enabled'}"]
        ok = False
        while time.monotonic() < deadline + (host_mcp.paused_seconds() if host_mcp else 0):
            note = session.poll_note(
                timeout=min(deadline + (host_mcp.paused_seconds() if host_mcp else 0) - time.monotonic(), 0.1 if host_mcp else 30.0))
            if note is None:
                continue
            method = note.get("method")
            if method == "_transport/closed":
                raise RuntimeError("Muse serve transport closed before turn completion.")
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
                    if kind == "toolCall":
                        detail += f" tool={str(item.get('tool') or 'unknown')[:80]}"
                        if item.get("failureKind"):
                            detail += f" failure={str(item['failureKind'])[:80]}"
                        # Keep machine evidence, never raw arguments, output,
                        # or arbitrary failure text that can contain secrets.
                        if "deny_unmatched" in str(item.get("failureReason") or ""):
                            detail += " deny_unmatched"
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
    except MusePolicyError:
        raise
    except Exception as exc:
        if turn_requested:
            raise MuseTurnError(
                f"Muse turn failed after submission; automatic replay was stopped: {exc}") from exc
        raise
    finally:
        try:
            track(None, proc)
        finally:
            if host_mcp:
                host_mcp.close()


if _HERMES_AVAILABLE:
    from agent.acp_openai_bridge import (
        completion_to_stream_chunks as _completion_to_stream_chunks,
        extract_tool_calls_from_text as _extract_tool_calls_from_text,
        render_tool_bridge_sections as _render_tool_bridge_sections,
    )

    class MuseCodeExecProfile(ProviderProfile):
        """Muse Spark via the attested Muse Code CLI (external process)."""

        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_SKIP_ASYNC_WRAP = True

        def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
            # Native transport omits `tools` when its authorized surface is
            # empty. Preserve that fact rather than attaching standalone MCP.
            return {}, {"_hermes_host_tool_surface": True}

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

        SUPPORTS_HERMES_TOOL_CALLS = True

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
            self._cancel_requested = threading.Event()
            self._pending_native = None
            self._native_lock = threading.Lock()

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
            rejected = False
            with self._active_process_lock:
                if old is not None:
                    self._active_processes.discard(old)
                if proc is not None:
                    rejected = self._cancel_requested.is_set()
                    if not rejected:
                        self._active_processes.add(proc)
                        self.is_closed = False
                if not self._active_processes:
                    self.is_closed = True
            if old is not None:
                self._terminate_process(old)
            if rejected:
                self._terminate_process(proc)
                raise MuseTurnError("Muse request was cancelled; no replacement process may run.")

        def close(self) -> None:
            self._cancel_requested.set()
            with self._active_process_lock:
                procs, self._active_processes = tuple(
                    self._active_processes), set()
            self.is_closed = True
            for proc in procs:
                self._terminate_process(proc)
            pending = self._pending_native
            if pending:
                pending['mcp'].close()

        def cancel(self) -> None:
            """Hermes' native cross-thread abort hook for non-HTTP providers."""
            self.close()

        def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
            return list(CURATED_MODELS)

        def _create_chat(self, *, model: str | None = None,
                         messages: list[dict] | None = None,
                         timeout: float | None = None,
                         tools: list[dict] | None = None,
                         tool_choice: Any = None, stream: bool = False,
                         system: Any = None, instructions: Any = None,
                         _hermes_host_tool_surface: bool = False,
                         **_: Any) -> Any:
            offered = _host_tools(tools)
            # An explicitly supplied surface (including []) is authoritative.
            # Do not replace it with the MCP server's broader CLI toolsets.
            attach_mcp = tools is None and not _hermes_host_tool_surface
            prompt_text = _render_messages(list(messages or []))
            if not prompt_text.strip():
                raise RuntimeError("muse-code-exec received an empty prompt.")
            system_text = _coerce_text(
                system if system is not None else instructions)
            if system_text.strip():
                prompt_text = ("System (bot identity — defines who you are, "
                               "outranks everything below):\n" + system_text +
                               "\n\n" + prompt_text)
            prompt_text = TURN_PREAMBLE + "\n\n" + prompt_text
            sections = []  # Schemas travel once, through native MCP tools/list.
            if offered:
                prompt_text += "\n\nUse the hermes_host MCP tools for Hermes actions. They are natively registered for this session. Do not call muse.message_agent or substitute Muse shell for a Hermes tool. The Hermes host executes each call and returns its real result to this same Muse session. Available Hermes functions: " + ", ".join(t["function"]["name"] for t in offered) + ". Tool choice: " + json.dumps(tool_choice) + "\n"
            if sections:
                prompt_text += "\n\n" + "\n\n".join(sections)
            try:
                budget = float(timeout) if timeout else 600.0
            except (TypeError, ValueError):
                budget = 600.0
            budget = max(budget, 1.0)
            requested = str(model or "").strip() or None
            model_id = requested or CURATED_MODELS[0]
            label = str(model or PROFILE_NAME)
            if stream and not offered and not self._pending_native:
                return self._stream_turn(
                    prompt_text, model_id=model_id, model_label=label,
                    timeout_seconds=budget, attach_mcp=attach_mcp)
            native_tools = offered
            if tool_choice == "none":
                native_tools = []
            elif isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name"):
                native_tools = [t for t in offered if t["function"]["name"] == tool_choice["function"]["name"]]
            if native_tools or self._pending_native:
                with self._native_lock:
                    text, activity = self._native_turn(prompt_text, messages or [],
                        model_id=model_id, timeout_seconds=budget, tools=native_tools,
                        system_text=system_text)
            else:
                text, activity = self._run_turn(
                    prompt_text, model_id=model_id,
                    timeout_seconds=budget, on_delta=None, attach_mcp=attach_mcp)
            # In native MCP mode, only a real MCP admission can request host
            # execution. A final answer quoting XML is ordinary text.
            admitted = offered if "transport: native-mcp-host-call" in activity else []
            completion = self._completion_namespace(text, activity, label, admitted,
                                                     tool_choice=tool_choice)
            return _completion_to_stream_chunks(completion) if stream else completion

        def _native_turn(self, prompt, messages, *, model_id, timeout_seconds, tools,
                         system_text=""):
            from gateway.session_context import get_session_env
            scope = (str(_harness_home().resolve()), self._cwd, _native_scope_cwd(),
                     *(get_session_env(name, '') for name in (
                         'HERMES_SESSION_ID', 'HERMES_SESSION_KEY', 'HERMES_SESSION_PROFILE')))
            pending = self._pending_native
            if pending:
                if pending.get('scope') != scope:
                    self.cancel()
                    raise MuseTurnError('Native caller scope changed; pending turn cancelled, not replayed.')
                result = next((msg for msg in reversed(messages)
                    if msg.get('role') == 'tool' and msg.get('tool_call_id') == pending.get('call_id')), None)
                if result is None or pending['model'] != model_id:
                    self.cancel()
                    raise MuseTurnError('Pending native tool call has no matching host result; no replay submitted.')
                count = pending['context_count']
                if _native_context_digest(messages[:count], system_text) != pending['context_digest']:
                    raise MusePolicyError('Native pending prefix/system context changed; no result released or turn replayed.')
                result_index = next(i for i in range(len(messages) - 1, -1, -1)
                    if messages[i] is result)
                if any(m.get('role') not in ('assistant', 'tool') for m in messages[count:result_index]):
                    raise MusePolicyError('Native context before the matching result is unsupported; original turn retained.')
                updates = messages[result_index + 1:]
                if any(m.get('role') != 'user' or not isinstance(m.get('content'), str)
                       or not m['content'].strip() for m in updates):
                    raise MusePolicyError('Only new text user rows support native steering; original turn retained.')
                update = pending.get('context_update')
                if update and update.get('attempted'):
                    raise MuseTurnError('Native steer outcome is uncertain; original command/payload retained, not replayed.')
                if updates:
                    parts = [{'type': 'text', 'text': row['content']} for row in updates]
                    if update and update['packet']['input'] != parts:
                        raise MusePolicyError('Native steer payload changed; original command retained.')
                    update = update or {'packet': {'commandId': uuid7(), 'input': parts}}
                    pending['context_update'] = update
                    if (not pending['mcp'].steer_ready.wait(timeout=min(timeout_seconds, 10.0))
                            or not callable(pending['mcp'].steer)):
                        raise MuseTurnError('Native steering channel unavailable; original result not released.')
                    update['attempted'] = True
                    try:
                        pending['mcp'].steer(update['packet'])
                    except Exception as exc:
                        raise MuseTurnError('Native steer outcome is uncertain; original command/payload retained, not replayed.') from exc
                    pending['context_update'] = None
                pending['mcp'].tools = {t['function']['name']: t['function'] for t in tools}
                content = result.get('content')
                pending['mcp'].results.put({'id': pending['call_id'],
                    'content': content if isinstance(content, str) else json.dumps(content)})
                pending['call_id'] = None
                pending['context_count'] = len(messages)
                pending['context_digest'] = _native_context_digest(messages, system_text)
            else:
                if self._cancel_requested.is_set():
                    raise MuseTurnError('Muse request cancelled; not submitted.')
                endpoint = _HostToolMCP(tools)
                pending = {'mcp': endpoint, 'call_id': None, 'model': model_id, 'scope': scope,
                           'context_count': len(messages),
                           'context_digest': _native_context_digest(messages, system_text)}
                self._pending_native = pending
                def runner():
                    try:
                        text, activity = self._run_turn(prompt, model_id=model_id,
                            timeout_seconds=timeout_seconds, on_delta=None,
                            attach_mcp=False, host_tools=tools, host_mcp=endpoint)
                        endpoint.calls.put(('final', (text, activity)))
                    except BaseException as exc:
                        endpoint.calls.put(('error', exc))
                context = contextvars.copy_context()
                pending['thread'] = threading.Thread(target=context.run, args=(runner,), daemon=True)
                pending['thread'].start()
            try:
                kind, value = pending['mcp'].calls.get(timeout=timeout_seconds + 15)
            except queue.Empty as exc:
                self.cancel()
                raise MuseTurnError('Native MCP request ended without a host call or final result; not replayed.') from exc
            if kind == 'call':
                pending['call_id'] = value['id']
                return '<tool_call>' + json.dumps(value) + '</tool_call>', ['transport: native-mcp-host-call']
            self._pending_native = None
            pending['mcp'].close()
            pending['thread'].join(timeout=2)
            if kind == 'error':
                raise value
            return value

        def _completion_namespace(self, text: str, activity: list[str],
                                  label: str, tools=None, *, tool_choice=None) -> SimpleNamespace:
            calls, cleaned = _extract_tool_calls_from_text(text) if tools else ([], text)
            allowed = {tool["function"]["name"] for tool in tools or []}
            if calls and (tool_choice == "none" or any(call.function.name not in allowed for call in calls)):
                raise MuseTurnError("Muse returned a host tool call outside the offered tool surface; not dispatched.")
            message = SimpleNamespace(
                content=cleaned, tool_calls=calls,
                reasoning="\n".join(activity) or None,
                reasoning_content="\n".join(activity) or None,
                reasoning_details=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message,
                                         finish_reason="tool_calls" if calls else "stop")],
                usage=SimpleNamespace(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
                model=label)

        def _run_turn(self, prompt_text: str, *, model_id: str,
                      timeout_seconds: float,
                      on_delta: Callable[[str], None] | None,
                      attach_mcp: bool = True, host_tools=None, host_mcp=None,
                      ) -> tuple[str, list[str]]:
            """MSP first; exec fallback only before a turn can execute."""
            if self._cancel_requested.is_set():
                raise MuseTurnError("Muse request was cancelled; not submitted.")
            try:
                return _msp_turn(
                    prompt_text, model_id=model_id, cwd=self._cwd,
                    timeout_seconds=timeout_seconds, on_delta=on_delta,
                    track=self._track_msp, attach_mcp=attach_mcp, host_tools=host_tools, host_mcp=host_mcp)
            except (MusePolicyError, MuseTurnError):
                # Preserve policy and any actions that may already have happened.
                raise
            except Exception as exc:  # noqa: BLE001
                if host_tools:
                    raise MusePolicyError("Native Hermes MCP tools could not attach; tool-less fallback refused.") from exc
                logger.warning("muse-code-exec MSP turn failed (%s); "
                               "falling back to muse exec.", exc)

            def _collect(text: str) -> None:
                if on_delta and text:
                    on_delta(text)

            if self._cancel_requested.is_set():
                raise MuseTurnError("Muse request was cancelled; exec fallback not submitted.")
            try:
                text = run_exec_completion(
                    prompt_text, model=model_id if model_id != PROFILE_NAME
                    else None, cwd=self._cwd, timeout_seconds=timeout_seconds,
                    track=self._track)
            except Exception as exc:
                raise MuseTurnError("Muse exec fallback failed; automatic replay was stopped: " + str(exc)) from exc
            _collect(text)
            return text, ["transport: exec-fallback"]

        def _track_msp(self, proc: subprocess.Popen[str] | None,
                       old: subprocess.Popen[str] | None = None) -> None:
            self._track(proc, old)

        def _stream_turn(self, prompt_text: str, *, model_id: str,
                         model_label: str, timeout_seconds: float,
                         attach_mcp: bool = True):
            chunks: queue.Queue = queue.Queue()
            outcome: dict[str, Any] = {}

            def _runner():
                try:
                    text, _ = self._run_turn(
                        prompt_text, model_id=model_id,
                        timeout_seconds=timeout_seconds,
                        on_delta=lambda t: chunks.put(("delta", t)),
                        attach_mcp=attach_mcp)
                    outcome["text"] = text
                    chunks.put(("done", "stop"))
                except Exception as exc:  # noqa: BLE001
                    outcome["error"] = exc
                    chunks.put(("done", "error"))

            # Profile routing and frozen worker policy are context-local in a
            # multiplexed gateway. A bare Thread reads the process default home.
            context = contextvars.copy_context()
            thread = threading.Thread(target=context.run, args=(_runner,), daemon=True)
            thread.start()
            emitted = False
            while True:
                kind, payload = chunks.get()
                if kind == "delta":
                    emitted = True
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
            if not emitted and outcome.get("text"):
                yield SimpleNamespace(choices=[SimpleNamespace(
                    index=0, delta=SimpleNamespace(role="assistant", content=outcome["text"],
                    tool_calls=None, reasoning_content=None, reasoning=None), finish_reason=None)],
                    model=model_label, usage=None)
            yield SimpleNamespace(choices=[SimpleNamespace(
                index=0, delta=SimpleNamespace(
                    role="assistant", content=None, tool_calls=None,
                    reasoning_content=None, reasoning=None),
                finish_reason="stop")], model=model_label, usage=None)

    muse_code_exec = MuseCodeExecProfile(
        name=PROFILE_NAME,
        display_name="Muse Bridge",
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
        classify_api_error=_classify_muse_error,
    )

    register_provider(muse_code_exec)


def register(ctx) -> None:
    """PluginManager entry point (also required by `hermes plugins validate`).

    Provider registration already happened at import; nothing cached to
    refresh (the CLI owns auth per spawned process).
    """
