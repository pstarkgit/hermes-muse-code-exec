"""Hermetic unit tests for the muse-code-exec provider (no Hermes install,
no Meta account, no network). Hermes imports are stubbed; subprocess is
mocked."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
PLUGIN_FILE = REPO / "__init__.py"


def _load_plugin(env=None):
    for var in ("HERMES_MUSE_CODE_EXEC_COMMAND", "MUSE_CLI_PATH"):
        if env and var in env:
            os.environ[var] = env[var]
        else:
            os.environ.pop(var, None)
    for name in ("_hermes_test_muse_exec", "agent", "agent.acp_openai_bridge",
                 "agent.runtime_cwd", "providers", "providers.base"):
        sys.modules.pop(name, None)

    agent = types.ModuleType("agent")
    bridge = types.ModuleType("agent.acp_openai_bridge")
    bridge.completion_to_stream_chunks = lambda completion: iter(
        [{"choices": [{"delta": {"content": completion.choices[0].message.content}}]}])
    bridge.extract_tool_calls_from_text = lambda text: ([], text)
    bridge.render_tool_bridge_sections = lambda *a, **k: ""
    agent.acp_openai_bridge = bridge
    runtime_cwd = types.ModuleType("agent.runtime_cwd")
    runtime_cwd.resolve_agent_cwd = lambda: Path("/tmp")
    agent.runtime_cwd = runtime_cwd

    registered = {}

    base_mod = types.ModuleType("providers.base")

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    base_mod.ProviderProfile = ProviderProfile
    providers = types.ModuleType("providers")
    providers.base = base_mod
    providers.register_provider = lambda p: registered.setdefault(p.name, p)

    with patch.dict(sys.modules, {
        "agent": agent,
        "agent.acp_openai_bridge": bridge,
        "agent.runtime_cwd": runtime_cwd,
        "providers": providers,
        "providers.base": base_mod,
    }):
        spec = importlib.util.spec_from_file_location(
            "_hermes_test_muse_exec", PLUGIN_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_hermes_test_muse_exec"] = module
        spec.loader.exec_module(module)
    return module, registered


def _jsonl(*events):
    return "\n".join(json.dumps(e) for e in events)


def _term(text, terminal="completed"):
    return {"payload": {"kind": "run_terminal", "terminal": terminal,
                        "text": text}}


def _delta(text):
    return {"payload": {"kind": "run_output_delta", "text": text}}


def test_provider_registered_with_expected_wiring():
    module, registered = _load_plugin()
    profile = registered["muse-code-exec"]
    assert profile.auth_type == "external_process"
    assert profile.base_url == "exec://muse-code"
    assert profile.api_mode == "chat_completions"
    assert "muse-spark-1.3" in module.CURATED_MODELS
    assert "muse-spark-1.3-contributor" in module.CURATED_MODELS
    assert module.CURATED_MODELS.index("muse-spark-1.3") < \
        module.CURATED_MODELS.index("muse-spark-1.3-contributor")


def test_render_messages_flattens_roles():
    module, _ = _load_plugin()
    out = module._render_messages([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ])
    assert "System:\nBe brief." in out
    assert "User:\nhi" in out


def test_run_exec_completion_prefers_terminal_text():
    module, _ = _load_plugin()
    proc = SimpleNamespace(returncode=0,
                           stdout=_jsonl(_delta("he"), _term("hello")),
                           stderr="")
    with patch("subprocess.run", return_value=proc):
        assert module.run_exec_completion(
            "hi", model="muse-spark-1.3", cwd="/tmp",
            timeout_seconds=30) == "hello"


def test_run_exec_completion_falls_back_to_deltas():
    module, _ = _load_plugin()
    proc = SimpleNamespace(returncode=0, stdout=_jsonl(
        _delta("he"), _delta("llo")), stderr="")
    with patch("subprocess.run", return_value=proc):
        assert module.run_exec_completion(
            "hi", model=None, cwd="/tmp",
            timeout_seconds=30) == "hello"


def test_run_exec_completion_raises_on_failure():
    module, _ = _load_plugin()
    proc = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    with patch("subprocess.run", return_value=proc):
        try:
            module.run_exec_completion("hi", model=None, cwd="/tmp",
                                       timeout_seconds=30)
        except RuntimeError as exc:
            assert "exit 1" in str(exc) and "boom" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_run_exec_completion_raises_on_empty_text():
    module, _ = _load_plugin()
    proc = SimpleNamespace(returncode=0, stdout=_jsonl(_term("  ")),
                           stderr="")
    with patch("subprocess.run", return_value=proc):
        try:
            module.run_exec_completion("hi", model=None, cwd="/tmp",
                                       timeout_seconds=30)
        except RuntimeError as exc:
            assert "no text" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_harness_posture_mapping():
    import os as _os
    module, _ = _load_plugin()
    for var in (module.ENV_APPROVAL_MODE, module.ENV_EXEC_APPROVAL,
                module.ENV_NO_SANDBOX):
        _os.environ.pop(var, None)
    with patch.object(module, "_read_harness_approvals_mode",
                      return_value="off"):
        assert module._msp_approval_mode() == "allowAll"
    with patch.object(module, "_read_harness_approvals_mode",
                      return_value="on-request"):
        assert module._msp_approval_mode() == "denyUnmatched"
    with patch.object(module, "_read_harness_approvals_mode",
                      return_value=None):
        assert module._msp_approval_mode() is None
    _os.environ[module.ENV_APPROVAL_MODE] = "promptUnmatched"
    with patch.object(module, "_read_harness_approvals_mode",
                      return_value="off"):
        assert module._msp_approval_mode() == "promptUnmatched"
    del _os.environ[module.ENV_APPROVAL_MODE]
    assert module._exec_approval_flag() == ["--approval-mode", "never"]
    _os.environ[module.ENV_EXEC_APPROVAL] = "on-request"
    assert module._exec_approval_flag() == ["--approval-mode", "on-request"]
    del _os.environ[module.ENV_EXEC_APPROVAL]
    assert module._serve_extra_flags() == []
    _os.environ[module.ENV_NO_SANDBOX] = "1"
    assert module._serve_extra_flags() == ["--disable-sandbox"]
    del _os.environ[module.ENV_NO_SANDBOX]


def test_uuid7_shape():
    import re
    module, _ = _load_plugin()
    value = module.uuid7()
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-"
        r"[0-9a-f]{12}", value)


def test_collect_model_ids():
    module, _ = _load_plugin()
    out = []
    module._collect_model_ids(
        {"models": [{"modelId": "muse-spark-1.3"}, {"id": "x"}]}, out)
    assert out == ["muse-spark-1.3", "x"]


def test_run_turn_falls_back_to_exec():
    module, _ = _load_plugin()
    client = module.MuseCodeExecClient()
    with patch.object(module, "_msp_turn",
                       side_effect=RuntimeError("no serve")):
        with patch.object(module, "run_exec_completion",
                           return_value="fallback-ok") as m:
            text, activity = client._run_turn(
                "hi", model_id="muse-spark-1.3", timeout_seconds=30,
                on_delta=None)
    assert text == "fallback-ok"
    assert activity == ["transport: exec-fallback"]
    assert m.call_args.kwargs["model"] == "muse-spark-1.3"


def test_dispatcher_routes_responses_and_notes():
    import json as _json
    import queue as _queue
    module, _ = _load_plugin()

    written: list = []

    class FakeStdout:
        def __iter__(self):
            import time as _t
            end = _t.monotonic() + 10
            while not written and _t.monotonic() < end:
                _t.sleep(0.01)
            rid = _json.loads(written[0])["id"]
            return iter([
                _json.dumps({"jsonrpc": "2.0", "id": rid,
                             "result": {"hello": "world"}}),
                _json.dumps({"jsonrpc": "2.0", "method": "item/delta",
                             "params": {"itemId": "i1", "delta": "hi"}}),
            ])

    class FakeStdin:
        def write(self, text):
            written.append(text)

        def flush(self):
            pass

    class FakeProc:
        stdin = FakeStdin()
        stdout = FakeStdout()
        stderr = []

        def poll(self):
            return None

    proc = FakeProc()
    import time as _time
    session = module._ServeSession(
        proc, _time.monotonic() + 30, on_release=lambda: None)
    try:
        result = session.request("initialize", {"clientInfo": {
            "name": "t", "version": "0"}}, timeout=5)
        assert result == {"hello": "world"}
        note = session.poll_note(timeout=5)
        assert note is not None and note.get("method") == "item/delta"
    finally:
        session.release()


def test_turn_preamble_present_and_safe():
    module, _ = _load_plugin()
    assert "muse-code-exec" in module.TURN_PREAMBLE
    lowered = module.TURN_PREAMBLE.lower()
    assert "never advise" in lowered or "never instruct" in lowered
    assert "disable sandbox" in lowered


def test_identity_stays_brief_without_runtime_lecture():
    module, _ = _load_plugin()
    lowered = module.TURN_PREAMBLE.lower()
    assert "one short sentence" in lowered
    assert "never lecture" in lowered
    assert "transport only" in lowered


def test_coerce_text_shapes():
    module, _ = _load_plugin()
    assert module._coerce_text(None) == ""
    assert module._coerce_text("You are Kira.") == "You are Kira."
    assert module._coerce_text(
        [{"type": "text", "text": "A"},
         {"type": "image", "text": "B"}]) == "A"


def test_system_kwarg_carried_into_prompt():
    module, _ = _load_plugin()
    client = module.MuseCodeExecClient()
    seen = {}

    def fake_run(prompt_text, **kwargs):
        seen["prompt"] = prompt_text
        return ("ok", [])

    with patch.object(module.MuseCodeExecClient, "_run_turn",
                       side_effect=fake_run):
        out = client._create_chat(
            model="muse-spark-1.3",
            messages=[{"role": "user", "content": "hi"}],
            system="You are Kira.")
    assert "System (bot identity" in seen["prompt"]
    assert "You are Kira." in seen["prompt"]
    assert out.choices[0].message.content == "ok"


def test_model_id_passed_through():
    module, _ = _load_plugin()
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return SimpleNamespace(returncode=0, stdout=_jsonl(_term("ok")),
                               stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        module.run_exec_completion("hi", model="muse-spark-1.2",
                                   cwd="/tmp", timeout_seconds=30)
    assert "--model" in seen["args"]
    assert "muse-spark-1.2" in seen["args"]


def test_streaming_keeps_each_callers_profile_context():
    """A multiplexed gateway's profile and frozen policy are ContextVars."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor
    module, _ = _load_plugin()
    profile = contextvars.ContextVar("test_profile", default="wrong-default")
    client = module.MuseCodeExecClient(exec_cwd=str(REPO))

    def run(prompt, **kwargs):
        kwargs["on_delta"](profile.get())
        return profile.get(), []

    def reply(name):
        token = profile.set(name)
        try:
            return "".join(chunk.choices[0].delta.content or "" for chunk in
                           client._stream_turn("probe", model_id="test",
                                               model_label="test", timeout_seconds=5))
        finally:
            profile.reset(token)

    with patch.object(client, "_run_turn", side_effect=run):
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(reply, ["kira", "other-profile"])) == ["kira", "other-profile"]


def _fake_msp(module, on_request=None, notes=None):
    """A deterministic host at the actual MSP transport seam; no inference."""
    import io
    from contextlib import ExitStack
    requests = []
    pending = list(notes or [
        {"method": "item/completed", "params": {"sessionId": "s", "item": {
            "kind": "agentMessage", "turnId": "t", "text": "ok"}}},
        {"method": "turn/completed", "params": {
            "sessionId": "s", "turnId": "t", "terminal": "completed"}},
    ])

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, params, **kwargs):
            requests.append((method, dict(params)))
            if on_request:
                result = on_request(method, params)
                if result is not None:
                    return result
            if method == "session/start":
                return {"session": {"sessionId": "s", "approvalMode": {"mode": "allowAll"}}}
            if method == "turn/start":
                return {"turnId": "t"}
            return {}

        def poll_note(self, timeout):
            return pending.pop(0)

    class Process:
        stdin = io.StringIO()
        stdout = []
        stderr = []

        def terminate(self):
            pass

        def wait(self, **kwargs):
            pass

    stack = ExitStack()
    stack.enter_context(patch.object(module.subprocess, "Popen", return_value=Process()))
    stack.enter_context(patch.object(module, "_ServeSession", Session))
    stack.enter_context(patch.object(module, "_msp_approval_mode", return_value="allowAll"))
    return stack, requests


def test_rejected_policy_is_not_dropped_or_replayed_through_exec():
    module, _ = _load_plugin()
    client = module.MuseCodeExecClient(exec_cwd=str(REPO))

    def reject(method, params):
        if method == "session/start" and "approvalMode" in params:
            raise RuntimeError("commandRejected -32030: approval mode unavailable")

    host, requests = _fake_msp(module, on_request=reject)
    with host, patch.object(module, "run_exec_completion", return_value="wrong fallback") as fallback:
        try:
            client._run_turn("probe", model_id="test", timeout_seconds=5, on_delta=None)
        except RuntimeError as exc:
            assert "approval" in str(exc).lower()
        else:
            raise AssertionError("Rejected approval policy was silently discarded")
        assert len([r for r in requests if r[0] == "session/start"]) == 1
        fallback.assert_not_called()


def test_host_must_confirm_the_requested_approval_mode_before_tools_run():
    module, _ = _load_plugin()
    for actual in ("denyUnmatched", None):
        def mismatch(method, params):
            if method == "session/start":
                session = {"sessionId": "s"}
                if actual:
                    session["approvalMode"] = {"mode": actual}
                return {"session": session}

        host, requests = _fake_msp(module, on_request=mismatch)
        with host:
            try:
                module._msp_turn("probe", model_id="test", cwd=str(REPO),
                                 timeout_seconds=5, on_delta=None, track=lambda *args: None)
            except RuntimeError as exc:
                assert "approval" in str(exc).lower()
            else:
                raise AssertionError(f"Unconfirmed approval policy {actual!r} admitted tools")
        assert not any(method == "turn/start" for method, _ in requests)


def test_exec_fallback_preserves_the_explicit_sandbox_setting():
    module, _ = _load_plugin()
    seen = []

    def execute(args, **kwargs):
        seen.append(args)
        return SimpleNamespace(returncode=0, stdout=_jsonl(_term("ok")), stderr="")

    with patch.dict(os.environ, {module.ENV_NO_SANDBOX: "1"}), \
            patch.object(module.subprocess, "run", side_effect=execute):
        module.run_exec_completion("probe", model="test", cwd=str(REPO), timeout_seconds=5)
    assert "--disable-sandbox" in seen[0]
    seen.clear()
    with patch.dict(os.environ, {module.ENV_NO_SANDBOX: "0"}), \
            patch.object(module.subprocess, "run", side_effect=execute):
        module.run_exec_completion("probe", model="test", cwd=str(REPO), timeout_seconds=5)
    assert "--disable-sandbox" not in seen[0]


def test_an_admitted_turn_is_not_replayed_after_a_transport_failure():
    module, _ = _load_plugin()
    client = module.MuseCodeExecClient(exec_cwd=str(REPO))
    host, _ = _fake_msp(module, notes=[
        {"method": "item/completed", "params": {"sessionId": "s", "item": {
            "kind": "toolCall", "tool": "shell", "status": "completed"}}},
        {"method": "turn/failed", "params": {"sessionId": "s", "turnId": "t"}},
    ])
    with host, patch.object(module, "run_exec_completion", return_value="replayed") as fallback:
        try:
            client._run_turn("do work", model_id="test", timeout_seconds=5, on_delta=None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("An admitted turn was silently executed again")
        fallback.assert_not_called()


def test_missing_approval_setting_is_not_the_string_none():
    module, _ = _load_plugin()
    config = types.ModuleType("hermes_cli.config_effective")
    config.load_user_config_effective = lambda path: {}
    with patch.dict(sys.modules, {"hermes_cli.config_effective": config}):
        assert module._read_harness_approvals_mode() is None


def test_trace_distinguishes_confirmed_policy_from_an_actual_tool_denial():
    module, _ = _load_plugin()
    host, _ = _fake_msp(module, notes=[
        {"method": "item/completed", "params": {"sessionId": "s", "item": {
            "kind": "toolCall", "tool": "shell", "status": "rejected",
            "failureKind": "approval_denied",
            "failureReason": "deny_unmatched: no policy rule allows this action private-content"}}},
        {"method": "item/completed", "params": {"sessionId": "s", "item": {
            "kind": "agentMessage", "turnId": "t", "text": "blocked"}}},
        {"method": "turn/completed", "params": {
            "sessionId": "s", "turnId": "t", "terminal": "completed"}},
    ])
    with host:
        _, activity = module._msp_turn("probe", model_id="test", cwd=str(REPO),
                                       timeout_seconds=5, on_delta=None, track=lambda *args: None)
    trace = "\n".join(activity)
    assert "approval=allowAll" in trace
    assert "tool=shell" in trace and "approval_denied" in trace and "deny_unmatched" in trace
    assert "private-content" not in trace
