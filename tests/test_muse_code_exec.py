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
