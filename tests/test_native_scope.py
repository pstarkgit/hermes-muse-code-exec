"""Offline pending-turn scope regressions; no model, gateway or database calls."""
import importlib.util
import contextvars
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scope_seams', ROOT/'tests/test_muse_code_exec.py')
seams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seams)


def runtime_cwd_for_test():
    """Optional real-core local replay; otherwise the portable runtime-cwd seam."""
    source = os.environ.get('HERMES_SCOPE_TEST_RUNTIME_CWD')
    if source:
        spec = importlib.util.spec_from_file_location('real_runtime_cwd', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    module = types.ModuleType('agent.runtime_cwd')
    declared = contextvars.ContextVar('test_declared_cwd', default='')
    module.set_session_cwd = declared.set
    module.scoped_session_cwd = declared.get
    module.scope_terminal_cwd = lambda: ''

    def resolve_agent_cwd():
        for raw in (module.scoped_session_cwd(), module.scope_terminal_cwd()):
            if raw and Path(raw).is_dir():
                return Path(raw)
        return Path.cwd()

    module.resolve_agent_cwd = resolve_agent_cwd
    return module


class Endpoint:
    def __init__(self, tools):
        self.calls, self.results = queue.Queue(), queue.Queue()
        self.closed = threading.Event()
        self.tools = {}

    def close(self):
        self.closed.set()
        self.results.put(None)


class NativeScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.getenv('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.workspace = self.base/'worker'
        self.workspace.mkdir()
        self.module, _ = seams._load_plugin()
        self.cwd = runtime_cwd_for_test()
        self.terminal_cwd = ''
        self.cwd.scope_terminal_cwd = lambda: self.terminal_cwd
        self.cwd.set_session_cwd(str(self.workspace))
        self.scope = {'HERMES_SESSION_ID': 'original-session',
                      'HERMES_SESSION_KEY': 'original-key',
                      'HERMES_SESSION_PROFILE': 'sage'}
        context = types.ModuleType('gateway.session_context')
        context.get_session_env = lambda name, default='': self.scope.get(name, default)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'agent.runtime_cwd': self.cwd,
                                 'gateway.session_context': context}).start()
        patch.object(self.module, '_harness_home', return_value=self.base/'sage').start()
        patch.object(self.module, '_HostToolMCP', Endpoint).start()
        self.client = self.module.MuseCodeExecClient(exec_cwd=str(self.workspace))
        self.addCleanup(self.client.close)
        self.observed = []

        def run(prompt, **kwargs):
            endpoint = kwargs['host_mcp']
            endpoint.calls.put(('call', {'id': 'actual-call', 'type': 'function',
                'function': {'name': 'kanban_complete', 'arguments': '{}'}}))
            self.observed.append(endpoint.results.get(timeout=3))
            return 'task result consumed', []

        self.runner = patch.object(self.client, '_run_turn', side_effect=run).start()
        self.tools = [{'type': 'function', 'function': {'name': 'kanban_complete'}}]

    def start(self):
        self.client._native_turn('original', [], model_id='original-model',
                                timeout_seconds=1, tools=self.tools)

    def resume(self, model='original-model'):
        return self.client._native_turn('continuation', [
            {'role': 'tool', 'tool_call_id': 'actual-call', 'content': 'task completed'}],
            model_id=model, timeout_seconds=1, tools=self.tools)

    def test_completed_scratch_cleanup_keeps_declared_session_scope(self):
        self.start()
        shutil.rmtree(self.workspace)  # kanban_complete's managed scratch cleanup
        final, activity = self.resume()
        self.assertEqual(final, 'task result consumed')
        self.assertEqual(self.observed, [{'id': 'actual-call', 'content': 'task completed'}])
        self.assertEqual(self.runner.call_count, 1)

    def test_completed_cleanup_keeps_declared_terminal_scope(self):
        self.cwd.set_session_cwd('')
        self.terminal_cwd = str(self.workspace)
        self.start()
        shutil.rmtree(self.workspace)
        self.assertEqual(self.resume()[0], 'task result consumed')
        self.assertEqual(self.runner.call_count, 1)

    def assert_rejected_without_replay(self, message='scope changed', model='original-model'):
        endpoint = self.client._pending_native['mcp']
        thread = self.client._pending_native['thread']
        with self.assertRaisesRegex(self.module.MuseTurnError, message):
            self.resume(model)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(endpoint.closed.is_set())
        self.assertNotIn({'id': 'actual-call', 'content': 'task completed'}, self.observed)
        self.assertEqual(self.runner.call_count, 1)

    def test_changed_profile_home_still_refuses(self):
        self.start()
        with patch.object(self.module, '_harness_home', return_value=self.base/'other-profile'):
            self.assert_rejected_without_replay()

    def test_changed_declared_session_workspace_still_refuses(self):
        self.start()
        self.cwd.set_session_cwd(str(self.base/'different-workspace'))
        self.assert_rejected_without_replay()

    def test_changed_declared_terminal_workspace_still_refuses(self):
        self.cwd.set_session_cwd('')
        self.terminal_cwd = str(self.workspace)
        self.start()
        self.terminal_cwd = str(self.base/'different-workspace')
        self.assert_rejected_without_replay()

    def test_changed_immutable_client_workspace_still_refuses(self):
        self.start()
        self.client._cwd = str(self.base/'different-client-workspace')
        self.assert_rejected_without_replay()

    def test_changed_session_id_still_refuses(self):
        self.start()
        self.scope['HERMES_SESSION_ID'] = 'different-session'
        self.assert_rejected_without_replay()

    def test_changed_session_key_still_refuses(self):
        self.start()
        self.scope['HERMES_SESSION_KEY'] = 'different-key'
        self.assert_rejected_without_replay()

    def test_changed_session_profile_still_refuses(self):
        self.start()
        self.scope['HERMES_SESSION_PROFILE'] = 'other-profile'
        self.assert_rejected_without_replay()

    def test_changed_model_still_refuses(self):
        self.start()
        self.assert_rejected_without_replay('no matching host result', model='different-model')


if __name__ == '__main__':
    unittest.main()
