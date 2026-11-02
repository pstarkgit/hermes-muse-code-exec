"""Offline native steering tests: no model, service, private history or database."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import threading
import time
import types
import unittest
import uuid
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('context_seams', ROOT/'tests/test_muse_code_exec.py')
seams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seams)


def owning_steer(messages, text):
    """Optional owning-core replay, otherwise the exact standalone-user seam."""
    import os
    core = os.getenv('HERMES_CONTEXT_TEST_AGENT_ROOT')
    if not core:
        messages.append({'role': 'user', 'content': text, 'display_kind': 'steer'})
        return
    namespace = {'Any': Any}
    wanted = {'STEER_MARKER_OPEN', 'STEER_MARKER_CLOSE', 'STEER_DISPLAY_KIND',
              'format_steer_marker', 'steer_user_row'}
    tree = ast.parse((Path(core)/'prompt_builder.py').read_text())
    nodes = [n for n in tree.body if
        (isinstance(n, ast.FunctionDef) and n.name in wanted) or
        (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in n.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'owning-steer-row', 'exec'), namespace)
    prompt = types.ModuleType('agent.prompt_builder')
    prompt.steer_user_row = namespace['steer_user_row']
    tree = ast.parse((Path(core)/'turn_iteration_prep.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_inject_steer_after_newest_tool_result')
    namespace = {'Any': Any, 'logger': types.SimpleNamespace(debug=lambda *a: None)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'owning-steer-injection', 'exec'), namespace)
    with patch.dict(sys.modules, {'agent.prompt_builder': prompt}):
        namespace['_inject_steer_after_newest_tool_result'](None, messages, text)


@contextlib.contextmanager
def native_fixture(fault=None, multi=False):
    module, _ = seams._load_plugin()
    f = types.SimpleNamespace(module=module, requests=[], results=[], fault=fault, calls=0,
                              results_seen_before_steer=[])
    f.session_id = '00000000-0000-4000-8000-000000000001'
    f.turn_id = '00000000-0000-4000-8000-000000000002'

    class Endpoint:
        def __init__(self, tools):
            f.endpoint = self
            self.calls, self.results = queue.Queue(), queue.Queue()
            self.closed, self.steer_ready = threading.Event(), threading.Event()
            self.tools, self.steer = {}, None
        def paused_seconds(self): return 900.0
        def config(self): return {'hermes_host': {'transport': 'streamableHttp', 'url': 'offline'}}
        def close(self): self.closed.set(); self.steer_ready.set()

    class Proc:
        def __init__(self): self.stdin, self.stdout, self.stderr = io.StringIO(), [], []
        def terminate(self): pass
        def wait(self, **kwargs): return 0
        def kill(self): pass

    class Session:
        def __init__(self, proc, deadline, on_release):
            self.called = False
            self.finished = False
        def request(self, method, params, timeout=None, **kwargs):
            f.requests.append((method, json.loads(json.dumps(params)), kwargs))
            if method == 'session/start': return {'session': {'sessionId': f.session_id}}
            if method == 'turn/start': return {'turnId': f.turn_id}
            if method == 'turn/steer':
                f.results_seen_before_steer.append(len(f.results))
                if f.fault == 'timeout': raise TimeoutError('synthetic ack lost after possible admission')
                ack = {'commandId': params['commandId'], 'turnId': f.turn_id, 'status': 'accepted'}
                if f.fault in ('commandId', 'turnId', 'status'): ack[f.fault] = 'wrong'
                return ack
            return {}
        def poll_note(self, timeout):
            if not self.called:
                self.called = True
                f.endpoint.calls.put(('call', {'id': 'same-call', 'type': 'function',
                    'function': {'name': 'process_manage', 'arguments': '{}'}}))
            if f.endpoint.closed.is_set(): return {'method': '_transport/closed', 'params': {}}
            if self.finished:
                return {'method': 'turn/completed', 'params': {
                    'sessionId': f.session_id, 'turnId': f.turn_id, 'terminal': 'completed'}}
            try: result = f.endpoint.results.get(timeout=.01)
            except queue.Empty: return None
            f.results.append(result)
            if multi and len(f.results) == 1:
                f.endpoint.calls.put(('call', {'id': 'second-call', 'type': 'function',
                    'function': {'name': 'process_manage', 'arguments': '{}'}}))
                return None
            self.finished = True
            return {'method': 'item/completed', 'params': {'sessionId': f.session_id,
                'item': {'kind': 'agentMessage', 'turnId': f.turn_id, 'text': 'original turn finished'}}}

    context = types.ModuleType('gateway.session_context')
    context.get_session_env = lambda name, default='': {
        'HERMES_SESSION_ID': 'same-session', 'HERMES_SESSION_KEY': 'same-key',
        'HERMES_SESSION_PROFILE': 'sage'}.get(name, default)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {'gateway.session_context': context}))
        for name, value in [('_harness_home', lambda: ROOT), ('_agent_cwd', lambda *a: str(ROOT)),
                            ('_native_scope_cwd', lambda: str(ROOT)), ('_HostToolMCP', Endpoint),
                            ('_ServeSession', Session), ('_build_subprocess_env', lambda: {}),
                            ('_msp_approval_mode', lambda: None)]:
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.object(module.subprocess, 'Popen', return_value=Proc()))
        f.client = module.MuseCodeExecClient(exec_cwd=str(ROOT))
        f.tools = [{'type': 'function', 'function': {'name': 'process_manage'}}]
        f.messages = [{'role': 'user', 'content': 'original request'}]
        f.client._create_chat(messages=f.messages, model='same-model', timeout=1, tools=f.tools)
        f.messages += [
            {'role': 'assistant', 'tool_calls': [{'id': 'same-call', 'type': 'function',
                'function': {'name': 'process_manage', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'same-call', 'content': 'unchanged actual result'}]
        f.resume = lambda **kw: f.client._create_chat(messages=f.messages,
            model='same-model', timeout=1, tools=f.tools, **kw)
        try: yield f
        finally:
            pending = f.client._pending_native
            f.client.close()
            if pending: pending['thread'].join(timeout=1)


class PendingContextTests(unittest.TestCase):
    def test_control_returns_exact_original_result_once_without_steer_or_replay(self):
        with native_fixture() as f:
            self.assertEqual(f.resume().choices[0].message.content, 'original turn finished')
            self.assertEqual(f.results, [{'id': 'same-call', 'content': 'unchanged actual result'}])
            self.assertEqual(sum(method == 'turn/start' for method, *_ in f.requests), 1)
            self.assertFalse(any(method == 'turn/steer' for method, *_ in f.requests))

    def test_new_user_steer_is_admitted_to_exact_original_turn_before_result(self):
        with native_fixture(multi=True) as f:
            correction = 'SYNTHETIC_CORRECTION_421c: finish without another poll'
            owning_steer(f.messages, correction)
            expected_text = f.messages[-1]['content']
            f.resume()
            f.messages += [
                {'role': 'assistant', 'tool_calls': [{'id': 'second-call', 'type': 'function',
                    'function': {'name': 'process_manage', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'second-call', 'content': 'second unchanged result'}]
            final = f.resume()
            steers = [(params, kwargs) for method, params, kwargs in f.requests if method == 'turn/steer']
            self.assertEqual(len(steers), 1)
            packet, kwargs = steers[0]
            self.assertEqual(packet['sessionId'], f.session_id)
            self.assertEqual(packet['expectedTurnId'], f.turn_id)
            self.assertEqual(packet['input'], [{'type': 'text', 'text': expected_text}])
            self.assertEqual(set(packet), {'commandId', 'sessionId', 'expectedTurnId', 'input'})
            self.assertEqual(uuid.UUID(packet['commandId']).version, 7)
            self.assertEqual(f.results_seen_before_steer, [0])
            self.assertEqual(kwargs['host_pause'], 900.0)
            self.assertEqual(final.choices[0].message.content, 'original turn finished')
            self.assertEqual(f.results, [{'id': 'same-call', 'content': 'unchanged actual result'},
                                        {'id': 'second-call', 'content': 'second unchanged result'}])
            self.assertEqual(sum(method == 'turn/start' for method, *_ in f.requests), 1)

    def test_system_or_prefix_changes_are_explicitly_refused_without_releasing_result(self):
        for variant in ('system-tail', 'prefix', 'separate-system'):
            with self.subTest(variant=variant), native_fixture() as f:
                if variant == 'system-tail': f.messages.append({'role': 'system', 'content': 'new system context'})
                if variant == 'prefix': f.messages[0]['content'] = 'changed historical user content'
                with self.assertRaises(f.module.MusePolicyError):
                    f.resume(**({'system': 'new system context'} if variant == 'separate-system' else {}))
                self.assertFalse(f.results)
                self.assertFalse(any(method == 'turn/steer' for method, *_ in f.requests))
                self.assertFalse(f.endpoint.closed.is_set())

    def test_uncertain_or_wrong_ack_retains_original_packet_and_never_retries(self):
        for fault in ('timeout', 'commandId', 'turnId', 'status'):
            with self.subTest(fault=fault), native_fixture(fault) as f:
                f.messages.append({'role': 'user', 'content': 'same immutable correction'})
                with self.assertRaisesRegex(f.module.MuseTurnError, 'uncertain'):
                    f.resume()
                saved = json.loads(json.dumps(f.client._pending_native['context_update']))
                self.assertEqual(saved['packet']['sessionId'], f.session_id)
                self.assertEqual(saved['packet']['expectedTurnId'], f.turn_id)
                with self.assertRaisesRegex(f.module.MuseTurnError, 'uncertain'):
                    f.resume()
                self.assertEqual(f.client._pending_native['context_update'], saved)
                self.assertEqual(sum(method == 'turn/steer' for method, *_ in f.requests), 1)
                self.assertFalse(f.results)
                self.assertEqual(sum(method == 'turn/start' for method, *_ in f.requests), 1)

    def test_rpc_deadline_excludes_host_wait_but_ack_wait_stays_finite(self):
        module, _ = seams._load_plugin()
        session = module._ServeSession.__new__(module._ServeSession)
        session._deadline = time.monotonic() - 100
        session._waiters, session._waiters_lock = {}, threading.Lock()
        packet = {'commandId': 'same-command', 'sessionId': 'same-session',
                  'expectedTurnId': 'same-turn', 'input': [{'type': 'text', 'text': 'one steer'}]}
        ack = {'commandId': 'same-command', 'turnId': 'same-turn', 'status': 'accepted'}
        class Input:
            def write(self, line):
                wire = json.loads(line)
                session._waiters.pop(wire['id']).put({'id': wire['id'], 'result': ack})
            def flush(self): pass
        session._proc = types.SimpleNamespace(stdin=Input(), poll=lambda: None)
        self.assertEqual(session.request('turn/steer', packet, timeout=.2, host_pause=900), ack)
        self.assertFalse(session._waiters)


if __name__ == '__main__':
    unittest.main()
