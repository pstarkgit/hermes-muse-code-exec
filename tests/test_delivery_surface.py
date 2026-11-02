"""Offline behavioral regressions at the Muse/Hermes completion boundary."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import hermes_bootstrap
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from agent.acp_openai_bridge import extract_tool_calls_from_text, render_tool_bridge_sections, completion_to_stream_chunks
from tools.environments.local import served_profile_child_env

spec = importlib.util.spec_from_file_location('baseline_seams', ROOT/'tests/test_muse_code_exec.py')
seams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seams)


def plugin():
    # The baseline unit helper replaces modules deliberately. Restore the real
    # core modules after loading the provider facade so integration helpers work.
    names = ('agent', 'agent.acp_openai_bridge', 'agent.runtime_cwd', 'providers', 'providers.base')
    previous = {n:sys.modules.get(n) for n in names}
    m,_ = seams._load_plugin()
    for name, value in previous.items():
        if value is not None: sys.modules[name] = value
        else: sys.modules.pop(name,None)
    m._extract_tool_calls_from_text = extract_tool_calls_from_text
    m._render_tool_bridge_sections = render_tool_bridge_sections
    m._completion_to_stream_chunks = completion_to_stream_chunks
    return m


def tool(name):
    return {'type':'function','function':{'name':name,'description':name+' current session tool',
            'parameters':{'type':'object','properties':{'target':{'type':'string'},'message':{'type':'string'}}}}}


def response(name='message_agent'):
    return '<tool_call>'+json.dumps({'id':'call_test_1','type':'function','function':{
        'name':name,'arguments':json.dumps({'target':'sage','message':'bounded exact candidate review'})}})+'</tool_call>'


class DeliverySurface(unittest.TestCase):
    def test_native_mcp_registers_exact_schemas_and_rejects_browser_origin(self):
        import urllib.request, urllib.error
        m=plugin();endpoint=m._HostToolMCP([tool('message_agent'),tool('future_plugin')])
        url=endpoint.config()['hermes_host']['url']
        try:
            packet={'jsonrpc':'2.0','id':1,'method':'tools/list'}
            request=urllib.request.Request(url,data=json.dumps(packet).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=2) as r: result=json.load(r)
            self.assertEqual([x['name'] for x in result['result']['tools']],['message_agent','future_plugin'])
            self.assertEqual(result['result']['tools'][0]['inputSchema'],tool('message_agent')['function']['parameters'])
            request=urllib.request.Request(url,data=json.dumps(packet).encode(),headers={'Origin':'https://untrusted.invalid'})
            with self.assertRaises(urllib.error.HTTPError) as error: urllib.request.urlopen(request,timeout=2)
            self.assertEqual(error.exception.code,403);error.exception.close()
        finally:endpoint.close()
        self.assertFalse(endpoint.thread.is_alive())

    def test_native_mcp_admits_one_call_without_executing_or_inventing_result(self):
        import urllib.request, threading, time
        m=plugin();endpoint=m._HostToolMCP([tool('message_agent')]);responses=[]
        packet={'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'message_agent','arguments':{'target':'dex','message':'same candidate'}}}
        def call():
            request=urllib.request.Request(endpoint.config()['hermes_host']['url'],data=json.dumps(packet).encode())
            with urllib.request.urlopen(request,timeout=3) as r:responses.append(json.load(r))
        t=threading.Thread(target=call);t.start()
        try:
            limit=time.monotonic()+2
            while endpoint.calls.empty() and time.monotonic()<limit: time.sleep(.01)
            envelope=endpoint.take();self.assertIsNotNone(envelope)
            calls,_=m._extract_tool_calls_from_text(envelope)
            self.assertEqual(calls[0].function.name,'message_agent')
            self.assertEqual(json.loads(calls[0].function.arguments),packet['params']['arguments'])
            self.assertIsNone(endpoint.take());self.assertFalse(responses)
        finally:endpoint.close();t.join(timeout=3)
        self.assertEqual(responses[0]['error']['code'],-32800)
        self.assertFalse(t.is_alive())

    def test_native_mcp_unoffered_call_never_reaches_host_queue(self):
        import urllib.request
        m=plugin();endpoint=m._HostToolMCP([tool('message_agent')])
        try:
            packet={'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'kanban_complete','arguments':{}}}
            request=urllib.request.Request(endpoint.config()['hermes_host']['url'],data=json.dumps(packet).encode())
            with urllib.request.urlopen(request,timeout=2) as r:result=json.load(r)
            self.assertEqual(result['error']['code'],-32602);self.assertIsNone(endpoint.take())
        finally:endpoint.close()

    def test_native_tools_never_fall_back_to_toolless_exec(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        with patch.object(m,'_msp_turn',side_effect=RuntimeError('MCP unavailable')),patch.object(m,'run_exec_completion') as fallback:
            with self.assertRaises(m.MusePolicyError):
                client._run_turn('test',model_id='test',timeout_seconds=1,on_delta=None,host_tools=[tool('message_agent')])
            fallback.assert_not_called()

    def test_native_same_session_receives_actual_host_result_without_replaying(self):
        import urllib.request
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));observed=[]
        def run(prompt,**kwargs):
            endpoint=kwargs['host_mcp']
            try:
                packet={'jsonrpc':'2.0','id':9,'method':'tools/call','params':{'name':'message_agent','arguments':{'target':'dex','message':'bounded'}}}
                request=urllib.request.Request(endpoint.config()['hermes_host']['url'],data=json.dumps(packet).encode())
                with urllib.request.urlopen(request,timeout=3) as r:observed.append(json.load(r)['result'])
                return 'actual delivery received',[]
            finally:endpoint.close()
        messages=[{'role':'user','content':'send one message'}]
        try:
            with patch.object(client,'_run_turn',side_effect=run) as runner:
                first=client._create_chat(messages=messages,tools=[tool('message_agent')],timeout=3)
                call=first.choices[0].message.tool_calls[0]
                self.assertTrue(client._pending_native['thread'].is_alive())
                messages.append({'role':'tool','tool_call_id':call.id,'content':'{"delivery_id":"actual_receipt_123"}'})
                final=client._create_chat(messages=messages,tools=[tool('message_agent')],timeout=3)
                self.assertEqual(runner.call_count,1)
                self.assertEqual(observed[0]['content'][0]['text'],messages[-1]['content'])
                self.assertEqual(final.choices[0].message.content,'actual delivery received')
                self.assertIsNone(client._pending_native)
        finally:client.close()

    def test_pending_native_call_without_matching_result_is_cancelled_not_replayed(self):
        import threading,time
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        endpoint=m._HostToolMCP([tool('message_agent')])
        client._pending_native={'mcp':endpoint,'call_id':'original_id','model':'test'}
        with patch.object(client,'_run_turn') as runner:
            with self.assertRaises(m.MuseTurnError):
                client._native_turn('test',[{'role':'tool','tool_call_id':'other_id','content':'wrong'}],model_id='test',timeout_seconds=1,tools=[tool('message_agent')])
            runner.assert_not_called()
        self.assertTrue(endpoint.closed.is_set())

    def test_native_final_text_quoting_xml_is_not_an_execution_request(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        with patch.object(client,'_native_turn',return_value=(response(),[])):
            final=client._create_chat(messages=[{'role':'user','content':'explain this call'}],tools=[tool('message_agent')])
        self.assertEqual(final.choices[0].finish_reason,'stop')
        self.assertFalse(final.choices[0].message.tool_calls)
        self.assertEqual(final.choices[0].message.content,response())

    def test_native_mcp_reconnect_reuses_same_call_and_actual_result(self):
        import urllib.request,threading,time
        m=plugin();endpoint=m._HostToolMCP([tool('message_agent')]);responses=[]
        packet={'jsonrpc':'2.0','id':'same-request','method':'tools/call','params':{'name':'message_agent','arguments':{'target':'dex','message':'one effect'}}}
        def request(payload):
            req=urllib.request.Request(endpoint.config()['hermes_host']['url'],data=json.dumps(payload).encode())
            with urllib.request.urlopen(req,timeout=3) as r:return json.load(r)
        def call():responses.append(request(packet))
        first=threading.Thread(target=call);second=threading.Thread(target=call)
        try:
            first.start();kind,call_id=endpoint.calls.get(timeout=2)
            self.assertEqual(kind,'call');second.start()
            time.sleep(.05);self.assertTrue(endpoint.calls.empty())
            endpoint.results.put({'id':call_id['id'],'content':'actual single delivery receipt'})
            first.join(timeout=2);second.join(timeout=2)
            self.assertEqual(len(responses),2);self.assertEqual(responses[0],responses[1])
            self.assertEqual(request(packet),responses[0]);self.assertTrue(endpoint.calls.empty())
            changed=json.loads(json.dumps(packet));changed['params']['arguments']['message']='different effect'
            self.assertEqual(request(changed)['error']['code'],-32602)
            self.assertTrue(endpoint.calls.empty())
        finally:endpoint.close();first.join(timeout=2);second.join(timeout=2)

    def test_changed_native_caller_scope_cannot_receive_a_pending_result(self):
        import threading
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));endpoint_holder=[]
        def run(prompt,**kwargs):
            endpoint=kwargs['host_mcp'];endpoint_holder.append(endpoint)
            endpoint.calls.put(('call',{'id':'actual-original','type':'function','function':{'name':'message_agent','arguments':'{}'}}))
            endpoint.closed.wait();return 'cancelled',[]
        try:
            with patch.object(client,'_run_turn',side_effect=run):
                first=client._create_chat(model='test',messages=[{'role':'user','content':'one bounded request'}],tools=[tool('message_agent')],timeout=2)
                messages=[{'role':'tool','tool_call_id':'actual-original','content':'private actual receipt'}]
                with patch.object(m,'_harness_home',return_value=ROOT/'different-profile'):
                    with self.assertRaisesRegex(m.MuseTurnError,'scope changed'):
                        client._create_chat(model='test',messages=messages,tools=[tool('message_agent')],timeout=2)
                self.assertTrue(endpoint_holder[0].results.empty())
                self.assertTrue(endpoint_holder[0].closed.is_set())
        finally:client.close()

    def test_internal_message_agent_is_a_host_tool_call_not_a_final_reply(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));seen={}
        def run(prompt,*args,**kwargs):seen['prompt']=prompt;return response(),['transport: native-mcp-host-call']
        with patch.object(client,'_native_turn',side_effect=run):
            out=client._create_chat(messages=[{'role':'user','content':'commission the review'}],tools=[tool('message_agent')])
        self.assertIn('message_agent',seen['prompt'])
        self.assertEqual(out.choices[0].finish_reason,'tool_calls')
        self.assertEqual(out.choices[0].message.tool_calls[0].function.name,'message_agent')

    def test_configured_plugin_and_protocol_tools_keep_the_actual_schema(self):
        for name in ['a2a_call','mcp_murmr_continuum_session_resume','kanban_show','delegate_task']:
            with self.subTest(name=name):
                m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));seen={}
                def run(prompt,*args,**kwargs):seen['prompt']=prompt;return response(name),['transport: native-mcp-host-call']
                with patch.object(client,'_native_turn',side_effect=run):
                    out=client._create_chat(messages=[{'role':'user','content':'continue'}],tools=[tool(name)])
                self.assertIn(name,seen['prompt'])
                self.assertEqual(out.choices[0].message.tool_calls[0].function.name,name)

    def test_streamed_message_agent_returns_one_structured_call(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        def run(prompt,*args,**kwargs):
            if kwargs.get('on_delta'):kwargs['on_delta'](response())
            return response(),['transport: native-mcp-host-call']
        with patch.object(client,'_native_turn',side_effect=run):
            chunks=list(client._create_chat(messages=[{'role':'user','content':'continue'}],tools=[tool('message_agent')],stream=True))
        calls=[call for chunk in chunks for choice in chunk.choices for call in (getattr(choice.delta,'tool_calls',None) or [])]
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].function.name,'message_agent')
        self.assertFalse(any('<tool_call>' in (getattr(choice.delta,'content',None) or '') for chunk in chunks for choice in chunk.choices))

    def test_unoffered_function_is_never_dispatched(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        with patch.object(client,'_native_turn',return_value=(response('unoffered_tool'),['transport: native-mcp-host-call'])):
            with self.assertRaises(m.MuseTurnError):
                client._create_chat(messages=[{'role':'user','content':'continue'}],tools=[tool('message_agent')])

    def test_future_plugins_and_process_session_tools_need_no_new_allowlist(self):
        names=['terminal','process','process_manage','session_search','session_spawn',
               'kanban_create','kanban_link','message_agent','future_plugin_v99']
        m=plugin();self.assertEqual([t['function']['name'] for t in m._host_tools([tool(n) for n in names])],names)

    def test_exact_host_tool_surface_prevents_a_second_mcp_toolset(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));seen={}
        def run(prompt,*args,**kwargs):seen.update(kwargs);return 'done',[]
        with patch.object(client,'_run_turn',side_effect=run):
            client._create_chat(messages=[{'role':'user','content':'continue'}],tools=[])
        self.assertIs(seen['attach_mcp'],False)

    def test_msp_reader_dispatcher_ends_at_eof(self):
        import io,time
        m=plugin();proc=type('Proc',(),{'stdin':io.StringIO(),'stdout':[],'stderr':[]})()
        session=m._ServeSession(proc,time.monotonic()+1,on_release=lambda:None)
        session._dispatch_thread.join(timeout=.5)
        self.assertFalse(session._dispatch_thread.is_alive())
        self.assertEqual(session.poll_note(.1)['method'],'_transport/closed')

    def test_transport_exit_after_submission_terminates_without_exec_replay(self):
        m=plugin();host,_=seams._fake_msp(m,notes=[{'method':'_transport/closed','params':{}}])
        client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        with patch.object(m,'_build_subprocess_env',return_value={}),host,patch.object(m,'run_exec_completion') as fallback:
            with self.assertRaisesRegex(m.MuseTurnError,'transport closed'):
                client._run_turn('probe',model_id='test',timeout_seconds=600,on_delta=None)
            fallback.assert_not_called()

    def test_native_cross_thread_abort_invokes_the_provider_cancel_hook(self):
        import threading
        from agent.client_lifecycle import ClientLifecycleMixin
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));events=[]
        class Proc:
            def terminate(self):events.append('terminate')
            def wait(self,**kwargs):return 0
        client._track(Proc())
        class Owner:
            def __init__(self):self.slot={'client':client,'poisoned':False};self.lock=threading.RLock()
            def _openai_client_lock(self):return self.lock
            def _request_slot(self,name):return self.slot
            def _client_log_context(self):return ''
            def _force_close_tcp_sockets(self,c):raise AssertionError('non-HTTP provider fell through to TCP abort')
        owner=Owner();ClientLifecycleMixin._abort_request_slot_client(owner,'_openai_request_client',client,reason='review probe')
        self.assertEqual(events,['terminate']);self.assertFalse(client._active_processes)
        self.assertTrue(owner.slot['poisoned'])
        with self.assertRaises(m.MuseTurnError):client._run_turn('probe',model_id='test',timeout_seconds=1,on_delta=None)

    def test_exec_fallback_is_tracked_so_cancel_cannot_leave_it_running(self):
        m=plugin();events=[]
        class Proc:
            returncode=0
            def communicate(self,**kwargs):events.append('communicate');return seams._jsonl(seams._term('ok')),''
        child=Proc()
        with patch.object(m,'_build_subprocess_env',return_value={}),patch.object(m.subprocess,'Popen',return_value=child):
            text=m.run_exec_completion('probe',model='test',cwd=str(ROOT),timeout_seconds=2,track=lambda new,old=None:events.append((new is child,old is child)))
        self.assertEqual(text,'ok');self.assertEqual(events,[(True,False),'communicate',(False,True)])

    def test_real_native_error_classifier_does_not_replay_ambiguous_turns(self):
        from agent.error_classifier import classify_api_error
        import providers
        m=plugin()
        with patch.object(providers,'get_provider_profile',return_value=m.muse_code_exec):
            for error in [m.MuseTurnError('submitted then failed'),m.MusePolicyError('policy refused')]:
                verdict=classify_api_error(error,provider='muse-code-exec',model='test')
                self.assertFalse(verdict.retryable);self.assertFalse(verdict.should_fallback)
                self.assertFalse(verdict.should_compress);self.assertFalse(verdict.should_rotate_credential)

    def test_cancel_before_tracking_terminates_late_child_and_refuses_fallback(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT));events=[]
        class Proc:
            def terminate(self):events.append('terminate')
            def wait(self,**kwargs):return 0
        client.cancel()
        with self.assertRaises(m.MuseTurnError):client._track(Proc())
        self.assertEqual(events,['terminate']);self.assertFalse(client._active_processes)
        with patch.object(m,'run_exec_completion') as fallback:
            with self.assertRaises(m.MuseTurnError):client._run_turn('probe',model_id='test',timeout_seconds=1,on_delta=None)
            fallback.assert_not_called()

    def test_real_native_empty_tool_transport_does_not_enable_standalone_mcp(self):
        from providers.base import ProviderProfile
        from agent.transports.chat_completions import ChatCompletionsTransport
        m=plugin();profile=ProviderProfile(name='muse-code-exec')
        profile.build_api_kwargs_extras=m.muse_code_exec.build_api_kwargs_extras
        kwargs=ChatCompletionsTransport().build_kwargs('test',[{'role':'user','content':'hi'}],tools=[],provider_profile=profile)
        self.assertNotIn('tools',kwargs)
        self.assertIs(kwargs['_hermes_host_tool_surface'],True)
        client=m.MuseCodeExecClient(exec_cwd=str(ROOT));seen={}
        def run(prompt,*args,**kwargs):seen.update(kwargs);return 'ok',[]
        with patch.object(client,'_run_turn',side_effect=run):client._create_chat(**kwargs)
        self.assertIs(seen['attach_mcp'],False)

    def test_call_history_keeps_identity_arguments_and_return_id(self):
        m=plugin();call=json.loads(response().removeprefix('<tool_call>').removesuffix('</tool_call>'))
        text=m._render_messages([{'role':'assistant','content':None,'tool_calls':[call]},
                                {'role':'tool','tool_call_id':'call_test_1','name':'message_agent','content':'{"status":"queued","delivery_id":"original-id"}'}])
        self.assertIn('message_agent',text);self.assertIn('call_test_1',text)
        self.assertIn('bounded exact candidate review',text);self.assertIn('original-id',text)

    def test_msp_subprocess_uses_the_context_profile_not_the_launch_profile(self):
        m=plugin()
        with tempfile.TemporaryDirectory(dir=os.getenv('HERMES_TEST_TMPDIR')) as temp:
            base=Path(temp);default=base/'default';active=base/'kira';default.mkdir();active.mkdir()
            (active/'config.yaml').write_text('approvals:\n  mode: off\n')
            token=set_hermes_home_override(active)
            try:
                with patch.dict(os.environ,{'HERMES_HOME':str(default)}):
                    host,_=seams._fake_msp(m)
                    with host:
                        m._msp_turn('probe',model_id='test',cwd=str(ROOT),timeout_seconds=5,on_delta=None,track=lambda *a:None)
                        actual=m.subprocess.Popen.call_args.kwargs['env']
                    self.assertEqual(Path(actual['HERMES_HOME']),active)
            finally:reset_hermes_home_override(token)

    def test_final_text_without_deltas_is_not_lost_in_stream(self):
        m=plugin();client=m.MuseCodeExecClient(exec_cwd=str(ROOT))
        with patch.object(client,'_run_turn',return_value=('final-only',[])):
            chunks=list(client._create_chat(messages=[{'role':'user','content':'hi'}],stream=True))
        self.assertEqual(''.join(getattr(choice.delta,'content',None) or '' for chunk in chunks for choice in chunk.choices),'final-only')

if __name__=='__main__':unittest.main(verbosity=2)
