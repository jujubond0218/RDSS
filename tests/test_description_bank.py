"""Offline tests: no paid requests, external networking or API keys required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/generate_description_bank.py'
spec = importlib.util.spec_from_file_location('bankgen',SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class BankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'base_url':'https://example.invalid/v1','model':'DeepSeek-V4.1-Flash',
                       'api_key':'TEST_SECRET_DO_NOT_EXPORT','temperature':.8,'max_tokens':4096,'timeout_seconds':10}
        self.config_path=self.root/'api.json'
        self.class_path=self.root/'classes.json'
        g.write_json(self.config_path,self.config)
        self.names=['petunia','mexican petunia','rose']
        g.write_json(self.class_path,{'dataset':'test','classes':[{'name':n,'hint':''} for n in self.names]})
        self.args=['--config',str(self.config_path),'--classes',str(self.class_path),
                   '--output',str(self.root/'out'),'--overview','2','--detail','2','--contrast','2','--batch-size','2']
        self.calls=[]

    def mock_api(self,config,body,record_dir,budget):
        if budget['used']>=budget['maximum']:
            raise g.RequestLimit('limit')
        budget['used']+=1
        self.calls.append(body)
        query=json.loads(body['messages'][1]['content'])
        name=query['target_class']['name']
        peers=[n for n in self.names if n!=name]
        if 'count' not in query:
            data={'superclass':'flower','attribute_axes':['shape','pattern'],'confusers':peers,'ambiguity_notes':''}
        else:
            data={'descriptions':[{'text':f'A photo of {name} with visible flower detail number {i}.',
                   'attribute':'shape','scope':'typical','compare_to':peers[i%len(peers)] if query['level']=='contrast' else None}
                   for i in range(query['count'])]}
        return {'data':data,'usage':{'total_tokens':12},'response_id':'mock','returned_model':'mock','attempt':1}

    def run_main(self,args=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return g.main(self.args if args is None else args)

    def test_dry_run_without_key_or_requests(self):
        self.config['api_key']=''
        g.write_json(self.config_path,self.config)
        with patch.object(g,'call_api',side_effect=AssertionError('No request allowed')):
            self.assertEqual(self.run_main(self.args+['--dry-run']),0)
        self.assertFalse((self.root/'out').exists())

    def test_explicit_pair_prompt_carries_balanced_assignments(self):
        item={'name':'pink primrose','hint':''}
        plan={'confusers':['primula','petunia','gaura','mexican petunia']}
        query=json.loads(g.description_prompt(item,plan,'contrast',20,[],0))
        self.assertEqual(query['comparison_targets'],plan['confusers']*5)
        self.assertIn('name BOTH',query['focus'])
        self.assertIn('same attribute axis',query['focus'])
        self.assertEqual(query['output_schema']['descriptions'][0]['compare_to'],'comparison_targets[i]')
        self.assertNotIn('5 to 36',json.dumps(query))

    def test_extra_metadata_keeps_generated_text(self):
        row={'text':'A pink primrose has four petals, whereas petunia has five fused lobes.',
             'attribute':'petal structure','scope':'typical','compare_to':'petunia','note':'extra'}
        value={'descriptions':[row],'compare_to':None}
        result=g.validate_descriptions(value,'pink primrose',{'confusers':['petunia']},'contrast',1)
        self.assertIs(result,value)
        self.assertIs(result['descriptions'][0],row)

    def test_end_to_end_exports_and_resume_no_paid_repeats(self):
        with patch.object(g,'call_api',side_effect=self.mock_api):
            self.assertEqual(self.run_main(self.args+['--limit','1']),0)
            self.assertEqual(len(self.calls),4)
            self.assertFalse(g.read_json(self.root/'out/status.json')['complete'])
            self.assertEqual(self.run_main(),0)
            self.assertEqual(len(self.calls),12)
            self.assertEqual(self.run_main(),0)
            self.assertEqual(len(self.calls),12)
        bank=g.read_json(self.root/'out/bank.json')
        flat=g.read_json(self.root/'out/prompts_all.json')
        self.assertEqual(set(flat),set(self.names))
        self.assertEqual(sum(map(len,flat.values())),18)
        self.assertEqual(bank['petunia']['descriptions'][-1]['level'],'contrast')
        self.assertTrue(g.read_json(self.root/'out/status.json')['complete'])
        for path in (self.root/'out').rglob('*.json'):
            self.assertNotIn(self.config['api_key'],path.read_text())

    def test_request_limit_preserves_completed_jobs(self):
        with patch.object(g,'call_api',side_effect=self.mock_api):
            self.assertEqual(self.run_main(self.args+['--max-requests','2']),2)
            self.assertEqual(len(self.calls),2)
            self.assertEqual(self.run_main(),0)
            self.assertEqual(len(self.calls),12)

    def test_changed_settings_cannot_mix_checkpoints(self):
        with patch.object(g,'call_api',side_effect=self.mock_api):
            self.run_main()
            with self.assertRaisesRegex(ValueError,'different generation'):
                self.run_main(self.args+['--overview','3'])
            with self.assertRaisesRegex(ValueError,'cannot shrink'):
                self.run_main(self.args+['--limit','1'])

    def test_comparison_metadata_is_not_content_filtered(self):
        plan={'superclass':'flower','attribute_axes':['shape','color'],'confusers':['banana'],'ambiguity_notes':''}
        self.assertIs(g.validate_plan(plan,'rose',self.names),plan)
        with self.assertRaisesRegex(ValueError,'nonempty list'):
            g.validate_plan({'confusers':[]},'rose',self.names)

    def test_overlapping_target_and_confuser_names(self):
        plan={'confusers':['petunia','rose']}
        value={'descriptions':[{'text':'A mexican petunia with wide purple petals and a pale center.',
                'attribute':'color','scope':'typical','compare_to':'petunia'}]}
        g.validate_descriptions(value,'mexican petunia',plan,'contrast',1)
        value['descriptions'][0]['text']='A mexican petunia unlike a petunia has purple petals.'
        g.validate_descriptions(value,'mexican petunia',plan,'contrast',1)
        inverse={'descriptions':[{'text':'A petunia unlike mexican petunia has ruffled pink petals.',
                 'attribute':'shape','scope':'typical','compare_to':'mexican petunia'}]}
        g.validate_descriptions(inverse,'petunia',{'confusers':['mexican petunia']},'contrast',1)

    def test_bad_response_never_becomes_completed_checkpoint(self):
        def invalid(*args):
            return {'data':{'broken':'response'}}
        with patch.object(g,'call_api',side_effect=invalid):
            self.assertEqual(self.run_main(),2)
        self.assertFalse(list((self.root/'out/jobs').glob('*.json')))

    def test_http_body_audit_and_secret_redaction(self):
        reply={'id':'fake','model':'alias-returned-by-gateway','choices':[{'finish_reason':'stop',
               'message':{'content':json.dumps({'ok':True})}}],'echo':self.config['api_key']}
        calls=[]
        def post(url,**kwargs):
            calls.append((url,kwargs))
            return types.SimpleNamespace(status_code=200,text=json.dumps(reply))
        module=types.SimpleNamespace(post=post,RequestException=Exception)
        raw=self.root/'raw'
        with patch.dict(sys.modules,{'requests':module}):
            result=g.call_api(self.config,{'model':self.config['model']},raw,{'used':0,'maximum':1})
        self.assertTrue(result['data']['ok'])
        self.assertEqual(calls[0][0],'https://example.invalid/v1/chat/completions')
        self.assertFalse(calls[0][1]['allow_redirects'])
        self.assertEqual(calls[0][1]['headers']['Authorization'],'Bearer '+self.config['api_key'])
        for path in raw.iterdir():
            self.assertNotIn(self.config['api_key'],path.read_text())

    def test_raw_response_reused_without_another_paid_request(self):
        output=self.root/'reused'
        raw=output/'raw'/'job'
        prompt='same request'
        body={'model':self.config['model'],'messages':[{'role':'system','content':g.SYSTEM},
              {'role':'user','content':prompt}],'temperature':self.config['temperature'],
              'max_tokens':self.config['max_tokens'],'stream':False}
        g.write_json(raw/'0001.request.json',{'body':body})
        g.write_json(raw/'0001.http.json',{'status_code':200})
        g.write_json(raw/'0001.response.txt',{'choices':[{'finish_reason':'stop',
                     'message':{'content':json.dumps({'confusers':['rose'],'extra':'preserved'})}}]})
        with patch.object(g,'call_api',side_effect=AssertionError('Already paid')):
            result=g.run_job('job',prompt,lambda v:g.validate_plan(v,'petunia',self.names),
                             self.config,output,{'used':0,'maximum':1})
        self.assertEqual(result,{'confusers':['rose'],'extra':'preserved'})
        self.assertTrue((output/'jobs/job.json').exists())

    def test_auth_error_does_not_retry(self):
        response=types.SimpleNamespace(status_code=401,text='Unauthorized')
        post=unittest.mock.Mock(return_value=response)
        with patch.dict(sys.modules,{'requests':types.SimpleNamespace(post=post,RequestException=Exception)}):
            with self.assertRaisesRegex(RuntimeError,'HTTP 401'):
                g.call_api(self.config,{},self.root/'raw',{'used':0,'maximum':10})
        self.assertEqual(post.call_count,1)


if __name__=='__main__':
    unittest.main()
