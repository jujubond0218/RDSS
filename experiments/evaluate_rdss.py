"""First paired RDSS experiment on the frozen Flowers-102 stream."""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import torch
from tqdm import tqdm

from evaluate_fixed_semantic import ROOT, REPO, PREVIOUS, DESCRIPTIONS, accuracy, digest, write_json
from rdss import RDSSCache
from semantic_cache import FixedSemanticCache, fuse_semantic_logits
from utils import get_config_file
from visual_cache import TemplateVisualCache

FIXED_RUN = ROOT/'runs/flowers_fixed_semantic_20261020_194908'
PARAMS = {'k':8, 'memory_capacity':32, 'decay':.9, 'entropy_threshold':.2,
          'discrimination_weight':1., 'redundancy_weight':.1}


@torch.no_grad()
def main():
    started = time.perf_counter()
    torch.manual_seed(1)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = ROOT/'runs'/('flowers_rdss_first_'+stamp)
    run.mkdir()
    (ROOT/'runs/latest_flowers_rdss_first.txt').write_text(str(run)+'\n')
    print('RUN_DIR',run,flush=True)
    write_json(run/'parameters.json',{'rdss':PARAMS,'hard_classes':3,'semantic_weight':1.,
               'reliability':'template entropy <= 0.2 and template/TDA top1 agreement',
               'timing':'predict first; then update semantic memory for future samples',
               'memory':'recent reliable visual evidence, separate from unchanged TDA cache slots',
               'recency_unit':'accepted arrivals of the same pseudo class; older entries decay by 0.9'})
    for source,target in [(REPO/'rdss.py','rdss.py'),(REPO/'semantic_cache.py','semantic_cache.py'),
                           (REPO/'visual_cache.py','visual_cache.py'),
                           (REPO/'tda_runner.py','tda_runner_source.py'),
                           (REPO/'tests/test_rdss.py','test_rdss.py'),
                           (REPO/'tests/test_fixed_semantic.py','test_fixed_semantic.py'),
                           (Path(__file__),'evaluate_rdss.py'),
                           (REPO/'experiments/evaluate_fixed_semantic.py','evaluate_fixed_semantic.py'),
                           (DESCRIPTIONS/'prompts_all.json','prompts_all.json'),
                           (PREVIOUS/'oxford_flowers.yaml','oxford_flowers.yaml')]:
        shutil.copy2(source,run/target)
    assert digest(PREVIOUS/'visual_stream.pt')=='a2567f1f255b26423daf89dc6881b6c63c753e2f1a0e05645119cfed6d2ea5b4'
    assert digest(run/'prompts_all.json')=='b4cdcea69694f27039fe365f304c4b8ce8f06b0b76c1e2087fa4a1b2b5e90029'
    assert digest(REPO/'configs/oxford_flowers.yaml')==digest(run/'oxford_flowers.yaml')
    stream = torch.load(PREVIOUS/'visual_stream.pt',map_location='cpu',weights_only=True)
    names,labels,order = stream['classnames'],stream['labels'],stream['dataset_indices']
    features = stream['features'].cuda()
    assert len(features)==len(labels)==2463 and sorted(order)==list(range(2463))
    template = torch.load(PREVIOUS/'template_text_prototypes.pt',map_location='cpu',weights_only=True)['initial'].cuda()
    saved = torch.load(DESCRIPTIONS/'description_embeddings.pt',map_location='cpu',weights_only=True)
    pool = json.loads((run/'prompts_all.json').read_text())
    assert set(names)==set(pool)==set(saved) and all(len(pool[n])==len(saved[n])==100 for n in names)
    embeddings = {n:saved[n].cuda() for n in names}
    fixed = FixedSemanticCache(embeddings,names)
    models = {'recent_topk':RDSSCache(embeddings,names,**{**PARAMS,'discrimination_weight':0.,'redundancy_weight':0.}),
              'rdss':RDSSCache(embeddings,names,**PARAMS)}
    for model in models.values(): assert torch.equal(model.prototypes,fixed.prototypes)
    visual = TemplateVisualCache(template,get_config_file(str(REPO/'configs'),'oxford_flowers'))
    predictions = {n:[] for n in ('template_tda','fixed_semantic','recent_topk','rdss')}
    logits = {n:[] for n in predictions}
    events = {n:[] for n in models}
    cache_trace = []
    torch.cuda.reset_peak_memory_stats()
    replay_started = time.perf_counter()
    with (run/'semantic_updates.jsonl').open('w') as output:
        for step in tqdm(range(len(features)),desc='Paired RDSS first round'):
            image = features[step:step+1]
            obs = visual.predict(image)
            scores = {'template_tda':obs.tda,
                      'fixed_semantic':fuse_semantic_logits(obs.tda,fixed.logits(image),1.)}
            for name,model in models.items():
                scores[name] = fuse_semantic_logits(obs.tda,model.logits(image),1.)
            for name,score in scores.items():
                predictions[name].append(int(score.argmax(1)))
                logits[name].append(score.cpu())
            cache_trace.append((step,obs.pseudo_label,obs.entropy,
                                sum(map(len,visual.positive.values())),sum(map(len,visual.negative.values()))))
            # Current sample cannot use the descriptions it is about to select.
            consistent = obs.pseudo_label == int(obs.tda.argmax(1))
            hard = [int(c) for c in obs.template.topk(4,dim=1).indices[0] if int(c)!=obs.pseudo_label][:3]
            for name,model in models.items():
                event = model.observe(image[0],obs.pseudo_label,obs.entropy,consistent,hard,step)
                if event is not None:
                    events[name].append(event)
                    output.write(json.dumps({'condition':name,**event})+'\n')
    torch.cuda.synchronize()
    seconds = time.perf_counter()-replay_started
    with (FIXED_RUN/'predictions.csv').open() as f: previous = list(csv.DictReader(f))
    for condition,column in [('template_tda','template_tda_prediction'),('fixed_semantic','fixed_semantic_prediction')]:
        assert predictions[condition]==[int(r[column]) for r in previous],condition+' control changed'
    results = {name:accuracy(preds,labels) for name,preds in predictions.items()}
    assert results['template_tda']['correct']==1761 and results['fixed_semantic']['correct']==1814
    with (run/'predictions.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stream_index','dataset_index','label',*predictions])
        writer.writerows((i,int(order[i]),int(labels[i]),*(predictions[n][i] for n in predictions)) for i in range(len(labels)))
    with (run/'visual_cache_trace.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['step','template_pseudo_label','entropy','positive_entries','negative_entries']); writer.writerows(cache_trace)
    paired = {}
    for ref in ('fixed_semantic','recent_topk'):
        gained = sum(a!=int(y) and b==int(y) for a,b,y in zip(predictions[ref],predictions['rdss'],labels))
        lost = sum(a==int(y) and b!=int(y) for a,b,y in zip(predictions[ref],predictions['rdss'],labels))
        paired[ref]={'gained':gained,'lost':lost,'net_correct':gained-lost,
                     'delta_pp':results['rdss']['top1']-results[ref]['top1']}
    final = {}
    for name,model in models.items():
        final[name] = {c:{'accepted':model.accepted[i],
                         'recent_count':len(model.memory.entries[i]),
                         'selected_indices':model.selected[i],
                         'selected_texts':None if model.selected[i] is None else [pool[c][j] for j in model.selected[i]]}
                       for i,c in enumerate(names)}
        torch.save({'initial':fixed.prototypes.cpu(),'final':model.prototypes.cpu()},run/(name+'_prototypes.pt'))
    write_json(run/'final_selections.json',final)
    torch.save({n:torch.cat(v) for n,v in logits.items()},run/'logits.pt')
    summary = {'results':results,'rdss_vs':paired,'parameters':PARAMS,
               'semantic_weight':1.,'hard_classes':3,
               'updates':{n:{'accepted':len(events[n]),'classes_updated':sum(x>0 for x in model.accepted),
                              'selection_changes':sum(e['selection_changed'] for e in events[n]),
                              'updates_with_negative_evidence':sum(bool(e['negative_classes_with_evidence']) for e in events[n])}
                          for n,model in models.items()},
               'checks':{'template_and_fixed_predictions_match_baseline':True,'cold_start_equals_fixed':True},
               'replay_seconds':seconds,'total_seconds':time.perf_counter()-started,
               'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
               'notes':['Same saved 2463-image stream, seed 1, frozen ViT-B/16 and description pool.',
                        'All conditions share the template-driven TDA visual state.',
                        'Template entropy <=0.2 and template/TDA top1 agreement gate semantic updates.',
                        'No labels enter RDSS memory or selection. All semantic updates follow current prediction.',
                        'Recent semantic evidence queue is separate from unchanged TDA best-entropy cache slots.',
                        'Parameters fixed before this run; no test-score search. Timings exclude feature extraction.']}
    write_json(run/'metrics.json',summary)
    write_json(run/'manifest.json',{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
               'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               'source_hashes':{n:digest(run/n) for n in ['rdss.py','visual_cache.py','semantic_cache.py','evaluate_rdss.py','evaluate_fixed_semantic.py','parameters.json']},
               'visual_stream_sha256':digest(PREVIOUS/'visual_stream.pt'),
               'description_embeddings_sha256':digest(DESCRIPTIONS/'description_embeddings.pt'),
               'texts_sha256':digest(run/'prompts_all.json'),
               'finished_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
    lines=['# Flowers-102 RDSS first round','', '| Condition | Top-1 | Correct/2463 | Mean per class |', '| --- | ---: | ---: | ---: |']
    for n,r in results.items(): lines.append(f"| {n} | {r['top1']:.4f}% | {r['correct']} | {r['mean_per_class_top1']:.4f}% |")
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print('RESULT',json.dumps({'results':results,'rdss_vs':paired,'updates':summary['updates']}),flush=True)
    print('COMPLETE',run,flush=True)


if __name__=='__main__':
    main()
