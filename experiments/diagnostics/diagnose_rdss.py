"""Post-hoc diagnostics; replay saved gates, without modifying production algorithms."""
from collections import deque
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import torch
from tqdm import tqdm

ROOT=Path('/root/autodl-tmp/project_R')
REPO=ROOT/'RDSS'
sys.path.insert(0,str(REPO))
from rdss import RDSSCache
from semantic_cache import FixedSemanticCache


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def acc(scores,labels):return int((scores.argmax(1)==labels).sum())


def margin(scores,labels):
    true=scores.gather(1,labels[:,None]).squeeze(1)
    other=scores.clone();other.scatter_(1,labels[:,None],-torch.inf)
    return true-other.max(1).values


@torch.no_grad()
def analyze(run_name,output):
    source=ROOT/'runs'/run_name
    flower=run_name.startswith('flowers')
    if flower:
        stream=torch.load(ROOT/'runs/flowers_cupl_manual_20261020_120643/visual_stream.pt',weights_only=True)
        saved=torch.load(ROOT/'runs/flowers_broad100_all_20261020_165242/description_embeddings.pt',weights_only=True)
        tda_key,fixed_key='template_tda','fixed_semantic'
        phases=[0]*len(stream['labels'])
    else:
        stream=torch.load(source/'visual_stream.pt',weights_only=True)
        saved=torch.load(source/'description_embeddings.pt',weights_only=True)
        tda_key,fixed_key='tda','fixed_all'
        phases=[r['phase'] for r in json.loads((source/'dataset_manifest.json').read_text())['stream']]
    names=stream['classnames'];labels=torch.tensor(stream['labels'])
    features=stream['features'].cuda()
    embeddings={n:saved[n].cuda() for n in names}
    snapshots=torch.load(source/'logits.pt',weights_only=True)
    tda=snapshots[tda_key]
    fixed=FixedSemanticCache(embeddings,names)
    params=dict(k=8,memory_capacity=32,decay=.9,entropy_threshold=.2)
    configs={'relevance_only':(0.,0.),'discrimination_only':(1.,0.),'redundancy_only':(0.,.1),'rdss':(1.,.1)}
    models={n:RDSSCache(embeddings,names,**params,discrimination_weight=d,redundancy_weight=r) for n,(d,r) in configs.items()}
    events=[json.loads(s) for s in (source/'semantic_updates.jsonl').read_text().splitlines()]
    if flower:events=[e for e in events if e['condition']=='rdss']
    bystep={e['step']:e for e in events}
    history=[deque(maxlen=32) for _ in names]
    combined={n:[] for n in models};semantic={n:[] for n in ['fixed','rdss']}
    trace=[];selection=[]
    for step in tqdm(range(len(features)),desc='Diagnosis '+run_name):
        image=features[step:step+1]
        fixed_sem=fixed.logits(image).cpu()
        assert torch.equal(tda[step:step+1]+fixed_sem,snapshots[fixed_key][step:step+1])
        semantic['fixed'].append(fixed_sem)
        for name,model in models.items():
            sem=model.logits(image).cpu()
            scores=tda[step:step+1]+sem
            combined[name].append(scores)
            if name=='rdss':
                semantic[name].append(sem)
                assert torch.equal(scores,snapshots['rdss'][step:step+1])
        y=int(labels[step]); queue=history[y]
        weights=[q*.9**(len(queue)-1-i) for i,(_,q,_) in enumerate(queue)]
        total=sum(weights)
        trace.append({'step':step,'phase':phases[step],'true_class_prior_updates':models['rdss'].accepted[y],
                      'same_phase_weight':None if not queue else sum(w for w,(p,_,_) in zip(weights,queue) if p==phases[step])/total,
                      'wrong_label_weight':None if not queue else sum(w for w,(_,_,wrong) in zip(weights,queue) if wrong)/total})
        if step in bystep:
            e=bystep[step];c=e['class_index']
            observed={}
            for name,model in models.items():
                result=model.observe(image[0],c,e['entropy'],True,e['hard_classes'],step)
                observed[name]=set(result['selected_indices'])
                if name=='rdss':assert result['selected_indices']==e['selected_indices']
            history[c].append((phases[step],1-e['entropy'],c!=y))
            if flower:
                selected=models['rdss'].selected[c]
                plain=models['relevance_only'].selected[c]
                rel=embeddings[names[c]].float()@models['rdss'].memory.anchor(c)
                selection.append({'step':step,'top8_set_overlap':len(observed['rdss']&observed['relevance_only'])/8,
                                  'rdss_minus_topk_relevance':float(rel[selected].mean()-rel[plain].mean())})
    combined={n:torch.cat(v) for n,v in combined.items()}
    semantic={n:torch.cat(v) for n,v in semantic.items()}
    if flower:assert torch.equal(combined['relevance_only'],snapshots['recent_topk'])
    fixed_scores=snapshots[fixed_key];dyn_scores=combined['rdss']
    fixed_ok=fixed_scores.argmax(1)==labels;dyn_ok=dyn_scores.argmax(1)==labels
    lost=fixed_ok&~dyn_ok;gained=~fixed_ok&dyn_ok
    true_sem_delta=(semantic['rdss']-semantic['fixed']).gather(1,labels[:,None]).squeeze(1)
    semantic_margin_delta=margin(semantic['rdss'].float(),labels)-margin(semantic['fixed'].float(),labels)
    output_dir=output/run_name;output_dir.mkdir()
    with (output_dir/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[*trace[0],'lost','gained','true_semantic_score_delta','semantic_margin_delta','fixed_margin','rdss_margin'])
        writer.writeheader()
        fm,dm=margin(fixed_scores.float(),labels),margin(dyn_scores.float(),labels)
        for i,row in enumerate(trace):
            writer.writerow({**row,'lost':bool(lost[i]),'gained':bool(gained[i]),
                             'true_semantic_score_delta':float(true_sem_delta[i]),'semantic_margin_delta':float(semantic_margin_delta[i]),
                             'fixed_margin':float(fm[i]),'rdss_margin':float(dm[i])})
    def subset(mask):
        indices=mask.nonzero().flatten().tolist()
        current=[trace[i]['same_phase_weight'] for i in indices if trace[i]['same_phase_weight'] is not None]
        wrong=[trace[i]['wrong_label_weight'] for i in indices if trace[i]['wrong_label_weight'] is not None]
        return {'count':len(indices),'true_semantic_score_increased':int((mask&(true_sem_delta>0)).sum()),
                'semantic_margin_decreased':int((mask&(semantic_margin_delta<0)).sum()),
                'mean_true_semantic_score_delta':float(true_sem_delta[mask].float().mean()),
                'mean_semantic_margin_delta':float(semantic_margin_delta[mask].mean()),
                'mean_current_phase_weight':None if not current else sum(current)/len(current),
                'mean_wrong_label_weight':None if not wrong else sum(wrong)/len(wrong)}
    phase_memory=[]
    for phase in sorted(set(phases)):
        rows=[r for r in trace if r['phase']==phase]
        def mean(chunk):
            values=[r['same_phase_weight'] for r in chunk if r['same_phase_weight'] is not None]
            return None if not values else sum(values)/len(values)
        phase_memory.append({'phase':phase,'first35_current_phase_weight':mean(rows[:35]),'last35_current_phase_weight':mean(rows[-35:])})
    summary={'samples':len(labels),'source':str(source),'control_logits_exactly_reproduced':True,
             'correct':{'tda':acc(tda,labels),'fixed':acc(fixed_scores,labels),**{n:acc(v,labels) for n,v in combined.items()}},
             'semantic_only_correct':{n:acc(v,labels) for n,v in semantic.items()},
             'overall_semantic_margin_delta':float(semantic_margin_delta.mean()),
             'lost_cases':subset(lost),'gained_cases':subset(gained),
             'phase_memory':phase_memory,
             'accepted_pseudo_label_errors':sum(e['class_index']!=int(labels[e['step']]) for e in events),
             'accepted':len(events)}
    if flower:
        summary['component_diagnostics']={'mean_top8_set_overlap':sum(s['top8_set_overlap'] for s in selection)/len(selection),
               'mean_selected_relevance_change':sum(s['rdss_minus_topk_relevance'] for s in selection)/len(selection)}
        dump(output_dir/'selection_diagnostics.json',selection)
    dump(output_dir/'summary.json',summary)
    torch.save({'semantic':semantic,'combined':combined},output_dir/'diagnostic_logits.pt')
    print('SUMMARY',json.dumps(summary),flush=True)
    return summary


@torch.no_grad()
def main():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output=ROOT/'runs'/('rdss_diagnosis_'+stamp);output.mkdir()
    (ROOT/'runs/latest_rdss_diagnosis.txt').write_text(str(output)+'\n')
    (output/'diagnose_rdss.py').write_bytes(Path(__file__).read_bytes())
    (output/'algorithm_sha256.txt').write_text(hashlib.sha256((REPO/'rdss.py').read_bytes()).hexdigest()+'\n')
    print('RUN_DIR',output,flush=True)
    summaries={}
    for name in ['flowers_rdss_first_20261020_201102','pacs_minimal_shift_20261020_210546','nico_minimal_shift_20261020_220735']:
        summaries[name]=analyze(name,output)
    dump(output/'summary.json',summaries)
    print('COMPLETE',output,flush=True)


if __name__=='__main__':main()
