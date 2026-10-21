"""Three-condition, 700-image PACS continual-domain smoke experiment."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import torch
from PIL import Image
from tqdm import tqdm

from evaluate_fixed_semantic import ROOT, REPO, write_json, digest
from evaluate_rdss import COMMON as PARAMS
from pacs_description_bank import build_bank
import clip
from rdss import RDSSCache
from semantic_cache import FixedSemanticCache, fuse_semantic_logits
from visual_cache import TemplateVisualCache
from utils import get_config_file


def stats(predictions, labels):
    hits = [int(p)==int(y) for p,y in zip(predictions,labels)]
    assert hits
    return {'samples':len(hits),'correct':sum(hits),'top1':100*sum(hits)/len(hits)}


@torch.no_grad()
def main():
    started = time.perf_counter()
    torch.manual_seed(1)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = ROOT/'runs'/('pacs_minimal_shift_'+stamp); run.mkdir()
    (ROOT/'runs/latest_pacs_minimal_shift.txt').write_text(str(run)+'\n')
    print('RUN_DIR',run,flush=True)
    data = ROOT/'data/pacs_minimal'
    manifest = json.loads((data/'manifest.json').read_text())
    stream,names = manifest['stream'],manifest['classnames']
    assert len(stream)==700 and len({r['source_row'] for r in stream})==700
    pool,meta = build_bank()
    assert set(names)==set(pool) and all(len(pool[n])==100 for n in names)
    write_json(run/'prompts_all.json',pool); write_json(run/'prompt_metadata.json',meta)
    write_json(run/'parameters.json',{'rdss':PARAMS,'semantic_weight':1.,'hard_classes':3,
               'backbone':'ViT-B/16','template':'a photo of a {}.',
               'domain_sequence':manifest['domain_sequence'],'per_class_per_phase':20,'seed':1,
               'description_source':'Fixed clauses composed in this conversation, 4 style renderings; no API.',
               'description_composition':'Each class: 20 overview, 60 detail, 20 contrast, 25 per style.',
               'visual_config':'Existing Oxford Flowers TDA settings reused unchanged, no PACS tuning.',
               'semantic_timing':'Predict first, then update. Neither labels nor domain IDs enter RDSS.'})
    for source,target in [(data/'manifest.json','dataset_manifest.json'),
                           (REPO/'configs/oxford_flowers.yaml','tda_config.yaml'),
                           (Path(__file__),'evaluate_pacs_minimal.py'),
                           (REPO/'experiments/pacs_description_bank.py','pacs_description_bank.py'),
                           (REPO/'experiments/prepare_pacs_minimal.py','prepare_pacs_minimal.py'),
                           (REPO/'rdss.py','rdss.py'),(REPO/'visual_cache.py','visual_cache.py'),
                           (REPO/'semantic_cache.py','semantic_cache.py'),(REPO/'tda_runner.py','tda_runner_source.py')]:
        shutil.copy2(source,run/target)
    model,preprocess = clip.load('ViT-B/16'); model.eval()
    features = []
    for start in tqdm(range(0,len(stream),32),desc='Encode 700 PACS images'):
        images=[]
        for row in stream[start:start+32]:
            path=data/row['path']; assert digest(path)==row['sha256']
            with Image.open(path) as image: images.append(preprocess(image.convert('RGB')))
        vectors = model.encode_image(torch.stack(images).cuda())
        vectors /= vectors.norm(dim=1,keepdim=True); features.append(vectors)
    features = torch.cat(features)
    template = model.encode_text(clip.tokenize(['a photo of a '+n+'.' for n in names]).cuda())
    template /= template.norm(dim=1,keepdim=True); template=template.T.contiguous()
    labels=[r['label'] for r in stream]
    config=get_config_file(str(REPO/'configs'),'oxford_flowers')
    smoke=TemplateVisualCache(template,config)
    smoke_preds=[int(smoke.predict(f.unsqueeze(0)).tda.argmax(1)) for f in features[:28]]
    smoke_result=stats(smoke_preds,labels[:28])
    write_json(run/'tda_smoke_28.json',smoke_result)
    print('TDA_SMOKE_28',json.dumps(smoke_result),flush=True)
    baseline=TemplateVisualCache(template,config)
    baseline_preds=[int(baseline.predict(f.unsqueeze(0)).tda.argmax(1)) for f in tqdm(features,desc='TDA baseline first')]
    print('TDA_BASELINE_700',json.dumps(stats(baseline_preds,labels)),flush=True)
    embeddings={}
    for n in names:
        vectors=model.encode_text(clip.tokenize(pool[n],truncate=False).cuda())
        vectors/=vectors.norm(dim=1,keepdim=True); embeddings[n]=vectors
    del model
    fixed=FixedSemanticCache(embeddings,names)
    dynamic=RDSSCache(embeddings,names,**PARAMS)
    assert torch.equal(fixed.prototypes,dynamic.prototypes)
    visual=TemplateVisualCache(template,config)
    torch.save({'features':features.cpu(),'labels':labels,'classnames':names,'template':template.cpu()},run/'visual_stream.pt')
    torch.save({n:v.cpu() for n,v in embeddings.items()},run/'description_embeddings.pt')
    predictions={n:[] for n in ('tda','fixed_all','rdss')}; logits={n:[] for n in predictions}
    events=[]; phase_selections={}
    torch.cuda.reset_peak_memory_stats(); replay_started=time.perf_counter()
    with (run/'semantic_updates.jsonl').open('w') as output:
        for step in tqdm(range(len(features)),desc='Paired full semantic/RDSS'):
            image=features[step:step+1]; obs=visual.predict(image)
            scores={'tda':obs.tda,'fixed_all':fuse_semantic_logits(obs.tda,fixed.logits(image),1.),
                    'rdss':fuse_semantic_logits(obs.tda,dynamic.logits(image, step),1.)}
            for name,score in scores.items():
                predictions[name].append(int(score.argmax(1))); logits[name].append(score.cpu())
            hard=[int(c) for c in obs.template.topk(4,dim=1).indices[0] if int(c)!=obs.pseudo_label][:3]
            event=dynamic.observe(image[0],obs.pseudo_label,obs.entropy,
                                  obs.pseudo_label==int(obs.tda.argmax(1)),hard,step)
            if event:
                # Domain information is added only to diagnostics after selection.
                styles=[meta[names[obs.pseudo_label]][j]['style'] for j in event['selected_indices']]
                event.update({'phase':stream[step]['phase'],'observed_domain':stream[step]['domain'],
                              'selected_styles':styles,'matching_style_count':styles.count(stream[step]['domain'])})
                events.append(event); output.write(json.dumps(event)+'\n')
            if (step+1)%140==0:
                phase_selections[str(stream[step]['phase'])]={c:dynamic.selected[i] for i,c in enumerate(names)}
    torch.cuda.synchronize(); replay_seconds=time.perf_counter()-replay_started
    assert predictions['tda']==baseline_preds
    results={n:stats(p,labels) for n,p in predictions.items()}
    phases=[]
    for phase,domain in enumerate(manifest['domain_sequence']):
        lo,hi=phase*140,(phase+1)*140
        es=[e for e in events if e['phase']==phase]
        phases.append({'phase':phase,'domain':domain,
                       'results':{n:stats(p[lo:hi],labels[lo:hi]) for n,p in predictions.items()},
                       'first_35':{n:stats(p[lo:lo+35],labels[lo:lo+35]) for n,p in predictions.items()},
                       'last_35':{n:stats(p[hi-35:hi],labels[hi-35:hi]) for n,p in predictions.items()},
                       'updates':len(es),'selected_style_counts':{style:sum(e['selected_styles'].count(style) for e in es) for style in ('photo','art_painting','cartoon','sketch')},
                       'matching_style_fraction':None if not es else sum(e['matching_style_count'] for e in es)/(8*len(es))})
    with (run/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','phase','domain','source_row','label',*predictions])
        writer.writerows([i,r['phase'],r['domain'],r['source_row'],r['label'],*(predictions[n][i] for n in predictions)] for i,r in enumerate(stream))
    write_json(run/'phase_selections.json',phase_selections)
    torch.save({n:torch.cat(v) for n,v in logits.items()},run/'logits.pt')
    torch.save({'fixed':fixed.prototypes.cpu(),
                'rdss_final':dynamic.read_prototypes(len(features)-1).cpu(),
                'rdss_selected':dynamic.prototypes.cpu()},run/'semantic_prototypes.pt')
    metrics={'results':results,'phases':phases,'accepted_updates':len(events),
             'accepted_wrong_pseudo_labels':sum(e['class_index']!=labels[e['step']] for e in events),
             'rdss_vs_fixed':{'gained':sum(a!=y and b==y for a,b,y in zip(predictions['fixed_all'],predictions['rdss'],labels)),
                              'lost':sum(a==y and b!=y for a,b,y in zip(predictions['fixed_all'],predictions['rdss'],labels))},
             'checks':{'tda_all_predictions_reproduced':True,'cold_start_matches_fixed':True,'unique_images':700},
             'replay_seconds':replay_seconds,'total_seconds':time.perf_counter()-started,
             'replay_gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    write_json(run/'metrics.json',metrics)
    write_json(run/'manifest.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               'source_hashes':{n:digest(run/n) for n in ['evaluate_pacs_minimal.py','pacs_description_bank.py','rdss.py','visual_cache.py','semantic_cache.py','prompts_all.json','parameters.json','dataset_manifest.json']}})
    lines=['# PACS 700-image continual-domain smoke test','','| Phase | TDA | Fixed all | RDSS | Matching-style selection |',
           '| --- | ---: | ---: | ---: | ---: |']
    for p in phases:
        r=p['results']; match=p['matching_style_fraction']
        lines.append(f"| {p['phase']} {p['domain']} | {r['tda']['top1']:.2f}% | {r['fixed_all']['top1']:.2f}% | {r['rdss']['top1']:.2f}% | {100*match:.2f}% |")
    lines.append('| Total | '+' | '.join(f"{results[n]['top1']:.2f}%" for n in predictions)+' | |')
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print('RESULT',json.dumps(metrics),flush=True); print('COMPLETE',run,flush=True)


if __name__=='__main__':
    main()
