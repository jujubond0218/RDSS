"""Paired, fixed-parameter attribute-mismatch noise stress test."""
import csv
from datetime import datetime
import gc
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import torch
from tqdm import tqdm

from evaluate_fixed_semantic import ROOT, REPO, PREVIOUS, DESCRIPTIONS, accuracy, digest, write_json
from evaluate_rdss import COMMON as PARAMS
import clip
from clip.simple_tokenizer import SimpleTokenizer
from rdss import RDSSCache
from semantic_cache import FixedSemanticCache, fuse_semantic_logits
from utils import get_config_file
from visual_cache import TemplateVisualCache

SEED = 20261020
LEVELS = (0, 25, 50)
CLEAN_RUN = ROOT/'runs/flowers_rdss_first_20261020_201102'


def build_noise(pool, names):
    rng = random.Random(SEED)
    eligible = {}
    patterns = {n:re.compile(r'(?<!\w)'+re.escape(n)+r'(?!\w)', re.I) for n in names}
    for n in names:
        eligible[n] = [i for i,t in enumerate(pool[n][:20]) if patterns[n].search(t)]
        assert eligible[n], 'No overview containing donor name: '+n
    generated, provenance = {}, []
    for target in names:
        generated[target] = []
        donors = [n for n in names if n != target]
        for index in range(max(LEVELS)):
            donor = rng.choice(donors)
            source_index = rng.choice(eligible[donor])
            original = pool[donor][source_index]
            text, replaced = patterns[donor].subn(lambda _:target, original)
            assert replaced > 0 and donor != target
            generated[target].append(text)
            provenance.append({'target_class':target,'noise_index':index,'donor_class':donor,
                               'donor_index':source_index,'original':original,'noise_text':text})
    return generated, provenance


@torch.no_grad()
def main():
    started = time.perf_counter()
    torch.manual_seed(1)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = ROOT/'runs'/('flowers_semantic_noise_'+stamp)
    run.mkdir()
    (ROOT/'runs/latest_flowers_semantic_noise.txt').write_text(str(run)+'\n')
    print('RUN_DIR',run,flush=True)
    write_json(run/'parameters.json',{'seed':SEED,'added_per_class':LEVELS,'rdss':PARAMS,
               'semantic_weight':1.,'hard_classes':3,
               'noise':'Retarget another flower overview by replacing only its class name; retain donor attributes.',
               'pool':'All original 100 retained; 25-noise condition is prefix of 50-noise condition.',
               'selection':'No test labels, image inspection, or accuracy feedback used to construct noise.'})
    for source,target in [(Path(__file__),'evaluate_semantic_noise.py'),
                           (REPO/'rdss.py','rdss.py'),(REPO/'semantic_cache.py','semantic_cache.py'),
                           (REPO/'visual_cache.py','visual_cache.py'),(REPO/'tda_runner.py','tda_runner_source.py'),
                           (REPO/'experiments/evaluate_rdss.py','evaluate_rdss.py'),
                           (REPO/'experiments/evaluate_fixed_semantic.py','evaluate_fixed_semantic.py'),
                           (PREVIOUS/'oxford_flowers.yaml','oxford_flowers.yaml'),
                           (DESCRIPTIONS/'prompts_all.json','prompts_original.json')]:
        shutil.copy2(source,run/target)
    assert digest(PREVIOUS/'visual_stream.pt')=='a2567f1f255b26423daf89dc6881b6c63c753e2f1a0e05645119cfed6d2ea5b4'
    assert digest(run/'prompts_original.json')=='b4cdcea69694f27039fe365f304c4b8ce8f06b0b76c1e2087fa4a1b2b5e90029'
    assert digest(REPO/'configs/oxford_flowers.yaml')==digest(run/'oxford_flowers.yaml')
    stream = torch.load(PREVIOUS/'visual_stream.pt',map_location='cpu',weights_only=True)
    names,labels = stream['classnames'],stream['labels']
    assert len(labels)==2463 and len(names)==102
    pool = json.loads((run/'prompts_original.json').read_text())
    noise, provenance = build_noise(pool,names)
    assert noise==build_noise(pool,names)[0]
    write_json(run/'noise_provenance.json',provenance)
    for level in LEVELS:
        combined = {n:pool[n]+noise[n][:level] for n in names}
        assert all(len(combined[n])==100+level and combined[n][:100]==pool[n] for n in names)
        write_json(run/f'prompts_add{level}.json',combined)
    tokenizer = SimpleTokenizer()
    all_texts = [t for n in names for t in noise[n]]
    lengths = [len(tokenizer.encode(t))+2 for t in all_texts]
    write_json(run/'token_lengths.json',{'max':max(lengths),'over_77':sum(x>77 for x in lengths),
                                       'encoding':'clip.tokenize truncate=True, same as original pool'})
    model,_ = clip.load('ViT-B/16'); model.eval()
    encoded = []
    for start in tqdm(range(0,len(all_texts),64),desc='Encode 5100 attribute-mismatch sentences'):
        vectors = model.encode_text(clip.tokenize(all_texts[start:start+64],truncate=True).cuda())
        vectors /= vectors.norm(dim=-1,keepdim=True)
        encoded.append(vectors.cpu())
    encoded = torch.cat(encoded)
    torch.save({n:encoded[i*50:(i+1)*50] for i,n in enumerate(names)},run/'noise_embeddings.pt')
    del model
    gc.collect(); torch.cuda.empty_cache()
    encoding_seconds = time.perf_counter()-started
    saved = torch.load(DESCRIPTIONS/'description_embeddings.pt',map_location='cpu',weights_only=True)
    fixed, models = {}, {}
    for level in LEVELS:
        embeddings = {n:torch.cat([saved[n],encoded[i*50:i*50+level]]).cuda() for i,n in enumerate(names)}
        fixed[level] = FixedSemanticCache(embeddings,names)
        models[level] = RDSSCache(embeddings,names,**PARAMS)
        assert torch.equal(fixed[level].prototypes,models[level].prototypes)
    template = torch.load(PREVIOUS/'template_text_prototypes.pt',map_location='cpu',weights_only=True)['initial'].cuda()
    visual = TemplateVisualCache(template,get_config_file(str(REPO/'configs'),'oxford_flowers'))
    images = stream['features'].cuda()
    predictions = {'template_tda':[],**{f'{method}_add{n}':[] for n in LEVELS for method in ('fixed','rdss')}}
    logits = {n:[] for n in predictions}
    updates = {n:[] for n in LEVELS}
    noise_selected = {n:0 for n in LEVELS}
    torch.cuda.reset_peak_memory_stats()
    replay_started = time.perf_counter()
    with (run/'semantic_updates.jsonl').open('w') as output:
        for step in tqdm(range(len(images)),desc='Paired clean/add25/add50'):
            image = images[step:step+1]
            obs = visual.predict(image)
            scores = {'template_tda':obs.tda}
            for level in LEVELS:
                scores[f'fixed_add{level}'] = fuse_semantic_logits(obs.tda,fixed[level].logits(image),1.)
                scores[f'rdss_add{level}'] = fuse_semantic_logits(obs.tda,models[level].logits(image, step),1.)
            for name,score in scores.items():
                predictions[name].append(int(score.argmax(1))); logits[name].append(score.cpu())
            consistent = obs.pseudo_label==int(obs.tda.argmax(1))
            hard = [int(c) for c in obs.template.topk(4,dim=1).indices[0] if int(c)!=obs.pseudo_label][:3]
            for level in LEVELS:
                event = models[level].observe(image[0],obs.pseudo_label,obs.entropy,consistent,hard,step)
                if event is not None:
                    count = sum(i>=100 for i in event['selected_indices'])
                    noise_selected[level] += count
                    updates[level].append(event)
                    output.write(json.dumps({'added_per_class':level,'selected_noise_count':count,**event})+'\n')
    torch.cuda.synchronize()
    replay_seconds = time.perf_counter()-replay_started
    previous = list(csv.DictReader((CLEAN_RUN/'predictions.csv').open()))
    for current,old in [('template_tda','template_tda'),('fixed_add0','fixed_semantic')]:
        assert predictions[current]==[int(r[old]) for r in previous],current+' changed'
    assert len(updates[0])==len(updates[25])==len(updates[50])==1482
    results = {n:accuracy(p,labels) for n,p in predictions.items()}
    diagnostics = {}
    for level in LEVELS:
        diagnostics[level] = {'pool_size':100+level,'added_pool_fraction':level/(100+level),
             'accepted_updates':len(updates[level]),'noise_selected_count':noise_selected[level],
             'noise_fraction_in_updated_selections':noise_selected[level]/(8*len(updates[level])),
             'fixed_delta_from_clean_pp':results[f'fixed_add{level}']['top1']-results['fixed_add0']['top1'],
             'rdss_delta_from_clean_pp':results[f'rdss_add{level}']['top1']-results['rdss_add0']['top1'],
             'rdss_minus_fixed_pp':results[f'rdss_add{level}']['top1']-results[f'fixed_add{level}']['top1']}
    with (run/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['step','dataset_index','label',*predictions])
        writer.writerows([i,int(stream['dataset_indices'][i]),int(labels[i]),*(predictions[n][i] for n in predictions)] for i in range(len(labels)))
    torch.save({n:torch.cat(v) for n,v in logits.items()},run/'logits.pt')
    write_json(run/'final_selections.json',{level:{c:models[level].selected[i] for i,c in enumerate(names)} for level in LEVELS})
    metrics={'results':results,'diagnostics':diagnostics,'checks':{'template_and_fixed_controls_match_first_round':True,
             'original_descriptions_preserved':True,'nested_noise_pools':True,'same_semantic_updates':True},
             'encoding_and_preparation_seconds':encoding_seconds,'replay_seconds':replay_seconds,
             'total_seconds':time.perf_counter()-started,'replay_gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
             'noise_type':'Synthetic cross-class attribute transfer, not measured natural LLM hallucination.'}
    write_json(run/'metrics.json',metrics)
    write_json(run/'manifest.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
             'source_hashes':{n:digest(run/n) for n in ['rdss.py','visual_cache.py','semantic_cache.py','evaluate_semantic_noise.py','parameters.json']},
             'visual_stream_sha256':digest(PREVIOUS/'visual_stream.pt'),'original_pool_sha256':digest(run/'prompts_original.json'),
             'noisy_pool_sha256':{n:digest(run/f'prompts_add{n}.json') for n in LEVELS}})
    lines=['# Flowers-102 synthetic attribute-noise test','','| Added/class | Fixed all | RDSS | Fixed delta | RDSS delta | Selected noise |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n,d in diagnostics.items():
        lines.append(f"| {n} | {results[f'fixed_add{n}']['top1']:.4f}% | {results[f'rdss_add{n}']['top1']:.4f}% | {d['fixed_delta_from_clean_pp']:+.4f} pp | {d['rdss_delta_from_clean_pp']:+.4f} pp | {100*d['noise_fraction_in_updated_selections']:.2f}% |")
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print('RESULT',json.dumps(metrics),flush=True)
    print('COMPLETE',run,flush=True)


if __name__=='__main__':
    main()
