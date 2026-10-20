"""Paired template TDA and template TDA plus a fixed semantic cache."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PREVIOUS = ROOT/'runs/flowers_cupl_manual_20261020_120643'
DESCRIPTIONS = ROOT/'runs/flowers_broad100_all_20261020_165242'
sys.path.insert(0, str(REPO))
from semantic_cache import FixedSemanticCache, fuse_semantic_logits
from tda_runner import update_cache, compute_cache_logits
from utils import get_config_file, softmax_entropy, get_entropy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def accuracy(predictions, labels):
    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)
    correct = predictions == labels
    counts = torch.bincount(labels, minlength=102)
    hits = torch.bincount(labels[correct], minlength=102)
    return {'samples':len(labels), 'correct':int(correct.sum()),
            'top1':float(correct.double().mean()*100),
            'mean_per_class_top1':float((hits.double()/counts).mean()*100)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-weight', type=float, default=1.)
    args = parser.parse_args()
    if not torch.isfinite(torch.tensor(args.semantic_weight)) or args.semantic_weight < 0:
        parser.error('semantic-weight must be finite and nonnegative')
    torch.manual_seed(1)
    started = time.perf_counter()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = ROOT/'runs'/('flowers_fixed_semantic_'+stamp)
    run.mkdir()
    print('RUN_DIR', run, flush=True)
    (ROOT/'runs/latest_flowers_fixed_semantic.txt').write_text(str(run)+'\n')
    assert digest(PREVIOUS/'visual_stream.pt') == 'a2567f1f255b26423daf89dc6881b6c63c753e2f1a0e05645119cfed6d2ea5b4'
    assert digest(DESCRIPTIONS/'prompts_all.json') == 'b4cdcea69694f27039fe365f304c4b8ce8f06b0b76c1e2087fa4a1b2b5e90029'
    assert digest(REPO/'configs/oxford_flowers.yaml') == digest(PREVIOUS/'oxford_flowers.yaml')
    for source, target in [(DESCRIPTIONS/'prompts_all.json','prompts_all.json'),
                           (PREVIOUS/'oxford_flowers.yaml','oxford_flowers.yaml'),
                           (REPO/'semantic_cache.py','semantic_cache.py'),
                           (REPO/'tda_runner.py','tda_runner_source.py'),
                           (REPO/'tests/test_fixed_semantic.py','test_fixed_semantic.py'),
                           (Path(__file__),'evaluate_fixed_semantic.py')]:
        shutil.copyfile(source, run/target)
    stream = torch.load(PREVIOUS/'visual_stream.pt', map_location='cpu', weights_only=True)
    names, labels, order = stream['classnames'], stream['labels'], stream['dataset_indices']
    images = stream['features'].cuda()
    template = torch.load(PREVIOUS/'template_text_prototypes.pt', map_location='cpu', weights_only=True)['initial'].cuda()
    all_embeddings = torch.load(DESCRIPTIONS/'description_embeddings.pt', map_location='cpu', weights_only=True)
    pool = json.loads((run/'prompts_all.json').read_text())
    assert len(images)==len(labels)==2463 and sorted(order)==list(range(2463))
    assert len(names)==102 and set(names)==set(pool)==set(all_embeddings)
    assert all(len(pool[n])==len(all_embeddings[n])==100 for n in names)
    cache = FixedSemanticCache({n:all_embeddings[n].cuda() for n in names}, names)
    previous_semantic = torch.load(DESCRIPTIONS/'generated_full100_text_prototypes.pt', map_location='cpu', weights_only=True)['initial'].cuda()
    assert torch.equal(cache.prototypes, previous_semantic), 'Semantic prototype changed from previous full100 experiment'
    torch.save({'classnames':names,'prototypes':cache.prototypes.cpu()},run/'semantic_prototypes.pt')
    cfg = get_config_file(str(REPO/'configs'), 'oxford_flowers')
    positive, negative = {}, {}
    pos, neg = cfg['positive'], cfg['negative']
    predictions = {'template_tda':[], 'fixed_semantic':[]}
    score_parts = {n:[] for n in ('template','positive','negative','semantic','template_tda','fixed_semantic')}
    trace = []
    torch.cuda.reset_peak_memory_stats()
    loop_started = time.perf_counter()
    for step in tqdm(range(len(images)), desc='Template TDA + fixed semantics'):
        image = images[step:step+1]
        original = 100. * image @ template
        loss = softmax_entropy(original)
        probabilities = original.softmax(1)
        pseudo = int(original.argmax(1))
        entropy = get_entropy(loss, template)
        if pos['enabled']:
            update_cache(positive, pseudo, [image,loss], pos['shot_capacity'])
        if neg['enabled'] and neg['entropy_threshold']['lower'] < entropy < neg['entropy_threshold']['upper']:
            update_cache(negative, pseudo, [image,loss,probabilities], neg['shot_capacity'], True)
        pos_logits, neg_logits = torch.zeros_like(original), torch.zeros_like(original)
        tda = original.clone()
        if pos['enabled'] and positive:
            pos_logits = compute_cache_logits(image,positive,pos['alpha'],pos['beta'],template)
            tda += pos_logits
        if neg['enabled'] and negative:
            neg_logits = compute_cache_logits(image,negative,neg['alpha'],neg['beta'],template,
                                              (neg['mask_threshold']['lower'],neg['mask_threshold']['upper']))
            tda -= neg_logits
        semantic = cache.logits(image)
        zero_control = fuse_semantic_logits(tda, semantic, 0.)
        assert torch.equal(zero_control, tda)
        final = fuse_semantic_logits(tda, semantic, args.semantic_weight)
        assert torch.equal(final, ((original+pos_logits)-neg_logits)+args.semantic_weight*semantic)
        predictions['template_tda'].append(int(tda.argmax(1)))
        predictions['fixed_semantic'].append(int(final.argmax(1)))
        for key,value in [('template',original),('positive',pos_logits),('negative',neg_logits),
                          ('semantic',semantic),('template_tda',tda),('fixed_semantic',final)]:
            score_parts[key].append(value.cpu())
        trace.append((step,int(order[step]),pseudo,float(entropy),
                      sum(map(len,positive.values())),sum(map(len,negative.values()))))
    torch.cuda.synchronize()
    replay_seconds = time.perf_counter()-loop_started
    with (PREVIOUS/'template_predictions.csv').open() as f:
        historical = list(csv.DictReader(f))
    assert predictions['template_tda'] == [int(r['tda_prediction']) for r in historical]
    control = accuracy(predictions['template_tda'],labels)
    result = accuracy(predictions['fixed_semantic'],labels)
    assert control['correct']==1761
    with (run/'predictions.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stream_index','dataset_index','label','template_tda_prediction','fixed_semantic_prediction'])
        writer.writerows((i,int(order[i]),int(labels[i]),predictions['template_tda'][i],predictions['fixed_semantic'][i]) for i in range(len(labels)))
    with (run/'visual_cache_trace.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stream_index','dataset_index','template_pseudo_label','normalized_entropy','positive_entries','negative_entries'])
        writer.writerows(trace)
    torch.save({n:torch.cat(v) for n,v in score_parts.items()},run/'logits.pt')
    gained = sum(a!=int(y) and b==int(y) for a,b,y in zip(predictions['template_tda'],predictions['fixed_semantic'],labels))
    lost = sum(a==int(y) and b!=int(y) for a,b,y in zip(predictions['template_tda'],predictions['fixed_semantic'],labels))
    summary = {'semantic_weight':args.semantic_weight,'template_tda':control,'fixed_semantic':result,
               'vs_template':{'gained':gained,'lost':lost,'net_correct':gained-lost,'delta_pp':result['top1']-control['top1']},
               'checks':{'zero_weight_matches_tda':True,'all_control_predictions_match_previous':True,
                         'semantic_prototypes_match_previous':True,'fusion_sum_matches':True,
                         'visual_cache_shared_and_driven_only_by_templates':True},
               'replay_seconds':replay_seconds,'total_seconds':time.perf_counter()-started,
               'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
               'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
               'notes':['Frozen ViT-B/16, saved 2463-image stream and seed-1 order reused.',
                        'All 100 descriptions/class equally fused, unchanged and fixed.',
                        'Template logits alone determine pseudo labels, entropy and visual cache writes.',
                        'Same TDA visual state shared by eta=0 control and fixed semantic condition.',
                        'Scale 100 for both text branches; no source image extraction included in timing.']}
    write_json(run/'metrics.json',summary)
    write_json(run/'manifest.json',{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
               'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               'git_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
               'stream_sha256':digest(PREVIOUS/'visual_stream.pt'),
               'template_prototypes_sha256':digest(PREVIOUS/'template_text_prototypes.pt'),
               'description_embeddings_sha256':digest(DESCRIPTIONS/'description_embeddings.pt'),
               'texts_sha256':digest(run/'prompts_all.json'),
               'config_sha256':digest(run/'oxford_flowers.yaml'),
               'semantic_module_sha256':digest(run/'semantic_cache.py'),
               'evaluation_script_sha256':digest(Path(__file__)),
               'tda_source_sha256':digest(REPO/'tda_runner.py'),
               'source_image_run':str(PREVIOUS),'source_semantic_run':str(DESCRIPTIONS),
               'finished_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
    report = (f"# Fixed semantic cache baseline\n\n"
              f"Template TDA: {control['top1']:.4f}% ({control['correct']}/2463).\n\n"
              f"Template TDA + fixed semantic cache, eta={args.semantic_weight}: "
              f"{result['top1']:.4f}% ({result['correct']}/2463).\n\n"
              f"Mean per-class accuracy: {result['mean_per_class_top1']:.4f}%.\n\n"
              f"All control predictions match the previous template TDA run. "
              f"Visual cache writes depend only on template predictions; semantic cache never updates.\n")
    (run/'report.md').write_text(report)
    print('RESULT',json.dumps(summary),flush=True)
    print('COMPLETE',run,flush=True)


if __name__=='__main__':
    main()
