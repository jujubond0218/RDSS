"""Evaluate all generated descriptions with the existing paired TDA stream."""
import json
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/root/autodl-tmp/project_R')
REPO = ROOT/'RDSS'
SOURCE = ROOT/'experiments/offline_description_bank/outputs/flowers102_broad100_20261020'
PREVIOUS = ROOT/'runs/flowers_cupl_manual_20261020_120643'
sys.path.insert(0,str(ROOT/'experiments/cupl_flowers'))
from flowers_experiment import add_metrics
from compare_prompts import prototypes, replay, write_json, digest, get_config_file
import clip
from clip.clip import _tokenizer
import torch
from tqdm import tqdm


def main():
    started = time.perf_counter()
    torch.manual_seed(1)
    assert json.loads((SOURCE/'status.json').read_text())['complete']
    assert digest(REPO/'configs/oxford_flowers.yaml') == digest(PREVIOUS/'oxford_flowers.yaml')
    assert digest(PREVIOUS/'visual_stream.pt') == 'a2567f1f255b26423daf89dc6881b6c63c753e2f1a0e05645119cfed6d2ea5b4'
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = ROOT/'runs'/('flowers_broad100_all_'+stamp)
    run.mkdir()
    print('RUN_DIR',run,flush=True)
    (ROOT/'runs/latest_flowers_broad100_all.txt').write_text(str(run)+'\n')
    shutil.copyfile(SOURCE/'prompts_all.json',run/'prompts_all.json')
    shutil.copyfile(SOURCE/'bank.json',run/'bank.json')
    shutil.copyfile(SOURCE/'manifest.json',run/'generation_manifest.json')
    for name in ('visual_stream.pt','oxford_flowers.yaml','requirements-lock.txt'):
        shutil.copyfile(PREVIOUS/name,run/name)
    shutil.copyfile(__file__,run/'evaluate_flowers_broad100.py')
    shutil.copyfile(ROOT/'experiments/cupl_dtd/compare_prompts.py',run/'tda_replay_source.py')
    stream = torch.load(run/'visual_stream.pt',map_location='cpu',weights_only=True)
    names = stream['classnames']
    features = stream['features'].cuda()
    labels,order = stream['labels'],stream['dataset_indices']
    pool = json.loads((run/'prompts_all.json').read_text())
    assert len(names)==102 and set(names)==set(pool) and all(len(pool[n])==100 for n in names)
    assert len(features)==len(labels)==2463 and sorted(order)==list(range(2463))
    model,_ = clip.load('ViT-B/16')
    model.eval()
    embeddings, token_counts = {}, {}
    torch.cuda.reset_peak_memory_stats()
    encode_started = time.perf_counter()
    with torch.no_grad():
        for name in tqdm(names,desc='Encode all 100 descriptions/class'):
            sentences = pool[name]
            token_counts[name] = sum(len(_tokenizer.encode(s))+2>77 for s in sentences)
            vectors = model.encode_text(clip.tokenize(sentences,truncate=True).cuda())
            embeddings[name] = vectors/vectors.norm(dim=-1,keepdim=True)
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter()-encode_started
    torch.save({n:v.cpu() for n,v in embeddings.items()},run/'description_embeddings.pt')
    write_json(run/'tokenization.json',{'context_length':77,'truncate':True,'sentences':10200,
               'over_context_count':sum(token_counts.values()),'over_context_by_class':token_counts,
               'note':'All sentences retained; standard CLIP tokenization, no semantic filtering or rewriting.'})
    config = get_config_file(str(REPO/'configs'),'oxford_flowers')
    result = replay('generated_full100',features,labels,order,prototypes(embeddings,names),config,run)
    add_metrics(result,labels,names,run)
    previous_template = json.loads((PREVIOUS/'template_metrics.json').read_text())
    previous_cupl = json.loads((PREVIOUS/'official_full20_metrics.json').read_text())
    summary = {'result':result,'previous_references':{'template_tda':previous_template['tda_top1'],
               'official_cupl20_tda':previous_cupl['tda_top1']},
               'vs_template_pp':result['tda_top1']-previous_template['tda_top1'],
               'vs_official_cupl20_pp':result['tda_top1']-previous_cupl['tda_top1'],
               'text_encoding_seconds':encode_seconds,'total_seconds':time.perf_counter()-started,
               'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
               'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
               'truncated_sentences':sum(token_counts.values()),
               'notes':['All 10200 original descriptions; no selection, deduplication or rewriting.',
                        'Normalize each text embedding, equally average 100/class, normalize the class vector.',
                        'Frozen ViT-B/16, all 2463 test samples, saved image features and seed-1 order reused.',
                        'Positive and negative TDA caches start empty; no dynamic text selection or gradients.',
                        'Previous reference accuracies are recorded historical runs, not rerun controls.',
                        'Timings exclude original image extraction and are not end-to-end inference timings.']}
    write_json(run/'metrics.json',summary)
    write_json(run/'manifest.json',{'backbone':'ViT-B/16','stream_seed':1,'source_run':str(PREVIOUS),
               'description_source':str(SOURCE),'texts_sha256':digest(run/'prompts_all.json'),
               'visual_stream_sha256':digest(run/'visual_stream.pt'),
               'config_sha256':digest(run/'oxford_flowers.yaml'),
               'script_sha256':digest(Path(__file__)),
               'replay_source_sha256':digest(run/'tda_replay_source.py'),
               'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               'finished_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
    report = (f"# Flowers-102: TDA + all 100 generated descriptions/class\n\n"
              f"TDA Top-1: {result['tda_top1']:.4f}% ({result['tda_correct']}/2463).\n\n"
              f"Text-only Top-1: {result['text_only_top1']:.4f}%. "
              f"Mean per-class TDA Top-1: {result['tda_mean_per_class_top1']:.4f}%.\n\n"
              f"Historical template TDA: {previous_template['tda_top1']:.4f}%; "
              f"historical official CuPL20 + TDA: {previous_cupl['tda_top1']:.4f}%.\n\n"
              f"All descriptions equally fused; no semantic filtering. "
              f"CLIP truncation: {sum(token_counts.values())}/10200 sentences.\n\n"
              f"Same saved 2463-image stream, seed 1; fresh positive/negative caches. "
              f"No dynamic RDSS selection. Timings exclude original image extraction.\n")
    (run/'report.md').write_text(report)
    print('RESULT',json.dumps({k:result[k] for k in ('tda_top1','tda_correct','text_only_top1','tda_mean_per_class_top1')}),flush=True)
    print('COMPLETE',run,flush=True)


if __name__=='__main__':
    main()
