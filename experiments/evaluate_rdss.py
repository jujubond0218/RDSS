"""Frozen-protocol, three-dataset component replay. No labels enter adaptation."""
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

ROOT = Path('/root/autodl-tmp/project_R')
REPO = ROOT / 'RDSS'
sys.path.insert(0, str(REPO))
from rdss import RDSSCache
from semantic_cache import FixedSemanticCache

SOURCES = {'flowers': 'flowers_rdss_first_20261020_201102',
           'pacs': 'pacs_minimal_shift_20261020_210546',
           'nico': 'nico_minimal_shift_20261020_220735'}
COMMON = dict(k=8, memory_capacity=32, decay=.9, entropy_threshold=.2,
              discrimination_weight=1., redundancy_weight=.1,
              time_half_life_per_class=4., prior_strength=4.)
CONDITIONS = {
    'aggregate_relevance': dict(discrimination_weight=0., time_half_life_per_class=0., prior_strength=0.),
    'margin_only': dict(time_half_life_per_class=0., prior_strength=0.),
    'margin_time': dict(time_half_life_per_class=4., prior_strength=0.),
    'margin_time_prior': dict(time_half_life_per_class=4., prior_strength=4.),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stats(scores, labels):
    predictions = scores.argmax(1)
    correct = predictions == labels
    classes = labels.unique()
    return {'samples': len(labels), 'correct': int(correct.sum()),
            'top1': 100 * float(correct.float().mean()),
            'class_mean_top1': 100 * float(torch.stack([correct[labels == c].float().mean() for c in classes]).mean())}


@torch.no_grad()
def evaluate(dataset, output, limit):
    started = time.perf_counter()
    source = ROOT / 'runs' / SOURCES[dataset]
    flowers = dataset == 'flowers'
    stream_path = (ROOT / 'runs/flowers_cupl_manual_20261020_120643/visual_stream.pt'
                   if flowers else source / 'visual_stream.pt')
    embeddings_path = (ROOT / 'runs/flowers_broad100_all_20261020_165242/description_embeddings.pt'
                       if flowers else source / 'description_embeddings.pt')
    stream = torch.load(stream_path, map_location='cpu', weights_only=True)
    saved = torch.load(embeddings_path, map_location='cpu', weights_only=True)
    original = torch.load(source / 'logits.pt', map_location='cpu', weights_only=True)
    names = stream['classnames']
    features = stream['features'].cuda()
    count = len(features) if limit == 0 else min(limit, len(features))
    # Scoring labels and phases are read only after all online predictions below.
    embeddings = {name: saved[name].cuda() for name in names}
    fixed = FixedSemanticCache(embeddings, names)
    tda_key, fixed_key = ('template_tda', 'fixed_semantic') if flowers else ('tda', 'fixed_all')
    tda = original[tda_key][:count]
    historical = original['rdss'][:count]
    events = [json.loads(line) for line in (source / 'semantic_updates.jsonl').read_text().splitlines()]
    if flowers:
        events = [event for event in events if event['condition'] == 'rdss']
    bystep = {event['step']: event for event in events}
    models = {name: RDSSCache(embeddings, names, **(COMMON | params))
              for name, params in CONDITIONS.items()}
    combined = {name: [] for name in models}
    semantic = {name: [] for name in models}
    full_semantic = []
    traces = []
    folder = output / dataset
    folder.mkdir()
    with (folder / 'semantic_updates.jsonl').open('w') as handle:
        for step in tqdm(range(count), desc='RDSS v2 ' + dataset):
            image = features[step:step + 1]
            full = fixed.logits(image).cpu()
            if not torch.equal(tda[step:step + 1] + full, original[fixed_key][step:step + 1]):
                raise AssertionError('Fixed full-pool control changed')
            full_semantic.append(full)
            for name, model in models.items():
                current = model.logits(image, step).cpu()
                semantic[name].append(current)
                combined[name].append(tda[step:step + 1] + current)
                if step == 0:
                    assert torch.equal(current, full), 'Cold start changed'
            event = bystep.get(step)
            if event is not None:
                for name, model in models.items():
                    updated = model.observe(image[0], event['class_index'], event['entropy'],
                                            True, event['hard_classes'], step)
                    assert updated is not None
                    handle.write(json.dumps({'condition': name, **updated}) + '\n')
            # Only unsupervised memory statistics; phase and truth are absent.
            for name, model in models.items():
                if event is not None:
                    c = event['class_index']
                    traces.append({'step': step, 'condition': name, 'class_index': c,
                                   'alpha': float(model.adaptation_weight(c, step))})
    torch.cuda.synchronize()
    labels = torch.as_tensor(stream['labels'], dtype=torch.long)[:count]
    phases = [0] * count if flowers else [row['phase'] for row in
             json.loads((source / 'dataset_manifest.json').read_text())['stream'][:count]]
    combined = {name: torch.cat(values) for name, values in combined.items()}
    semantic = {name: torch.cat(values) for name, values in semantic.items()}
    scores = {'tda': tda, 'fixed_all': original[fixed_key][:count],
              'rdss_first_round': historical, **combined}
    fixed_ok = scores['fixed_all'].argmax(1) == labels
    comparisons = {}
    for name, values in combined.items():
        current_ok = values.argmax(1) == labels
        comparisons[name] = {'gained': int((~fixed_ok & current_ok).sum()),
                             'lost': int((fixed_ok & ~current_ok).sum())}
    phase_stats = []
    for phase in sorted(set(phases)):
        indices = torch.tensor([i for i, p in enumerate(phases) if p == phase])
        phase_stats.append({'phase': phase,
                            'all': {name: stats(value[indices], labels[indices]) for name, value in scores.items()},
                            'first35': {name: stats(value[indices[:35]], labels[indices[:35]]) for name, value in scores.items()}})
    angles = {}
    for name, model in models.items():
        prototypes = model.read_prototypes(count - 1).float()
        cosine = (prototypes * fixed.prototypes.float()).sum(0).clamp(-1, 1)
        angles[name] = float(torch.rad2deg(torch.acos(cosine)).mean())
    summary = {'source': str(source), 'results': {name: stats(value, labels) for name, value in scores.items()},
               'semantic_only': {'fixed': stats(torch.cat(full_semantic), labels),
                                 **{name: stats(value, labels) for name, value in semantic.items()}},
               'vs_fixed': comparisons, 'phase_results': phase_stats,
               'final_mean_prototype_angle_deg': angles,
               'checks': {'fixed_logits_exactly_match_saved_control': True,
                          'cold_start_exactly_matches_fixed': True,
                          'accepted_events_equal_saved_template_gate': sum(step < count for step in bystep),
                          'labels_and_domains_used_only_after_prediction': True},
               'seconds': time.perf_counter() - started,
               'input_hashes': {str(path): digest(path) for path in
                                [stream_path, embeddings_path, source / 'logits.pt', source / 'semantic_updates.jsonl']}}
    with (folder / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'phase', 'label', *scores])
        predictions = {name: value.argmax(1).tolist() for name, value in scores.items()}
        writer.writerows([i, phases[i], int(labels[i]), *[predictions[name][i] for name in scores]] for i in range(count))
    dump(folder / 'summary.json', summary)
    dump(folder / 'adaptation_weights.json', traces)
    dump(folder / 'final_selections.json', {name: {c: model.selected[i] for i, c in enumerate(names)}
                                          for name, model in models.items()})
    torch.save({'combined': scores, 'semantic': semantic}, folder / 'logits.pt')
    print('RESULT', dataset, json.dumps(summary['results']), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    torch.manual_seed(1)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'runs' / (('rdss_v2_smoke_' if args.limit else 'rdss_v2_') + stamp)
    output.mkdir()
    print('RUN_DIR', output, flush=True)
    # Parameters and protocol are saved before loading test labels or scoring.
    dump(output / 'protocol.json', {'common': COMMON, 'conditions': CONDITIONS,
         'timing': 'Predict from history; write trusted evidence after prediction.',
         'margin_reference': 'Fixed full-pool prototypes; CLIP scale 100; balanced logistic class margins.',
         'time_half_life': '4 * class count stream steps, replacing per-class arrival decay.',
         'prior': 'alpha = concentration * decayed_reliability_mass / (mass + 4)',
         'selection': 'Greedy k=8 normalized aggregate; redundancy weight .1 unchanged.',
         'gate': 'Replay unchanged template entropy <= .2 and template/TDA agreement events.',
         'evaluation_status': 'Exploratory replay of already observed test streams, not independent validation.',
         'historical_control': 'Read saved first-round logits; do not reimplement old algorithm.'})
    sources = [REPO / 'rdss.py', REPO / 'semantic_cache.py', Path(__file__), REPO / 'tests/test_rdss.py']
    for path in sources:
        shutil.copy2(path, output / path.name)
    dump(output / 'source_manifest.json', {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
        'sha256': {path.name: digest(path) for path in sources}})
    summary = {dataset: evaluate(dataset, output, args.limit) for dataset in args.datasets}
    dump(output / 'summary.json', summary)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
