"""Independent saved-artifact audit and post-hoc evidence-age statistics."""
import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    manifest = json.loads((run / 'source_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert digest(run / name) == expected, name
    protocol = json.loads((run / 'protocol.json').read_text())
    conditions = protocol['conditions']
    summaries = json.loads((run / 'summary.json').read_text())
    result = {}
    for dataset, summary in summaries.items():
        folder = run / dataset
        logits = torch.load(folder / 'logits.pt', map_location='cpu', weights_only=True)
        rows = list(csv.DictReader((folder / 'predictions.csv').open()))
        labels = torch.tensor([int(row['label']) for row in rows])
        for name, values in logits['combined'].items():
            assert bool(torch.isfinite(values).all()), name
            predictions = values.argmax(1).tolist()
            assert predictions == [int(row[name]) for row in rows], name
            assert int((values.argmax(1) == labels).sum()) == summary['results'][name]['correct'], name
        for path, expected in summary['input_hashes'].items():
            assert digest(Path(path)) == expected, path
        events = [json.loads(line) for line in (folder / 'semantic_updates.jsonl').read_text().splitlines()]
        bycondition = {name: [e for e in events if e['condition'] == name] for name in conditions}
        accepted = [(e['step'], e['class_index']) for e in bycondition[next(iter(conditions))]]
        for name, current in bycondition.items():
            assert [(e['step'], e['class_index']) for e in current] == accepted
            for event in current:
                indices = event['selected_indices']
                assert len(indices) == len(set(indices)) == 8
                assert min(indices) >= 0 and max(indices) < 100
                assert 0 <= event['adaptation_weight'] <= 1
        # This section is post-hoc only: true classes/phase IDs never enter RDSS.
        stream_path = next(Path(path) for path in summary['input_hashes'] if Path(path).name == 'visual_stream.pt')
        stream = torch.load(stream_path, map_location='cpu', weights_only=True)
        names = stream['classnames']
        memory = [deque(maxlen=32) for _ in names]
        event_map = {e['step']: e for e in bycondition[next(iter(conditions))]}
        traces = []
        for step, row in enumerate(rows):
            phase, truth = int(row['phase']), int(row['label'])
            entries = memory[truth]
            trace = {'step': step, 'phase': phase}
            for mode in ('arrival_decay', 'global_time'):
                weights = [(1. - ent) * (.9 ** (len(entries) - 1 - i) if mode == 'arrival_decay'
                           else 2. ** (-(step - when) / (4. * len(names))))
                           for i, (when, _, ent) in enumerate(entries)]
                total = sum(weights)
                trace[mode] = (sum(weight for weight, (_, p, _) in zip(weights, entries) if p == phase) / total
                               if total else None)
            traces.append(trace)
            event = event_map.get(step)
            if event is not None:
                memory[event['class_index']].append((step, phase, event['entropy']))
        phases = []
        for phase in sorted({int(row['phase']) for row in rows}):
            subset = [r for r in traces if r['phase'] == phase]
            entry = {'phase': phase}
            for mode in ('arrival_decay', 'global_time'):
                for window, selected in [('first35', subset[:35]), ('last35', subset[-35:])]:
                    values = [r[mode] for r in selected if r[mode] is not None]
                    entry[mode + '_' + window] = sum(values) / len(values) if values else None
            phases.append(entry)
        result[dataset] = {'samples': len(rows), 'accepted': len(accepted),
                           'csv_logits_counts_hashes_and_gate_alignment': True,
                           'phase_evidence_weights_posthoc': phases}
    (run / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    audit(parser.parse_args().run)
