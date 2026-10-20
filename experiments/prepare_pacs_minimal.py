"""Run with the already-installed base PyArrow to extract a fixed 700-image stream."""
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['dog','elephant','giraffe','guitar','horse','house','person']
DOMAINS = ['photo','art_painting','cartoon','sketch','photo']


def main():
    source = ROOT/'downloads/pacs_39411307.parquet'
    target = ROOT/'data/pacs_minimal'
    target.mkdir(exist_ok=False)
    records = pq.read_table(source).to_pylist()
    available = {}
    counts = {}
    for domain in set(DOMAINS):
        for label in range(len(NAMES)):
            indices = [i for i,r in enumerate(records) if r['domain']==domain and r['label']==label]
            available[domain,label] = indices
            counts[f'{domain}/{NAMES[label]}'] = len(indices)
    # Iterate a sorted domain set when shuffling to avoid dependence on Python hash seed.
    rng = random.Random(1)
    for key in sorted(available):
        indices = sorted(available[key]); rng.shuffle(indices); available[key] = indices
    stream = []
    used = set()
    for phase,domain in enumerate(DOMAINS):
        selected = []
        for label in range(len(NAMES)):
            offset = 20 if phase==4 else 0
            selected.extend(available[domain,label][offset:offset+20])
        assert len(selected)==140
        rng.shuffle(selected)
        for index in selected:
            assert index not in used
            used.add(index)
            record = records[index]
            image = record['image']['bytes']
            path = Path('images')/domain/NAMES[record['label']]/f'{index:05d}.jpg'
            (target/path).parent.mkdir(parents=True,exist_ok=True)
            (target/path).write_bytes(image)
            stream.append({'phase':phase,'domain':domain,'label':record['label'],'source_row':index,
                           'path':path.as_posix(),'sha256':hashlib.sha256(image).hexdigest()})
    assert len(stream)==len(used)==700
    result = {'source':'https://huggingface.co/datasets/flwrlabs/pacs',
              'source_revision':'394113073258ead631f617d2e13bb377c0715c4b',
              'source_parquet_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'seed':1,'classnames':NAMES,'domain_sequence':DOMAINS,'original_counts':counts,
              'samples_per_phase_per_class':20,'stream':stream}
    (target/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PREPARED',target,'images',len(stream),flush=True)


if __name__=='__main__':
    main()
