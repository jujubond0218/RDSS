"""Extract a deterministic, class-balanced NICO++ context stream from a pinned ZIP."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = ['grass','water','rock','autumn','dim','outdoor','grass']
SOURCE_SHA = 'c5531759655342b7adf8b2ede51a97954e8114816106d64a100ed6774bd3eafa'


def main():
    source = ROOT/'downloads/nico_66bea990.zip'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA
    target = ROOT/'data/nico_minimal'
    target.mkdir(exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        groups=defaultdict(list)
        for entry in archive.namelist():
            parts=PurePosixPath(entry).parts
            if len(parts)==4 and parts[0]=='NICOpp' and parts[-1].lower().endswith(('.jpg','.jpeg','.png')):
                groups[parts[1],parts[2]].append(entry)
        names=sorted({c for _,c in groups}); assert len(names)==20
        assert {d for d,_ in groups}==set(CONTEXTS)
        rng=random.Random(1)
        for key in sorted(groups):
            groups[key].sort(); rng.shuffle(groups[key])
        seen_paths,seen_hashes=set(),set()
        stream=[]
        for phase,domain in enumerate(CONTEXTS):
            selected=[]
            for label,name in enumerate(names):
                accepted=0
                for entry in groups[domain,name]:
                    if entry in seen_paths:continue
                    image=archive.read(entry); checksum=hashlib.sha256(image).hexdigest()
                    if checksum in seen_hashes:continue
                    seen_paths.add(entry);seen_hashes.add(checksum)
                    path=Path('images')/domain/name/PurePosixPath(entry).name
                    (target/path).parent.mkdir(parents=True,exist_ok=True);(target/path).write_bytes(image)
                    selected.append({'phase':phase,'domain':domain,'label':label,'source_member':entry,
                                     'source_row':len(seen_paths)-1,'path':path.as_posix(),'sha256':checksum})
                    accepted+=1
                    if accepted==8:break
                assert accepted==8,(domain,name)
            rng.shuffle(selected);stream.extend(selected)
        assert len(stream)==len(seen_hashes)==1120
        manifest={'source':'https://huggingface.co/datasets/2018LZY/NICOplusplus',
                  'source_revision':'66bea9905018646633b3c3664027e7e80cdcb687',
                  'source_type':'Unofficial 20-class, 6-context reorganization of NICO++',
                  'source_zip_sha256':SOURCE_SHA,'classnames':names,'seed':1,
                  'domain_sequence':CONTEXTS,'samples_per_phase_per_class':8,
                  'original_counts':{d+'/'+c:len(v) for (d,c),v in sorted(groups.items())},'stream':stream}
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('PREPARED',target,'images',len(stream),'classes',names,flush=True)


if __name__=='__main__':
    main()
