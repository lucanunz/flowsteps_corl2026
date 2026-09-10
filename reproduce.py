#!/usr/bin/env python3
"""Reproduce the existing paper figures from packed JSON inputs in isolation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent


def materialize(work):
    shutil.copytree(ROOT/'scripts/figures',work,dirs_exist_ok=True)
    documents=json.loads((ROOT/'shared_inputs.json').read_text())
    for p in sorted((ROOT/'data').glob('*.json')):
        for name,doc in json.loads(p.read_text()).get('source_documents',{}).items():
            if name in documents and documents[name]!=doc:raise ValueError(f'Conflicting source {name}')
            documents[name]=doc
    for name,doc in documents.items():
        path=work/name
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError(f'Unsafe source path: {name}')
        path.parent.mkdir(parents=True,exist_ok=True)
        payload=doc['payload']
        text='\n'.join(json.dumps(row,ensure_ascii=False) for row in payload) if doc['encoding']=='jsonl' else json.dumps(payload,ensure_ascii=False)
        path.write_text(text+'\n')
    shutil.copy2(ROOT/'task_metadata.json',work/'task_metadata.json')
    return set(documents)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('target',nargs='?',default='all',choices=['all','scatter','tost','simulation','violins','glyphs'])
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
    with tempfile.TemporaryDirectory(prefix='corl-figures-') as tmp:
        work=Path(tmp);inputs=materialize(work);env['MPLCONFIGDIR']=str(work/'.mplconfig')
        subprocess.run([sys.executable,str(work/'reproduce.py'),a.target],cwd=work,env=env,check=True)
        n=0
        for path in work.rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(work)
            if str(rel) in inputs or path.suffix not in ('.png','.pdf','.svg','.tex','.md','.csv'):continue
            dest=a.output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);n+=1
    print(f'{n} outputs written to {a.output}')

if __name__=='__main__':main()
