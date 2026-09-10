#!/usr/bin/env python3
"""Validate schema, pairing coverage, and optional lossless source payloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from paired import key,load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    output=a.output or a.root/'validation.json'
    prior=json.loads(output.read_text()) if output.exists() else {}
    settings=set();datasets={};documents=json.loads((a.root/'shared_inputs.json').read_text());rows=[]
    for path in sorted((a.root/'data').glob('*.json')):
        d=load(path);s=d['setting'];sk=json.dumps(s,sort_keys=True)
        if sk in settings:raise ValueError(f'Duplicate setting: {s}')
        settings.add(sk);datasets[path.name]=d
        rows.append(dict(file=path.name,setting=s,episodes=len(d['episodes']),pairing_bases=dict(Counter(r['pairing_basis'] for r in d['episodes'])),task_counts=len(d.get('task_counts',[])),conflicts=len(d.get('normalization',{}).get('conflicting_episode_keys',[]))))
        for name,doc in d.get('source_documents',{}).items():
            if name in documents:raise ValueError(f'Duplicated source document: {name}')
            documents[name]=doc
    verified=0
    verified_logs=0
    if a.source:
        for name,doc in documents.items():
            path=a.source/name
            if not doc.get('source_sha256'):continue
            if not path.exists():raise FileNotFoundError(f'Source document missing from --source tree: {name}')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=doc['source_sha256']:raise ValueError(f'Source hash changed: {name}')
            payload=json.loads(raw) if doc['encoding']=='json' else [json.loads(line) for line in raw.splitlines() if line.strip()]
            if json.dumps(payload,ensure_ascii=False,separators=(',',':'))!=json.dumps(doc['payload'],ensure_ascii=False,separators=(',',':')):raise ValueError(f'Source payload changed: {name}')
            verified+=1
    if a.source:
        for data in datasets.values():
            for name,digest in data.get('parsed_log_sources',{}).items():
                if not (a.source/name).exists():raise FileNotFoundError(f'Log source missing from --source tree: {name}')
                if hashlib.sha256((a.source/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Log source hash changed: {name}')
                verified_logs+=1
    coverage=[]
    for name,low in datasets.items():
        s=low['setting']
        if s['evaluation']!='primary' or s['nsteps']!=1:continue
        for high_name,high in datasets.items():
            hs=high['setting']
            if hs['model']!=s['model'] or hs['benchmark']!=s['benchmark'] or hs['evaluation']!=s['evaluation'] or hs['nsteps']<=1:continue
            lo={key(r):r for r in low['episodes']};hi={key(r):r for r in high['episodes']}
            shared=lo.keys()&hi.keys();valid=[k for k in shared if lo[k]['success'] is not None and hi[k]['success'] is not None]
            coverage.append(dict(low=name,high=high_name,matched=len(valid),low_only=len(lo.keys()-hi.keys()),high_only=len(hi.keys()-lo.keys()),missing_outcomes=len(shared)-len(valid)))
    # Source verification needs the original log tree. Without --source the counts
    # cannot be recomputed, so the recorded ones are carried forward rather than
    # overwritten with zeros.
    report=dict(settings=len(settings),episodes=sum(r['episodes'] for r in rows),source_documents=len(documents),verified_source_documents=verified if a.source else prior.get('verified_source_documents',0),verified_log_sources=verified_logs if a.source else prior.get('verified_log_sources',0),settings_inventory=rows,pairing_coverage=coverage)
    output.write_text(json.dumps(report,indent=2)+'\n')
    checked=f"{verified} source documents" if a.source else f"{report['verified_source_documents']} source documents carried forward (no --source)"
    print(f"Validated {len(settings)} settings, {report['episodes']} episodes, {checked}; report: {output}")

if __name__=='__main__':main()
