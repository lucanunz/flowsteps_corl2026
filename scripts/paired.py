"""Strict alignment and paired statistics for schema v1 setting JSON files."""
import dataclasses
import json
import math
from pathlib import Path
from scipy.stats import binomtest
from equivalence import paired_tost,paired_t_tost


def key(row):
    return json.dumps([row['suite'],row['task'],row['episode_id']],sort_keys=True,separators=(',',':'),ensure_ascii=False)


def load(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema_version')!=1:raise ValueError(f'Unsupported schema: {path}')
    setting=data['setting']
    if type(setting['nsteps']) is not int or setting['nsteps']<1:raise ValueError('nsteps must be a positive integer')
    seen=set()
    for row in data['episodes']:
        k=key(row)
        if k in seen:raise ValueError(f'Duplicate pairing key in {path}: {k}')
        seen.add(k)
        if row['success'] is not None and type(row['success']) is not bool:raise ValueError('success must be boolean or null')
        if 'completion' in row and (type(row['completion']) is not int or not 0<=row['completion']<=5):raise ValueError('Invalid CALVIN completion score')
    return data


def align(low,high,*,allow_order=False,allow_partial=False,suite=None,task=None):
    for name in ('model','benchmark','checkpoint','evaluation'):
        if low['setting'].get(name)!=high['setting'].get(name):raise ValueError(f'Incompatible {name}')
    issues={side:data.get('normalization',{}) for side,data in [('low',low),('high',high)]}
    unresolved=any(v.get('conflicting_episode_keys') or v.get('log_aggregate_mismatches') for v in issues.values())
    if unresolved and not allow_partial:raise ValueError('Unresolved source conflicts or log/aggregate mismatches; inspect normalization and use --allow-partial only for an explicit subset analysis')
    def select(data):
        rows=[r for r in data['episodes'] if (suite is None or r['suite']==suite) and (task is None or r['task']==task)]
        result={key(r):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate pairing keys')
        return result
    lo,hi=select(low),select(high)
    shared=[k for k in lo if k in hi]
    null=[k for k in shared if lo[k]['success'] is None or hi[k]['success'] is None]
    null_set=set(null)
    matched=[k for k in shared if k not in null_set]
    diagnostic=dict(low_total=len(lo),high_total=len(hi),matched=len(matched),low_only=len(lo.keys()-hi.keys()),high_only=len(hi.keys()-lo.keys()),missing_outcomes=len(null))
    if unresolved:diagnostic['source_issues']=issues
    if not allow_partial and (diagnostic['low_only'] or diagnostic['high_only'] or null):raise ValueError(f'Incomplete pairing; use --allow-partial to explicitly analyze the intersection: {diagnostic}')
    if not matched:raise ValueError('No matched observed outcomes; aggregate totals cannot supply paired data')
    if not allow_order and any(lo[k]['pairing_basis']=='assumed_order' or hi[k]['pairing_basis']=='assumed_order' for k in matched):raise ValueError('Pairing relies on identical rollout ordering; use --allow-order only if that experimental assumption is justified')
    for k in matched:
        if lo[k]['pairing_basis']!=hi[k]['pairing_basis']:raise ValueError('Mismatched pairing bases')
        if lo[k]['pairing_basis']=='session_and_pair_id':
            for field in ('policy_seed','initial_position'):
                if lo[k].get(field)!=hi[k].get(field):raise ValueError(f'Mismatched real-world {field}: {k}')
    return [lo[k] for k in matched],[hi[k] for k in matched],diagnostic


def compare(low,high,*,margin=.03,chain_margin=.15,alpha=.05,**kwargs):
    if not 0<alpha<.5 or not math.isfinite(margin) or not 0<margin<=1 or not math.isfinite(chain_margin) or chain_margin<=0:raise ValueError('Invalid alpha or equivalence margin')
    lo,hi,diag=align(low,high,**kwargs)
    a=sum(x['success'] and y['success'] for x,y in zip(lo,hi))
    b=sum(x['success'] and not y['success'] for x,y in zip(lo,hi))
    c=sum(not x['success'] and y['success'] for x,y in zip(lo,hi))
    d=len(lo)-a-b-c
    exact=float(binomtest(b,b+c,.5).pvalue) if b+c else 1.0
    result=dict(pairing=diag,table=dict(both_success=a,low_only_success=b,high_only_success=c,both_failure=d),
                difference_direction='high minus low',mcnemar_exact_p=exact,
                binary_tost=dataclasses.asdict(paired_tost(a,b,c,d,delta=margin,alpha=alpha)))
    if all('completion' in x for x in lo+hi):
        tost=paired_t_tost([x['completion'] for x in lo],[x['completion'] for x in hi],delta=chain_margin,alpha=alpha)
        result['chain_tost']=dataclasses.asdict(tost) if tost else None
    return result
