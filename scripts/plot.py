#!/usr/bin/env python3
"""Lightweight success curves from normalized JSON, grouped by benchmark/model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/".mplconfig"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paired import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data');p.add_argument('--output',type=Path,default=Path('figures/overview'));p.add_argument('--evaluation',default='primary');a=p.parse_args()
    groups=defaultdict(lambda:defaultdict(list))
    for path in sorted(a.data.glob('*.json')):
        data=load(path);s=data['setting']
        if s['evaluation']!=a.evaluation:continue
        rows=[r for r in data['episodes'] if r['success'] is not None]
        agg=data.get('aggregates',{}).get('overall')
        if agg:n,succ=agg['trials'],agg['successes']
        elif data.get('task_counts'):n,succ=sum(c['trials'] for c in data['task_counts']),sum(c['successes'] for c in data['task_counts'])
        elif rows:n,succ=len(rows),sum(r['success'] for r in rows)
        else:continue
        label=s['model']
        if s.get('checkpoint'):label+=' / '+s['checkpoint']
        if s.get('horizon') is not None:label+=' / horizon '+str(s['horizon'])
        if n:groups[s['benchmark']][label].append((s['nsteps'],100*succ/n))
    a.output.mkdir(parents=True,exist_ok=True)
    for benchmark,models in groups.items():
        fig,ax=plt.subplots(figsize=(6,4))
        for model,points in sorted(models.items()):
            x,y=zip(*sorted(points));ax.plot(x,y,'o-',label=model)
        ax.set(xlabel='Inference steps',ylabel='Success (%)',title=benchmark,ylim=(0,100));ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(a.output/f'{benchmark}.pdf',metadata={'CreationDate':None});fig.savefig(a.output/f'{benchmark}.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
