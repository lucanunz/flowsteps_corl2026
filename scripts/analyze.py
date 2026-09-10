#!/usr/bin/env python3
"""Analyze two setting JSON files, joining trial identities explicitly."""
import argparse
import json
from pathlib import Path
from paired import compare,load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('low',type=Path);p.add_argument('high',type=Path)
    p.add_argument('--margin',type=float,default=.03,help='Success-rate margin in proportion units (default .03 = 3pp)')
    p.add_argument('--chain-margin',type=float,default=.15,help='CALVIN margin in subtasks')
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--suite');p.add_argument('--task')
    p.add_argument('--allow-order',action='store_true');p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=compare(load(a.low),load(a.high),margin=a.margin,chain_margin=a.chain_margin,alpha=a.alpha,suite=a.suite,task=a.task,allow_order=a.allow_order,allow_partial=a.allow_partial)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    else:print(text,end='')

if __name__=='__main__':main()
