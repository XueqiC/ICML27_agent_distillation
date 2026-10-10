#!/usr/bin/env python3
"""Regenerate every figure twice and verify deterministic bytes and layout."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

sys.dont_write_bytecode = True
# Reuse the generator's CPU/network/filesystem guards before importing plotting libraries.
from make_figures import BASENAMES, HERE, ROOT, matplotlib, np, sp


def snapshot():
    paths=[HERE/f'{name}.{suffix}' for name in BASENAMES for suffix in ('pdf','png')]
    paths += [HERE/'README.md',HERE/'text_audit.json']
    return {p.name:p.read_bytes() for p in paths}


def main():
    first=None
    for run in (1,2):
        print(f'Full regeneration {run}/2',flush=True)
        with (HERE/f'verification_run_{run}.log').open('w') as log:
            subprocess.run([sys.executable,'-B',str(HERE/'make_figures.py')],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        current=snapshot()
        if first is None:first=current
    different=[name for name in first if first[name]!=current[name]]
    if different:raise RuntimeError(f'Outputs differ between runs: {different}')
    audit=json.loads(current['text_audit.json'])
    if len(audit)!=20:raise RuntimeError('Expected audits for all 20 figures')
    raw=[name for name,row in audit.items() if row['raw_math']]
    if raw:raise RuntimeError(f'Unrendered formula text: {raw}')
    files=dict(current)
    for name in ('make_figures.py','verify_figures.py'):files[name]=(HERE/name).read_bytes()
    result=dict(byte_identical=True,compared_files=len(current),
                head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                evidence_blob=subprocess.check_output(['git','rev-parse','HEAD:theory/v015_checks/results_v015_independent.json'],cwd=ROOT,text=True).strip(),
                versions=dict(python=sys.version.split()[0],matplotlib=matplotlib.__version__,numpy=np.__version__,sympy=sp.__version__),
                text_artists=sum(row['text_count'] for row in audit.values()),
                text_overlaps={name:len(row['overlaps']) for name,row in audit.items() if row['overlaps']},
                outside_canvas={name:row['outside_canvas'] for name,row in audit.items() if row['outside_canvas']},
                files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in sorted(files.items())})
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(current)} outputs byte-identical across two full regenerations.',flush=True)
    print(f'Text: {result["text_artists"]} artists; overlaps: {result["text_overlaps"]}; outside: {result["outside_canvas"]}',flush=True)
    for name,row in result['files'].items():print(f'{name}: {row["bytes"]:,} bytes')
    return 0


if __name__=='__main__':
    sys.exit(main())
