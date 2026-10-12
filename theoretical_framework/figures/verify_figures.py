#!/usr/bin/env python3
"""Verify deterministic figures, provenance, every text font, and visible layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
# Install the generator's CPU, network and filesystem guards before libraries.
from make_figures import BASENAMES, HERE, ROOT, SOURCES, matplotlib, np, sp, committed_bytes
from PIL import Image


def snapshot():
    paths = [HERE / f'{name}.{suffix}' for name in BASENAMES for suffix in ('pdf', 'png')]
    paths += [HERE / 'README.md', HERE / 'text_audit.json']
    return {p.name: p.read_bytes() for p in paths}


def assert_audits(audit):
    if set(audit) != set(BASENAMES):
        raise AssertionError('Expected audits for all 25 figures')
    minima = {}
    for name, row in audit.items():
        width = row['fig_width_in']
        assert width <= 10, f'{name}: width exceeds 10 inches'
        assert row['panel_columns'] <= 2, f'{name}: more than two panels per row'
        assert row['font_artists'], f'{name}: no text artists audited'
        for artist in row['font_artists']:
            size = artist['fontsize_pt']
            assert size >= 12, f'{name}: text below 12 pt: {artist["text"]!r}'
            assert size * 900 / (width * 72) >= 15, f'{name}: text below 15 px at 900 px width'
            assert not re.search(r'\b(?:N\d{2}|W[123])\b', artist['text']), f'{name}: old node label'
        assert not row['raw_math'], f'{name}: unrendered formula text'
        assert not row['outside_canvas'], f'{name}: text outside canvas'
        assert not row['overlaps'], f'{name}: overlapping text boxes'
        with Image.open(HERE / f'{name}.png') as im:
            assert im.width == round(width * 300), f'{name}: PNG width differs from audited width'
            assert im.info.get('Description'), f'{name}: missing caption metadata'
            assert im.info.get('Title') == name, f'{name}: wrong metadata title'
            assert not re.search(r'\b(?:N\d{2}|W[123])\b', im.info['Description']), f'{name}: node label in caption'
        minima[name] = dict(min_fontsize_pt=row['min_fontsize_pt'],
                            min_fontsize_px_at_900=row['min_fontsize_px_at_900'],
                            width_in=width, text_artists=len(row['font_artists']))
    return minima


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', help='Regenerate only these short IDs twice; preserve README and audit all figures')
    args = parser.parse_args()
    ids = ['_'.join(name.split('_')[:2]) for name in BASENAMES]
    requested = set(ids if args.only is None else [value.strip() for value in args.only.split(',')])
    if not requested or not requested.issubset(ids):
        parser.error('--only must contain new short figure IDs')
    regenerated = [name for name, short_id in zip(BASENAMES, ids) if short_id in requested]
    before = snapshot() if args.only is not None else None
    command = [sys.executable, '-B', str(HERE / 'make_figures.py')]
    if args.only is not None:
        command += ['--only', args.only]
    started = time.monotonic()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    first = None
    runs = []
    for run in (1, 2):
        start = time.monotonic()
        print(f'{"Full" if args.only is None else "Subset"} regeneration {run}/2', flush=True)
        with (HERE / f'verification_run_{run}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs.append(time.monotonic() - start)
        current = snapshot()
        if before is not None:
            outputs = {f'{name}.{suffix}' for name in regenerated for suffix in ('pdf', 'png')}
            for name in before.keys() - outputs - {'text_audit.json'}:
                assert before[name] == current[name], f'Subset regeneration changed {name}'
            old_audit, new_audit = (json.loads(snap['text_audit.json']) for snap in (before, current))
            for name in set(BASENAMES) - set(regenerated):
                assert old_audit[name] == new_audit[name], f'Subset regeneration changed audit for {name}'
        if first is None:
            first = current
    changed = [name for name in first if first[name] != current[name]]
    if changed:
        raise AssertionError(f'Outputs differ between runs: {changed}')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == head, 'HEAD changed during verification'
    audit = json.loads(current['text_audit.json'])
    minima = assert_audits(audit)
    source_hashes = {path: hashlib.sha256(committed_bytes(path)).hexdigest() for path in SOURCES.values()}
    v016 = json.loads(committed_bytes(SOURCES['U']))
    files = dict(current)
    for p in [HERE/'make_figures.py', HERE/'verify_figures.py', *sorted((HERE/'specs').glob('fig*.md'))]:
        files[str(p.relative_to(HERE))] = p.read_bytes()
    result = dict(status='PASS', byte_identical=True, compared_files=len(current), head=head,
                  regenerated_figures=regenerated,
                  source_sha256=source_hashes, raw_artifact=v016['raw_artifact'],
                  scientific_status=v016['status'],
                  preserved_failures={r['claim_id']:r['failures'] for r in v016['checks'] if r['failures']},
                  versions=dict(python=sys.version.split()[0], matplotlib=matplotlib.__version__,
                                numpy=np.__version__, sympy=sp.__version__),
                  runtime_seconds=time.monotonic()-started, regeneration_seconds=runs,
                  figures=minima, text_overlaps={}, outside_canvas={},
                  files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                         for name,data in sorted(files.items())})
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(current)} outputs byte-identical; all 25 font/layout audits passed.', flush=True)
    for name, row in minima.items():
        print(f'{name}: min {row["min_fontsize_pt"]:g} pt / {row["min_fontsize_px_at_900"]:g} px')
    for name, row in result['files'].items():
        if name.endswith(('.png', '.pdf')):
            print(f'{name}: {row["bytes"]:,} bytes')
    print(f'Runtime: {result["runtime_seconds"]:.2f} seconds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
