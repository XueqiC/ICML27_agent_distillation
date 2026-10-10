#!/usr/bin/env python3
"""Independent v0.15 entry point. Writes only inside this directory; no network."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def inside(path):
    return Path(path).resolve().is_relative_to(HERE)


def audit(event,args):
    if event.startswith('socket.') and event not in ('socket.__new__',):
        raise RuntimeError('Independent checks forbid network access')
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(path)).resolve()
            if path.name in ('verify_v015_extensions.py','results_v015.json'):
                raise RuntimeError('Independent checks forbid reading the package verifier or its results')
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if writing and not inside(path):
                raise RuntimeError('Independent checks forbid writes outside theory/v015_checks/')
    if event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime'):
        if not inside(args[0]):raise RuntimeError('Out-of-scope filesystem mutation')
    if event in ('os.rename','os.link','os.symlink'):
        if not all(inside(path) for path in args[:2]):raise RuntimeError('Out-of-scope filesystem mutation')


sys.addaudithook(audit)
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
os.environ['MPLBACKEND']='Agg'
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import json
import importlib.metadata
from datetime import datetime,timezone
import platform
import time
import traceback
import common as C
from mathematics import check_V1,check_V2,check_V3,check_V4,check_V5,check_V6,check_V7
from experiments import check_E0,check_E1,check_E2,check_E3,check_E4,check_E5,check_E6,check_E7

CHECKS={f'V{i}':globals()[f'check_V{i}'] for i in range(1,8)}
CHECKS.update({f'E{i}':globals()[f'check_E{i}'] for i in (0,1,3,2,4,5,6,7)})


class Tee:
    def __init__(self,stream,file):self.stream=stream;self.file=file
    def write(self,text):self.stream.write(text);self.file.write(text);return len(text)
    def flush(self):self.stream.flush();self.file.flush()


def key_number(row):
    d=row['data'];id=row['id']
    if not d:return 'No completed numerical result'
    if id=='V1':return f"old bound {d['regression']['old_bound']:.8f} < gap 1; best A4 ratio {row['best_ratio']:.5g}"
    if id=='V2':return f"{len(d['quotients'])} quotient alphabets; inverse derivative reaches {d['conditioning'][-1]['inverse_derivative']}"
    if id=='V3':return f"largest A9 gap/bound {row['best_ratio']:.6g}"
    if id=='V4':return f"opposite-step witnesses on {d['grid_fraction']:.0%} of beta grid"
    if id=='V5':return f"{len(d['cases'])} transcript families; strict Jensen {d['strict_Jensen']['E_positive']} > {d['strict_Jensen']['positive_E']}"
    if id=='V6':return f"best A15 tightness ratio {row['best_ratio']:.8g}"
    if id=='V7':return f"linear witness: {d['linear_failure']['bound']} > {d['linear_failure']['linear']}"
    if id=='E0':return '; '.join(f"R={r['R']}: gap {r['gap']:.9f}" for r in d['gaps'] if r['eta']==.1)
    if id=='E1':return f"{len(d['ablations'])} ablation cells; max relative map error {d['maximum_measured_relative_error']:.3g}"
    if id=='E2':return f"{len(d['persistent_frozen_gaps'])}/{len(d['comparisons'])} frozen-Adam gaps persist"
    if id=='E3':return f"{sum(w is not None for w in d['neural_witnesses'])} neural epsilon-grid witnesses"
    if id=='E4':return f"{d['failures']}/{len(d['observations'])} anchor prediction failures at 0.001"
    if id=='E5':return f"{len(d['curves'])} exact K/query cells; replay gain zero"
    if id=='E6':return f"{d['held_out_rows']} held-out rows; {len(d['negative_results'])} proxy/split failures to beat best simple control"
    if id=='E7':return f"Hoeffding n={d['Hoeffding']['n']}; pilot empirical-Bernstein forecast n={d['empirical_Bernstein']['n_forecast']}"
    return ''


def readme(report):
    lines=['# Independent v0.15 checks','',
        'All mathematics is derived from the three documents named in SPEC.md. The package verifier and its result file are forbidden by a runtime read guard. Only the expressly permitted v0.14 independent C23 helpers are imported, unchanged. A runtime filesystem guard restricts writes to this directory; network access is disabled. No repository tests are collected.',
        '', 'Run:', '', '```bash',
        "CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json",
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 theory/.venv/bin/python -m pytest -c /dev/null --rootdir=theory/v015_checks --confcutdir=theory/v015_checks -p no:cacheprovider --basetemp=theory/v015_checks/quick/pytest_tmp theory/v015_checks/tests/test_v015_quick.py -q",'```','',
        '`--quick` reduces random counts and the E2 matrix; it retains every group and every failure assertion. `--only V3,E2` selects groups. Quick artifacts go in `quick/`; full artifacts and figures remain intact. Every FAIL makes the command and the quick test fail. Expected failures of the old theorem and the linear recursion shortcut are affirmative regressions, not skipped assertions.',
        '',f"Master seed: {C.MASTER}. Every random family uses `SeedSequence([150150, group_number])`, with V1–V7 numbered 1–7 and E0–E7 numbered 100–107. Runtime: {report['runtime_seconds']:.2f} s. One CPU process; BLAS threads capped at one.",
        '', '| Group | Status | Instances | Key result |','|---|---|---:|---|']
    for row in report['groups']:lines.append(f"| {row['id']} | {row['status']} | {row['n_instances']} | {key_number(row)} |")
    lines+=['','## Methods','']
    for row in report['groups']:
        lines += [f"- **{row['id']} — {row['statement']}.** {row['method']}."]
    lines += ['', '## Reading the evidence', '',
        '`PASS` means the implemented mathematical or numerical consistency assertions passed. `INFO` marks empirical diagnostics or forecasts; it is not a gate pass or a sufficiency certificate. A scientific negative result remains in the raw results even when numerical consistency checks pass. All assertion failures retain exact inputs under `counterexamples`; rational numbers are strings. LP tolerance is 1e-8, Adam-map tolerance is a predeclared relative 1e-12, neural arithmetic is float64.',
        '', 'The C23 bundle includes exact laws, matched moments, raw gradients/updates/moments, the exact historical Theorem 19 wording and source SHA, 23 independent check definitions and their historical independent results. The supplied 19-file package manifest is metadata: its package verifier/results are deliberately not reread or rerun. Reproduction here refers to the available independent C23 implementation, not an unspecified external model run.',
        '', 'E2 uses all finite histories when the imported helper permits (at most 4096), otherwise 2000 independent histories per source in the full run. The same histories feed all five branches. E4 measures finite message panels at genuinely reached states; its prediction failures do not falsify the repaired bound and do not provide uniform coverage.',
        '', 'E5 uses integer subset Möbius inversion and exact rational risks, then actual tabular Adam/SGD logits. K counts independent complete trajectories; additional queries and replay are separate. Distinct subcontexts are not distinct tasks.',
        '', 'E6 writes its split manifest before any measurements and freezes all proxy coefficients and pilot abstention thresholds before held-out evaluation. Gold is exact finite-history integration for one candidate boundary plus seven reference updates. State, task, epsilon, learning-rate and fresh/warm transfers are reported separately; no held-out result changes a proxy. Negative selection results are explicit. Empirical calibration and cell bootstraps are not uniform certificates.',
        '', 'E7 retains shared-reference covariance, separately enumerates block/suffix/deployment variance, and reports the 1016-replica Hoeffding allocation. Its Maurer–Pontil allocation uses measured pilot variance as a forecast; a future independent fixed-n audit must compute its own radius. Supplied hpg/Gemma timings are scenario inputs only, not measured here.',
        '', '## Scope and claim ledger', '',
        '- Reproduced: available C23 controlled network and old coverage counterexample.',
        '- Exact finite evidence: rational mathematics, hidden-bit enumeration, optimizer-history moments. Numerical checks are not formal proofs.',
        '- Empirical: neural comparator gaps, anchor diagnostics, held-out proxy results.',
        '- Forecast: Gemma costs from supplied hpg segment timings.',
        '- No Gemma or micro-LoRA pilot ran: this task authorizes the offline CPU controlled suite only, with no network, GPU, checkpoint access or model substitution. No teacher is called.',
        '- Unresolved: useful uniform neural coverage/response certificates and transfer to actual agent training.',
        '', '## Artifacts', '',
        '`results_v015_independent.json` contains provenance, source/code SHA-256s, every group, counterexamples, and artifact hashes. `full_run.log` is the complete progress log. `c23_bundle.json` contains reproduction records. `raw/` holds per-history arrays, proxy rows and paired-pilot data; `figs/E0` through `figs/E7` have PNG and SVG versions. `proxy_split_freeze.json` and `proxy_models_frozen.json` preserve the split and fitted proxy contract. No commits or repository test runs are performed.', '']
    (HERE/'README.md').write_text('\n'.join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--only',help='comma-separated group IDs, e.g. V3,E2')
    args=parser.parse_args(argv)
    if args.output is None:
        args.output=(HERE/'quick'/'results.json') if args.quick else (
            HERE/'selected_results.json' if args.only else HERE/'results_v015_independent.json')
    if not inside(args.output):parser.error('--output must be under theory/v015_checks/')
    selected=list(CHECKS) if not args.only else args.only.split(',')
    if any(key not in CHECKS for key in selected):parser.error('unknown group ID')
    C.QUICK=args.quick;C.OUT=HERE/'quick' if args.quick else HERE
    C.OUT.mkdir(parents=True,exist_ok=True)
    log_path=C.OUT/('selected_run.log' if args.only else 'full_run.log')
    began=time.perf_counter();original_stdout=sys.stdout
    report=dict(schema_version=1,master_seed=C.MASTER,quick=args.quick,groups=[],counterexamples=[],
        started_UTC=datetime.now(timezone.utc).isoformat(),
        command=dict(output=str(args.output),quick=args.quick,only=selected),
        versions=dict(python=platform.python_version(),numpy=C.np.__version__,scipy=C.scipy.__version__,
                      sympy=C.sp.__version__,matplotlib=C.matplotlib.__version__,
                      pytest=importlib.metadata.version('pytest')),
        provenance=dict(network=False,CUDA_VISIBLE_DEVICES='',processes=1,BLAS_threads=1,
            teacher=None,student='controlled NumPy tanh/sigmoid; no pretrained checkpoint',
            environment='finite constructed mathematical models and controlled neural learner',
            model_calls=0,training_tokens=0,split_manifest='proxy_split_freeze.json',
            independence='Never imports or reads the package verifier or its results. C23 helper reuse expressly authorized.'),
        source_sha256s={},runtime_seconds=None)
    for path in [HERE/'SPEC.md',*sorted(HERE.glob('*.py')),*sorted((HERE/'tests').glob('*.py')),
        C.ROOT/'theory/independent_checks/check_v014_independent.py',
        *[C.ROOT/'docs/direction/v015_revision'/name for name in
            ('Environment_Information_Theory_EN.md','Optimizer_State_and_Few_Shot_Theory_v015_EN.md','Coding_Agent_Followup_v015_EN.md')]]:
        report['source_sha256s'][str(path.relative_to(C.ROOT))]=C.digest(path)
    with log_path.open('w') as logfile:
        sys.stdout=Tee(original_stdout,logfile)
        try:
            print(f'Independent v0.15 | master seed {C.MASTER} | quick={args.quick}',flush=True)
            for id in selected:
                start=time.perf_counter();print(f'START {id}',flush=True)
                try:
                    row=CHECKS[id](args.quick)
                except Exception as exc:
                    # Pure local code only: no SDK/network exception can occur.
                    failure=dict(group=id,seed=[C.MASTER,int(id[1:])+(100 if id[0]=='E' else 0)],
                        assertion='group execution completed',exception_type=type(exc).__name__,
                        local_traceback=traceback.format_exc())
                    if C.ACTIVE is not None and C.ACTIVE.id==id:
                        row=C.ACTIVE.finish();row['status']='FAIL';row['counterexamples'].append(failure)
                    else:
                        row=dict(id=id,statement='Execution failure retained',method='see counterexample',kind='execution',
                            n_instances=0,n_assertions=1,worst_residual=0,status='FAIL',notes=[],data={},counterexamples=[failure])
                row['runtime_seconds']=time.perf_counter()-start
                report['counterexamples'].extend(row.pop('counterexamples'))
                report['groups'].append(row)
                print(f"{id} {row['status']} | {key_number(row)} | {row['runtime_seconds']:.2f}s",flush=True)
                # Persist after every group, including failures and partial runs.
                report['runtime_seconds']=time.perf_counter()-began
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(C.serial(report),indent=2,allow_nan=False)+'\n')
            print(f"Completed in {report['runtime_seconds']:.2f}s; {len(report['counterexamples'])} unexpected failures",flush=True)
        finally:
            sys.stdout=original_stdout
    report['artifact_sha256s']={str(path.relative_to(HERE)):C.digest(path)
        for pattern in ('raw/*','figs/*','c23_bundle.json','proxy*json') for path in C.OUT.glob(pattern) if path.is_file()}
    report['runtime_seconds']=time.perf_counter()-began
    report['interpretation_withheld']=any(r['id']=='E1' and r['status']=='FAIL' for r in report['groups'])
    args.output.write_text(json.dumps(C.serial(report),indent=2,allow_nan=False)+'\n')
    if not args.quick and not args.only:readme(report)
    return int(any(r['status']=='FAIL' for r in report['groups']))


if __name__=='__main__':
    raise SystemExit(main())
