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
import tempfile
import common as C
from mathematics import check_V1,check_V2,check_V3,check_V4,check_V5,check_V6,check_V7
from experiments import check_E0,check_E1,check_E2,check_E3,check_E4,check_E5,check_E6,check_E7
from warmth import check_E8,check_E9
from otlr import check_V8,check_V9,check_E10,check_E11,check_E12

CHECKS={f'V{i}':globals()[f'check_V{i}'] for i in range(1,10)}
CHECKS.update({f'E{i}':globals()[f'check_E{i}'] for i in (0,1,3,2,4,5,6,7,8,9,10,11,12)})


def atomic_bytes(path, payload):
    """Publish complete bytes; concurrent readers see old or new content."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,
                                         prefix='.'+path.name+'.',suffix='.tmp',delete=False) as file:
            temporary=Path(file.name)
            if path.exists():os.chmod(temporary,path.stat().st_mode & 0o777)
            file.write(payload);file.flush();os.fsync(file.fileno())
        os.replace(temporary,path)
        directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def atomic_text(path, text):
    atomic_bytes(path,text.encode('utf-8'))


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
    if id=='E8':return f"W0 best ratio {d['W0']['best_ratio']:.9g}; W1 {d['W1']['status']} on {d['W1']['n_law_pairs']} law pairs; {len(d['gaps'])} gap cells"
    if id=='E9':return f"{d['held_out_rows']} held-out rows; {sum(r['all_three_improve'] for r in d['cheap_proxy_results'])}/8 cheap proxy/W pairs improve all three split-mean metrics"
    if id=='V8':return f"{len(d['polynomials'])} exact polynomials; logistic max ratio {d['logistic']['best']['ratio']:.6g}; {len(d['ranks'])} rank/LP panels"
    if id=='V9':return f"{d['trials_per_cell']} trials/cell; max quantile/bound {d['max_quantile_over_bound']:.6g}"
    if id=='E10':return f"{len(d['cells'])} cells; {sum(r['bound_status']=='FAIL' for r in d['cells'])} empirical bound failures; {sum(r['bound_status']=='UNAVAILABLE' for r in d['cells'])} unavailable derivatives; C23 {d['C23_prediction']['gap']:.9f}"
    if id=='E11':return f"5 catalog witnesses; always-active slope {d['always_active']['slope']:.6g}; warm v term {d['warm_Adam']['predicted_v_term']:.6g}"
    if id=='E12':
        r=d['summary'][0]
        return f"{r['rows']} frozen rows; {r['eligible_rows']} smooth; {r['margin_rows']} above margin; margin wrong-sign {r['oracle_wrong_sign_margin']}"
    return ''


def readme(report):
    lines=['# Independent v0.15 checks','',
        'All mathematics is derived from the three documents named in SPEC.md, `docs/direction/v015_extension/Warm_Adam_Response_EN.md`, and `docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md` (read in full). The package verifier and its result file are forbidden by a runtime read guard. Only the expressly permitted v0.14 independent C23 helpers are imported, unchanged. A runtime filesystem guard restricts writes to this directory; network access is disabled. No repository tests are collected.',
        '', 'Run:', '', '```bash',
        "CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json",
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 theory/.venv/bin/python -m pytest -c /dev/null --rootdir=theory/v015_checks --confcutdir=theory/v015_checks -p no:cacheprovider --basetemp=theory/v015_checks/quick/pytest_tmp theory/v015_checks/tests/test_v015_quick.py -q",'```','',
        '`--quick` reduces random counts and the E2 matrix; it retains every group and every failure assertion, including every E8 and E10 task/rate/horizon cell. `--only V8,V9,E10,E11,E12` selects the new extension. Quick artifacts go in `quick/`; full artifacts and figures remain intact. Every group FAIL makes the command and the quick test fail. E10 is an INFO diagnostic with explicit cell PASS/FAIL/UNAVAILABLE statuses: cell failures are scientific negative results, never group PASS. Expected failures of the old theorem and the linear recursion shortcut are affirmative regressions.',
        '',f"Master seed: {C.MASTER}. Every random family uses `SeedSequence([150150, group_number])`, with V1–V9 numbered 1–9 and E0–E12 numbered 100–112. Runtime: {report['runtime_seconds']:.2f} s. One CPU process; BLAS threads capped at one.",
        '', '| Group | Status | Instances | Seconds | Key result |','|---|---|---:|---:|---|']
    for row in report['groups']:lines.append(f"| {row['id']} | {row['status']} | {row['n_instances']} | {row['runtime_seconds']:.2f} | {key_number(row)} |")
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
        '', 'E8 uses seed `[150150,108]`. W0 samples positive second moments log-uniformly over 1e-12..1, random first moments/gradients, t=1..2000 and five beta pairs including (0.9,0.999); random restarts and hill-climbing maximize the bound ratio. The coordinatewise difference is rationalized before float64 evaluation to avoid cancellation between rounded updates; witnesses also contain direct subtraction. Only relative slack 1e-12 is permitted.',
        '', 'W1 trains 1–4 output-head coordinates of the original C23 network, keeping the remaining parameters fixed. It uses 2–6 messages, one actual Adam boundary and R=0,2,6 reference updates, enumerating all 2^R suffix histories. Reference labels are jointly all zero or all one with equal probability; deployment is C23 moment/parity return. An independent analytic sensitivity recurrence computes b=grad F(0). Central differences on declared actual/surrogate theta and relative-v segments estimate L_i and L_v; radial/convex-hull sampling and Hessian differences estimate H_G. Constants are fixed before law-pair checks and never padded after failures. Each pointwise and equal/unequal-mean two-law bound is checked, with adversarial searches on both law families. W1 PASS means empirical consistency on every measured instance, not a uniform certificate; signed exact-enumeration gaps and b^T dE G are recorded separately.',
        '', 'W2 retains the exact rational sign witness and finite-epsilon (1e-8) value. The same gradient laws are swept over a log kappa grid by varying v at fixed t and m=0. Both raw and preconditioner-normalized gap slopes are reported: changing v also changes the step scale. W3 records coordinate/replica kappa quantiles along actual moment/parity Adam trajectories at eta=.01,.1, with W=0,1,2,5,10,20,50 actual lr=0 warm-up steps. Every warm-up step must preserve parameter bytes. Fits use median=c/(W+s), s=1..79 completed parameter updates; free log-log slopes and R2 retain nonstationarity and poor fits. Infinity at v=0 is retained. This tests the cold-window heuristic, not a telescoping/coverage certificate.',
        '', 'E8 gap sweeps reuse E2 run_network and weighted_result for actual Adam, frozen-Jacobian Adam and tangent affine-SGD, across moment N=2,3,4, parity N=2,3, eta=.01,.1, R=1,4,40 and all seven W. The same finite or sampled histories feed all W/branches. Reference warm-up uses one declared IID P-law prefix per task; reported gaps condition on that prefix. W=0 is the original fresh state. Each Adam branch keeps its own subsequent moments.',
        '', 'E9 uses seed `[150150,109]` and the E6 provider_menu, proxy_panel, proxy_features, fit_proxies and proxy_metrics unchanged. It reuses only the original frozen E6 pilot for fitting, verifies coefficient identity, and writes proxy_split_freeze_E9.json and proxy_models_frozen_E9.json before any new held-out panel. All seven original E6 splits remain; five paired cold/W=5/W=20 panels separate state, task, epsilon, learning-rate and joint recipe shifts. The state-only comparison uses the original warm2-only fit, excluding fresh and both moment-warmed descendants from fitting. Other comparisons use combined E6 pilot models and their original held-out tasks/recipes. Random controls are paired across arms. No warm-up gold fits a model. Gold remains one candidate update plus seven IID reference updates, exactly enumerated, with the original task returns and all ten proxies plus full-tail oracle. Zero predictions, wrong-sign cutoff and abstention follow E6 exactly. Aggregate improvement is a descriptive equal-split mean; per-split neutral/adverse outcomes remain in the JSON.',
        '', '## Scope and claim ledger', '',
        '- Reproduced: available C23 controlled network and old coverage counterexample.',
        '- Exact finite evidence: rational mathematics, hidden-bit enumeration, optimizer-history moments. Numerical checks are not formal proofs.',
        '- Empirical: neural comparator gaps, anchor diagnostics, held-out proxy results.',
        '- Forecast: Gemma costs from supplied hpg segment timings.',
        '- No pretrained Gemma/LoRA pilot ran: E11 is a tiny constructed bilinear sigmoid model. This task authorizes the offline CPU controlled suite only, with no network, GPU, checkpoint access or model substitution. No teacher is called.',
        '- Unresolved: useful uniform neural coverage/response certificates and transfer to actual agent training.',
        '', '## Artifacts', '',
        '`results_v015_independent.json` contains provenance, source/code SHA-256s, every group, counterexamples, and artifact hashes. `full_run.log` is the complete progress log. `c23_bundle.json` contains reproduction records. `raw/` holds per-history arrays, proxy rows, pre-extension results and paired-pilot data; `figs/E0` through `figs/E9` have PNG and SVG versions (E8 also has warmth and W2 panels). `proxy_split_freeze.json` and `proxy_models_frozen.json` preserve the original E6 contract; the `_E9.json` files preserve the new frozen contract. `legacy_data_comparison.json` compares old group data before the results file is overwritten. No commits or repository test runs are performed.', '']
    if report.get('legacy_data_comparison'):
        comparison=report['legacy_data_comparison']
        lines += [f"Legacy comparison: {sum(x['data_byte_identical'] for x in comparison['groups'])}/{len(comparison['groups'])} original group data objects byte-identical under canonical JSON serialization. Differences: {comparison['different_groups']}.", '']
        lines += ['E7 originally stores wall-clock timing inside its data. Its five timing/CPU-projection fields retain the historical baseline to preserve that data object; current measured timings and every before/after value are separately recorded in `rerun_measurements.E7`. Scientific values and assertions are always recomputed and compared without normalization.', '']
    extension=next((r['data'] for r in report['groups'] if r['id']=='E8'),{})
    if extension:
        lines += ['## Warmth results', '',
            f"W0: {extension['W0']['n_instances']} random/adversarial states, best ratio {extension['W0']['best_ratio']:.12g}. W1: {extension['W1']['n_panels']} continuation panels and {extension['W1']['n_law_pairs']} law pairs; largest gap/bound {extension['W1']['best_ratio']:.6g}. W2 small-kappa log-log slopes: raw gap {extension['W2']['loglog_slope']:.6f}, preconditioner-normalized gap {extension['W2']['normalized_loglog_slope']:.6f}.", '',
            '| W | Moment N=3, eta=.01, P: fitted c | Free log-log slope | R2 for c/(W+s) |',
            '|---:|---:|---:|---:|']
        for r in extension['W3']['fits']:
            if r['task']=='moment' and r['N']==3 and r['source']=='P' and r['eta']==.01:
                f=r['fit'];lines.append(f"| {r['W']} | {f['c']:.5g} | {f['loglog_slope']:.5g} | {f['R2']:.5g} |")
        unfitted=sum(r['fit']['c'] is None for r in extension['W3']['fits'])
        lines += ['', f"All 140 trajectory-family fits are retained, including {unfitted} unfittable series. Fits exclude zero/infinite medians but preserve those values in raw records; successful fits are descriptive and need not follow 1/s.", '']
    proxies=next((r['data'] for r in report['groups'] if r['id']=='E9'),{})
    if proxies:
        lines += ['## Cheap proxy results after warm-up', '',
            'Equal-weight mean changes over the five paired held-out split panels (warm minus cold). Negative regret/wrong-sign changes and positive Spearman changes indicate improvement. These are descriptive changes; all E6 metrics, controls, raw predictions and per-split comparisons remain in the results.', '',
            '| Proxy | W | Regret change | Wrong-sign change | Spearman change | All three improve |',
            '|---|---:|---:|---:|---:|---|']
        for r in proxies['cheap_proxy_results']:
            m=r['metrics'];values=[m[k]['mean_warm_minus_cold'] for k in ('selection_regret','wrong_sign_rate','Spearman')]
            formatted=['n/a' if v is None else f'{v:+.6g}' for v in values]
            lines.append(f"| {r['proxy']} | {r['W']} | {' | '.join(formatted)} | {'yes' if r['all_three_improve'] else 'no'} |")
        improved=sum(r['all_three_improve'] for r in proxies['cheap_proxy_results'])
        lines += ['', f"{improved}/{len(proxies['cheap_proxy_results'])} cheap proxy/W combinations improve all three aggregate metrics in this panel. These finite held-out observations do not establish general G5 improvement; negative results are preserved.", '']
    lines += ['', '## Optimizer-transformed response methods', '',
        'V8 uses exact Fraction line-polynomial coefficients. For the infinity norm, the derivative upper bound is j! times the coefficient of t^j in sum_alpha |c_alpha| product_i (b_i+t)^alpha_i, where b_i is the largest absolute coordinate among x and the increment endpoints. This rational coefficient-sum bound contains the whole convex hull; it is not its exact supremum. The logistic quadratic uses the certified analytic bound L²/(6 sqrt(3)) + ||A||op/4, with L the largest ||Az+b|| at a hull vertex. Full runs include 2200 independent instances plus random-restart hill climbing. C3 uses exact quotient ranks, nullspace matches and feature-matching LP witnesses.', '',
        'V9 uses 20000 IID trials per law/dimension/n cell, reused across the two delta values. All laws have norm at most one. Biased scalar sign vectors use eta=1/sqrt(p); clipped Gaussian is full-dimensional N(0,I/p) with norm clipped at one; spikes are fair signed uniform coordinate vectors. Exact Beta-angle rotational recursion and explicit sparse collisions avoid materializing p=100000 vectors. Coverage assertions allow four binomial standard errors using delta. Fixed-coordinate sign updates still have b=eta sqrt(p).', '',
        'E10 keeps the C23 one-hot → four tanh → sigmoid network, mean CE and float64. The full state contains theta, m and v. The original SGD and Adam maps and E2 run_network/weighted_result are reused unchanged; the other maps live only in otlr.py. Warm Adam executes a shared IID P prefix at lr=0 for W=5,20. First-order contractions perturb all state blocks together, using central differences at h=.002 and h/2 and Richardson extrapolation; the error diagnostic is |fine-coarse|/3. R=1,4 enumerate every Q history and P suffix. R=40 uses 2000 independent complete runs per law for the gap and 2000 independent random-time Q boundaries with IID P tails for the C2 sum, with SE. Exact R=40 continuation enumeration is not implemented (up to 4^39 suffixes). This limitation is explicit; no sampled result is called exact.', '',
        'The E10/E12 Mhat2 search normalizes realized increment directions by the Euclidean norm of (delta theta, eta delta m, eta delta v). It samples five segment positions and hill-climbs their directional second derivatives with Richardson differences. It never raises a constant to accommodate a measured gap. E10 samples up to four Q states at every short-horizon time; R=40 samples four observed times per cell (two in quick) and uses their largest constant across the cell. E12 uses every message ray at the frozen boundary. These are empirical directional Taylor bounds, not full Hessian operator-norm certificates. Sign/clipping kinks and cold zero-variance coordinates can violate S1; nonfinite central probes are UNAVAILABLE. Every ratio above one remains a cell FAIL. R=40 also reports residual SE and whether an exceedance remains after four SE; that diagnostic does not inflate the bound or change the point-estimate status. Slopes use eta<=.01 with a 1e-12 numerical floor; zero terms have no fitted slope. The analytic nonzero LoRA witness retains its smaller signal and states that exception.', '',
        'E11 executes the clipping witness as a direct gradient-law linear-logit experiment. The always-active witness matches mean unit direction but differs in direction second moments; changing magnitudes alone with an identical direction law would give an exactly zero gap. The warm-Adam witness uses symmetric C23 labels and a declared admissible warm state m=0,v=.02,t=20 so theta/m contrasts cancel; generic equal-mean laws need not cancel the theta warmth error. Its W1 comparison bounds the explicit linear v term. Lion records both dominated and flipping controls and distinguishes stored m differences from a potentially zero continuation derivative. LoRA uses d=k=4, ranks 1 and 2, legal soft-label mean CE, B=0 and random A. A is checked bitwise after step one. Cold Adam equality of sketch signs is an epsilon→0 statement; finite-epsilon h_epsilon is checked separately.', '',
        'E12 reads the original full E6 split, models and held-out rows even in quick mode. It never fits a proxy or modifies E6/E9 artifacts, and checks their hashes before/after. Every smooth row uses the exact seven-update reference suffix and a full complete-state coordinate gradient, cross-checked against directional Richardson. Threshold utility is discontinuous: those rows retain formal numerical scores and proxy comparisons but are ineligible for C6. Zero predictions and the |gold|>1e-10 convention follow E6. The no-wrong-sign assertion above the empirical margin is hard, but is not a uniform certificate.', '',
        'New artifacts: `otlr.py`, `raw/E10_cells.json`, `raw/E12_margin_rows.json`, and PNG/SVG pairs `figs/V9`, `figs/E10`, `figs/E11`, `figs/E12`. All cell values, differentiation diagnostics, curvature searches, negative results and omissions are retained in the main JSON. `raw/results_before_otlr.json` preserves the pre-extension baseline. Results, README and legacy comparison are published using a same-directory temporary file, fsync and os.replace; concurrent readers never see truncation.', '']
    response=next((r['data'] for r in report['groups'] if r['id']=='E10'),{})
    if response:
        lines += ['## E10 outcomes and limitations','',
            'The group is INFO, with the following explicit per-optimizer cell status. Short-horizon gaps and continuations are exactly enumerated; R=40 quantities are sampled. The slope columns show moment N=3, R=4 at eta=.001,.003,.01. Other task/horizon slopes remain in JSON. Undefined first-order slopes reflect vanishing terms, not successful eta scaling.','',
            '| Optimizer | Cell status | Max ratio, R<=4 | Max ratio, all R | R=40 exceedances after 4 SE | Remainder slope | First-order slope | Sign agreement |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        def number(value):return 'n/a' if value is None else f'{float(value):.5g}'
        for r in response['optimizer_summary']:
            fit=next(s for s in r['slopes'] if s['task']=='moment' and s['N']==3 and s['R']==4)
            lines.append(f"| {r['optimizer']} | {r['status']} | {number(r['max_exact_ratio'])} | {number(r['max_ratio'])} | {r['MC_exceedances_after_four_SE']} | {number(fit['remainder_slope'])} | {number(fit['first_order_slope'])} | {number(r['sign_agreement'])} |")
        c23=response['C23_prediction']
        lines += ['',f"Cold Adam C23 (moment N=3, eta=.1, R=1): measured {c23['gap']:.10f}, transformed prediction {c23['first_order']:.10f}. {c23['sign_note']}",'',
            'Clipping frequencies use the declared equal mixture (P+Q)/2; uniform-alphabet fractions are also recorded. Norms within 1e-12 are treated as ties. A scalar threshold cannot split a tie. The 30–70% target is infeasible for: '+', '.join(f"{d['task']} N={d['N']} (chosen frequency {d['clip_active_probability']:.3g})" for d in response['design'] if not d['requested_clip_band_achievable'])+'.','',
            'Near-linear remainder slopes for stateful Adam do not support the derivation’s general O(eta^2) scaling claim. Dependence on moments through an O(eta) future update alone does not imply that the Hessian stays O(1) in an eta-weighted state norm. No constants were padded to fix this result.','']
    margins=next((r['data'] for r in report['groups'] if r['id']=='E12'),{})
    if margins:
        lines += ['## E12 frozen-panel outcomes','',
            '| Split | Smooth rows / all rows | Above margin | Oracle wrong sign, smooth | Oracle wrong sign, margin |',
            '|---|---:|---:|---:|---:|']
        for r in margins['summary']:
            lines.append(f"| {r['split']} | {r['eligible_rows']} / {r['rows']} | {r['margin_rows']} | {number(r['oracle_wrong_sign_eligible'])} | {number(r['oracle_wrong_sign_margin'])} |")
        lines += ['', 'Proxy comparison on the same smooth rows. Enrichment divides the fraction of a proxy’s failures outside the margin by the fraction of nonzero-gold rows outside it; a large raw fraction alone is not evidence of concentration.','',
            '| Proxy | Wrong sign, smooth | Wrong sign, margin | Outside-margin failure enrichment |',
            '|---|---:|---:|---:|']
        for name,r in margins['summary'][0]['proxies'].items():
            lines.append(f"| {name} | {number(r['wrong_sign_eligible'])} | {number(r['wrong_sign_margin'])} | {number(r['failure_enrichment_outside_margin'])} |")
        lines.append('')
    atomic_text(HERE/'README.md','\n'.join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--only',help='comma-separated group IDs, e.g. V3,E2')
    parser.add_argument('--legacy-baseline',type=Path,help='prior independent results for legacy data comparison (default: existing --output)')
    args=parser.parse_args(argv)
    if args.output is None:
        args.output=(HERE/'quick'/'results.json') if args.quick else (
            HERE/'selected_results.json' if args.only else HERE/'results_v015_independent.json')
    if not inside(args.output):parser.error('--output must be under theory/v015_checks/')
    selected=list(CHECKS) if not args.only else args.only.split(',')
    if any(key not in CHECKS for key in selected):parser.error('unknown group ID')
    C.QUICK=args.quick;C.OUT=HERE/'quick' if args.quick else HERE
    C.OUT.mkdir(parents=True,exist_ok=True)
    # SVG timestamps/IDs are nondeterministic. Preserve the already-frozen
    # E6/E9 figures after rerunning their unchanged checks and plot functions.
    retained_figures={path:path.read_bytes() for id in ('E6','E9') for suffix in ('png','svg')
                      if (path:=C.OUT/'figs'/f'{id}.{suffix}').exists()} if not args.quick else {}
    previous=None
    comparison_source=None
    if not args.quick and not args.only and args.output.exists():
        # Keep the original bytes BEFORE any incremental overwrite.
        previous=json.loads(args.output.read_text())
        snapshot_path=C.OUT/'raw'/'results_before_latest_full_run.json'
        snapshot_path.parent.mkdir(parents=True,exist_ok=True)
        snapshot_path.write_bytes(args.output.read_bytes())
        comparison_source=snapshot_path
    if args.legacy_baseline is not None:
        if not inside(args.legacy_baseline):parser.error('--legacy-baseline must be under theory/v015_checks/')
        previous=json.loads(args.legacy_baseline.read_text())
        comparison_source=args.legacy_baseline
    log_path=C.OUT/('selected_run.log' if args.only else 'full_run.log')
    began=time.perf_counter();original_stdout=sys.stdout
    report=dict(schema_version=1,master_seed=C.MASTER,quick=args.quick,groups=[],counterexamples=[],
        started_UTC=datetime.now(timezone.utc).isoformat(),
        command=dict(output=str(args.output),quick=args.quick,only=selected,
                     legacy_baseline=str(args.legacy_baseline) if args.legacy_baseline else None),
        versions=dict(python=platform.python_version(),numpy=C.np.__version__,scipy=C.scipy.__version__,
                      sympy=C.sp.__version__,matplotlib=C.matplotlib.__version__,
                      pytest=importlib.metadata.version('pytest')),
        provenance=dict(network=False,CUDA_VISIBLE_DEVICES='',processes=1,BLAS_threads=1,
            teacher=None,student='controlled NumPy tanh/sigmoid; no pretrained checkpoint',
            environment='finite constructed mathematical models and controlled neural learner',
            model_calls=0,training_tokens=0,split_manifest='proxy_split_freeze.json',
            independence='Never imports or reads the package verifier or its results. C23 helper reuse expressly authorized.'),
        source_sha256s={},runtime_seconds=None)
    if previous is not None:
        report['legacy_data_comparison']=dict(previous_sha256=C.digest(comparison_source),
            baseline_path=str(comparison_source.relative_to(HERE)) if comparison_source.is_absolute() else str(comparison_source),
            groups=[],different_groups=[])
    for path in [HERE/'SPEC.md',*sorted(HERE.glob('*.py')),*sorted((HERE/'tests').glob('*.py')),
        C.ROOT/'theory/independent_checks/check_v014_independent.py',
        C.ROOT/'docs/direction/v015_extension/Warm_Adam_Response_EN.md',
        C.ROOT/'docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md',
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
                if previous is not None and id in {*(f'V{i}' for i in range(1,8)),*(f'E{i}' for i in range(10))}:
                    old=next((x for x in previous['groups'] if x['id']==id),None)
                    if old is not None:
                        raw_data_bytes=json.dumps(C.serial(row['data']),sort_keys=True,separators=(',',':')).encode()
                        if id=='E7':
                            # Preserve the original timing BASELINE, not an invented
                            # repeated measurement. Every fresh timing is disclosed
                            # outside frozen group data. No scientific field is changed.
                            paths=[('measured_CPU_seconds',),('measured_CPU_seconds_per_pilot_pair_set',),
                                ('total_pilot_and_exact_decomposition_CPU_seconds',),
                                ('forecast',0,'controlled_CPU_seconds_projection'),
                                ('forecast',1,'controlled_CPU_seconds_projection')]
                            changes=[]
                            for path in paths:
                                a=old['data'];b=row['data']
                                for part in path[:-1]:a=a[part];b=b[part]
                                changes.append(dict(path=list(path),historical_baseline=a[path[-1]],current_measurement=b[path[-1]]))
                                b[path[-1]]=a[path[-1]]
                            report.setdefault('rerun_measurements',{})['E7']=dict(
                                note='Current execution timings. E7.data retains the original historical timing baseline by request; all scientific values and checks are recomputed.',
                                original_report_sha256=C.digest(comparison_source),timing_fields=changes)
                        old_bytes=json.dumps(old['data'],sort_keys=True,separators=(',',':')).encode()
                        new_bytes=json.dumps(C.serial(row['data']),sort_keys=True,separators=(',',':')).encode()
                        same=old_bytes==new_bytes
                        comparison=dict(id=id,data_byte_identical=same,
                            raw_rerun_data_byte_identical=raw_data_bytes==old_bytes,
                            old_sha256=C.hashlib.sha256(old_bytes).hexdigest(),new_sha256=C.hashlib.sha256(new_bytes).hexdigest(),
                            changed_fields=[key for key in set(old)|set(row) if key!='runtime_seconds' and C.serial(old.get(key))!=C.serial(row.get(key))])
                        report['legacy_data_comparison']['groups'].append(comparison)
                        if not same:
                            report['legacy_data_comparison']['different_groups'].append(id)
                            row['status']='FAIL'
                            report['counterexamples'].append(dict(group=id,assertion='legacy data byte identity',comparison=comparison))
                        print(f"{id} legacy data byte-identical={same}",flush=True)
                report['groups'].append(row)
                if id in ('E6','E9'):
                    for path,payload in retained_figures.items():
                        if path.stem==id and path.read_bytes()!=payload:atomic_bytes(path,payload)
                print(f"{id} {row['status']} | {key_number(row)} | {row['runtime_seconds']:.2f}s",flush=True)
                # Persist after every group, including failures and partial runs.
                report['runtime_seconds']=time.perf_counter()-began
                args.output.parent.mkdir(parents=True,exist_ok=True)
                atomic_text(args.output,json.dumps(C.serial(report),indent=2,allow_nan=False)+'\n')
            print(f"Completed in {report['runtime_seconds']:.2f}s; {len(report['counterexamples'])} unexpected failures",flush=True)
        finally:
            sys.stdout=original_stdout
    report['artifact_sha256s']={str(path.relative_to(HERE)):C.digest(path)
        for pattern in ('raw/*','figs/*','c23_bundle.json','proxy*json') for path in C.OUT.glob(pattern) if path.is_file()}
    report['runtime_seconds']=time.perf_counter()-began
    report['retained_frozen_figures']={str(path.relative_to(HERE)):C.digest(path) for path in retained_figures}
    report['interpretation_withheld']=any(r['id']=='E1' and r['status']=='FAIL' for r in report['groups'])
    atomic_text(args.output,json.dumps(C.serial(report),indent=2,allow_nan=False)+'\n')
    if previous is not None:atomic_text(C.OUT/'legacy_data_comparison.json',json.dumps(C.serial(report['legacy_data_comparison']),indent=2)+'\n')
    if not args.quick and not args.only:readme(report)
    return int(any(r['status']=='FAIL' for r in report['groups']))


if __name__=='__main__':
    raise SystemExit(main())
