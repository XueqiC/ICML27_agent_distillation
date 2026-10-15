"""Collect the numbers behind docs/proposals/P1_empirical_findings*.md (read-only; CPU only).

Inputs
- reports/runs/eval/...: rai copies of the hpg dev140 evaluations (Seed-SFT sweep, AgentGym BC/AgentEvol).
- --r1-dir: R1 rows copied from $B/reports/runs/r03/r1 (p1p3-*.jsonl, p2p4-0.jsonl, merge_p1p3_*.json).
- --probe: $B/reports/runs/pilot-warm-1010/probe/cold.json (Adam response probe).
Numbers that exist only in notes/exp_log.md (paired counts of pilot-warm-1010, LEAP K50/K100) are copied below with
their log line, not recomputed.

Output: docs/proposals/figs/p1_findings_data.json
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVAL = os.path.join(REPO, 'reports', 'runs', 'eval')
TYPES = ['look_at_obj_in_light', 'pick_and_place_simple', 'pick_clean_then_place_in_recep',
         'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_two_obj_and_place']


def task_type(task_id):
    return re.search(r'/(' + '|'.join(TYPES) + r')-', task_id).group(1)


def cell(eval_dir):
    s = json.load(open(os.path.join(eval_dir, 'summary.json')))
    assert s['episodes'] == 140, eval_dir
    per_type = defaultdict(lambda: [0, 0])
    n = 0
    for f in glob.glob(os.path.join(eval_dir, 'episodes', '*', 'episode.json')):
        e = json.load(open(f))
        per_type[task_type(e['task_id'])][0] += e['success'] is True
        per_type[task_type(e['task_id'])][1] += 1
        n += 1
    out = {'successes': s['successes'], 'unknown': s.get('unknown_success', 0)}
    if n == 140:  # some direct evaluations were copied without episodes; their per-type counts are not used
        out['per_type'] = dict(per_type)
    return out


def adapter_tokens(server_dir):
    p = json.load(open(os.path.join(server_dir, 'provenance.json')))
    return {a['name']: int(re.search(r'checkpoint-tokens-(\d+)', a['path']).group(1))
            for a in p['config']['server']['lora_adapters']}


def sweep():
    out, base = {}, None
    reeval = os.path.join(EVAL, 'hpg-seedsft-sweep-bi-reeval-s5')
    # later roots override earlier ones (v2k12-r2 is the rerun of v2k12 cells with an error episode)
    for root in ['hpg-seedsft-sweep-bi', 'hpg-seedsft-sweep-bi-v2k12', 'hpg-seedsft-sweep-bi-v2k12-r2',
                 'hpg-seedsft-sweep-bi-v2k50']:
        for run_dir in sorted(glob.glob(os.path.join(EVAL, root, 'k*s*'))):
            run = os.path.basename(run_dir)
            for name, tokens in adapter_tokens(os.path.join(run_dir, 'server')).items():
                ev, source = os.path.join(run_dir, name, 'eval'), root
                if not os.path.exists(os.path.join(ev, 'summary.json')):
                    ev, source = os.path.join(reeval, name, 'eval'), 'hpg-seedsft-sweep-bi-reeval-s5'
                    if not os.path.exists(os.path.join(ev, 'summary.json')):
                        continue  # equivalence gate failed and no valid re-evaluation: cell not used
                out.setdefault(run, {})[name.split('-', 1)[1]] = dict(cell(ev), tokens=tokens, source=source)
            b = os.path.join(run_dir, 'google', 'gemma-4-12B-it', 'eval')
            if base is None and root == 'hpg-seedsft-sweep-bi' and os.path.exists(os.path.join(b, 'summary.json')):
                base = dict(cell(b), source=os.path.relpath(b, REPO))
    return out, base


def agentevol():
    out = {}
    for ev in glob.glob(os.path.join(EVAL, 'hpg-baselines', 'baselines', 'agentgym-k*s*', 'stages', 'eval-*',
                                     'attempt-*', 'output', '*', 'eval')):
        if not os.path.exists(os.path.join(ev, 'summary.json')):
            continue  # failed in-pipeline evaluation (context overflow before the 10-06 fix)
        run = re.search(r'(agentgym-k\d+s\d)', ev).group(1)
        it = os.path.basename(os.path.dirname(ev))
        out.setdefault(run, {})[it] = dict(cell(ev), source='in-pipeline evaluation')
    # direct evaluations (run after the 10-06 context-overflow fix) take precedence
    for ev in glob.glob(os.path.join(EVAL, 'hpg-baselines', 'eval', 'baselines', 'agentgym-k*s*-*', 'agentgym-*', 'eval')):
        m = re.match(r'(agentgym-k\d+s\d)-(bc|iter-\d)$', os.path.basename(os.path.dirname(ev)))
        if m and os.path.exists(os.path.join(ev, 'summary.json')):
            out.setdefault(m.group(1), {})[m.group(2)] = dict(cell(ev), source='direct evaluation')
    return out


def leap():
    out = {}
    for ev in glob.glob(os.path.join(EVAL, 'hpg-baselines', 'baselines', 'leap-privileged-k*s0', 'stages', 'eval-iter-*',
                                     'attempt-*', 'output', '*', 'eval')):
        if not os.path.exists(os.path.join(ev, 'summary.json')):
            continue
        run = re.search(r'leap-privileged-(k\d+s0)/', ev).group(1)
        it = re.search(r'eval-(iter-\d)', ev).group(1)
        out.setdefault(run, {})[it] = cell(ev)['successes']
    # exp_log 10-08 14:30 / 14:28 CDT (evaluations read on hpg; not copied to rai)
    out['k50s0'] = {'iter-0': 52, 'iter-1': 53, 'iter-2': 75, 'iter-3': 81}
    out['k100s0'] = {'iter-0': 49, 'iter-1': 68, 'iter-2': 92, 'iter-3': 82}
    return out


def r1(r1_dir):
    rows = [json.loads(line) for f in sorted(glob.glob(os.path.join(r1_dir, 'p1p3-*.jsonl'))) +
            [os.path.join(r1_dir, 'p2p4-0.jsonl')] for line in open(f)]
    by = {(r['candidate_id'], r['replica']): r for r in rows}
    static = {}
    for c in sorted({c for c, _ in by}):
        msgs = [by[k]['message'] for k in by if k[0] == c]
        static[c] = {'replicas': len(msgs),
                     'student_loss': st.mean(m['weighted_token_mean_loss'] for m in msgs),
                     'pre_clip_grad_norm': st.mean(m['pre_clip_grad_norm'] for m in msgs),
                     'update_norm': st.mean(m['update_norm'] for m in msgs),
                     'supervised_tokens': st.mean(m['supervised_tokens'] for m in msgs),
                     'examples': st.mean(m['examples'] for m in msgs),
                     'clip_active': st.mean(1.0 if m['clip_active'] else 0.0 for m in msgs)}
    paired = {}
    for metric in ['F_tilde_geo_prob', 'F_tilde_per_token']:
        for k in ['P3', 'P2']:
            reps = sorted(r for c, r in by if c == k)
            assert all(by[('P1', r)]['tail_seed'] == by[(k, r)]['tail_seed'] for r in reps)
            z = [by[(k, r)][metric] - by[('P1', r)][metric] for r in reps]  # candidate minus replay
            paired[f'{metric}:{k}-P1'] = {'replicas': reps, 'Z': z, 'mean': st.mean(z),
                                          'se': st.stdev(z) / math.sqrt(len(z)), 'positive': sum(v > 0 for v in z)}
    merge = json.load(open(os.path.join(r1_dir, 'merge_p1p3_F_tilde_geo_prob.json')))['pairs'][0]
    return {'static': static, 'paired': paired,
            'merge_p1p3_geo_prob': {k: merge[k] for k in ['S_hat', 'radius_a_priori', 'radius_observed_range_DIAGNOSTIC']}}


def probe(path):
    d = json.load(open(path))
    out = []
    for s in d['states']:
        row = {'state': s['state'], 'optimizer_step': s['optimizer_step']}
        for opt, key in [('AdamW_clipped_float64_reference', 'adamw_clipped'),
                         ('AdamW_unclipped_float64_reference', 'adamw_unclipped'),
                         ('SGD_unclipped_float64_reference', 'sgd_unclipped')]:
            row[key + '_momentum_share'] = [p[opt]['momentum_share'] for p in s['pairs']]
            row[key + '_nu_centered'] = [p[opt]['nu_centered'] for p in s['pairs']]
        out.append(row)
    return out


PILOT_WARM = {  # exp_log 10-11 11:40 CDT, eval 45476444 (one server, batch-invariant, BI dev140)
    'p10': {'cold': 66, 'mw32': 77, 'lw32': 74, 'mw32_vs_cold_discordant': [19, 8], 'mw32_vs_cold_p': 0.052},
    'p40': {'cold': 112, 'mw32': 99, 'lw32': 104, 'mw32_vs_cold_discordant': [6, 19], 'mw32_vs_cold_p': 0.015},
    'noise_reference_flips_across_servers': [8, 7],
}
ETO_FROM_P40 = {  # exp_log 10-07 02:09 CDT, job 45008705 (ETO dose control on the Seed-SFT k20s0-p40 init), train log
    'explore_1_pairs': 3, 'dpo_1_updates': 9, 'dpo_loss_approx': 1e-13, 'grad_norm_approx': 1e-9,
}
REFERENCES = {'teacher_dev140': 114, 'noise_reference_dev140_cross_config': 7,
              'kang_k20s0_training_seed_successes': [63, 74, 86]}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--r1-dir', required=True)
    ap.add_argument('--probe', required=True)
    args = ap.parse_args()
    sw, base = sweep()
    data = {'sweep': sw, 'base': base, 'agentevol': agentevol(), 'leap': leap(), 'r1': r1(args.r1_dir),
            'probe': probe(args.probe), 'pilot_warm': PILOT_WARM, 'eto_from_p40': ETO_FROM_P40, 'references': REFERENCES}
    out = os.path.join(REPO, 'docs', 'proposals', 'figs', 'p1_findings_data.json')
    json.dump(data, open(out, 'w'), indent=1)
    print('wrote', out, os.path.getsize(out), 'bytes; base', base['successes'])
