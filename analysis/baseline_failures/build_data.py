"""Numbers behind docs/analysis/baseline_failure_analysis*.md (read-only; CPU only; dev140 = valid_seen only).

Every evaluated dev140 cell of every baseline that has a rai copy is read episode by episode:
- success per task type;
- for failed episodes, the furthest task subgoal the agent reached (take the target object -> clean/heat/cool it or switch on
  the lamp -> put it in the target receptacle; pick_two needs two distinct instances), read from the executed legal actions;
- behaviour: steps, inadmissible actions, parser errors, context overflow, repeated actions.
Training-process records (dose, self-generated or corrected data sizes, preference pairs) come from each run's iteration-*.json.
Numbers that exist only in notes/exp_log.md (runs whose directories were not copied to rai) are copied below with their log line.

Run: python3 analysis/baseline_failures/build_data.py   ->   docs/analysis/figs/baseline_failures_data.json
"""
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVAL = os.path.join(REPO, 'reports', 'runs', 'eval')
HB = os.path.join(EVAL, 'hpg-baselines')
TEACHER = os.path.join(REPO, 'reports', 'runs', 'teacher-dev140')
OUT = os.path.join(REPO, 'docs', 'analysis', 'figs', 'baseline_failures_data.json')
TYPES = ['pick_and_place_simple', 'look_at_obj_in_light', 'pick_clean_then_place_in_recep',
         'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_two_obj_and_place']
TASK_TYPE_RE = re.compile(r'/(' + '|'.join(TYPES) + r')-')  # also matches train/ ids in process records
TASK_RE = re.compile(r'valid_seen/(' + '|'.join(TYPES) + r')-([A-Za-z]+)-([A-Za-z]+)-([A-Za-z]+)-\d+/')
TRANSFORM = {'pick_clean_then_place_in_recep': 'clean', 'pick_heat_then_place_in_recep': 'heat',
             'pick_cool_then_place_in_recep': 'cool'}


def subgoal_stage(task_id, actions):
    """Furthest ordered subgoal reached, from executed legal actions only (inadmissible actions do not change the state).

    Returns one of: 'search' (target object never taken), 'transform' (taken, never cleaned/heated/cooled after taking it,
    or the lamp never used in a look task), 'place' (ready but never put in a target receptacle), 'second' (pick_two: one
    placed, second not), 'all_subgoals' (every subgoal action executed, yet the environment did not register success).
    A few dev tasks register success without the final put (e.g. a clean mug for the coffee machine); successes are never
    classified, so this only matters for failures of those tasks.
    """
    ttype, obj, _, recep = TASK_RE.search(task_id).groups()
    obj, recep = obj.lower(), recep.lower()
    take = re.compile(rf'^take {obj} (\d+) from ')
    put = re.compile(rf'^move {obj} (\d+) to {recep} \d+$')
    held, done, stage = None, 0, 0
    if ttype == 'pick_two_obj_and_place':
        placed = set()
        for a in actions:
            m = take.match(a)
            if m:
                held = m.group(1)
            m = put.match(a)
            if m and m.group(1) == held:
                placed.add(held)
                held = None
        took = any(take.match(a) for a in actions)
        return 'search' if not took else ('place' if not placed else ('second' if len(placed) < 2 else 'all_subgoals'))
    need = TRANSFORM.get(ttype)
    if ttype == 'look_at_obj_in_light':  # the lamp may be switched on before or after taking the object
        lamp = re.compile(rf'^use {recep} \d+$')
        took = any(take.match(a) for a in actions)
        return 'search' if not took else ('all_subgoals' if any(lamp.match(a) for a in actions) else 'transform')
    tr = re.compile(rf'^{need} {obj} \d+ with ') if need else None
    for a in actions:
        if take.match(a):
            stage = max(stage, 1)
        elif tr is not None and stage >= 1 and tr.match(a):
            stage = max(stage, 2)
        elif put.match(a) and stage >= (2 if need else 1):
            done = 1
    if done:
        return 'all_subgoals'
    if stage == 0:
        return 'search'
    if need and stage == 1:
        return 'transform'
    return 'place'


def episode_record(e):
    steps = e['steps']
    legal = [s['action_parsed'] for s in steps if s.get('action_outcome') == 'legal' and s.get('action_parsed')]
    tail = [s.get('action_parsed') for s in steps[-10:]]
    rec = {'task_id': e['task_id'], 'type': TASK_RE.search(e['task_id']).group(1), 'success': e['success'] is True,
           'steps': len(steps), 'inadmissible': sum(s.get('action_outcome') == 'inadmissible' for s in steps),
           'parser_errors': sum(s.get('action_outcome') == 'parser_error' for s in steps),
           'overflow': bool(e.get('metrics', {}).get('context_overflow_episodes', 0)),
           'repeat_share': (1 - len(set(legal)) / len(legal)) if legal else 0.0,
           'tail_loop': len(steps) >= 10 and len(set(tail)) <= 3}
    if not rec['success']:
        rec['stage'] = subgoal_stage(e['task_id'], legal)
    return rec


def load_eval(eval_dir):
    s = json.load(open(os.path.join(eval_dir, 'summary.json')))
    assert s['episodes'] == 140, eval_dir
    files = glob.glob(os.path.join(eval_dir, 'episodes', '*', 'episode.json'))
    eps = [episode_record(json.load(open(f))) for f in files]
    assert all(r['task_id'].startswith('valid_seen/') for r in eps), eval_dir  # dev only; the sealed split is never read
    out = {'successes': s['successes'], 'source': os.path.relpath(eval_dir, REPO)}
    if len(eps) == 140:
        assert sum(r['success'] for r in eps) == s['successes'], eval_dir
        out['episodes'] = eps
    return out


def adapter_tokens(server_dir):
    p = json.load(open(os.path.join(server_dir, 'provenance.json')))
    return {a['name']: int(re.search(r'checkpoint-tokens-(\d+)', a['path']).group(1))
            for a in p['config']['server']['lora_adapters']}


def seedsft():
    """Same cell rules as analysis/p1_findings/build_data.py: later roots override, failed equivalence gates use re-eval s5."""
    cells, base = {}, None
    reeval = os.path.join(EVAL, 'hpg-seedsft-sweep-bi-reeval-s5')
    for root in ['hpg-seedsft-sweep-bi', 'hpg-seedsft-sweep-bi-v2k12', 'hpg-seedsft-sweep-bi-v2k12-r2',
                 'hpg-seedsft-sweep-bi-v2k50']:
        for run_dir in sorted(glob.glob(os.path.join(EVAL, root, 'k*s*'))):
            run = os.path.basename(run_dir)
            for name, tokens in adapter_tokens(os.path.join(run_dir, 'server')).items():
                ev = os.path.join(run_dir, name, 'eval')
                if not os.path.exists(os.path.join(ev, 'summary.json')):
                    ev = os.path.join(reeval, name, 'eval')
                    if not os.path.exists(os.path.join(ev, 'summary.json')):
                        continue
                cells[(run, name.split('-', 1)[1])] = (ev, tokens)
            b = os.path.join(run_dir, 'google', 'gemma-4-12B-it', 'eval')
            if base is None and root == 'hpg-seedsft-sweep-bi' and os.path.exists(os.path.join(b, 'summary.json')):
                base = b
    out = []
    for (run, label), (ev, tokens) in sorted(cells.items()):
        k, s = map(int, re.match(r'k(\d+)s(\d+)', run).groups())
        out.append(dict(load_eval(ev), method='Seed-SFT', K=k, seed=s, dose=label, tokens=tokens))
    return out, base


def agentgym():
    found = {}
    for ev in glob.glob(os.path.join(HB, 'baselines', 'agentgym-k*s*', 'stages', 'eval-*', 'attempt-*', 'output', '*', 'eval')):
        if os.path.exists(os.path.join(ev, 'summary.json')):
            found[(re.search(r'agentgym-(k\d+s\d)', ev).group(1), os.path.basename(os.path.dirname(ev)))] = ev
    for ev in glob.glob(os.path.join(HB, 'eval', 'baselines', 'agentgym-k*s*-*', 'agentgym-*', 'eval')):
        m = re.match(r'agentgym-(k\d+s\d)-(bc|iter-\d)$', os.path.basename(os.path.dirname(ev)))
        if m and os.path.exists(os.path.join(ev, 'summary.json')):
            found[(m.group(1), m.group(2))] = ev  # direct evaluations (after the 10-06 overflow fix) take precedence
    out = []
    for (cell, it), ev in sorted(found.items()):
        k, s = map(int, re.match(r'k(\d+)s(\d)', cell).groups())
        out.append(dict(load_eval(ev), method='AgentGym', K=k, seed=s, iteration=0 if it == 'bc' else int(it[-1])))
    return out


def kang():
    out = []
    for ev in glob.glob(os.path.join(HB, 'eval', 'baselines', 'kang-k*s*-tp*', 'kang-*', 'eval')):
        name = os.path.basename(os.path.dirname(os.path.dirname(ev)))
        m = re.match(r'kang-k(\d+)s(\d)-tp(?:-seed(\d))?-(sag|greedy)$', name)
        if m:
            out.append(dict(load_eval(ev), method='Kang', K=int(m.group(1)), seed=int(m.group(2)),
                            training_seed=int(m.group(3) or 0), decoding=m.group(4)))
    sag = os.path.join(HB, 'eval', 'baselines', 'sag-on-seedsft', 'k20s0-p40-sag', 'eval')
    if os.path.exists(os.path.join(sag, 'summary.json')):
        out.append(dict(load_eval(sag), method='SAG on Seed-SFT', K=20, seed=0, decoding='sag'))
    return out


def leap():
    out = []
    for ev in glob.glob(os.path.join(HB, 'baselines', 'leap-*-k*s0*', 'stages', 'eval-iter-*', 'attempt-*', 'output', '*', 'eval')):
        m = re.search(r'leap-(privileged|zero)-k(\d+)s0(-m5|-init)?/stages/eval-iter-(\d)', ev)
        if m and os.path.exists(os.path.join(ev, 'summary.json')):
            variant = {'privileged': 'LEAP privileged', 'zero': 'LEAP 0% privileged'}[m.group(1)]
            if m.group(3) == '-init':
                variant = 'LEAP privileged (Seed-SFT p40 init)'
            out.append(dict(load_eval(ev), method=variant, K=int(m.group(2)), seed=0, iteration=int(m.group(4))))
    return out


def iteration_records():
    rec = defaultdict(dict)
    for f in glob.glob(os.path.join(HB, 'baselines', '*', 'iteration-*.json')):
        run = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        keep = {k: d[k] for k in ['F_m', 'D_m', 'D_next', 'supervised_tokens_seen', 'corrections', 'failed_rollouts',
                                  'rejected_corrections', 'excluded_no_privileged_state', 'inadmissible_corrections',
                                  'rollouts', 'P', 'tie', 'lose', 'win'] if k in d}
        if isinstance(d.get('dose'), dict):
            keep['updates'] = d['dose'].get('total_steps')
            keep['N_examples'] = d['dose'].get('N_examples')
        for k in ['P_by_kind', 'agent_chosen', 'expert_chosen']:  # IPR pair composition
            if k in d:
                keep[k] = d[k]
        agg = d.get('aggregation')
        if isinstance(agg, dict):
            keep['aggregation_before'] = agg.get('before')
        rec[run][d['iteration']] = keep
    # AgentEvol: which of the K seed TRAIN tasks the current iterate re-solved (accepted into F_m), by task type
    for f in glob.glob(os.path.join(HB, 'baselines', 'agentgym-k*', 'stages', 'filter-*', 'attempt-000', 'output', 'report.json')):
        run = f.split(os.sep)[-6]
        d = json.load(open(f))
        decisions = d['decisions']
        it = d['exploration_iteration']
        typ = lambda t: TASK_TYPE_RE.search(t).group(1)
        rec[run].setdefault(it, {})['explored_types'] = dict(Counter(typ(x['task_id']) for x in decisions))
        rec[run][it]['accepted_types'] = dict(Counter(typ(x['task_id']) for x in decisions if x['decision'] == 'accepted'))
    # Kang: one SFT stage; its training state records updates and supervised tokens
    for f in glob.glob(os.path.join(HB, 'baselines', 'kang-k*-tp*', 'stages', 'train-sft', 'attempt-000', 'output',
                                    'checkpoint', 'training_state.json')):
        d = json.load(open(f))
        rec[f.split(os.sep)[-7]]['sft'] = {k: d[k] for k in ['updates', 'supervised_tokens', 'passes']}
    return rec


# Numbers recorded only in notes/exp_log.md (run directories not copied to rai); the log line is given for each.
LOG_ONLY = {
    'leap_dev140': {  # exp_log 10-06 lines 71/73 (K20 m5), 10-08 14:30 / 14:28 CDT (K50, K100)
        'LEAP privileged K20': [58, 62, 57, 60], 'LEAP 0% privileged K20': [58, 59, 52, 58],
        'LEAP privileged K50': [52, 53, 75, 81], 'LEAP privileged K100': [49, 68, 92, 82]},
    'leap_corrections': {  # exp_log lines 69, 72, 119, 122, 125
        'LEAP privileged K20': {'corrections': [8, 4, 4], 'aggregate_N': [640, 480, 640]},
        'LEAP 0% privileged K20': {'corrections': [11, 7, 8], 'rejected': [2, 5, 7], 'aggregate_N': [764, 840, 1120]},
        'LEAP privileged K50 round 3': {'rollouts': 50, 'failed': 29, 'corrections': 20, 'rejected': 1, 'no_privileged_state': 8},
        'LEAP privileged K100 round 2': {'rollouts': 100, 'failed': 58, 'corrections': 42, 'rejected': 2, 'no_privileged_state': 14,
                                         'aggregate_N': 5040},
        'LEAP privileged K100 round 3': {'rollouts': 100, 'failed': 31, 'corrections': 18, 'rejected': 4, 'no_privileged_state': 9,
                                         'aggregate_N': 2880, 'updates': 90},
        'LEAP privileged K20, Seed-SFT p40 init': {'rollouts': 20, 'failed': 6, 'corrections': 0, 'no_privileged_state': 6}},
    'eto_explore_k20s0': {  # successes of the SFT model on its own 20 TRAIN tasks = 20 - P (exp_log lines 34, 40, 86, 116;
        # tp5/tp6/init iteration files are read directly into 'iterations')
        'tp2 (2 GPU)': {'P': 11, 'tie': 9}, 'tp3 (4 GPU)': {'P': 4, 'tie': 16}, 'tp4': {'P': 15}},
    'sft_updates_eto_ipr_k20s0': 30,  # 3 epochs x 10 updates (IPR r4 train-sft training_state.json on hpg; ETO uses the same recipe)
    'eto_dpo1_updates_tp6': 27,  # exp_log line 116
    'eto_dpo': {  # exp_log lines 89 and 110
        'tp4 DPO-1': {'pairs': 15, 'updates': 45, 'loss_epoch2': 2e-21},
        'Seed-SFT p40 init DPO-1': {'pairs': 3, 'updates': 9, 'loss': 1e-13, 'grad_norm': 1e-9}},
    'ipr_pairs_k20s0': {  # exp_log lines 116 and 121; r4 hashcheck line 133 gives the rollout counts
        'r3 iteration 1': {'P': 95, 'step_level': 83, 'trajectory_level': 12, 'agent_chosen': 25, 'expert_chosen': 70},
        'r4 iteration 1': {'P': 84, 'step_level': 71, 'trajectory_level': 13, 'agent_chosen': 8, 'expert_chosen': 76},
        'r4 rollouts per iteration': {'expert_mc': 1910, 'explore': 382, 'agent_mc': 1910},
        'dpo hours per iteration (2xB200, estimated)': [9, 13]},
    'kang_training_seeds_k20s0': [63, 74, 86],  # exp_log line 97
    'noise_reference_dev140': 7,  # same adapter, two identically configured servers (P1 findings, common setting)
}


TASK_INDEX = {}  # dev140 task id -> position in data['tasks']


def summarize(e):
    """Per-cell aggregates; episodes are dropped from the output to keep the snapshot small."""
    out = {k: v for k, v in e.items() if k != 'episodes'}
    eps = e.get('episodes')
    if not eps:
        return out
    pt = defaultdict(lambda: [0, 0])
    stages = Counter()
    stages_by_type = defaultdict(Counter)
    for r in eps:
        pt[r['type']][0] += r['success']
        pt[r['type']][1] += 1
        if not r['success']:
            key = 'overflow' if r['overflow'] else r['stage']
            stages[key] += 1
            stages_by_type[r['type']][key] += 1
    fail = [r for r in eps if not r['success']]
    succ = [r for r in eps if r['success']]
    out.update(per_type=dict(pt), failure_stage=dict(stages), failure_stage_by_type={t: dict(c) for t, c in stages_by_type.items()},
               inadmissible_per_episode=st.mean(r['inadmissible'] for r in eps),
               parser_errors_per_episode=st.mean(r['parser_errors'] for r in eps),
               overflow_episodes=sum(r['overflow'] for r in eps),
               failed_hit_step_cap=sum(r['steps'] >= 40 for r in fail),
               failed_tail_loop=sum(r['tail_loop'] for r in fail),
               failed_repeat_share=st.mean(r['repeat_share'] for r in fail) if fail else None,
               success_steps_median=st.median(r['steps'] for r in succ) if succ else None,
               solved=sorted(TASK_INDEX[r['task_id']] for r in succ))
    return out


if __name__ == '__main__':
    sft, base_dir = seedsft()
    cells = sft + agentgym() + kang() + leap()
    base = dict(load_eval(base_dir), method='base')
    t = json.load(open(os.path.join(TEACHER, 'summary.json')))
    teacher_eps = [episode_record(json.load(open(f))) for f in glob.glob(os.path.join(TEACHER, 'episodes', '*', 'episode.json'))]
    assert len(teacher_eps) == 140 and sum(r['success'] for r in teacher_eps) == t['successes']
    TASK_INDEX.update({tid: i for i, tid in enumerate(sorted(r['task_id'] for r in teacher_eps))})
    teacher ={'method': 'teacher', 'successes': t['successes'], 'episodes': teacher_eps,
               'source': os.path.relpath(TEACHER, REPO)}
    data = {'base': summarize(base), 'teacher': summarize(teacher), 'cells': [summarize(c) for c in cells],
            'iterations': iteration_records(), 'log_only': LOG_ONLY, 'types': TYPES,
            'tasks': sorted(TASK_INDEX, key=TASK_INDEX.get)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, 'w'), indent=1)
    n_ep = sum('per_type' in c for c in data['cells'])
    print('wrote', OUT, os.path.getsize(OUT), 'bytes;', len(cells), 'cells,', n_ep, 'with episodes; base', base['successes'],
          'teacher', t['successes'])
