"""Build the small JSON tables behind the motivation result pages (run on rai from the project root).

Inputs (local artifacts, not all committed):
  reports/runs/repair_pilot/cov_theta0_k20_extra2, cov_theta0_umech      (M1 coverage rollouts)
  data/verified/seeds_full_train_r2/subsets/k*/                           (teacher demonstrations)
  HK_EPISODES = compact dev140 episodes of every E4B grid cell + teacher, produced on hpg by
                reports/analysis/main_grid_seeds/teacher_only/hk_extract.py (29 MB, not committed)
  reports/runs/repair_pilot/cg_probe/run1/rollouts/records                (M3 rollouts)
Outputs: docs/method/motivation/data/{m1,m2,m3}.json
"""
import collections, glob, json, os, re, statistics, sys

HK_EPISODES = sys.argv[1]
OUT = 'docs/method/motivation/data'
fm = json.load(open('reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
TEACHER_OK = {t for t, e in fm['teacher'].items() if e['success']}


def episodes(root):
    out = collections.defaultdict(dict)
    for att in sorted(glob.glob(root + '/attempt-*')):
        for p in glob.glob(att + '/episodes/*/episode.json'):
            e = json.load(open(p))
            out[e['task_id']][os.path.basename(att)] = e
    return out


# ---------------- M1 ----------------
k20 = episodes('reports/runs/repair_pilot/cov_theta0_k20_extra2/coverage')
um = episodes('reports/runs/repair_pilot/cov_theta0_umech/coverage')
demo = {}
m = json.load(open('data/verified/seeds_full_train_r2/subsets/k20/manifest.json'))
for a in m['attempts']:
    if a.get('won'):
        e = json.load(open('data/verified/seeds_full_train_r2/' + a['episode_dir'] + '/episode.json'))
        demo[a['task_id']] = [s.get('action_parsed') for s in e['steps']]
acts = lambda e: [s['action_parsed'] for s in e['steps']]
m1 = {'k20': {'tasks': len(k20), 'rollouts': sum(len(d) for d in k20.values()),
              'successes': sum(e['success'] for d in k20.values() for e in d.values()),
              'identical_across_decodings': sum(len({tuple(acts(e)) for e in d.values()}) == 1 for d in k20.values()),
              'equal_to_teacher_demo': sum(acts(d['attempt-01-greedy']) == demo[t] for t, d in k20.items()),
              'demo_lengths': sorted(len(demo[t]) for t in k20),
              'by_decoding': {a: sum(d[a]['success'] for d in k20.values()) for a in sorted(next(iter(k20.values())))}},
      'umech': {'tasks': len(um), 'rollouts': sum(len(d) for d in um.values()),
                'successes': sum(e['success'] for d in um.values() for e in d.values()),
                'identical_greedy_sampled': sum(len({tuple(acts(e)) for e in d.values()}) == 1 for d in um.values()),
                'by_decoding': {a: sum(d[a]['success'] for d in um.values()) for a in sorted(next(iter(um.values())))},
                'per_task_pattern': {str(k): v for k, v in collections.Counter(
                    tuple(d[a]['success'] for a in sorted(d)) for d in um.values()).items()},
                'by_type': {}}}
for t, d in um.items():
    typ = t.split('/')[1].split('-')[0]
    r = m1['umech']['by_type'].setdefault(typ, {'tasks': 0, 'rollouts': 0, 'successes': 0})
    r['tasks'] += 1; r['rollouts'] += len(d); r['successes'] += sum(e['success'] for e in d.values())
# failure types of the 15 failed U_mech tasks (greedy failure trajectory, rules of failure_modes.py)
m1['umech']['failed_task_types'] = {'never_saw_target': 7, 'saw_not_taken': 3, 'took_not_completed': 3, 'placed_not_completed': 2,
    'source': 'reports/analysis/repair_v01_evidence/mechanism_pilot_report_1026_ZH.md section 4.2 (U_mech table)'}
json.dump(m1, open(OUT + '/m1.json', 'w'), indent=1)

# ---------------- M2 ----------------
E = json.load(open(HK_EPISODES))
APP = {'pick_cool_then_place_in_recep': 'cool', 'pick_heat_then_place_in_recep': 'heat', 'pick_clean_then_place_in_recep': 'clean'}


def parse(t):
    typ, obj, _, rec, scene = t.split('/')[1].split('-')
    return typ, obj.lower(), rec.lower(), scene


def mode(t, ep):
    typ, obj, rec, _ = parse(t)
    L = [((s['a'] or '').lower(), s['o'].lower()) for s in ep['steps'] if s['o'] != 'Nothing happens.']
    if ep['success']:
        return 'success'
    seen = bool(re.search(rf'\b{obj} \d+', ep['init'].lower())) or any(re.search(rf'\b{obj} \d+', o) for _, o in L)
    took = any(re.match(rf'take {obj} \d+ from', a) for a, _ in L)
    placed = any(re.match(rf'(move|put) {obj} \d+ (to|in|on|in/on) {rec} \d+', a) for a, _ in L)
    if not seen: return 'A_never_seen'
    if not took: return 'B_seen_not_taken'
    if typ in APP:
        if not any(re.match(rf'{APP[typ]} {obj} \d+ with', a) for a, _ in L):
            return 'D_placed_without_transform' if placed else 'C_took_no_transform'
        return 'E_transformed_not_placed'
    if typ == 'look_at_obj_in_light': return 'C_took_no_lamp'
    if typ == 'pick_two_obj_and_place': return 'F_two_incomplete'
    return 'E_took_not_placed'


Ks = ['10', '20', '30', '100']
cellK = lambda c: c.split('|')[1] if '|' in c else None
cells = {K: sorted({c for t in E for c in E[t] if cellK(c) == K and c != 'SAD|10'}) for K in Ks}
modes = {K: collections.Counter() for K in Ks}
refused = {K: [0, 0] for K in Ks}
for t in TEACHER_OK:
    typ, obj, rec, _ = parse(t)
    for c, ep in E[t].items():
        K = cellK(c)
        if K is None or c == 'SAD|10': continue
        md = mode(t, ep); modes[K][md] += 1
        if md != 'success':
            refused[K][1] += 1
            refused[K][0] += any(s['o'] == 'Nothing happens.' and re.match(rf'(take|move|put|heat|cool|clean|use) .*{obj} \d+', (s['a'] or '').lower()) for s in ep['steps'])
goals = json.load(open('reports/analysis/main_grid_seeds/teacher_only/dev_goals.json'))
phr = json.load(open('reports/analysis/main_grid_seeds/teacher_only/dev_phr.json'))
groups = collections.defaultdict(list)
for t in goals:
    groups[(t.split('/')[1].split('-')[0], phr[t])].append(t)
phrasing = []
for (typ, p), ts in sorted(groups.items()):
    phrasing.append({'type': typ, 'phrasing': p, 'tasks': len(ts), 'teacher_ok': sum(t in TEACHER_OK for t in ts),
                     'solve_rate': {K: sum(E[t][c]['success'] for t in ts for c in cells[K] if c in E[t]) /
                                    sum(1 for t in ts for c in cells[K] if c in E[t]) for K in Ks}})
allfail = {K: [t.split('/')[1] for t in sorted(TEACHER_OK) if all(not E[t][c]['success'] for c in cells[K] if c in E[t])] for K in Ks}
m2 = {'cells_per_K': {K: len(v) for K, v in cells.items()}, 'cells': cells,
      'mode_counts': {K: dict(v) for K, v in modes.items()},
      'refused_target_action_in_failures': refused, 'phrasing': phrasing, 'teacher_ok_all_cells_fail': allfail,
      'demos_by_type': {K: json.load(open(f'data/verified/seeds_full_train_r2/subsets/k{K}/selection.json'))['counts_by_task_type'] for K in Ks}}
json.dump(m2, open(OUT + '/m2.json', 'w'), indent=1)

# ---------------- M3 ----------------
recs = [json.load(open(p)) for p in glob.glob('reports/runs/repair_pilot/cg_probe/run1/rollouts/records/*.json')]
m3 = {'per_case': {}, 'branch_totals': {}, 'steps_to_find': {}}
for r in recs:
    if r['decoding'] != 'sampled': continue
    g = r['group']; b = r['branch']; mt = r['metrics']
    d = m3['per_case'].setdefault(g, {}).setdefault(b, {'found': 0, 'picked': 0, 'success': 0, 'n': 0})
    d['n'] += 1; d['found'] += mt['found']; d['picked'] += mt['picked']; d['success'] += mt['success']
    tb = m3['branch_totals'].setdefault(b, {'found': 0, 'picked': 0, 'success': 0, 'n': 0, 'followed': 0, 'first_action_differs_from_A': 0})
    tb['n'] += 1; tb['found'] += mt['found']; tb['picked'] += mt['picked']; tb['success'] += mt['success']
    tb['followed'] += bool(mt['followed_suggestion'])
    a = next(x for x in recs if x['group'] == g and x['branch'] == 'A' and x['decoding'] == 'sampled' and x['seed'] == r['seed'])
    tb['first_action_differs_from_A'] += mt['first_post_branch_action'] != a['metrics']['first_post_branch_action']
    m3['steps_to_find'].setdefault(b, []).append(mt['steps_to_find'])
json.dump(m3, open(OUT + '/m3.json', 'w'), indent=1)
print('ok', {k: len(v) for k, v in cells.items()})
