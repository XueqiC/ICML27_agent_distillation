"""Search breadth: for student 'never saw the target' failures on teacher-solved tasks, compare the longest run of
consecutive same-type receptacle visits and the number of receptacle types visited with the teacher's episode on the same task."""
import collections, json, re, statistics, sys
E = json.load(open(sys.argv[1]))
m2 = json.load(open('docs/method/motivation/data/m2.json'))
fm = json.load(open('reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
TEACHER_OK = {t for t, e in fm['teacher'].items() if e['success']}
cellK = lambda c: c.split('|')[1] if '|' in c else None


def visits(ep, obj, stop_at_seen):
    v = []
    for s in ep['steps']:
        g = re.fullmatch(r'go to (\w+) \d+', (s['a'] or '').lower())
        if g and s['o'] != 'Nothing happens.': v.append(g[1])
        if stop_at_seen and re.search(rf'\b{obj} \d+', s['o'].lower()): break
    return v


def longest_run(v):
    best = cur = 0; prev = None
    for x in v:
        cur = cur + 1 if x == prev else 1; best = max(best, cur); prev = x
    return best


res = {}
for K in ['10', '20', '30', '100']:
    st, te, n = [], [], 0
    for t in TEACHER_OK:
        obj = t.split('/')[1].split('-')[1].lower()
        tv = visits(E[t]['teacher'], obj, True)
        for c, ep in E[t].items():
            if cellK(c) != K or c == 'SAD|10' or ep['success']: continue
            if re.search(rf'\b{obj} \d+', ep['init'].lower()) or any(re.search(rf'\b{obj} \d+', s['o'].lower()) for s in ep['steps']):
                continue
            sv = visits(ep, obj, False); n += 1
            st.append((longest_run(sv), len(set(sv)), len(sv)))
            te.append((longest_run(tv), len(set(tv)), len(tv)))
    res[K] = {'episodes': n,
              'student_never_seen': {'median_longest_same_type_run': statistics.median(x[0] for x in st),
                                     'median_types_visited': statistics.median(x[1] for x in st),
                                     'median_visits': statistics.median(x[2] for x in st)},
              'teacher_same_tasks_until_seen': {'median_longest_same_type_run': statistics.median(x[0] for x in te),
                                                'median_types_visited': statistics.median(x[1] for x in te),
                                                'median_visits': statistics.median(x[2] for x in te)}}
    print(K, res[K])
m2['search_runs'] = res
json.dump(m2, open('docs/method/motivation/data/m2.json', 'w'), indent=1)
