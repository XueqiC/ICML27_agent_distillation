"""Prevalence statistics for the four phenomena over ALL teacher-solved dev140 tasks x E4B cells (adds to data/m2.json)."""
import collections, json, re, sys
E = json.load(open(sys.argv[1]))
m2 = json.load(open('docs/method/motivation/data/m2.json'))
fm = json.load(open('reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
TEACHER_OK = {t for t, e in fm['teacher'].items() if e['success']}
Ks = ['10', '20', '30', '100']
cellK = lambda c: c.split('|')[1] if '|' in c else None


def go_seq(steps):
    """Consecutive legal 'go to <type> n' pairs within the same receptacle type: fraction with m = n+1."""
    gos = [re.fullmatch(r'go to (\w+) (\d+)', (s['a'] or '').lower()) for s in steps if s['o'] != 'Nothing happens.']
    gos = [g for g in gos if g]
    same = [(a, b) for a, b in zip(gos, gos[1:]) if a[1] == b[1]]
    return sum(int(b[2]) == int(a[2]) + 1 for a, b in same), len(same), len(gos)


out = {}
for who in Ks + ['teacher']:
    rep_ref = [0, 0]; seq = [0, 0]; types_before_seen = []
    for t in TEACHER_OK:
        obj = t.split('/')[1].split('-')[1].lower()
        for c, ep in E[t].items():
            if (who == 'teacher' and c != 'teacher') or (who != 'teacher' and (cellK(c) != who or c == 'SAD|10')):
                continue
            st = ep['steps']
            if not ep['success']:
                refused = [(s['a'] or '').lower() for s in st if s['o'] == 'Nothing happens.']
                rep_ref[1] += 1
                rep_ref[0] += any(n >= 2 for n in collections.Counter(refused).values())
            a, b, _ = go_seq(st); seq[0] += a; seq[1] += b
            # distinct receptacle types visited before the target is first seen (successful episodes only)
            if ep['success']:
                seen = set()
                for s in st:
                    g = re.fullmatch(r'go to (\w+) \d+', (s['a'] or '').lower())
                    if g and s['o'] != 'Nothing happens.': seen.add(g[1])
                    if re.search(rf'\b{obj} \d+', s['o'].lower()): break
                types_before_seen.append(len(seen))
    out[who] = {'failed_eps_with_repeated_refused_action': rep_ref,
                'same_type_goto_pairs_in_numeric_order': seq,
                'mean_types_before_target_seen_successes': sum(types_before_seen) / max(1, len(types_before_seen))}
m2['prevalence'] = out
json.dump(m2, open('docs/method/motivation/data/m2.json', 'w'), indent=1)
for k, v in out.items():
    r, s = v['failed_eps_with_repeated_refused_action'], v['same_type_goto_pairs_in_numeric_order']
    print(k, f"repeated refused {r[0]}/{r[1]} ({r[0]/max(1,r[1]):.0%})", f"numeric-order same-type go-to {s[0]}/{s[1]} ({s[0]/max(1,s[1]):.0%})", f"types before seen {v['mean_types_before_target_seen_successes']:.2f}")
