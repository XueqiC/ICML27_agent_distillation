"""Precondition violations in failed episodes on teacher-solved tasks (adds to data/m2.json):
 transform_refused: heat/cool/clean task failures containing a refused '<verb> <target> N with ...'
 transform_after_putting_inside: ... of which the target had been moved INTO the appliance before the refused transform
 second_take_refused: two-object task failures containing a refused 'take <target> N' while a target was held (taken, not yet placed)."""
import json, re, sys
E = json.load(open(sys.argv[1]))
m2 = json.load(open('docs/method/motivation/data/m2.json'))
fm = json.load(open('reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
T = {t for t, e in fm['teacher'].items() if e['success']}
V = {'pick_heat_then_place_in_recep': ('heat', 'microwave'), 'pick_cool_then_place_in_recep': ('cool', 'fridge'),
     'pick_clean_then_place_in_recep': ('clean', 'sinkbasin')}
cellK = lambda c: c.split('|')[1] if '|' in c else None
res = {}
for K in ['10', '20', '30', '100']:
    tr = [0, 0, 0]; two = [0, 0]
    for t in T:
        typ, obj = t.split('/')[1].split('-')[0], t.split('/')[1].split('-')[1].lower()
        for c, ep in E[t].items():
            if cellK(c) != K or c == 'SAD|10' or ep['success']: continue
            st = [((s['a'] or '').lower(), s['o']) for s in ep['steps']]
            if typ in V:
                v, app = V[typ]; tr[1] += 1
                bad = [i for i, (a, o) in enumerate(st) if o == 'Nothing happens.' and re.match(rf'{v} {obj} \d+ with', a)]
                if bad:
                    tr[0] += 1
                    inside = any(re.match(rf'(move|put) {obj} \d+ (to|in|in/on) {app} \d+', a) and o != 'Nothing happens.' for a, o in st[:bad[0]])
                    tr[2] += inside
            if typ == 'pick_two_obj_and_place':
                two[1] += 1; held = 0
                for a, o in st:
                    if re.match(rf'take {obj} \d+ from', a):
                        if o == 'Nothing happens.' and held: two[0] += 1; break
                        if o != 'Nothing happens.': held += 1
                    if re.match(rf'(move|put) {obj} \d+ ', a) and o != 'Nothing happens.': held = max(0, held - 1)
    res[K] = {'transform_task_failures': tr[1], 'with_refused_transform': tr[0], 'refused_transform_after_putting_target_inside_appliance': tr[2],
              'two_object_failures': two[1], 'second_take_refused_while_holding': two[0]}
    print(K, res[K])
m2['precondition'] = res
json.dump(m2, open('docs/method/motivation/data/m2.json', 'w'), indent=1)
