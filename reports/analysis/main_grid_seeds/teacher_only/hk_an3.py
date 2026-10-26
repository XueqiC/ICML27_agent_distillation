import json,re,collections
E=json.load(open('hk_episodes_all.json')); PH=json.load(open('dev_phr.json'))
exec(open('hk_an2.py').read().split('def Kof')[0].split("E=json.load")[0])  # imports only
APP={'pick_cool_then_place_in_recep':('cool','fridge'),'pick_heat_then_place_in_recep':('heat','microwave'),'pick_clean_then_place_in_recep':('clean','sinkbasin')}
def parse(t):
    typ,obj,_,rec,scene=t.split('/')[1].split('-'); return typ,obj.lower(),rec.lower(),scene
def legal(s): return s['o']!='Nothing happens.'
def mode(t,ep):
    typ,obj,rec,scene=parse(t); st=ep['steps']
    L=[(i,(s['a'] or '').lower(),s['o'].lower()) for i,s in enumerate(st) if legal(s)]
    if ep['success']: return 'success'
    seen = bool(re.search(rf'\b{obj} \d+',ep['init'].lower())) or any(re.search(rf'\b{obj} \d+',o) for _,_,o in L)
    took=[i for i,a,o in L if re.match(rf'take {obj} \d+ from',a)]
    pl=[i for i,a,o in L if re.match(rf'(move|put) {obj} \d+ (to|in|on|in/on) {rec} \d+',a)]
    if not seen: return 'A_never_seen'
    if not took: return 'B_seen_not_taken'
    if typ in APP:
        v,app=APP[typ]; tr=[i for i,a,o in L if re.match(rf'{v} {obj} \d+ with',a)]
        if not tr: return 'D_placed_without_transform' if pl else 'C_took_no_transform'
        return 'E_transformed_not_placed'
    if typ=='look_at_obj_in_light': return 'C_took_no_lamp'
    if typ=='pick_two_obj_and_place': return 'F_two_incomplete'
    return 'E_took_not_placed'
fm=json.load(open('/home/xueqi/hq/projects/ICML27_agent_distillation/reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
T={t for t,e in fm['teacher'].items() if e['success']}
def K(c): return c.split('|')[1] if '|' in c else None
cnt=collections.defaultdict(collections.Counter); tot=collections.Counter()
for t in T:
    for c,ep in E[t].items():
        k=K(c)
        if k is None or c=='SAD|10': continue
        m=mode(t,ep); cnt[k][m]+=1; tot[k]+=1
print('teacher-ok tasks (114) x non-collapsed cells; share of episodes')
modes=sorted({m for k in cnt for m in cnt[k]})
print(f"{'mode':28}"+''.join(f'K{k:>6}' for k in ('10','20','30','100')))
for m in modes: print(f"{m:28}"+''.join(f"{cnt[k][m]/tot[k]:7.2f}" for k in ('10','20','30','100')))
print('episodes',dict(tot))
# misreading signature in thoughts for adjective phrasings
pat=re.compile(r"not (?:the|a|an) (hot|cool|cold|clean)\b|(?:already|is) (hot|cool|cold|clean)\b|(hot|cool|cold|clean) (\w+) (?:isn|is not|must be|might be|could be)",re.I)
sig=collections.defaultdict(lambda:[0,0])
for t in T:
    typ,obj,rec,_=parse(t); ph=PH[t]
    if ph not in ('hot X','put a cool X','put a clean X'): continue
    for c,ep in E[t].items():
        k=K(c)
        if k is None or c=='SAD|10' or ep['success']: continue
        txt=' '.join(s['raw'] for s in ep['steps'])
        hit=bool(re.search(rf"not (?:the|a|an) (?:hot|cool|cold|clean) {obj}|(?:hot|cool|cold|clean) {obj} (?:isn|is not|must|might|could|elsewhere)|already (?:hot|cool|cold|clean)",txt,re.I))
        sig[(ph,k)][0]+=hit; sig[(ph,k)][1]+=1
print('\nfailed episodes whose Thoughts treat the adjective as an object property to be found ("not the hot plate", "already cool"):')
for (ph,k),(h,n) in sorted(sig.items()): print(f"  {ph:14} K{k:4} {h}/{n}")
