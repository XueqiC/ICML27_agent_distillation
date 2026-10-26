import json,re,collections
E=json.load(open('hk_episodes.json')); S=json.load(open('hk_sets.json'))
APP={'pick_cool_then_place_in_recep':('cool','fridge'),'pick_heat_then_place_in_recep':('heat','microwave'),'pick_clean_then_place_in_recep':('clean','sinkbasin')}
def parse(t):
    typ,obj,_,rec,scene=t.split('/')[1].split('-'); return typ,obj.lower(),rec.lower(),scene
def legal(s): return s['o']!='Nothing happens.'
def mode(t,ep):
    typ,obj,rec,scene=parse(t)
    st=ep['steps']; L=[(i,(s['a'] or '').lower(),s['o'].lower()) for i,s in enumerate(st) if legal(s)]
    seen = bool(re.search(rf'\b{obj} \d+',ep['init'].lower())) or any(re.search(rf'\b{obj} \d+',o) for _,_,o in L)
    took=[i for i,a,o in L if re.match(rf'take {obj} \d+ from',a)]
    if ep['success']: return 'success'
    if not seen: return 'never_seen'
    if not took: return 'seen_not_taken'
    if typ in APP:
        v,app=APP[typ]
        tr=[i for i,a,o in L if re.match(rf'{v} {obj} \d+ with',a)]
        pl=[i for i,a,o in L if re.match(rf'(move|put) {obj} \d+ (to|in|on|in/on) {rec} \d+',a)]
        bad_tr=[i for i,s in enumerate(st) if not legal(s) and re.match(rf'{v} {obj} \d+ with',(s['a'] or '').lower())]
        if not tr:
            if pl: return 'placed_without_'+v
            return f'took_no_{v}'+('(tried_illegal)' if bad_tr else '')
        return 'transformed_not_placed'
    if typ=='look_at_obj_in_light':
        return 'took_no_lamp_use'
    if typ=='pick_two_obj_and_place':
        pl=[i for i,a,o in L if re.match(rf'(move|put) {obj} \d+ (to|in|on|in/on) {rec} \d+',a)]
        return f'two:placed{len(pl)}'
    pl=[i for i,a,o in L if re.match(rf'(move|put) {obj} \d+ (to|in|on|in/on) {rec} \d+',a)]
    return 'took_not_placed' if not pl else 'placed_other'
def Kof(c): return c.split('|')[1] if '|' in c else c
res={}
for name in ('R','P'):
    for t in S[name]:
        d=collections.defaultdict(collections.Counter); steps=collections.defaultdict(list)
        for c,ep in E[t].items():
            if c.startswith('SAD|10') or c.startswith('SAD|30'): continue   # collapsed cells (8/140, 58/140)
            k=Kof(c); m=mode(t,ep); d[k][m]+=1
            if ep['success']: steps[k].append(len(ep['steps']))
        res[t]={k:dict(v) for k,v in d.items()}
        print(name, t.split('/')[1])
        for k in ('10','20','30','100','teacher','base'):
            if k in d: print(f"   K{k:8} {dict(d[k])}  success steps {sorted(steps[k])}")
json.dump(res,open('hk_modes.json','w'))
