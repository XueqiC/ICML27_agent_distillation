import json,re,collections,sys
exec(open('hk_an3.py').read().split("fm=json.load")[0])
want_typ=sys.argv[1]; want_mode=sys.argv[2]; Ks=sys.argv[3].split(','); lim=int(sys.argv[4])
fm=json.load(open('/home/xueqi/hq/projects/ICML27_agent_distillation/reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
T={t for t,e in fm['teacher'].items() if e['success']}
n=0; pat=collections.Counter()
for t in sorted(T):
    typ,obj,rec,_=parse(t)
    if typ!=want_typ: continue
    for c,ep in sorted(E[t].items()):
        k=c.split('|')[1] if '|' in c else c
        if k not in Ks or c=='SAD|10' or mode(t,ep)!=want_mode: continue
        acts=[(s['a'] or '').lower() for s in ep['steps']]
        i=next(i for i,a in enumerate(acts) if re.match(rf'take {obj} \d+ from',a) and ep['steps'][i]['o']!='Nothing happens.')
        post=[re.sub(r'\d+','N',a) for a in acts[i+1:]]
        verbs=collections.Counter(a.split()[0] for a in post if a)
        pat[tuple(sorted(verbs))]+=1
        if n<lim:
            print('==',t.split('/')[1],c,'| after take at',i)
            for j,s in enumerate(ep['steps'][i:i+14]):
                print(f"   {i+j:2d} {s['a']!s:40} | {s['o'][:80]} || {s['raw'].splitlines()[0][:90]}")
        n+=1
print('total',n)
