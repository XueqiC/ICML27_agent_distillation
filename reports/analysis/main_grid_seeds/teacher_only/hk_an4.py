import json,re,collections,sys
exec(open('hk_an3.py').read().split("fm=json.load")[0])
G=json.load(open('dev_goals.json'))
fm=json.load(open('/home/xueqi/hq/projects/ICML27_agent_distillation/reports/analysis/main_grid_seeds/fm_e4b_seed0.json'))
T={t for t,e in fm['teacher'].items() if e['success']}
want=sys.argv[1]
for t in sorted(T):
    if PH[t]!=want: continue
    d=collections.defaultdict(collections.Counter)
    for c,ep in E[t].items():
        k=c.split('|')[1] if '|' in c else c
        if c=='SAD|10': continue
        d[k][mode(t,ep)]+=1
    print('==',t.split('/')[1],'|',G[t])
    for k in ('10','20','30','100','base'):
        if k in d: print(f"   K{k:5} succ {d[k]['success']}/{sum(d[k].values())}  {dict((m,n) for m,n in d[k].items() if m!='success')}")
