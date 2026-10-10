"""Standalone quick integration test; never imports the repository package."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_v015_quick():
    here=Path(__file__).resolve().parents[1]
    output=here/'quick'/'test_results.json'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(here/'.mplconfig'),MPLBACKEND='Agg',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    process=subprocess.run([sys.executable,str(here/'check_v015.py'),'--quick','--output',str(output)],
        env=env,cwd=here.parents[1],capture_output=True,text=True,timeout=1800)
    assert output.exists(),process.stdout+process.stderr
    report=json.loads(output.read_text())
    assert {r['id'] for r in report['groups']}=={*(f'V{i}' for i in range(1,10)),*(f'E{i}' for i in range(13))}
    failures=[r['id'] for r in report['groups'] if r['status']=='FAIL']
    assert not failures,json.dumps(report['counterexamples'],indent=2)
    assert process.returncode==0,process.stdout+process.stderr
    assert not report['counterexamples']
    groups={r['id']:r for r in report['groups']}
    for name,number in (('V8',8),('V9',9),('E10',110),('E11',111),('E12',112)):
        assert groups[name]['seed']==[150150,number]
    assert len(groups['E10']['data']['cells'])==11*6*5*3
    assert groups['E10']['data']['C23_prediction']['gap']>0.0769
    assert groups['E12']['data']['frozen_sha256_before']==groups['E12']['data']['frozen_sha256_after']
    assert all(not r['wrong_sign'] for r in groups['E12']['data']['rows'] if r['margin'])
    for name in ('V9','E10','E11','E12'):
        for suffix in ('png','svg'):
            assert (here/'quick'/'figs'/f'{name}.{suffix}').is_file()
