"""Verify the comparison-only revision against accepted source on one native runner."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,shutil
root=Path.cwd();source=root/'accepted';out=root/'results';out.mkdir();env=os.environ.copy();env['PYTHONPATH']=str(source/'src');env['PYTHONDONTWRITEBYTECODE']='1';target=source/'tests/test_model_selection/test_online/test_validation.py';results=[]
for variant in ['accepted','comparison']:
    if variant=='comparison':shutil.copy2(root/'published/tests/test_model_selection/test_online/test_validation.py',target)
    command=[sys.executable,'-m','pytest','-q','-n=0','-p','no:cacheprovider','tests/test_model_selection/test_online/test_validation.py::TestOnlinePredictNamedEstimators::test_matches_individual_runs','--junitxml='+str(out/(variant+'.xml'))]
    with (out/(variant+'.log')).open('w') as log:p=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    results.append({'variant':variant,'returncode':p.returncode,'test_sha256':hashlib.sha256(target.read_bytes()).hexdigest()});print(variant,p.returncode,flush=True)
    if p.returncode not in (0,1):raise RuntimeError('Setup/collection failed')
(out/'results.json').write_text(json.dumps(results,indent=2))
raise SystemExit(results[-1]['returncode'])
