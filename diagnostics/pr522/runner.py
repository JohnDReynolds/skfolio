"""Paired numerical diagnostics for upstream PR522; no production source edits."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,platform,importlib.metadata,time

def child():
    import numpy as np,scipy,pytest
    import skfolio.prior._entropy_pooling as ep
    variant=sys.argv[2];out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
    if variant in ("expm1", "accepted-strict", "expm1-strict", "normalized", "normalized-contract"):
        files={"expm1":"candidate_dual.py","accepted-strict":"accepted_strict_dual.py","expm1-strict":"expm1_strict_dual.py","normalized":"normalized_dual.py","normalized-contract":"normalized_dual.py"}
        source_file=Path(__file__).parent/files[variant];code=source_file.read_text();namespace=ep.__dict__.copy();exec(compile(code,str(source_file),"exec"),namespace);ep.EntropyPooling._solve_dual=namespace["_solve_dual"]
    trace=[]
    original_minimize=ep.sco.minimize
    def traced(*args,**kwargs):
        sol=original_minimize(*args,**kwargs)
        if kwargs.get("method")=="TNC":
            trace.append({"test":os.environ.get("PYTEST_CURRENT_TEST"),"options":kwargs.get("options"),"dimension":len(sol.x),"success":bool(sol.success),"message":str(sol.message),"fun":float(sol.fun),"gradient":sol.jac.tolist(),"multipliers":sol.x.tolist()})
        return sol
    ep.sco.minimize=traced
    import threadpoolctl
    identity={"variant":variant,"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"packages":{x.metadata["Name"]:x.version for x in importlib.metadata.distributions()},"blas":threadpoolctl.threadpool_info(),"source":ep.__file__,"experimental_source_sha256":hashlib.sha256(code.encode()).hexdigest() if variant in ("expm1","accepted-strict","expm1-strict","normalized","normalized-contract") else None,"test_sha256":hashlib.sha256(Path("tests/test_prior/test_entropy_pooling.py").read_bytes()).hexdigest()}
    (out/"identity.json").write_text(json.dumps(identity,indent=2))
    target="tests/test_prior/test_normalized_contract.py" if variant=="normalized-contract" else "tests/test_prior/test_entropy_pooling.py"
    code=pytest.main(["-q","-n=0","-p","no:cacheprovider","--junitxml="+str(out/"tests.xml"),target])
    (out/"tnc-trace.json").write_text(json.dumps(trace,indent=2,default=str))
    (out/"result.json").write_text(json.dumps({"pytest_returncode":int(code)}))
    raise SystemExit(code)

def parent():
    root=Path.cwd();source_test=root/"pr/tests/test_prior/test_entropy_pooling.py";data=source_test.read_bytes();results=[]
    for variant in ["accepted","pr","normalized","normalized-contract"]:
        checkout=root/("accepted" if variant!="pr" else "pr")
        target=checkout/"tests/test_prior/test_entropy_pooling.py";target.write_bytes(data)
        (checkout/"tests/test_prior/test_normalized_contract.py").write_bytes((Path(__file__).parent/"test_normalized_contract.py").read_bytes())
        out=root/"results"/variant;out.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env["PYTHONPATH"]=str(checkout/"src");env["PYTHONDONTWRITEBYTECODE"]="1"
        with (out/"pytest.log").open("w") as log:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--child",variant,str(out)],cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append({"variant":variant,"returncode":proc.returncode});print(variant,proc.returncode,flush=True)
        if proc.returncode not in (0,1):raise RuntimeError("Diagnostic setup or collection failure: "+str(out/"pytest.log"))
    (root/"results/paired-summary.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":
    child() if sys.argv[1:2]==["--child"] else parent()
