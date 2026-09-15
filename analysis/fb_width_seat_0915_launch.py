"""Capture immutable launch provenance and run six vectorized arms in parallel."""
import hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy,torch
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/fb_width_seat_0915"
ARMS=["LRoff0_1216","LRwf21_1216","FB21LRoff0_1216","FB21LRwf21_1216","LRwi21_1216","FB21LRwi21_1216"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"run_logs").mkdir(exist_ok=True)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); dirty=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)
    src=ROOT/"src/fb_width_seat_0915.py"; spec=ROOT/"specs/spec_fb_width_seat_0915.md"; cfg=ROOT/"configs/fb_width_seat_0915.yaml"
    prov=dict(experiment="fb_width_seat_0915",started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),git_hash=head,git_dirty=dirty,source_sha256=sha(src),spec_sha256=sha(spec),config_sha256=sha(cfg),python=sys.version,numpy=numpy.__version__,torch=torch.__version__,platform=platform.platform(),cpu=platform.processor(),g1_reference="/home/issan/Projects/obsidian-research-data/act_offset_review_0908/full/logs",arms=ARMS,seeds=list(range(10)),parallel_unit="arm")
    (OUT/"launch_provenance.json").write_text(json.dumps(prov,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONPATH=str(ROOT)); ps=[]
    py="/home/issan/Projects/claude/proj_004_drift/.venv/bin/python"
    for a in ARMS:
        f=open(OUT/"run_logs"/f"{a}.log","w"); ps.append((a,subprocess.Popen([py,"-m","src.fb_width_seat_0915","--arm",a],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f))
    status={}
    for a,p,f in ps: status[a]=p.wait(); f.close()
    (OUT/"launch_status.json").write_text(json.dumps(status,indent=2))
    if any(status.values()): raise SystemExit(1)
if __name__=="__main__":main()
