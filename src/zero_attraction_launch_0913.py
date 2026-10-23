"""Bounded launcher for the registered learning grid."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys,time,resource,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/zero_attraction_learning_0913.yaml"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--parallel",type=int,default=6);args=ap.parse_args()
 cfg=json.loads(CONFIG.read_text())
 phase="preflight" if args.preflight else "main"
 out=ROOT/("results/_preflight_zero_attraction_learning_0913" if args.preflight else cfg["output"]["dir"])
 out.mkdir(parents=True,exist_ok=True)
 statefile=out/"launch_status.json";names=[a["name"] for a in cfg["arms"]]
 env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONPATH=str(ROOT))
 def job(name):
  logfile=out/f"{name}.stdout.log"
  cmd=["/usr/bin/time","-v",sys.executable,"-m","src.edge_law_0905","--config",str(CONFIG),"--arm",name,"--outdir",str(out)]
  if args.preflight:cmd+=["--steps","30000"]
  started=time.time()
  with logfile.open("w") as f:ret=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  statuspath=out/"arm_status"/f"{name}_done.json"
  st=json.loads(statuspath.read_text()) if statuspath.exists() else {}
  peak=None
  for line in logfile.read_text().splitlines():
   if "Maximum resident set size" in line:peak=int(line.rsplit(":",1)[1].strip())
  return {"arm":name,"returncode":ret,"status":st.get("status","NO_STATUS"),"wall_seconds":time.time()-started,"peak_rss_kib":peak}
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 state={"phase":phase,"started":time.time(),"git_head":head,"parallel":args.parallel,"jobs_total":len(names),"completed":[],"python":sys.version,"platform":platform.platform()}
 statefile.write_text(json.dumps(state,indent=2))
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
  fs=[pool.submit(job,n) for n in names]
  for f in concurrent.futures.as_completed(fs):
   item=f.result();state["completed"].append(item)
   statefile.write_text(json.dumps(state,indent=2));print(json.dumps(item),flush=True)
 state["finished"]=time.time();state["all_complete"]=all(x["status"]=="COMPLETE" and x["returncode"]==0 for x in state["completed"])
 statefile.write_text(json.dumps(state,indent=2))
 print("ALL_COMPLETE",state["all_complete"],flush=True)
 raise SystemExit(0 if state["all_complete"] else 1)
if __name__=="__main__":main()
