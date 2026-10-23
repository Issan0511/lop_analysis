#!/usr/bin/env python3
"""Read-only source-checkpoint validation; no optimization or new task outcome."""
import hashlib,json
from pathlib import Path
import torch,numpy as np
import elu_environment_0913 as base
import elu_sunk_rescue_0913 as old
from elu_response_engine_0913 import Engine,MODELS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/elu_response_anchor_0913"
SOURCE=Path("/home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
    old.cfg()
    ck=SOURCE/"prefix/checkpoint_20.pt";sp=SOURCE/"RL_t20_l2/selections.json"
    assert sha(ck)=="4b62bad9ad48a8940d81e9d709c4d987abc74eb2dec7d0b563cbad19f4d2f0c9"
    assert sha(sp)=="4e561bae77f004679f2c81ac6b8b13e832a637c802371adc1213dbc34fc119f7"
    q=torch.load(ck,map_location="cpu",weights_only=False);sels=json.loads(sp.read_text())["seeds"];x,y,sub=old.load_data();e=Engine()
    for j,m in enumerate(MODELS):
        s=m["seed"];idx=old.mid(s,"RL")
        assert np.array_equal(sub[s],q["subset"][s])
        for group,key in ((e.p,"parameters"),(e.m,"adam_m"),(e.v,"adam_v")):
            for dest,src in zip(group,q[key]):dest[j].copy_(src[idx])
        e.cx[j].copy_(torch.as_tensor(x[sub[s]],device="cuda"))
    e.t.copy_(q["t"])
    initial=base.forward(e.p,e.cx,e.elu);d=torch.zeros_like(e.delta)
    for j,m in enumerate(MODELS):
        s=sels[m["seed"]];assert len(s["target20"])==20
        d[j,s["target20"]]=torch.as_tensor(s["deltas20"],device="cuda")
    e.set_anchors(initial[2].clone(),d)
    result=dict(status="PASS",checkpoint_sha256=sha(ck),selection_sha256=sha(sp),engine_sha256=sha(ROOT/"src/elu_response_engine_0913.py"),code_sha256=sha(Path(__file__)),checks={})
    for name,ix in (("full",None),("first16",torch.arange(16,device="cuda").expand(18,-1))):
        out=e.forward_full() if ix is None else e.forward_indexed(ix)
        for s in range(3):
            for aa,bb in ((0,1),(2,3),(4,5)):
                for pos,label in ((3,"activation"),(4,"logit")):
                    a=out[pos][s*6+aa];b=out[pos][s*6+bb]
                    err=float((a-b).abs().max());ok=bool(torch.allclose(a,b,atol=2e-4,rtol=1e-6))
                    result["checks"][f"{name}_s{s}_{aa}_{bb}_{label}"]=dict(maxabs=err,allclose=ok)
                    assert ok,(name,s,aa,bb,label,err)
        for j,m in enumerate(MODELS):
            zero=e.delta[j].eq(0);arg=out[2][j]
            assert torch.equal(out[3][j,:,zero],base.activ(arg,torch.tensor(True,device="cuda"))[:,zero])
    result["full_maxabs"]=max(v["maxabs"] for k,v in result["checks"].items() if k.startswith("full"))
    result["batch_maxabs"]=max(v["maxabs"] for k,v in result["checks"].items() if k.startswith("first16"))
    result["unselected_exact"]=True;result["optimizer_updates"]=0
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/"initial_independent_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))
if __name__=="__main__":main()
