import sys, time, json, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0,'/home/kubo/project/Nakatsuka/lop_detlabel/analysis/chargpt_detlabel_1023/j6')
import measure as M
C=M.C; torch=M.torch
SP=Path(sys.argv[1]); res={}
# 参照: π を ChainM で作る道（RUN_DIR は A2 の置き場。pi.npz は無い）
ref=M.Inputs(); assert M.read_pi(0,1) is None
t0=time.monotonic(); refd={t:ref.data(0,t) for t in [1,100]}; res['chainm_seconds_alpha0.01_2chains']=time.monotonic()-t0
for fmt in ['a','b']:
    d=SP/f'fmt_{fmt}'; (d/'seed0').mkdir(parents=True,exist_ok=True)
    pis={t:C.ChainM(0,t,0.01) for t in [1,100]}
    if fmt=='a':
        np.savez(d/'seed0/pi.npz',**{f't{t}':c.pi.reshape(-1) for t,c in pis.items()},**{f't{t}_iters':np.int64(c.pi_iters) for t,c in pis.items()},
                 **{f't{t}_rule_sha':np.str_(hashlib.sha256(c.P.tobytes()).hexdigest()) for t,c in pis.items()})
    else:
        np.savez(d/'seed0/pi.npz',tasks=np.array([1,100]),pi=np.stack([pis[t].pi for t in [1,100]]),pi_iters=np.array([pis[t].pi_iters for t in [1,100]]),
                 rule_sha=np.array([hashlib.sha256(pis[t].P.tobytes()).hexdigest() for t in [1,100]]),pi_sha=np.array([hashlib.sha256(pis[t].pi.tobytes()).hexdigest() for t in [1,100]]))
    M.RUN_DIR=d
    inp=M.Inputs(); t0=time.monotonic()
    got={t:inp.data(0,t,check_pi=True) for t in [1,100]}
    res[f'fmt_{fmt}']=dict(seconds_2chains_with_check=time.monotonic()-t0,pi_checks=inp.pi_checks,pi_used={str(k):v for k,v in inp.pi_used.items()},
        probe_equal=all(torch.equal(got[t][0],refd[t][0]) and torch.equal(got[t][1],refd[t][1]) and torch.equal(got[t][2],refd[t][2]) and got[t][3]==refd[t][3] for t in [1,100]))
    inp2=M.Inputs(); t0=time.monotonic(); [inp2.data(0,t) for t in [1,100]]; res[f'fmt_{fmt}']['seconds_2chains_no_check']=time.monotonic()-t0
# 腕の α の効き: α=1e-4 の 1 連鎖（π を反復で作る道）の時間

print(json.dumps(res,indent=1,default=str))
