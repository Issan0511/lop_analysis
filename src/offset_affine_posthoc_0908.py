
from pathlib import Path
import numpy as np,torch,csv,json
p=Path('/home/issan/Projects/claude/collective_kick_0908/results/offset_compensation_0908')
with np.load(p/'trajectories.npz') as f:d={k:f[k] for k in f.files}
o=torch.load(p/'oracle_states.pt',weights_only=False)
rows=[]
for i in range(10):
 x=d['x'][:,i];A=np.column_stack([np.ones(32),x])
 W=d['W'][0,0,i];b=d['b'][0,0,i];v=d['v'][0,0,i];z=x@W.T+b;h=np.where(z>0,z,.1*z)-2
 U=d['group_U'][i];L=d['group_L'][i]
 HU=np.column_stack([np.ones(32),h[:,U]])
 projected=A-HU@np.linalg.lstsq(HU,A,rcond=1e-12)[0]
 residual=o[str(i)+'_oracle_KUc']['residual'][:,0]
 gb=.2*v*residual.mean();gw=.2*v[:,None]*(residual[:,None]*x).mean(0)[None]
 rows.append({'seed':i,'U_plus_constant_rank':int(np.linalg.matrix_rank(HU,tol=1e-10)),
   'affine_rank':int(np.linalg.matrix_rank(A,tol=1e-10)),
   'affine_span_projection_relative_error':float(np.linalg.norm(projected)/np.linalg.norm(A)),
   'oracle_KUc_leak_full_hidden_gradient_max_abs':float(max(abs(gb[L]).max(),abs(gw[L]).max()))})
with (p/'posthoc_affine_span.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(rows)
