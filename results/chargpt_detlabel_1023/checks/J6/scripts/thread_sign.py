import sys, numpy as np
sys.path.insert(0,'/home/kubo/project/Nakatsuka/lop_detlabel/analysis/chargpt_detlabel_1023/j6')
import components as CO
M=CO.M; torch=M.torch
z=np.load(M.OUT/'raw_components/A2_seed0.npz'); r1=np.load(M.OUT/'raw/A2_seed0.npz')
inp=M.Inputs()
for t in [9,50]:
    al=z[f't{t}_alive']; base=z[f't{t}_loaderror_other']; sc=np.abs(base[al]).max()
    tiny=np.flatnonzero(al&(np.abs(base)<=1e-12*sc)&r1[f't{t}_gradient_active'])
    for n in [1,2,4,8,16]:
        torch.set_num_threads(n)
        row,acc=CO.measure(inp,'A2',0,t)
        out=[]
        for e in ['original','loaderror']:
            v=acc[f'{e}_other'].numpy()
            out.append(f"{e}: s_other={row[f's_{e}_other']:.6f} tiny_units_pos={int((v[tiny]>0).sum())}/{len(tiny)} max|tiny|/scale={np.abs(v[tiny]).max()/np.abs(v[al]).max():.2e}")
        print('task',t,'threads',n,' | '.join(out),flush=True)
