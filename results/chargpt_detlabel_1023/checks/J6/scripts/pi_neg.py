import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'/home/kubo/project/Nakatsuka/lop_detlabel/analysis/chargpt_detlabel_1023/j6')
import measure as M
C=M.C
d=Path(sys.argv[1]); (d/'seed0').mkdir(parents=True,exist_ok=True)
pi=C.ChainM(0,1,0.01).pi.copy(); pi[0,0]=np.nextafter(pi[0,0],1)
np.savez(d/'seed0/pi.npz',tasks=np.array([1]),pi=pi[None])
M.RUN_DIR=d
try:
    M.Inputs().data(0,1,check_pi=True); print('NOT STOPPED')
except AssertionError as e: print('stopped:', str(e)[:60])
