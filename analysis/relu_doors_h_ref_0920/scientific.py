"""Frozen decision arithmetic of spec §5. No outcome-dependent thresholds."""
import math
import numpy as np
import pandas as pd
DELTA=0.0007606370276951454
THETA=0.49249997734999995
KEYS=('M_H','N_C','R_LAYER','D_HR')

def classify(k,d):
 d=np.sort(np.asarray(d,float))
 assert d.shape==(10,) and np.isfinite(d).all()
 m='H_ALONE_RESCUES' if k>=9 else 'H_ALONE_FAILS'
 n='C_NEEDED' if d[8]<-DELTA else 'C_REDUNDANT' if k>=9 and d[1]>-DELTA else 'TIE'
 return m,n

def valid(frame):
 required=['seed','task','cond','online_acc','memo_acc','dead_frac_l1','gate_zero_frac_l2']
 if not all(c in frame for c in required) or len(frame)!=500:return False
 if not (frame.cond=='raw').all() or frame.duplicated(['seed','task']).any():return False
 if set(map(tuple,frame[['seed','task']].to_numpy()))!={(s,t) for s in range(10) for t in range(1,51)}:return False
 return bool(np.isfinite(frame[['online_acc','memo_acc','dead_frac_l1','gate_zero_frac_l2']].to_numpy(float)).all())

def verdict(frames,qualified=True,hashes_ok=True):
 if not qualified or not hashes_ok or set(frames)!={'ref','H','C','CH'} or not all(valid(d) for d in frames.values()):
  return {'labels':dict.fromkeys(KEYS,'INAPPLICABLE'),'status':'INAPPLICABLE'}
 windows={a:d[d.task.between(31,50)].groupby('seed').online_acc.mean().sort_index().to_numpy() for a,d in frames.items()}
 k={a:int(np.sum(w>=.5)) for a,w in windows.items()}
 result={'windows':{a:w.tolist() for a,w in windows.items()},'medians':{a:float(np.median(w)) for a,w in windows.items()},'K':k}
 if k['CH']<9 or k['ref']>1 or k['C']>1:
  return {**result,'status':'NOT_REPRODUCED','labels':dict.fromkeys(KEYS,'NOT_REPRODUCED')}
 dhc=windows['H']-windows['CH'];dhr=windows['H']-windows['ref']
 m,n=classify(k['H'],dhc)
 t1={a:d[d.task==1].sort_values('seed').dead_frac_l1.to_numpy() for a,d in frames.items()}
 gate={a:d[d.task==50].sort_values('seed').gate_zero_frac_l2.to_numpy() for a,d in frames.items()}
 layer_checks={'H_l1_dies':int(np.sum(t1['H']>THETA))>=9,'C_l1_saved':int(np.sum(t1['C']<THETA))>=9,
 'C_l2_dies':int(np.sum(gate['C']==1))>=9,'CH_l2_not_all_zero':int(np.sum(gate['CH']<1))>=9}
 layer='LAYER_CORRESPONDENCE' if all(layer_checks.values()) else 'LAYER_CORRESPONDENCE_NOT_SHOWN'
 pos=int(np.sum(dhr>0));neg=int(np.sum(dhr<0));nn=pos+neg
 p=min(1.,2*sum(math.comb(nn,j) for j in range(min(pos,neg)+1))/2**nn) if nn else None
 dlabel=('H_HELPS' if pos>neg else 'H_HURTS') if nn and p<.05 and min(pos,neg)<=1 else 'TIE'
 return {**result,'status':'COMPLETE','labels':dict(zip(KEYS,[m,n,layer,dlabel])),
 'details_H':'RESCUED' if k['H']>=9 else 'COLLAPSED' if k['H']<=1 else 'SPLIT',
 'd_HC':dhc.tolist(),'d_HR':dhr.tolist(),'median_d_HC':float(np.median(dhc)),
 'median_d_HR':float(np.median(dhr)),'I_HC':np.sort(dhc)[[1,8]].tolist(),
 'delta_C':DELTA,'theta1':THETA,'layer_conditions':layer_checks,
 'R1_H':'L1_DIES' if layer_checks['H_l1_dies'] else 'L1_SAVED' if np.sum(t1['H']<THETA)>=9 else 'TIE',
 'sign_HR':{'pos':pos,'neg':neg,'n':nn,'p':p}}
