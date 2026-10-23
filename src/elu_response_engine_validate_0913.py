#!/usr/bin/env python3
"""Synthetic QA for the response-anchor engine; no experiment checkpoints."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
import elu_environment_0913 as base
import elu_response_engine_0913 as mod
OUT=Path(__file__).resolve().parents[1]/'results/elu_response_anchor_0913'

def ma(a,b):return float((a-b).detach().abs().max())

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 c={};gen=torch.Generator().manual_seed(913131)
 # Float64 true-gradient and finite-difference check across the four algebraic cells.
 M=4;models=[dict(seed=0,env='RL',act='ELU1',iv=b,branch=b) for b in 'ABCD']
 p=[]
 for x in base.initial(0):p.append(torch.stack([x.double() for _ in range(M)]).requires_grad_())
 x=torch.rand(M,7,784,generator=gen,dtype=torch.float64);yi=torch.randint(10,(M,7),generator=gen);y=torch.nn.functional.one_hot(yi,10).double()
 z20=base.forward(p,x,torch.ones(M,1,1,dtype=torch.bool))[2].detach();delta=torch.zeros(M,100,dtype=torch.float64);delta[:,3:11]=torch.rand(M,8,generator=gen,dtype=torch.float64)*2
 F,K,_=mod._fk(models,'cpu',torch.float64);a0=base.activ(z20,torch.ones_like(F));a1=base.activ(z20+delta[:,None,:],torch.ones_like(F));got,_,_=mod.anchored_gradients(p,x,y,a0,a1,delta,F,K)
 logits=mod.anchored_forward(p,x,a0,a1,delta,F,K)[-1];loss=(logits.logsumexp(-1)-(logits*y).sum(-1)).mean(-1).sum()
 ref=torch.autograd.grad(loss,p);c['float64_gradient_maxabs']=max(ma(a,b) for a,b in zip(got,ref));c['float64_gradient_maxrel']=max(float(((a-b).abs()/b.abs().clamp_min(1e-12)).detach().max()) for a,b in zip(got,ref))
 direction=[torch.randn(q.shape,generator=gen,dtype=torch.float64) for q in p];analytic=sum((g*d).sum() for g,d in zip(got,direction));eps=1e-6
 def objective(sign):
  pp=[q.detach()+sign*eps*d for q,d in zip(p,direction)];logit=mod.anchored_forward(pp,x,a0,a1,delta,F,K)[-1]
  return (logit.logsumexp(-1)-(logit*y).sum(-1)).mean(-1).sum()
 numeric=(objective(1)-objective(-1))/(2*eps);c['finite_difference_abs']=float((numeric-analytic).detach().abs());c['finite_difference_rel']=float(((numeric-analytic).abs()/analytic.abs().clamp_min(1e-12)).detach())
 # CUDA algebra, optimizer-history freeze, and graph/eager checks with all six branches.
 dev='cuda';models=[dict(seed=0,env='RL',act='ELU1',iv=b,branch=b) for b in mod.BRANCHES];e=mod.Engine(models,dev)
 cx=torch.rand(1,base.N,784,generator=gen).expand(e.M,-1,-1).to(dev);labels=torch.randint(10,(1,base.N),generator=gen).expand(e.M,-1);cy=torch.nn.functional.one_hot(labels,10).float().to(dev);e.cx.copy_(cx);e.cy.copy_(cy)
 # Replicate identical checkpoint state across branches.
 for q in e.p+e.m+e.v:q.copy_(q[0:1].expand_as(q))
 z0=base.forward([q for q in e.p],e.cx,e.elu)[2];d=torch.zeros(e.M,100,device=dev);d[:,5:25]=torch.linspace(.2,2,20,device=dev)
 e.set_anchors(z0,d);ef=e.forward_full();bf=base.forward(e.p,e.cx,e.elu);out=ef[-1];orig=bf[-1]
 c['A_original_logits_maxabs']=ma(out[0],orig[0]);c['AF_original_logits_maxabs']=ma(out[4],orig[4])
 c['F0_pair_initial_logits_maxabs']=max(ma(out[0],out[1]),ma(out[4],out[5]));c['F1_pair_initial_logits_maxabs']=ma(out[2],out[3])
 unselected=d[0].eq(0);c['offdiagonal_unselected_activation_maxabs']=max(ma(ef[3][1,:,unselected],bf[3][1,:,unselected]),ma(ef[3][2,:,unselected],bf[3][2,:,unselected]))
 pp=[q.clone() for q in e.p];pp[3]=pp[3]+d;explicit=base.forward(pp,e.cx,e.elu)[-1]
 c['D_explicit_bias_shift_abs']=ma(out[3],explicit[3]);c['D_explicit_bias_shift_rel']=float(((out[3]-explicit[3]).abs()/explicit[3].abs().clamp_min(1e-6)).max());c['D_explicit_bias_shift_allclose']=bool(torch.allclose(out[3],explicit[3],atol=1e-5,rtol=1e-5))
 # A same-geometry gradients and Adam trajectory reproduce the original engine.
 one=[dict(seed=0,env='RL',act='ELU1',iv='A',branch='A')];ea=mod.Engine(one,dev);eb=base.Engine([dict(seed=0,env='RL',act='ELU1',iv='ref')],dev)
 xx=torch.rand(1,base.BATCH,784,generator=gen).to(dev);yy=torch.nn.functional.one_hot(torch.randint(10,(1,base.BATCH),generator=gen),10).float().to(dev);dd=torch.zeros(1,100,device=dev);dd[:,2:9]=1.3;ea.set_anchors(torch.zeros(1,base.N,100,device=dev),dd)
 ga,_,_=ea.gradients(xx,yy,ea.anchor_base[:,:base.BATCH],ea.anchor_shift[:,:base.BATCH]);gb,_,_=base.gradients(eb.p,xx,yy,eb.elu);c['A_base_gradient_maxabs']=max(ma(q,r) for q,r in zip(ga,gb))
 ea.t.fill_(17);eb.t.fill_(17)
 for _ in range(3):ea.step(xx,yy,ea.anchor_base[:,:base.BATCH],ea.anchor_shift[:,:base.BATCH]);eb.step(xx,yy)
 c['A_base_adam_p_maxabs']=max(ma(q,r) for q,r in zip(ea.p,eb.p));c['A_base_adam_m_maxabs']=max(ma(q,r) for q,r in zip(ea.m,eb.m));c['A_base_adam_v_maxabs']=max(ma(q,r) for q,r in zip(ea.v,eb.v))
 # Full-feature-freeze null: within-F K pairs have identical readout updates.
 nm=[dict(seed=0,env='RL',act='ELU1',iv=b,branch=b) for b in 'ABCD'];ne=mod.Engine(nm,dev)
 for q in ne.p+ne.m+ne.v:q.copy_(q[0:1].expand_as(q))
 nx=torch.rand(1,base.BATCH,784,generator=gen).expand(4,-1,-1).to(dev);ny=torch.nn.functional.one_hot(torch.randint(10,(1,base.BATCH),generator=gen).expand(4,-1),10).float().to(dev);nz=base.forward(ne.p,nx,ne.elu)[2];nd=torch.zeros(4,100,device=dev);nd[:,7:17]=.8
 nza=torch.zeros(4,base.N,100,device=dev);nza[:,:base.BATCH].copy_(nz);ne.set_anchors(nza,nd)
 for i in range(4):ne.freeze_masks[i].fill_(True);ne.frozen_reference[i].copy_(ne.p[i])
 ne.step(nx,ny,ne.anchor_base[:,:base.BATCH],ne.anchor_shift[:,:base.BATCH])
 c['full_feature_freeze_F0_readout_maxabs']=max(ma(q[0],q[1]) for q in ne.p[4:]+ne.m[4:]+ne.v[4:]);c['full_feature_freeze_F1_readout_maxabs']=max(ma(q[2],q[3]) for q in ne.p[4:]+ne.m[4:]+ne.v[4:])
 # Establish nonzero optimizer history, then reset anchors/references and test AF/BF actual freeze.
 ix=torch.randint(base.N,(e.M,base.BATCH),generator=gen).to(dev);xb=e.cx[e.mid,ix];yb=e.cy[e.mid,ix];ab=e.anchor_base[e.mid,ix];ash=e.anchor_shift[e.mid,ix]
 for _ in range(3):e.step(xb,yb,ab,ash)
 e.set_anchors(e.anchor_z2,e.delta);ab=e.anchor_base[e.mid,ix];ash=e.anchor_shift[e.mid,ix];before=[q.clone() for q in e.p];mb=[q.clone() for q in e.m];vb=[q.clone() for q in e.v];e.step(xb,yb,ab,ash)
 frozen=e.upstream_frozen;c['L1_frozen_parameter_maxabs']=max(ma(e.p[i][frozen],before[i][frozen]) for i in (0,1));c['L1_frozen_moment_change_min']=min(float((e.m[i][frozen]-mb[i][frozen]).abs().max()) for i in (0,1));c['L1_frozen_v_change_min']=min(float((e.v[i][frozen]-vb[i][frozen]).abs().max()) for i in (0,1));c['trainable_parameter_change']=max(float((e.p[i][~frozen]-before[i][~frozen]).abs().max()) for i in (0,1))
 # diagnose_step must expose true values without changing state.
 snap=e.snapshot();bp=[q.clone() for q in e.p];diag=e.diagnose_step(xb,yb,ab,ash);e.step(xb,yb,ab,ash);actual=[q-p for q,p in zip(e.p,bp)];c['diagnose_parameter_delta_vs_step_maxabs']=max(ma(q,p) for q,p in zip(diag['parameter_delta'],actual));e.restore(snap)
 oldpos=e.online_pos.clone();e.online_pos.fill_(base.STEPS);snap=e.snapshot();diag=e.diagnose_step(xb,yb,ab,ash);c['diagnose_restore_maxabs']=max(float(torch.logical_xor(a,b).any()) if a.dtype==torch.bool else ma(a,b) for a,b in zip(e.snapshot(),snap));c['diagnose_at_full_buffer']=bool(e.online_pos==base.STEPS);c['diagnose_gradient_arrays']=len(diag['gradients']);c['diagnose_parameter_delta_arrays']=len(diag['parameter_delta']);c['diagnose_decomposition_arrays']=min(len(diag[k]) for k in ('current_gradient_numerator','history_numerator','decomposition_error','denominator','predicted_update'));c['diagnose_decomposition_rounding_maxabs']=max(float(q.abs().max()) for q in diag['decomposition_error']);e.online_pos.copy_(oldpos)
 e.indices.copy_(torch.randint(base.N,e.indices.shape,generator=gen).to(dev));e.t.fill_(73);e.capture();s=e.snapshot();e.block();want=e.snapshot();e.restore(s);e.graph.replay();torch.cuda.synchronize();gotstate=e.snapshot()
 c['graph_eager_maxabs']=max(float(torch.logical_xor(a,b).any()) if a.dtype==torch.bool else ma(a,b) for a,b in zip(gotstate,want));c['graph_counter_increment']=float(e.t-s[18]);c['graph_anchor_unchanged']=torch.equal(e.anchor_z2,s[21]) and torch.equal(e.anchor_base,s[22]) and torch.equal(e.anchor_shift,s[23]) and torch.equal(e.delta,s[24]);c['graph_online_entries']=int(e.online_pos-s[27]);c['graph_online_finite']=bool(torch.isfinite(e.online_ce[:,int(s[27]):int(e.online_pos)]).all() and torch.isfinite(e.online_acc[:,int(s[27]):int(e.online_pos)]).all())
 assert c['float64_gradient_maxabs']<1e-7 and c['float64_gradient_maxrel']<1e-5,c
 assert c['finite_difference_rel']<1e-4,c
 assert c['A_original_logits_maxabs']==0 and c['AF_original_logits_maxabs']==0,c
 assert c['F0_pair_initial_logits_maxabs']==0 and c['F1_pair_initial_logits_maxabs']==0,c
 assert c['offdiagonal_unselected_activation_maxabs']==0,c
 assert c['D_explicit_bias_shift_allclose'],c
 assert c['A_base_gradient_maxabs']<=1e-6 and c['A_base_adam_p_maxabs']<=1e-6 and c['A_base_adam_m_maxabs']<=1e-6 and c['A_base_adam_v_maxabs']<=1e-6,c
 assert c['full_feature_freeze_F0_readout_maxabs']==0 and c['full_feature_freeze_F1_readout_maxabs']==0,c
 assert c['L1_frozen_parameter_maxabs']==0 and c['L1_frozen_moment_change_min']>0 and c['L1_frozen_v_change_min']>0 and c['trainable_parameter_change']>0,c
 assert c['diagnose_parameter_delta_vs_step_maxabs']<=1e-6 and c['diagnose_restore_maxabs']==0 and c['diagnose_at_full_buffer'] and c['diagnose_gradient_arrays']==6 and c['diagnose_parameter_delta_arrays']==6 and c['diagnose_decomposition_arrays']==6,c
 assert c['graph_eager_maxabs']<5e-5 and c['graph_counter_increment']==base.BLOCK and c['graph_anchor_unchanged'] and c['graph_online_entries']==base.BLOCK and c['graph_online_finite'],c
 c['rounding_note']='D uses z2+delta inside activation; explicit bias addition forms the same sum through a different float32 operation order, so abs/rel 1e-5 is the preregistered numerical tolerance.'
 c['engine_sha256']=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest();c['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();c['status']='PASS'
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'engine_validation.json').write_text(json.dumps(c,indent=2)+'\n')
 print(json.dumps(c,indent=2))

if __name__=='__main__':main()
