"""S2 registered checks. Every named mutant must fail the same positive predicate."""
from common import *
import common, copy, math, shutil, resource, time, argparse
import scientific as S
import launch as L
DEV=torch.device('cuda')
U=2.**-24
G=lambda n:n*U/(1-n*U)
PHI='return a - self.m[layer][:, None, :]'
UPDATE='mi.mul_(1.0 - self.beta).add_(torch.clamp(z, min=0.0).mean(1), alpha=self.beta)'

def attempt(fn):
 try:return bool(fn()),None
 except (RuntimeError,ValueError,KeyError,IndexError,SystemExit) as e:return False,str(e)

def s_off(prov):
 old=frozen('e7069acf585ece6914e2d28b3b14def4e94aff11','rlcifar_mlp_battle_0918')
 a=run(old,'R','off_parent',snapshots=False);b=run(E,'ref','off_current',snapshots=False)
 yes,n=equal_rows(df(a),df(b));yes &= equal_state(state(a),state(b),act=False)
 mods={'C_on':(E,'C'),
 'labels_shift':(mutated('off_labels',[('lab = {s: RC.task_labels(g_lab[s]) for s in useeds}','lab = {s: (RC.task_labels(g_lab[s]) + 1) % N_CLASSES for s in useeds}')]),'ref'),
 'adam_time':(mutated('off_adam',[('tc += 1','tc += 2')]),'ref')}
 muts={}
 for name,(mod,arm) in mods.items():
  c=run(mod,arm,'off_'+name,snapshots=False)
  muts[name]=not (equal_rows(df(a),df(c))[0] and equal_state(state(a),state(c),act=False))
 return part('S-off',yes,muts,n+sum(q.numel() for q in state(a)['P']),{},prov)

def route_pred(mod):
 act=mod.make_act('H');act.init_state(10,DEV,'fixture')
 if act.door_c or not act.door_h or act.door_b!='none':return False
 P=[torch.stack([q[i].detach() for q in [E.H.init_params(s,DEV,E.DIMS) for s in SEEDS]]) for i in range(6)]
 P[1].fill_(.125)
 X=torch.stack([E.slot_inputs(common.CIFAR,s,'raw',DEV,center=act.door_c)[:16] for s in SEEDS])
 ref=E.make_act('ref');ref.init_state(10,DEV,'fixture')
 if not torch.equal(mod.forward(P,X,act)[0],E.forward(P,X,ref)[0]):return False
 with torch.no_grad():
  act.m[0].copy_(torch.linspace(.125,1,100,device=DEV).expand(10,-1))
  act.m[1].copy_(torch.linspace(.25,2,100,device=DEV).expand(10,-1))
  tr=mod.forward(P,X,act,True);ev=mod.forward(P,X,act,False)
  if not all(torch.equal(a,b) for a,b in zip(tr,ev)):return False
  for li,(z,aa) in enumerate(((tr[0],tr[1]),(tr[2],tr[3]))):
   if not torch.equal(aa,z.clamp(min=0)-act.m[li][:,None,:]):return False
   # Independent float64 dot products: subtracting m changes the next affine by -Wm.
   W,b=P[2*(li+1)],P[2*(li+1)+1]
   raw=z.clamp(min=0)
   before=torch.baddbmm(b[:,None,:],raw,W.transpose(1,2));after=torch.baddbmm(b[:,None,:],aa,W.transpose(1,2))
   expected=-torch.bmm(act.m[li].double()[:,None,:],W.double().transpose(1,2))
   magnitude=torch.bmm((raw.abs()+aa.abs()).double(),W.abs().double().transpose(1,2))+2*b.abs().double()[:,None,:]
   # Each affine is a sum of n products plus bias, aa has one subtraction rounding.
   bound=G(2*W.shape[-1]+2)*magnitude+U*torch.bmm((raw.abs()+act.m[li][:,None,:].abs()).double(),W.abs().double().transpose(1,2))
   if not torch.all(((after.double()-before.double())-expected).abs()<=bound):return False
  bias=[P[i].clone() for i in (1,3,5)];act.post_update(P,.001)
  if not all(torch.equal(x,P[i]) for x,i in zip(bias,(1,3,5))):return False
 return True

def s_route(prov):
 mods={'C_on':mutated('route_C',[('"H":    dict(c=False, h=True,  b="none")','"H":    dict(c=True, h=True,  b="none")')]),
 'skip_m1':mutated('route_m1',[(PHI,'return a if layer == 0 else a - self.m[layer][:, None, :]')]),
 'skip_m2':mutated('route_m2',[(PHI,'return a if layer == 1 else a - self.m[layer][:, None, :]')]),
 'eval_m0':mutated('route_eval',[(PHI,'return a - self.m[layer][:, None, :] if train else a')]),
 'zero_b1':mutated('route_b1',[('if self.door_b == "none":\n            return','if self.door_b == "none":\n            P[1].zero_()\n            return')])}
 return part('S-H-route',route_pred(E),{k:not route_pred(m) for k,m in mods.items()},10*16*200,{'bound':'gamma_(2n+2) affine absolute sum + one subtraction rounding'},prov)

def ema_fixture(mod):
 act=mod.make_act('H');act.init_state(10,DEV,'constant')
 z=torch.linspace(.125,4,100,device=DEV).expand(10,16,100).contiguous()
 mu=z.double().mean(1);expected=torch.zeros_like(mu);bound=torch.zeros_like(mu)
 with torch.no_grad():
  for _ in range(math.ceil(math.log(U)/math.log(.99))):
   bound=.99*bound+G(16+6)*(.99*expected.abs()+.01*mu.abs())
   expected=.99*expected+.01*mu
   act.update(z,z)
  for m in act.m:
   if not torch.all((m.double()-expected).abs()<=bound):return False
   if not torch.all((mu-m.double()).abs()<=(.99**1656)*mu.abs()+bound):return False
 return True

def trace_pred(mod,tag):
 # Observe actual forward, optimizer parameter list, and EMA on 300 real steps.
 orig=mod.forward;original_grad=torch.autograd.grad
 expected=None;bound=None;previous_mu=None;envelope=None;count=0;ok=True;grad_ok=True
 logits=[];post=[];pending=None;nonzero=0
 Xfull=torch.stack([E.slot_inputs(common.CIFAR,s,'raw',DEV,False) for s in SEEDS])
 def observed(P,X,act,train=False):
  nonlocal expected,bound,previous_mu,envelope,count,ok,pending,nonzero
  out=orig(P,X,act,train)
  if train:
   with torch.no_grad():
    muhat=[z.detach().double().clamp(min=0).mean(1) for z in (out[0],out[2])]
    # Full population means only on the first 6 steps: enough to exercise drift and sampling terms.
    mu=[z.double().clamp(min=0).mean(1) for z in (orig(P,Xfull,act,False)[0],orig(P,Xfull,act,False)[2])] if count<6 else None
    if expected is None:
     expected=[torch.zeros_like(m) for m in muhat];bound=[torch.zeros_like(m) for m in muhat]
     envelope=[m.abs() for m in mu]
    for li in (0,1):
     ok &= bool(torch.all((act.m[li].double()-expected[li]).abs()<=bound[li]))
     if mu is not None and previous_mu is not None:
      envelope[li]=.99*envelope[li]+(mu[li]-previous_mu[li]).abs()+.01*(previous_mu[li]-pending[li]).abs()
     if mu is not None:
      ok &= bool(torch.all((mu[li]-act.m[li].double()).abs()<=envelope[li]+bound[li]+G(4)*(mu[li].abs()+act.m[li].double().abs())))
     bound[li]=.99*bound[li]+G(22)*(.99*expected[li].abs()+.01*muhat[li].abs())
     expected[li]=.99*expected[li]+.01*muhat[li]
     nonzero+=int(torch.count_nonzero(act.m[li]))
    previous_mu=mu;pending=muhat;count+=1
    logits.append(out[-1].detach().cpu().clone())
  return out
 def grad(outputs,inputs,*a,**kw):
  nonlocal grad_ok
  grad_ok &= len(inputs)==6 and all(q.requires_grad for q in inputs)
  return original_grad(outputs,inputs,*a,**kw)
 mod.forward=observed;torch.autograd.grad=grad
 debug={}
 try:p=run(mod,'H',tag,debug=debug,graph=False,snapshots=False)
 finally:mod.forward=orig;torch.autograd.grad=original_grad
 for li,m in enumerate(state(p)['act_state']['m']):
  ok &= bool(torch.all((m.to(DEV).double()-expected[li]).abs()<=bound[li]))
 labels=debug['labels'];orders=debug['orders'];ind=[]
 for k,l in enumerate(logits):
  t=k//150;e=k//75;j=k%75
  y=labels[t].gather(1,orders[e][:,j*16:(j+1)*16])
  ind.append((l.argmax(-1)==y).float().mean(1))
 online=torch.stack(ind)
 matches=torch.equal(online,torch.stack(debug['online']))
 for t in (1,2):
  values=online[(t-1)*150:t*150].sum(0).double()/150
  recorded=df(p).query('task==@t').online_acc.to_numpy()
  matches &= np.array_equal(np.array([float(f'{v:.10g}') for v in values]),recorded)
 return bool(ok and count==300 and nonzero>0),{'count':count,'nonzero_m_observations':nonzero,'actual_optimizer_P':grad_ok,'independent_online':bool(matches),'path':str(p)}

def s_ema(prov):
 mods={'beta0':mutated('ema_beta0',[('self.lam, self.beta, self.widths = lam, beta, widths','self.lam, self.beta, self.widths = lam, 0.0, widths')]),
 'centered':mutated('ema_centered',[(UPDATE,'mi.mul_(1.0 - self.beta).add_((torch.clamp(z, min=0.0) - mi[:, None, :]).mean(1), alpha=self.beta)')]),
 'stop_m1':mutated('ema_stop1',[('for mi, z in zip(self.m, (z1, z2)):','for mi, z in zip(self.m[1:], (z2,)):')]),
 'stop_m2':mutated('ema_stop2',[('for mi, z in zip(self.m, (z1, z2)):','for mi, z in zip(self.m[:1], (z1,)):')])}
 yes,detail=trace_pred(E,'trace_H')
 muts={k:not ema_fixture(m) for k,m in mods.items()}
 # Move the EMA update to immediately before the actual training forward, using its incoming P.
 pre=mutated('ema_preforward',[
 ('z1, a1, z2, a2, z3 = forward(P, xb, act, train=True)',
  'with torch.no_grad():\n            zz = forward(P, xb, act, train=False)\n            act.update(zz[0], zz[2])\n        z1, a1, z2, a2, z3 = forward(P, xb, act, train=True)'),
 ('act.update(z1.detach(), z2.detach())   # door H: advance the EMA of phi(z)','pass   # deliberately moved EMA before forward')])
 muts['pre_forward']=not trace_pred(pre,'trace_preforward')[0]
 write(CHECK/'trace_details.json',detail)
 return part('S-H-EMA',yes and ema_fixture(E),muts,300*10*200+1656*10*200,detail,prov)

def grad_pred(mod):
 a=mod.make_act('H');a.init_state(10,DEV,'grad')
 z=torch.linspace(-2,2,1600,device=DEV).reshape(1,16,100).repeat(10,1,1).requires_grad_(True)
 a.m[0].fill_(.5);a.m[1].fill_(.25)
 extra=a.extra_params(10,DEV)
 if any(m is p for m in a.m for p in extra):return False
 up=torch.linspace(.25,3,z.numel(),device=DEV).reshape_as(z)
 for li in (0,1):
  y=a.phi(z,li,True);g=torch.autograd.grad((y*up).sum(),z,retain_graph=True)[0]
  if not torch.equal(g,up*(z>=0)) or a.m[li].requires_grad:return False
 return True

def s_grad(prov):
 a=mutated('grad_P',[('if self.norm != "layernorm":\n            return []','if self.norm != "layernorm":\n            return self.m if self.door_h else []')])
 b=mutated('grad_mean',[(PHI,'return a - a.mean(1, keepdim=True)')])
 tr=json.loads((CHECK/'trace_details.json').read_text())
 return part('S-grad',grad_pred(E) and tr['actual_optimizer_P'],{'m_in_P':not grad_pred(a),'batchmean_grad':not grad_pred(b)},32000,{'actual_run_parameter_count':6,'chain_rule':'upstream * 1[z>=0], m detached'},prov)

def child(cfg,name):
 p=CHECK/'worker_configs'/(name+'.json');write(p,cfg)
 log=CHECK/'worker_logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('w') as f:
  rc=subprocess.run([sys.executable,str(Path(__file__).with_name('worker.py')),str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 return rc

def s_resume_graph(prov):
 full=run(E,'H','H_full',snapshots=True)
 one=run(E,'H','H_one',tasks=1,snapshots=True)
 ck=state(one);nonzero=all(torch.count_nonzero(m)>0 for m in ck['act_state']['m'])
 target=CHECK/'runs/H_resume';shutil.copytree(one,target)
 rc=child({'arm':'H','tag':'H_resume','resume':True,'snapshots':True},'resume')
 def matches(p):
  if not (p/'ckpt.pt').exists():return False
  good=equal_state(state(full),state(p)) and equal_rows(df(full),df(p))[0]
  for seed in SEEDS:
   for t in (0,1,2):
    aa=E.snapshot_path(full,'H','raw',seed,t);bb=E.snapshot_path(p,'H','raw',seed,t)
    if not bb.exists():return False
    with np.load(aa) as a,np.load(bb) as b:
     good &= a.files==b.files and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
  return good
 yes=rc==0 and nonzero and matches(target);muts={};details={'nonzero_checkpoint_m1_m2':nonzero,'resume_rc':rc}
 for op in ('missing','zero','skipload'):
  for li in (0,1):
   name=op+str(li+1);tag='resume_'+name;dest=CHECK/'runs'/tag;shutil.copytree(one,dest)
   cfg={'arm':'H','tag':tag,'resume':True,'snapshots':True}
   if op in ('missing','zero'):
    st=state(dest)
    if op=='missing':st['act_state']['m'].pop(li)
    else:st['act_state']['m'][li].zero_()
    torch.save(st,dest/'ckpt.pt')
    write(CHECK/'mutants'/(name+'_state.json'),{'target':'ckpt.act_state.m','layer':li,'operation':op,'count':1})
   else:
    mod=mutated('resume_'+name,[('for dst, src in zip(self.m, st["m"]):\n                dst.copy_(src)',
     'for layer, (dst, src) in enumerate(zip(self.m, st["m"])):\n                if layer != '+str(li)+':\n                    dst.copy_(src)')])
    cfg['source']=mod.__file__
   code=child(cfg,name);muts[name]=code!=0 or not matches(dest)
   details[name+'_rc']=code
 mod=mutated('resume_rng',[('g_lab[s].set_state(st["g_lab"][s])','pass # deliberately omit label RNG restore')])
 dest=CHECK/'runs/resume_rng';shutil.copytree(one,dest)
 code=child({'arm':'H','tag':'resume_rng','resume':True,'snapshots':True,'source':mod.__file__},'rng')
 muts['rng']=code!=0 or not matches(dest)
 dest=CHECK/'runs/resume_wrong';shutil.copytree(one,dest)
 code=child({'arm':'H','tag':'resume_wrong','resume':True,'snapshots':True,'beta':.02},'wrong')
 muts['wrong_config']=code!=0
 res=part('S-resume',yes,muts,sum(q.numel() for q in ck['P']),details,prov)
 eager=run(E,'H','H_eager',graph=False,snapshots=False)
 correct=equal_state(state(eager),state(full)) and equal_rows(df(eager),df(full))[0]
 mod=mutated('graph_restore',[('act.load_state(keep_act)','pass # deliberately omit capture EMA restore')])
 bad=run(mod,'H','H_badgraph',snapshots=False)
 graph=part('S-graph',correct,{'skip_restore':not(equal_state(state(full),state(bad)) and equal_rows(df(full),df(bad))[0])},sum(q.numel() for q in ck['P']),{},prov)
 return res and graph

def eval_metrics(P,values):
 z1,a1,z2,a2,logits,Y=values
 ans={'memo_acc':(logits.argmax(-1)==Y).float().mean(1)}
 for li,z in enumerate((z1,z2)):
  tag='l'+str(li+1);gate=z>0
  ans['dead_frac_'+tag]=(~gate.any(1)).float().mean(1)
  ans['gate_zero_frac_'+tag]=(~gate).float().mean((1,2))
  ans['zbar_'+tag]=torch.stack([v.mean(0).median() for v in z])
  ans['zsd_'+tag]=torch.stack([v.std(0).median() for v in z])
  ans['sink_ratio_'+tag]=torch.stack([(v.mean(0)/v.std(0).clamp(min=1e-12)).median() for v in z])
  ans['bias_over_sd_'+tag]=torch.stack([P[2*li+1][r].abs().mean()/float(z[r].std(0).median().clamp(min=1e-12)) for r in range(10)])
 ans['r_a1']=torch.stack([row.mean(0).norm()/(row-row.mean(0)).square().sum(1).mean().sqrt() for row in a1])
 return ans

def round10(v):return float(f'{float(v):.10g}')
def metric_match(metrics,record):
 for k,v in metrics.items():
  for r in range(10):
   a=round10(v[r]);b=float(record.iloc[r][k])
   if a!=b and not(math.isnan(a) and math.isnan(b)):return False
 return True

def s_eval(prov):
 p=CHECK/'runs/H_full';slots=[(s,'raw') for s in SEEDS];rec=df(p);yes=True;count=0;mut={k:False for k in L.EXPECTED['S-eval']};details={}
 for t in (1,2):
  values=E.replay_stack(p,'H',slots,t,DEV,common.CIFAR)
  snaps=[np.load(E.snapshot_path(p,'H','raw',s,t)) for s in SEEDS]
  P=[torch.stack([torch.from_numpy(d[k]) for d in snaps]).to(DEV) for k in ('W1','b1','W2','b2','W3','b3')]
  metrics=eval_metrics(P,values);rt=rec[rec.task==t].reset_index(drop=True)
  yes &= metric_match(metrics,rt);count+=10*len(metrics)
  X=torch.stack([E.slot_inputs(common.CIFAR,s,'raw',DEV,False) for s in SEEDS])
  for name,layers in [('no_H',(0,1)),('skip_m1',(0,)),('skip_m2',(1,))]:
   act=E.make_act('H');act.init_state(10,DEV,'eval')
   act.m=[torch.stack([torch.from_numpy(d[k]) for d in snaps]).to(DEV) for k in ('m1','m2')]
   for li in layers:act.m[li].zero_()
   bad=(*E.forward(P,X,act,False),values[-1]);mut[name]|=not metric_match(eval_metrics(P,bad),rt)
  bad=dict(metrics);bad['sink_ratio_l2']=metrics['zbar_l2']/metrics['zsd_l2'].clamp(min=1e-12)
  mut['ratio_medians'] |= not metric_match(bad,rt)
  bad=dict(metrics);bad['dead_frac_l1']=metrics['dead_frac_l2'];bad['zbar_l1']=metrics['zbar_l2'];mut['swap_layers']|=not metric_match(bad,rt)
  details['t'+str(t)+'_z_exact0']=[int((v==0).sum()) for v in (values[0],values[2])]
 tr=json.loads((CHECK/'trace_details.json').read_text());yes &= tr['independent_online']
 # Re-evaluate each actual batch AFTER its Adam/EMA update in the negative control.
 mod=mutated('eval_post_online',[
 ('acc_sum.add_(hit)\n        last_hit.copy_(hit)','pass # deliberately move online below update'),
 ('act.update(z1.detach(), z2.detach())   # door H: advance the EMA of phi(z)',
  'act.update(z1.detach(), z2.detach())\n            post_logits = forward(P, xb, act, train=False)[-1]\n            hit_post = (post_logits.argmax(-1) == yb).float().mean(1)\n            acc_sum.add_(hit_post)\n            last_hit.copy_(hit_post)')])
 _,post=trace_pred(mod,'trace_post_online');mut['post_update']=not post['independent_online']
 z=torch.tensor([-1.,0.,1.],device=DEV,requires_grad=True);act=E.make_act('ref');act.init_state(1,DEV,'zero')
 grad=torch.autograd.grad(act.phi(z).sum(),z)[0]
 yes &= torch.equal(grad,torch.tensor([0.,1.,1.],device=DEV)) and torch.equal(act.dphi(z),torch.tensor([0.,0.,1.],device=DEV))
 details['zero_point']={'train':[0,1,1],'diagnostic':[0,0,1]}
 return part('S-eval',yes,mut,count+300*10,details,prov)

def s_diverge(prov):
 good=CHECK/'runs/H_full';bad=run(E,'H','H_nan',nan_slot=0,snapshots=False)
 frame=df(bad);normal=df(good);other=frame[frame.seed!=200].reset_index(drop=True);wanted=normal[normal.seed!=200].reset_index(drop=True)
 def predicate(frame,st):
  other=frame[frame.seed!=200].reset_index(drop=True)
  return equal_rows(other,wanted)[0] and len(st['diverged'])==1 and not bool(st['alive'][0]) and frame[frame.seed==200].online_acc.isna().all()
 st=state(bad);yes=predicate(frame,st)
 # Actual trajectories in the other nine slots must also match.
 yes &= all(torch.equal(a[1:],b[1:]) for k in ('P','m','v') for a,b in zip(state(good)[k],st[k]))
 fake=frame.copy();fake.loc[fake.seed==200,'online_acc']=.9;wrong=copy.deepcopy(st);wrong['alive'][0]=True;wrong['diverged']=[]
 return part('S-diverge',yes,{'nan_success':not predicate(fake,wrong),'stop_all':not predicate(frame[frame.seed==200],st)},9*2*len(frame.columns),{},prov)

def synthetic(k=0):
 frames={}
 for arm in ('ref','H','C','CH'):
  rows=[]
  for seed in range(10):
   w=.987 if arm=='CH' else .9 if arm=='H' and seed<k else .11
   for t in range(1,51):rows.append({'seed':seed,'task':t,'cond':'raw','online_acc':w,'memo_acc':w,
    'dead_frac_l1':.985 if arm in ('ref','H') else 0.,'gate_zero_frac_l2':1. if arm=='C' else .5})
  frames[arm]=pd.DataFrame(rows)
 return frames

def s_verdict(prov):
 yes=True;count=0;details={}
 for k in (0,1,2,8,9,10):
  for d,expected in [(-2*S.DELTA,'C_NEEDED'),(-S.DELTA,'TIE'),(0.,'C_REDUNDANT' if k>=9 else 'TIE')]:
   got=S.classify(k,np.full(10,d));yes &= got==('H_ALONE_RESCUES' if k>=9 else 'H_ALONE_FAILS',expected);count+=1
  v=S.verdict(synthetic(k));yes &= v['labels']['M_H']==('H_ALONE_RESCUES' if k>=9 else 'H_ALONE_FAILS');count+=1
 f=synthetic();v=S.verdict(f);yes &= v['labels']['D_HR']=='TIE' and v['sign_HR']['n']==0 and v['sign_HR']['p'] is None
 for case in ('missing','duplicate','nan','hash','checks','not_reproduced'):
  ff=copy.deepcopy(f);kw={};expected='INAPPLICABLE'
  if case=='missing':ff['H']=ff['H'].iloc[1:]
  elif case=='duplicate':ff['H']=pd.concat([ff['H'],ff['H'].iloc[:1]])
  elif case=='nan':ff['H'].loc[0,'online_acc']=np.nan
  elif case=='hash':kw['hashes_ok']=False
  elif case=='checks':kw['qualified']=False
  else:ff['ref']['online_acc']=.9;expected='NOT_REPRODUCED'
  yes &= S.verdict(ff,**kw)['status']==expected;count+=1
 # Nonuniform paired differences make median(H-ref) differ from median(H)-median(ref).
 f=synthetic();rr=np.array([0.,0.,0.,0.,0.,.2,.2,.2,.2,.2]);hh=np.array([.3,.3,.3,.3,.3,.21,.21,.21,.21,.21])
 for arm,w in [('ref',rr),('H',hh)]:f[arm]['online_acc']=f[arm].seed.map(dict(enumerate(w)))
 v=S.verdict(f);yes &= v['median_d_HR']==float(np.median(hh-rr))
 # Layer boundary equality is neither dies nor saved. Nine and eight strict signs differ.
 f=synthetic();f['H']['dead_frac_l1']=S.THETA;yes &= S.verdict(f)['R1_H']=='TIE'
 muts={'boundary_ge':S.classify(9,np.full(10,-S.DELTA))[1]!='C_NEEDED',
 'median_difference':v['median_d_HR']!=float(np.median(hh)-np.median(rr)),
 'tie_redundant':S.classify(9,np.full(10,-S.DELTA))[1]!='C_REDUNDANT',
 'drop_missing':S.verdict({**f,'H':f['H'].iloc[1:]})['status']=='INAPPLICABLE'}
 # Execute actual source mutants against the same expected boundary/paired/missing predicates.
 source=Path(S.__file__).read_text()
 replacements={
 'boundary_ge':('d[8]<-DELTA','d[8]<=-DELTA'),
 'median_difference':("float(np.median(dhr))","float(np.median(windows['H'])-np.median(windows['ref']))"),
 'tie_redundant':("else 'TIE'\n return m,n","else 'C_REDUNDANT'\n return m,n"),
 'drop_missing':("or not all(valid(d) for d in frames.values())","")}
 for name,(old,new) in replacements.items():
  assert source.count(old)==1
  path=CHECK/'mutants'/('verdict_'+name+'.py');path.write_text(source.replace(old,new));mod=module(path)
  if name in ('boundary_ge','tie_redundant'):muts[name]=mod.classify(9,np.full(10,-S.DELTA))!=S.classify(9,np.full(10,-S.DELTA))
  elif name=='median_difference':
   ff=synthetic()
   for arm,w in [('ref',rr),('H',hh)]:ff[arm]['online_acc']=ff[arm].seed.map(dict(enumerate(w)))
   muts[name]=mod.verdict(ff)['median_d_HR']!=S.verdict(ff)['median_d_HR']
  else:muts[name]=mod.verdict({**f,'H':f['H'].iloc[1:]})['status']!='INAPPLICABLE'
  write(path.with_suffix('.json'),{'replacements':[[old,new]],'count':1,'source_sha256':sha(S.__file__),'mutant_sha256':sha(path)})
 # Sign reversal and dispersion-zero decision fixtures, with prescribed order interval.
 f=synthetic();f['H']['online_acc']=.12;yes &= S.verdict(f)['labels']['D_HR']=='H_HELPS'
 f['H']['online_acc']=.1;yes &= S.verdict(f)['labels']['D_HR']=='H_HURTS'
 return part('S-verdict',yes,muts,count+8,details,prov)

def provenance_pred(folder):
 p=json.loads((folder/'provenance.json').read_text());start=json.loads((folder/'provenance_start.json').read_text())
 return p['run_id']==RUN and p['prereg_commit']==PREREG and start['run_id']==RUN and start['prereg_commit']==PREREG and p['git_hash']==start['git_hash'] and p['launch_provenance']==start

def s_cli(prov):
 base=CHECK/'runs/cli_H';argv=[sys.executable,str(ROOT/'src/relu_doors_0919.py'),'run','--arm','H','--seeds','200-209','--conds','raw','--tasks','2','--epochs','2','--run-id',RUN,'--prereg-commit',PREREG,'--out',str(base)]
 log=CHECK/'worker_logs/cli.log'
 with log.open('w') as f:rc=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 full=CHECK/'runs/H_full';yes=rc==0 and equal_state(state(base),state(full)) and equal_rows(df(base),df(full))[0] and provenance_pred(base)
 old=mutated('cli_runid',[('"run_id": run_id or EXPERIMENT','"run_id": EXPERIMENT')])
 dest=run(old,'H','cli_oldid',run_id=RUN,prereg_commit=PREREG,snapshots=False)
 muts={'old_runid':not provenance_pred(dest)}
 end=mutated('cli_endgit',[('**git_states[0], "git_states": git_states','**git_state(), "git_states": git_states')])
 original=end.git_state;calls=0
 def moving_git():
  nonlocal calls
  calls+=1;d=original()
  if calls>1:d['git_hash']='deliberate_late_git_state'
  return d
 end.git_state=moving_git
 dest=run(end,'H','cli_endgit',run_id=RUN,prereg_commit=PREREG,snapshots=False)
 muts['end_git']=not provenance_pred(dest)
 stopbase=BASE/'_smoke/stop';(stopbase/'_launch').mkdir(parents=True,exist_ok=True);(stopbase/'_launch/STOP').write_text('fixture')
 try:L.guard(stopbase,{});muts['STOP']=False
 except RuntimeError as e:muts['STOP']='STOP file' in str(e)
 # Mutation removes STOP check and proceeds to the distinct mandatory-check error.
 src=Path(L.__file__).read_text();oldline="if (base/'_launch/STOP').exists():raise RuntimeError('STOP file present')"
 assert src.count(oldline)==1;path=CHECK/'mutants/launch_no_STOP.py';path.write_text(src.replace(oldline,'pass # STOP ignored'))
 m=module(path)
 try:m.guard(stopbase,{});detected=False
 except RuntimeError as e:detected='STOP file' in str(e)
 muts['STOP'] &= not detected
 # Protected path fails before directory creation or any training.
 forbidden=ROOT/'results/relu_doors_0919/H_ILLEGAL_TEST'
 assert not forbidden.exists()
 good,_=attempt(lambda:E.run('H',SEEDS,['raw'],2,2,DEV,forbidden,run_id=RUN,prereg_commit=PREREG))
 muts['old_output']=not good and not forbidden.exists()
 return part('S-CLI',yes,muts,100,{'cli_rc':rc,'start_end_provenance_match':provenance_pred(base)},prov)

def collect(prov):
 parts={n:json.loads((CHECK/(n+'.json')).read_text()) for n in REQUIRED-{'S-collect'} if (CHECK/(n+'.json')).exists()}
 yes=L.qualified(parts,False);miss=dict(parts);miss.pop('S-off',None);stale=copy.deepcopy(parts)
 if stale:next(iter(stale.values()))['provenance']['source_sha256']['src/relu_doors_0919.py']='bad'
 muts={'missing':not L.qualified(miss,False),'empty':not L.qualified({},False),'stale':not L.qualified(stale,False)}
 part('S-collect',yes,muts,len(parts),{'expected':sorted(REQUIRED-{'S-collect'})},prov)
 parts={n:json.loads((CHECK/(n+'.json')).read_text()) for n in REQUIRED if (CHECK/(n+'.json')).exists()}
 passed=L.qualified(parts)
 write(CHECK/'checks.json',{'all_pass':passed,'checks':parts,'prereg_commit':PREREG,'source_commit_at_collection':E.H.git_hash()})
 return passed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only',default='all');a=ap.parse_args()
 torch.set_num_threads(2);E.H.setup('cuda');common.CIFAR=E.RC.Cifar10()
 sources=[Path(__file__),Path(S.__file__),Path(L.__file__),Path(__file__).with_name('worker.py')]
 prov=provenance(sources);assert not prov['dirty_src_analysis'],prov['dirty_src_analysis']
 write(CHECK/'suite_start.json',prov)
 started=time.monotonic()
 cases={'off':s_off,'route':s_route,'ema':s_ema,'grad':s_grad,'resume_graph':s_resume_graph,'eval':s_eval,'diverge':s_diverge,'verdict':s_verdict,'cli':s_cli}
 for name,fn in cases.items():
  if a.only not in ('all',name):continue
  print('START',name,flush=True)
  if not fn(prov):raise SystemExit('check failed: '+name)
 cost={'wall_s':time.monotonic()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'H_step_ms':json.loads((CHECK/'runs/H_full/provenance.json').read_text())['step_ms_last_task'] if (CHECK/'runs/H_full/provenance.json').exists() else None}
 write(CHECK/'S-cost.json',cost)
 if a.only in ('all','collect') and not collect(prov):raise SystemExit('incomplete check suite')
if __name__=='__main__':main()
