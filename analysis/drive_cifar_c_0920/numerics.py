"""A2 read-only float64 decomposition and conservative, independently checked bounds.

All arrays have a leading R axis. A captured call contains no host synchronization.
The target is the *stored float32* Adam parameter difference at a fixed native
full-dataset feature mean. Measured defects are admissible only after independent
forward-error checks. They are never used as their own acceptance tolerance.
"""
import torch

U32, U64 = 2.**-24, 2.**-53
TINY32 = torch.finfo(torch.float32).tiny  # also covers a flushed subnormal
TINY64 = torch.finfo(torch.float64).tiny

def gamma(n, u=U64):
    return n*u/(1-n*u)

def augmented(w, b):
    return torch.cat((w.double(), b.double().unsqueeze(-1)), -1)

def dot(a, b):
    value = (a*b).sum(-1)
    error = gamma(2*a.shape[-1]+2)*(a.abs()*b.abs()).sum(-1) + TINY64
    return value, error

def full_features(P, X):
    z = torch.baddbmm(P[1][:,None,:], X, P[0].transpose(1,2))
    h = z.clamp(min=0)
    z2 = torch.baddbmm(P[3][:,None,:], h, P[2].transpose(1,2))
    hd = h.double()
    mu = hd.mean(1)
    mu_error = gamma(h.shape[1]+2)*hd.abs().mean(1) + TINY64
    # Bound native B1200 affine/reduction against the algebraic mean of h.
    native_error = gamma(h.shape[2]+3,U32)*(torch.bmm(hd.abs(),P[2].double().abs().transpose(1,2))+P[3].double().abs()[:,None,:]).mean(1)
    native_error += gamma(h.shape[1]+2)*z2.double().abs().mean(1) + TINY32*(h.shape[2]+3)
    return mu, mu_error, z2.double().mean(1), native_error

def decompose(h, z, logits, J, old, new):
    h, z, logits, J = h.double(), z.double(), logits.double(), J.double()
    R,B,_ = h.shape
    ar = torch.arange(R,device=h.device)[:,None]
    jo, jn = J[ar,old], J[ar,new]
    p = torch.softmax(logits, -1)
    d = jo-jn
    e = torch.bmm(p,J)-jo
    eps = 1-p.gather(-1,old[...,None]).squeeze(-1)
    L = (J[:,None,:,:]-jo[:,:,None,:]).abs().amax(2)
    # Host ReLUDoors uses clamp(min=0), whose native derivative at exactly 0 is 1.
    gate = (z >= 0).double()
    ht = torch.cat((h,torch.ones(R,B,1,device=h.device,dtype=h.dtype)),-1)
    g = torch.bmm(((d+e)*gate).transpose(1,2),ht)/B
    uconf = torch.bmm(p-1/J.shape[1],J)
    gc = torch.bmm((uconf*gate).transpose(1,2),ht)/B
    # CE softmax/backward + 10-class product + batch16 reduction. 128 operations
    # dominates these chains (including exp/log/division ulps); native defects
    # must pass this scale-aware bound before projection into the certificate.
    absu = torch.bmm(p,J.abs()) + jn.abs()
    gb = gamma(128,U32)*torch.bmm((absu*gate).transpose(1,2),ht.abs())/B + 128*TINY32
    eb = gamma(64)*(torch.bmm(p,J.abs())+jo.abs()+eps[...,None]*L)+TINY64
    return dict(ht=ht,gate=gate,d=d,e=e,eps=eps,L=L,g=g,gconf=gc,gb=gb,eb=eb)

def measure(before, after, mprev, mnew, vnew, gactual, old_features, new_features,
            dec, inv1, inv2, conf, hist):
    mu, mue, native0, native0e = old_features
    mu1, mu1e, native1, native1e = new_features
    R,I,D = before.shape; B=dec['ht'].shape[1]
    k = torch.cat((mu,torch.ones_like(mu[:,:1])),-1)
    ke = torch.cat((mue,torch.zeros_like(mue[:,:1])),-1)
    kd = k[:,None,:]
    V = vnew.double()
    inv1d, inv2d = inv1.double(), inv2.double()
    pre = 1/(torch.sqrt(V*inv2d)+1e-8)
    K = torch.bmm(kd*pre,dec['ht'].transpose(1,2))
    gate = dec['gate'].transpose(1,2)
    d = dec['d'].transpose(1,2)
    epsL = (dec['eps'][...,None]*dec['L']).transpose(1,2)
    T = (K*gate*d).mean(-1)
    rad = (K.abs()*gate*epsL).mean(-1)
    H, He = dot(kd*pre,mprev)
    qm, qp = .9*H+.1*(T-rad), .9*H+.1*(T+rad)
    # This independently tests d+e and the CE gradient, even though the observed
    # discrepancy is subsequently included in the numerical certificate radius.
    gd = gactual.double()-dec['g']
    bad_grad = (gd.abs()>dec['gb']).any(-1)
    bad_e = (dec['e'].abs()>dec['eps'][...,None]*dec['L']+dec['eb']).any(1)
    model_m = .9*mprev + .1*gactual.double()
    md = mnew.double()-model_m
    mb = gamma(4,U32)*(.9*mprev.abs()+.1*gactual.double().abs())+4*TINY32
    bad_m = (md.abs()>mb).any(-1)
    ideal = -.001*inv1d*pre*mnew.double()
    actual = after-before
    pd = actual-ideal
    pb = gamma(12,U32)*(before.abs()+ideal.abs())+12*TINY32
    bad_p = (pd.abs()>pb).any(-1)
    # Triangle bound for denominator/EMA/gradient/reduction and stored subtraction.
    # Actual denominator and inv correction are read, never recomputed per component.
    qscale = .9*(kd*pre*mprev).abs().sum(-1)+.1*(K.abs()*gate*(d.abs()+epsL)).mean(-1)
    qerr = (gamma(2*D+2*B+128)+gamma(12,U32))*qscale + He
    qerr += .1*(K.abs()*gate*dec['eb'].transpose(1,2)).mean(-1)
    qerr += (pre*(.1*gd.abs()+md.abs())*kd.abs()).sum(-1)
    qerr += (pre*(.9*mprev.abs()+.1*dec['g'].abs())*ke[:,None,:]).sum(-1)
    move_error = .001*inv1d*qerr+(kd.abs()*pd.abs()).sum(-1)
    S, Se = dot(actual,kd)
    Se += (actual.abs()*ke[:,None,:]).sum(-1)
    U, Ue = dot(after[...,:-1],(mu1-mu)[:,None,:])
    Ue += (after[...,:-1].abs()*(mue+mu1e)[:,None,:]).sum(-1)
    newmean,newerr = dot(after,torch.cat((mu1,torch.ones_like(mu1[:,:1])),-1)[:,None,:])
    oldmean,olderr = dot(before,kd)
    total = newmean-oldmean
    closure_error=Se+Ue+newerr+olderr+gamma(4)*(newmean.abs()+oldmean.abs()+S.abs()+U.abs())
    bad_closure=(total-S-U).abs()>closure_error
    native_error = native0e+native1e+closure_error
    bad_native = ((native1-native0)-total).abs()>native_error
    hi = -.001*inv1d*qm + move_error + Se
    lo = -.001*inv1d*qp - move_error - Se
    down, up = hi<0, lo>0
    violation = (down & (S+Se>=0)) | (up & (S-Se<=0)) | (down & up)
    cm = conf*.9 + dec['gconf']*.1
    hm = hist*.9
    lm = mnew.double()-cm-hm
    sc,_=dot(-.001*inv1d*pre*cm,kd)
    sh,_=dot(-.001*inv1d*pre*hm,kd)
    sl,_=dot(-.001*inv1d*pre*lm,kd)
    component_error=(pd.abs()*kd.abs()).sum(-1)+gamma(2*D+32)*(sc.abs()+sh.abs()+sl.abs())+Se
    bad_components=(S-sc-sh-sl).abs()>component_error
    sgdk = torch.bmm(k[:,None,:],dec['ht'].transpose(1,2)).expand(-1,I,-1)
    fail=bad_grad|bad_e|bad_m|bad_p|bad_closure|bad_native|violation|bad_components
    out=dict(S=S,U=U,total=total,S_conf=sc,S_label=sl,S_hist=sh,
             cert_down=down.double(),cert_up=up.double(),uncertain=(~(down|up)).double(),
             cert_total_down=(hi+U+Ue<0).double(),raw_down=(qm>0).double(),raw_up=(qp<0).double(),
             Qminus=qm,Qplus=qp,T=T,R=rad,H=H,move_error=move_error,S_error=Se,
             native_closure=((native1-native0)-total).abs(),closure=(total-S-U).abs(),
             grad_defect=gd.abs().amax(-1),grad_bound=dec['gb'].amax(-1),
             adam_defect=pd.abs().amax(-1),component_defect=(S-sc-sh-sl).abs(),
             SGD_T=(sgdk*gate*d).mean(-1),SGD_R=(sgdk.abs()*gate*epsL).mean(-1),
             fail=fail.double(),certificate_violation=violation.double())
    finite=torch.stack([torch.isfinite(x) for x in out.values()]).all(0)
    out['nonfinite']=(~finite).double()
    return out,cm,hm

COUNT_KEYS=('cert_down','cert_up','uncertain','cert_total_down','raw_down','raw_up','fail','certificate_violation','nonfinite')
TRANSPORT_KEYS=('S','U','total','S_conf','S_label','S_hist')
DIAG_KEYS=('Qminus','Qplus','T','R','H','move_error','S_error','native_closure','closure','grad_defect','grad_bound','adam_defect','component_defect','SGD_T','SGD_R')
KEYS=COUNT_KEYS+tuple(f'{k}_{suffix}' for k in TRANSPORT_KEYS for suffix in ('sum','positive','negative','positive_count','negative_count'))+DIAG_KEYS

def pack(d):
    out={k:d[k] for k in COUNT_KEYS+DIAG_KEYS}
    for k in TRANSPORT_KEYS:
        x=d[k];out.update({k+'_sum':x,k+'_positive':x.clamp(min=0),k+'_negative':x.clamp(max=0),k+'_positive_count':(x>0).double(),k+'_negative_count':(x<0).double()})
    return torch.stack([out[k] for k in KEYS],-1)
