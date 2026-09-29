"""Check a target-selected, finite, true-ELU optimum on four shared inputs."""
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import expit
import torch

OUT=Path('results/two_group_logistic_0929')
D,P,A0,H0,B0,R0=4.,.02,1.4,1.,-3.,.5
X=np.array([[D,0.],[0.,-1.],[0.,0.],[0.,1.]])
MASS=np.array([P,(1-P)/4,(1-P)/2,(1-P)/4])
INITIAL=np.array([(H0-B0)/D,R0,B0,0.])


def forward(theta,a):
    z=X@theta[:2]+theta[2]
    phi=np.where(z>0,z,np.expm1(np.minimum(z,0)))
    return z,theta[3]+a*phi


def statistic(z):
    order=np.argsort(z)
    median=z[order[np.searchsorted(np.cumsum(MASS[order]),.5)]]
    mean=MASS@z
    sd=np.sqrt(MASS@(z-mean)**2)
    return float((z.max()-median)/sd),float(sd),float(median)


def selected(a):
    b=B0-np.log(a/A0)
    c=a-A0
    h=A0*(H0+1)/a-1
    return np.array([(h-b)/D,R0,b,c])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    z0,targets=forward(INITIAL,A0);q=expit(targets)
    beta=(targets[3]*targets[1]-targets[2]**2)/(targets[3]+targets[1]-2*targets[2])
    kval=targets[2]-beta
    r=np.log((targets[3]-beta)/kval)
    assert abs(beta+A0)<1e-12 and abs(r-R0)<1e-12
    rows=[];curve=[];checks=[]
    for a in [.5,1.,1.4,2.]:
        theta=selected(a);z,logits=forward(theta,a)
        tail,sd,median=statistic(z)
        derivative=-z[0]*(1-P)*(R0*R0/2)/sd**3
        step=1e-5
        finite=(statistic(forward(selected(a*np.exp(step)),a*np.exp(step))[0])[0]
                -statistic(forward(selected(a*np.exp(-step)),a*np.exp(-step))[0])[0])/(2*step)
        root=least_squares(lambda t:forward(t,a)[1]-targets,INITIAL.copy(),
            jac='3-point',xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=3000,x_scale='jac')
        root_logits=forward(root.x,a)[1]
        tt=torch.tensor(theta,requires_grad=True)
        xx=torch.tensor(X);ww=torch.tensor(MASS);qq=torch.tensor(q)
        def tloss(t):
            zz=xx@t[:2]+t[2]
            ll=t[3]+a*torch.nn.functional.elu(zz)
            return torch.sum(ww*(torch.nn.functional.softplus(ll)-qq*ll))
        hh=torch.autograd.functional.hessian(tloss,tt).detach().numpy()
        prime=np.exp(np.minimum(z,0))
        jac=np.column_stack([a*prime[:,None]*X,a*prime,np.ones(4)])
        exact_h=jac.T@np.diag(MASS*q*(1-q))@jac
        grad=torch.autograd.grad(tloss(tt),tt)[0].detach().numpy()
        eig=np.linalg.eigvalsh(hh)
        row=dict(gain=a,h=float(z[0]),b=float(theta[2]),r=float(theta[1]),c=float(theta[3]),
            w1=float(theta[0]),gap=float(z[0]-theta[2]),T=tail,sd=sd,body_sd=R0/np.sqrt(2),
            loss=float(tloss(tt).detach()),analytic_dT_dlogv=float(derivative),
            finite_difference_dT_dlogv=float(finite),root_success=bool(root.success),
            root_nfev=int(root.nfev),root_max_logit_error=float(np.max(abs(root_logits-targets))),
            root_max_parameter_error=float(np.max(abs(root.x-theta))),
            min_hessian_eigenvalue=float(eig.min()),max_hessian_eigenvalue=float(eig.max()))
        rows.append(row)
        check=dict(gain=a,closed_form_logit_max_error=float(np.max(abs(logits-targets))),
            gradient_max_abs=float(np.max(abs(grad))),hessian_max_error=float(np.max(abs(hh-exact_h))),
            derivative_error=float(abs(derivative-finite)),jacobian_determinant=float(np.linalg.det(jac)),
            physical_branches=bool(z[0]>0 and z[1:].max()<0),
            root_physical_branches=bool(forward(root.x,a)[0][0]>0 and forward(root.x,a)[0][1:].max()<0))
        checks.append(check)
        assert check['physical_branches'] and check['root_physical_branches']
        assert check['closed_form_logit_max_error']<1e-13
        assert check['gradient_max_abs']<1e-13 and check['hessian_max_error']<1e-12
        assert check['derivative_error']<1e-8 and eig.min()>0
        assert root.success and row['root_max_logit_error']<1e-9 and row['root_max_parameter_error']<1e-7
    # This curve evaluates the derived expression, not additional fitting.
    for a in np.geomspace(.15,2.75,180):
        theta=selected(a);z,_=forward(theta,a);tail,sd,_=statistic(z)
        curve.append(dict(gain=float(a),h=float(z[0]),b=float(theta[2]),r=R0,T=tail,
            sd=sd,body_sd=R0/np.sqrt(2),physical_branches=bool(z[0]>0 and z[1:].max()<0)))
    for name,data in [('exact_elu_endpoints.csv',rows),('exact_elu_curve.csv',curve)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    (OUT/'exact_elu_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    provenance=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
        initial_state=INITIAL.tolist(),old_gain=A0,inputs=X.tolist(),masses=MASS.tolist(),
        fixed_target_logits=targets.tolist(),fixed_target_probabilities=q.tolist(),
        recovered_beta=float(beta),recovered_K=float(kval),recovered_r=float(r),
        scope='New realizable soft targets; distinct from the equal-bulk-target model. No MNIST fit.',
        root_initialization='same original state in all four arms, no extra starts',
        gate_interval=[float(A0*np.exp(B0+R0)),float(A0*(H0+1))])
    (OUT/'exact_elu_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(endpoints=rows,checks=checks)))


if __name__=='__main__':main()
