"""Registered 32-arm finite Adam check for one shared two-group neuron.

No seed/geometry sweep. Fixed-branch activation is externally fixed and its
physical sign validity is measured separately. True ELU uses exp(z) backward.
Zero Adam denominator with zero moments receives zero update (eps -> 0+).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

D, P, QA, QB, H0, R0, C0 = 4.0, 0.02, 0.8, 0.2, 1.0, 2.0, 0.0
GAINS = (0.1, 0.3, 1.0, 2.0)
EPSILONS = (0.0, 1e-8)
BIASES = (-12.0, -24.0)
ACTIVATIONS = ("saturated", "exact_elu")
LR, BETA1, BETA2, STEPS = 0.001, 0.9, 0.999, 20000
CHECKPOINTS = (0, 1, 10, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000)
X = np.array([[D, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0]])
MASS = np.array([P, (1-P)/4, (1-P)/2, (1-P)/4])
TARGET = np.array([QA, QB, QB, QB])
TARGET_LOGITS = np.log(TARGET / (1-TARGET))
DELTA = TARGET_LOGITS[0] - TARGET_LOGITS[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def design():
    metadata, theta = [], []
    for activation in ACTIVATIONS:
        for b0 in BIASES:
            for eps in EPSILONS:
                for gain in GAINS:
                    metadata.append({"arm": len(metadata), "activation": activation, "b0": b0, "epsilon": eps, "gain": gain})
                    theta.append([(H0-b0)/D, R0, b0, C0])
    return metadata, np.array(theta, dtype=np.float64)


def forward_and_gradient(theta, gain, is_elu):
    z = theta[:, :2] @ X.T + theta[:, 2:3]
    elu = np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))
    elu_prime = np.exp(np.minimum(z, 0.0))
    saturated = np.full_like(z, -1.0); saturated[:, 0] = z[:, 0]
    sat_prime = np.zeros_like(z); sat_prime[:, 0] = 1.0
    phi = np.where(is_elu[:, None], elu, saturated)
    slope = np.where(is_elu[:, None], elu_prime, sat_prime)
    logits = gain[:, None] * phi + theta[:, 3:4]
    probability = np.exp(-np.logaddexp(0.0, -logits))
    error = probability - TARGET
    weighted_error = error * MASS
    dz = weighted_error * gain[:, None] * slope
    grad = np.column_stack([dz @ X[:, 0], dz @ X[:, 1], dz.sum(axis=1), weighted_error.sum(axis=1)])
    loss = ((np.logaddexp(0.0, logits) - TARGET * logits) * MASS).sum(axis=1)
    return grad, {"z": z, "logits": logits, "error": error, "loss": loss}


def adam_step(theta, grad, first, second, step, epsilon):
    first *= BETA1; first += (1-BETA1) * grad
    second *= BETA2; second += (1-BETA2) * grad**2
    mhat = first / (1-BETA1**step)
    rms = np.sqrt(second / (1-BETA2**step))
    denominator = rms + epsilon[:, None]
    normalized = np.divide(mhat, denominator, out=np.zeros_like(mhat), where=denominator > 0)
    delta = -LR * normalized
    theta += delta
    return delta, rms


def measured_rows(metadata, theta, initial, step, gradient, info):
    result = []
    for i, meta in enumerate(metadata):
        z = info["z"][i]
        order = np.argsort(z, kind="stable")
        median = z[order[np.flatnonzero(np.cumsum(MASS[order]) >= 0.5)[0]]]
        mean = float(MASS @ z)
        variance = float(MASS @ ((z-mean)**2))
        sd = np.sqrt(variance)
        gap = z[0] - theta[i, 2]
        theoretical_sd = np.sqrt((1-P)*theta[i, 1]**2/2 + P*(1-P)*gap**2)
        physical = bool(z[0] > 0 and np.max(z[1:]) < 0)
        hstar = DELTA/meta["gain"] - 1
        kappa = D/(D+1)
        predicted_b = meta["b0"] + (hstar-H0)/(D+1)
        predicted_gap = hstar-predicted_b
        predicted_t = predicted_gap / np.sqrt((1-P)*R0**2/2 + P*(1-P)*predicted_gap**2)
        row = dict(meta, step=step, w1=float(theta[i, 0]), r=float(theta[i, 1]), b=float(theta[i, 2]), c=float(theta[i, 3]),
                   h=float(z[0]), D_gap=float(gap), body_sd=float(abs(theta[i, 1])/np.sqrt(2)),
                   mixture_mean=mean, mixture_median=float(median), mixture_sd=float(sd),
                   T=float((z.max()-median)/sd), T_formula=float(gap/theoretical_sd), pplus=float(MASS @ (z>0)),
                   body_max=float(z[1:].max()), body_min=float(z[1:].min()), physical_branches_valid=physical,
                   loss=float(info["loss"][i]), max_abs_probability_residual=float(np.max(np.abs(info["error"][i]))),
                   max_abs_logit_residual=float(np.max(np.abs(info["logits"][i]-TARGET_LOGITS))),
                   open_logit=float(info["logits"][i,0]), body_logit_min=float(info["logits"][i,1:].min()),
                   body_logit_max=float(info["logits"][i,1:].max()),
                   logit_gap_to_weighted_body=float(info["logits"][i,0]-np.array([.25,.5,.25])@info["logits"][i,1:]),
                   grad_w1=float(gradient[i,0]), grad_r=float(gradient[i,1]), grad_b=float(gradient[i,2]), grad_c=float(gradient[i,3]),
                   gradient_w1_minus_d_b=float(gradient[i,0]-D*gradient[i,2]),
                   invariant_w1_minus_b_error=float((theta[i,0]-theta[i,2])-(initial[i,0]-initial[i,2])),
                   r_change=float(theta[i,1]-R0), saturated_hstar=float(hstar),
                   saturated_h_error=float(z[0]-hstar), saturated_cstar=float(TARGET_LOGITS[1]+meta["gain"]),
                   saturated_adam_eps0_predicted_T=float(predicted_t),
                   saturated_adam_eps0_predicted_physical=bool(hstar>0 and predicted_b+abs(R0)<0))
        result.append(row)
    return result


def implementation_check():
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    class ExactELU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z):
            ctx.save_for_backward(z)
            return torch.where(z>0, z, torch.expm1(z.clamp_max(0)))
        @staticmethod
        def backward(ctx, g):
            (z,) = ctx.saved_tensors
            return g * torch.exp(z.clamp_max(0))
    theta=np.array([[(H0+12)/D,R0,-12.,C0]])
    gain=np.array([.3]); flag=np.array([True]); epsilon=np.array([1e-8])
    tp=torch.tensor(theta[0],requires_grad=True); tx=torch.tensor(X); tw=torch.tensor(MASS); tq=torch.tensor(TARGET)
    optim=torch.optim.Adam([tp],lr=LR,betas=(BETA1,BETA2),eps=1e-8)
    def loss_t():
        z=tx@tp[:2]+tp[2]
        logits=.3*ExactELU.apply(z)+tp[3]
        return (tw*(torch.nn.functional.softplus(logits)-tq*logits)).sum()
    grad,_=forward_and_gradient(theta,gain,flag)
    loss_t().backward()
    gradient_error=float(np.max(np.abs(grad[0]-tp.grad.numpy())))
    fd_error=0.
    for j in range(4):
        plus=theta.copy();minus=theta.copy();plus[0,j]+=1e-5;minus[0,j]-=1e-5
        fd=(forward_and_gradient(plus,gain,flag)[1]["loss"][0]-forward_and_gradient(minus,gain,flag)[1]["loss"][0])/2e-5
        fd_error=max(fd_error,float(abs(fd-grad[0,j])))
    first=np.zeros_like(theta);second=np.zeros_like(theta);max_error=0.
    first_delta=None
    for step in range(1,129):
        grad,_=forward_and_gradient(theta,gain,flag)
        delta,_=adam_step(theta,grad,first,second,step,epsilon)
        optim.zero_grad();loss_t().backward();optim.step()
        max_error=max(max_error,float(np.max(np.abs(theta[0]-tp.detach().numpy()))))
        if step==1:first_delta=delta[0].tolist()
    check={"case":{"activation":"exact_elu","gain":.3,"b0":-12.,"epsilon":1e-8},
           "torch":torch.__version__,"gradient_max_absolute_error":gradient_error,
           "finite_difference_max_absolute_error":fd_error,"adam_128_steps_max_parameter_error":max_error,
           "first_step_delta":first_delta,
           "first_moment_max_error":float(np.max(np.abs(first[0]-optim.state[tp]["exp_avg"].numpy()))),
           "second_moment_max_error":float(np.max(np.abs(second[0]-optim.state[tp]["exp_avg_sq"].numpy())))}
    assert gradient_error<1e-12 and fd_error<1e-8 and max_error<1e-10,check
    return check


def run(out):
    metadata,theta=design(); initial=theta.copy()
    gain=np.array([m["gain"] for m in metadata]);epsilon=np.array([m["epsilon"] for m in metadata])
    flag=np.array([m["activation"]=="exact_elu" for m in metadata]);sat=~flag
    config={"model":"one shared neuron z=w1*x1+r*x2+b; train w1,r,b,c; fixed positive readout a",
            "d":D,"p":P,"qA":QA,"qB":QB,"h0":H0,"r0":R0,"c0":C0,
            "inputs":X.tolist(),"masses":MASS.tolist(),"targets":TARGET.tolist(),
            "parameter_order":["w1","r","b","c"],"arms":metadata,"initial_parameters":initial.tolist(),
            "learning_rate":LR,"betas":[BETA1,BETA2],"steps":STEPS,"checkpoints":CHECKPOINTS,"dtype":"float64",
            "exact_elu_backward":"exp(z) on negative branch, 1 on positive branch",
            "zero_denominator":"zero moments / zero gradient uses zero update, the eps->0+ limit",
            "weighted_median":"first ordered atom with cumulative mass >= 0.5",
            "interpretation":"Finite trajectory check only; constant-step Adam convergence is not asserted.",
            "code_sha256":sha(__file__),"frozen_before_updates":True}
    (out/"adam_config.json").write_text(json.dumps(config,indent=2)+"\n")
    checks=implementation_check()
    first=np.zeros_like(theta);second=np.zeros_like(theta)
    grad,info=forward_and_gradient(theta,gain,flag)
    trajectories=measured_rows(metadata,theta,initial,0,grad,info)
    first_grad=grad.copy();first_delta=None
    audit={
        "gradient_proportionality_absmax":np.zeros(len(metadata)),
        "epsilon0_equal_hidden_step_absmax":np.zeros(len(metadata)),
        "epsilon_corrected_step_ratio_abs_error_max":np.zeros(len(metadata)),
        "epsilon_corrected_D_h_relation_abs_error_max":np.zeros(len(metadata)),
        "r_displacement_absmax":np.zeros(len(metadata)),
        "invariant_w1_minus_b_absmax":np.zeros(len(metadata)),
        "raw_gradient_r_absmax":np.zeros(len(metadata)),
        "r_step_absmax":np.zeros(len(metadata)),
        "r_step_absmax_when_gradient_lt_1e_minus8":np.zeros(len(metadata)),
        "first_physical_branch_violation_step":np.full(len(metadata),-1,dtype=int),
        "physical_branch_violation_steps":np.zeros(len(metadata),dtype=int),
    }
    for step in range(1,STEPS+1):
        grad,info=forward_and_gradient(theta,gain,flag)
        delta,rms=adam_step(theta,grad,first,second,step,epsilon)
        if step==1:first_delta=delta.copy()
        rb=rms[:,2]
        rho=np.divide(D*(rb+epsilon),D*rb+epsilon,out=np.ones_like(rb),where=D*rb+epsilon>0)
        kappa=D*rho/(D*rho+1)
        audit["gradient_proportionality_absmax"]=np.maximum(audit["gradient_proportionality_absmax"],abs(grad[:,0]-D*grad[:,2]))
        audit["epsilon0_equal_hidden_step_absmax"]=np.maximum(audit["epsilon0_equal_hidden_step_absmax"],abs(delta[:,0]-delta[:,2]))
        audit["epsilon_corrected_step_ratio_abs_error_max"]=np.maximum(audit["epsilon_corrected_step_ratio_abs_error_max"],abs(delta[:,0]-rho*delta[:,2]))
        audit["epsilon_corrected_D_h_relation_abs_error_max"]=np.maximum(audit["epsilon_corrected_D_h_relation_abs_error_max"],abs(D*delta[:,0]-kappa*(D*delta[:,0]+delta[:,2])))
        audit["r_displacement_absmax"]=np.maximum(audit["r_displacement_absmax"],abs(theta[:,1]-R0))
        audit["invariant_w1_minus_b_absmax"]=np.maximum(audit["invariant_w1_minus_b_absmax"],abs((theta[:,0]-theta[:,2])-(initial[:,0]-initial[:,2])))
        audit["raw_gradient_r_absmax"]=np.maximum(audit["raw_gradient_r_absmax"],abs(grad[:,1]))
        audit["r_step_absmax"]=np.maximum(audit["r_step_absmax"],abs(delta[:,1]))
        tiny=abs(grad[:,1])<1e-8
        audit["r_step_absmax_when_gradient_lt_1e_minus8"]=np.maximum(audit["r_step_absmax_when_gradient_lt_1e_minus8"],np.where(tiny,abs(delta[:,1]),0.))
        z=theta[:,:2]@X.T+theta[:,2:3]
        valid=(z[:,0]>0)&(z[:,1:].max(axis=1)<0)
        first_bad=(~valid)&(audit["first_physical_branch_violation_step"]<0)
        audit["first_physical_branch_violation_step"][first_bad]=step
        audit["physical_branch_violation_steps"]+=(~valid)
        if step in CHECKPOINTS:
            grad_after,info_after=forward_and_gradient(theta,gain,flag)
            trajectories.extend(measured_rows(metadata,theta,initial,step,grad_after,info_after))
    assert np.all(np.isfinite(theta)) and np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    assert audit["gradient_proportionality_absmax"][sat].max()<1e-12
    assert audit["r_displacement_absmax"][sat].max()==0
    assert audit["epsilon0_equal_hidden_step_absmax"][sat&(epsilon==0)].max()<1e-14
    assert audit["epsilon_corrected_step_ratio_abs_error_max"][sat].max()<1e-14
    audit_rows=[]
    for i,meta in enumerate(metadata):
        audit_rows.append(dict(meta,**{k:float(v[i]) for k,v in audit.items()},
                               first_raw_gradient_r=float(first_grad[i,1]),first_update_r=float(first_delta[i,1]),
                               first_gradient_w1=float(first_grad[i,0]),first_gradient_b=float(first_grad[i,2]),
                               first_update_w1=float(first_delta[i,0]),first_update_b=float(first_delta[i,2])))
    write_csv(out/"adam_trajectories.csv",trajectories)
    write_csv(out/"adam_endpoints.csv",[r for r in trajectories if r["step"]==STEPS])
    write_csv(out/"adam_audits.csv",audit_rows)
    np.savez_compressed(out/"adam_final_states.npz",theta=theta,first_moment=first,second_moment=second,gains=gain,epsilons=epsilon,exact_elu=flag,initial=initial)
    checks["all_arms_finite"]=True
    checks["saturated_gradient_proportionality_max"]=float(audit["gradient_proportionality_absmax"][sat].max())
    checks["saturated_epsilon0_equal_step_max"]=float(audit["epsilon0_equal_hidden_step_absmax"][sat&(epsilon==0)].max())
    checks["saturated_epsilon_corrected_ratio_error_max"]=float(audit["epsilon_corrected_step_ratio_abs_error_max"][sat].max())
    (out/"adam_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    provenance={"code_sha256":sha(__file__),"git_hash":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "numpy":np.__version__,"python":platform.python_version(),"config_sha256":sha(out/"adam_config.json"),
                "arms":len(metadata),"updates_per_arm":STEPS,"scope":"Registered small deterministic finite-trajectory check; no original RL-MNIST inference."}
    (out/"adam_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(checks,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=Path("results/two_group_logistic_0929"));args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args.out)


if __name__=="__main__":
    main()
