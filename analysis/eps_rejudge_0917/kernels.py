#!/usr/bin/env python3
"""eps_rejudge_0917 section 1.2 (K1-K4) and check S0: the float32 floors of this machine.

    OMP_NUM_THREADS=1 python3 analysis/eps_rejudge_0917/kernels.py

K1/K2  largest float32 z whose training factor of where(z>0, z, expm1(min(z,0))) is exactly 0,
       CPU (flush off and on) and CUDA, found on the float32 grid with nextafter.
K3     largest float32 z whose exp is exactly 0 (the hand-written gate of the GPU lineage).
K4     Adam's moments under a zero gradient, stepped exactly as the host does
       (m.mul_(b1).add_(0, alpha=1-b1); v.mul_(b2).addcmul_(0, 0, value=1-b2)):
       where they stop without flush, and after how many steps they reach 0 with flush.
S0     K1 equals the value resp_ee_0917 measured; the next float32 up has factor 2**-24; the mutant
       factor exp(z) (the GPU lineage's gate) has no zero above -87.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "eps_rejudge_0917"
F32 = torch.float32
TINY = 2.0 ** -149
K1_EXPECTED = -16.635534286499023          # resp_ee_0917 section 5 / src/resp_ee_0917.py ZERO_Z32 comment


def factor_autograd(z: torch.Tensor) -> torch.Tensor:
    zz = z.detach().clone().requires_grad_(True)
    y = torch.where(zz > 0, zz, torch.expm1(zz.clamp(max=0.0)))
    g, = torch.autograd.grad(y.sum(), zz)
    return g


def factor_exp(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z > 0, torch.ones_like(z), z.clamp_max(0).exp())


def up(t: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(t, torch.zeros_like(t))


def largest_zero(fn, dev: str, lo: float, hi: float, n: int) -> dict:
    z = torch.linspace(lo, hi, n, dtype=F32, device=dev)
    zero = fn(z) == 0
    if not bool(zero.any()):
        return {"largest_zero_z": None}
    t = z[torch.nonzero(zero).flatten().max()].reshape(1)
    for _ in range(100000):                         # walk up the float32 grid to the last zero
        u = up(t)
        if float(fn(u).item()) != 0.0:
            break
        t = u
    u = up(t)
    return {"largest_zero_z": float(t.item()), "next_z": float(u.item()),
            "factor_at_next": float(fn(u).item())}


def moment_decay(beta: float, start: float, flush: bool, square: bool, max_steps: int = 2_000_000) -> dict:
    torch.set_flush_denormal(flush)
    x = torch.tensor([start], dtype=F32)
    g = torch.zeros(1, dtype=F32)
    steps = 0
    prev = None
    while steps < max_steps:
        if square:
            x.mul_(beta).addcmul_(g, g, value=1 - beta)
        else:
            x.mul_(beta).add_(g, alpha=1 - beta)
        steps += 1
        cur = x.clone()
        if float(cur.item()) == 0.0:
            break
        if prev is not None and torch.equal(cur, prev):
            break
        prev = cur
    torch.set_flush_denormal(False)
    val = x.double().item()
    return {"beta": beta, "start": start, "flush": flush, "steps": steps, "final": val,
            "final_in_tiny_quanta": val / TINY, "reached_zero": val == 0.0}


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
           "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "ln2m24": math.log(2.0 ** -24), "ln2m25": math.log(2.0 ** -25), "ln2m149": math.log(TINY)}
    k = {}
    for flush in (False, True):
        torch.set_flush_denormal(flush)
        tag = "cpu_flush" if flush else "cpu"
        k[f"K1_{tag}"] = largest_zero(factor_autograd, "cpu", -40.0, -10.0, 3_000_001)
        k[f"K3_{tag}"] = largest_zero(factor_exp, "cpu", -110.0, -80.0, 6_000_001)
        k[f"S0mut_{tag}"] = largest_zero(factor_exp, "cpu", -80.0, -1.0, 7_900_001)
    torch.set_flush_denormal(False)
    if torch.cuda.is_available():
        k["K2_cuda"] = largest_zero(factor_autograd, "cuda", -40.0, -10.0, 3_000_001)
        k["K3_cuda"] = largest_zero(factor_exp, "cuda", -110.0, -80.0, 6_000_001)
    # K4: m starts at a typical first-moment size, v at its square
    k["K4"] = [moment_decay(0.9, 1e-6, False, False), moment_decay(0.9, 1e-6, True, False),
               moment_decay(0.999, 1e-12, False, True), moment_decay(0.999, 1e-12, True, True)]
    res["kernels"] = k
    # S0
    k1 = k["K1_cpu"]
    s0 = {
        "K1_matches_resp_ee": k1["largest_zero_z"] == K1_EXPECTED,
        "K1_next_factor_is_2m24": k1["factor_at_next"] == 2.0 ** -24,
        "K1_flush_same": k["K1_cpu_flush"]["largest_zero_z"] == K1_EXPECTED,
        "mutant_exp_has_no_zero_above_m80": k["S0mut_cpu"]["largest_zero_z"] is None
        and k["S0mut_cpu_flush"]["largest_zero_z"] is None,
        "K4_m_sticks_without_flush": (not k["K4"][0]["reached_zero"]) and k["K4"][0]["final_in_tiny_quanta"] <= 4.5,
        "K4_m_zero_with_flush": k["K4"][1]["reached_zero"],
        "K4_v_zero_with_flush": k["K4"][3]["reached_zero"],
    }
    s0["pass"] = all(s0.values())
    res["S0"] = s0
    (OUT / "kernels.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not s0["pass"]:
        sys.exit("S0 failed (kernels.json written)")


if __name__ == "__main__":
    main()
