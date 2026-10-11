#!/usr/bin/env python3
"""adamw_dose2_1011: measure the float32 lines of native F.elu on the run's device (spec 1.1 (a), 14.4, 14.9).

    python -m analysis.adamw_dose2_1011.lines [--out FILE]

- zf: the largest float32 z with F.elu(z) == -1 (the forward output saturates; CUDA -16.982107, CPU -16.636)
- z0: the largest float32 z whose local derivative d/dz F.elu(z) (autograd, dy = 1) is exactly 0, and the next float32
  above it (the derivative's first non-zero value there): the rule's zero line ln 2^-150 = -103.972 lies between
- z0_dy: the same for dy = 1e-4 and 1e-8 (the product dy * e^z, which is what a parameter gradient carries;
  Codex measured -94.76 and -85.55 on the CPU)
Writes the raw values, the device, torch, CUDA and the driver.  Used by checks.py S-line and by launch.sh.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _f32(x: float) -> float:
    return float(np.float32(x))


def _bisect(pred, lo: float, hi: float) -> float:
    """The largest float32 z in [lo, hi] with pred(z) True, given pred(lo) True and pred(hi) False (monotone)."""
    lo, hi = np.float32(lo), np.float32(hi)
    while True:
        nxt = np.nextafter(lo, np.float32(np.inf))
        if nxt >= hi:
            return float(lo)
        mid = np.float32((float(lo) + float(hi)) / 2)
        if mid <= lo or mid >= hi:
            mid = nxt
        if pred(float(mid)):
            lo = mid
        else:
            hi = mid


def measure(device: str = "cuda") -> dict:
    dev = torch.device(device)

    def fwd_is_minus_one(z: float) -> bool:
        t = torch.tensor([z], dtype=torch.float32, device=dev)
        return F.elu(t, 1.0).item() == -1.0

    def grad_is_zero(z: float, dy: float = 1.0) -> bool:
        t = torch.tensor([z], dtype=torch.float32, device=dev, requires_grad=True)
        g = torch.autograd.grad(F.elu(t, 1.0), t, grad_outputs=torch.tensor([dy], dtype=torch.float32, device=dev))[0]
        return g.item() == 0.0

    zf = _bisect(fwd_is_minus_one, -40.0, -1.0)
    z0 = _bisect(grad_is_zero, -200.0, -50.0)
    z0_up = float(np.nextafter(np.float32(z0), np.float32(np.inf)))
    out = {"device": str(dev), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None,
           "zf": zf, "z0_last_zero": z0, "z0_first_nonzero": z0_up,
           "z0_first_nonzero_grad": None, "ln2_m150": math.log(2.0 ** -150), "ln2_m24": math.log(2.0 ** -24),
           "ln2_m25": math.log(2.0 ** -25), "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    t = torch.tensor([z0_up], dtype=torch.float32, device=dev, requires_grad=True)
    out["z0_first_nonzero_grad"] = torch.autograd.grad(F.elu(t, 1.0).sum(), t)[0].item()
    for dy in (1e-4, 1e-8):
        out[f"z0_dy{dy:g}_last_zero"] = _bisect(lambda z: grad_is_zero(z, dy), -200.0, -50.0)
    try:
        out["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        out["driver"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    r = measure(a.device)
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
