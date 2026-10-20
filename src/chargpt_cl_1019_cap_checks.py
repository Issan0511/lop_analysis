#!/usr/bin/env python3
"""cap 腕・SiLU 腕（下見）の走る前の検査。変異対照つき。
  python src/chargpt_cl_1019_cap_checks.py --out results/chargpt_cl_1019/cap_checks
1. S-silu   : SiLU の φ′ が自動微分と一致。変異（φ′ = σ(z)）は外れる。
2. S-base   : C3 をこのモジュールで回しても本走モジュールとビット一致（短い 2 課題）。
3. S-cap1   : 課題 1 の間は SiLUcap と SiLU がビット一致（cap は課題 2 から）。
4. S-cap    : 課題 2・3 の末で cap = √V(課題 1 末)、全 unit で √V ≤ cap、縛っている unit がある。
              変異対照: cap なしの SiLU は課題 3 末の probe の幅が 1.2 cap を超える unit があり、SiLUcap には無い。
5. S-vtrack : 縛ったあとも √V が probe で測った幅と同じ桁（V を直さないと縮めすぎる）。
6. S-ckpt   : CHARGPT_DECOMP_CKPT=1 で u100_task{t}.pt と課題末の checkpoint が出る。
"""
import argparse, json, os, subprocess, sys, csv
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019_kune as K
C = K.C
PY = sys.executable
SRC = Path(__file__).resolve().parent


def runcli(mod, arm, out, tasks, n, env=None):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([PY, str(SRC / mod), "run", "--box", "M", "--arm", arm, "--seed", "1000", "--tasks", str(tasks),
                        "--n_per_task", str(n), "--alpha", "0.01", "--out", str(out)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); R = {}
    # 1
    z = torch.linspace(-12, 8, 4001, dtype=torch.float64, requires_grad=True)
    torch.nn.functional.silu(z).sum().backward()
    m = K.MLPKune("SiLU").double()
    err = float((m.dphi(z.detach()) - z.grad).abs().max())
    mut = float((torch.sigmoid(z.detach()) - z.grad).abs().max())
    R["S-silu"] = dict(err=err, mutant_err=mut, ok=err < 1e-12 and mut > 1e-3)
    # 2
    runcli("chargpt_cl_1019.py", "C3", out / "runs/base_prod", 2, 300)
    runcli("chargpt_cl_1019_kune.py", "C3", out / "runs/base_kune", 2, 300)
    same = all(open(out / f"runs/base_prod/{f}", "rb").read() == open(out / f"runs/base_kune/{f}", "rb").read() for f in ("per_task.csv", "online.npz"))
    R["S-base"] = dict(ok=same)
    # 3・4・5・6
    runcli("chargpt_cl_1019_kune.py", "SiLU", out / "runs/silu", 3, 300, env=dict(CHARGPT_DECOMP_CKPT="1"))
    runcli("chargpt_cl_1019_kune.py", "SiLUcap", out / "runs/silucap", 3, 300, env=dict(CHARGPT_DECOMP_CKPT="1"))
    on_s = np.load(out / "runs/silu/online.npz")["loss"]; on_c = np.load(out / "runs/silucap/online.npz")["loss"]
    R["S-cap1"] = dict(task1_equal=bool((on_s[0] == on_c[0]).all()), task2_differs=bool((on_s[1] != on_c[1]).any()))
    R["S-cap1"]["ok"] = R["S-cap1"]["task1_equal"] and R["S-cap1"]["task2_differs"]
    ck = lambda run, name: torch.load(out / f"runs/{run}/ckpts/{name}.pt", map_location="cpu")
    c1, c2, c3 = ck("silucap", "task1"), ck("silucap", "task2"), ck("silucap", "task3")
    s3 = ck("silu", "task3")
    probe3 = C.task_cpu_M(1000, 3, 10, 0.01)["probe"]

    def probe_sd(arm, sd_):
        mm = C.build_model(arm, "M", 1000, "cpu"); mm.load_state_dict(sd_); mm.eval(); cc = []
        with torch.no_grad(): mm.body(probe3, cc)
        return [cc[2 * l + 1].reshape(-1, C.FF).double().std(0) for l in range(C.L)]
    psd_mut, psd_cap = probe_sd("SiLU", s3), probe_sd("SiLUcap", c3)
    rows = []
    for l in range(C.L):
        cap_set = float((c2[f"h.{l}.mlp.cap"] - c1[f"h.{l}.mlp.V"].sqrt()).abs().max())
        over = max(float((c[f"h.{l}.mlp.V"].sqrt() / c[f"h.{l}.mlp.cap"]).max()) for c in (c2, c3))
        bound = float((c3[f"h.{l}.mlp.V"].sqrt() >= c3[f"h.{l}.mlp.cap"] * (1 - 1e-4)).double().mean())
        capv = c2[f"h.{l}.mlp.cap"].double()
        mut_over = float((psd_mut[l] > 1.2 * capv).double().mean()); cap_over = float((psd_cap[l] > 1.2 * capv).double().mean())
        rows.append(dict(layer=l, cap_minus_sqrtV_task1=cap_set, max_s_over_cap=over, frac_bound_task3=bound,
                         mutant_frac_probe_over_1p2cap=mut_over, capped_frac_probe_over_1p2cap=cap_over))
    ok4 = all(r["cap_minus_sqrtV_task1"] < 1e-6 and r["max_s_over_cap"] <= 1 + 1e-5 for r in rows) and any(r["frac_bound_task3"] > 0 for r in rows) \
        and any(r["mutant_frac_probe_over_1p2cap"] > 0.05 for r in rows) and all(r["capped_frac_probe_over_1p2cap"] < 0.02 for r in rows)
    R["S-cap"] = dict(rows=rows, ok=ok4)
    # 5: 課題 3 末の網で probe の幅を測り、√V と比べる
    mdl = C.build_model("SiLUcap", "M", 1000, "cpu"); mdl.load_state_dict(c3); mdl.eval()
    probe = C.task_cpu_M(1000, 3, 10, 0.01)["probe"]; cap_ = []
    with torch.no_grad(): mdl.body(probe, cap_)
    ratios = []
    for l, blk in enumerate(mdl.h):
        sd = cap_[2 * l + 1].reshape(-1, C.FF).double().std(0)
        ratios.append(float((blk.mlp.V.double().sqrt() / sd).median()))
    R["S-vtrack"] = dict(sqrtV_over_probe_sd_median=ratios, ok=all(0.5 < x < 2.0 for x in ratios))
    names = sorted(p.name for p in (out / "runs/silucap/ckpts").glob("*.pt"))
    need = [f"u100_task{t}.pt" for t in (1, 2, 3)] + [f"task{t}.pt" for t in (1, 2, 3)]
    R["S-ckpt"] = dict(found=names, ok=all(n in names for n in need))
    R["all_ok"] = all(v["ok"] for k, v in R.items() if isinstance(v, dict) and "ok" in v)
    (out / "cap_checks.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    for k, v in R.items():
        print(k, json.dumps(v, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
