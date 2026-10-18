#!/usr/bin/env python3
"""sna_grow_1017 -- registered mechanism readout (spec §4.1 副（機構）, REPORT_ONLY).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python analysis/sna_grow_1017/mechanism.py

From the task-20 checkpoints of this run (results/sna_grow_1017/cnn/ckpt/<arm>_seed<s>_t20.pt, ':' -> '_'),
replay task 21 and record, at 0 / 10 / 75 updates and 3 / 5 / 10 (and 2, 25, 50, 100, 400) epochs, the mean
slope of every site: mean over all inputs (1200 images x positions x channels, or x units) of
1 + a sin(2 alpha z) (a = 1 in both arms).  Also, on the same trajectory, the slope with alpha held at its value at
the switch (the dist_anim/chase_cf counterfactual), the median alpha / alpha_switch, the median unit width and the
CE / accuracy on the task-21 labels.

Replay: all 20 slots of the run, in the run's slot order (provenance.json), are rebuilt with
`src/sna_grow_1017_cnn.load_runs` (install()ed BundleSnakeGrow; V, V_max, grow_on, ada, cval, fixA from each
checkpoint, verified against the grow rule; the label and batch generators advanced through task 20 exactly as
the template `dist_anim/extract.py` part2 does).  Same R and order as the run, so the replay is the run's own
trajectory; this is checked at the end against the task-21 row of per_task.csv (switch_ce before training,
online_acc / ep_first99 / acc_ep1 / acc_ep10 and every end-of-task evaluation column after 400 epochs).
Epoch 1 is stepped eagerly (to record at 10 and 75 updates); epochs 2-400 replay the CUDA graph (K-cnn-graph:
graph and eager are bit-identical).  The registered summary is the mean over seeds 10-12 (as the template);
seeds 10-19 are given as well.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

from src import pmnist_0905 as H                     # noqa: E402
from src import sna_grow_1017_cnn as G               # noqa: E402

E = G.E
RES = ROOT / "results" / "sna_grow_1017"
CNN = RES / "cnn"
OUT = RES / "cnn_mechanism.json"
TASK = 20                                            # checkpoint; the replayed task is TASK + 1
EPOCHS = 400
SITES = ("c1", "c2", "f1", "f2")
MAIN_SEEDS = (10, 11, 12)
UPD_EP1 = (10, 75)
REC_EP = (2, 3, 5, 10, 25, 50, 100, 400)
KNOWN = Path("/tmp/claude-1000/-home-issan-Projects-claude/ecf63120-82f2-4867-ade7-17734283f67e/"
             "scratchpad/dist_anim/chase_cf.json")    # sna_cnn_cause_1009 bundle A, SNA, seeds 10-12 (read only)


def ckpt_path(arm: str, seed: int, task: int) -> Path:
    return CNN / "ckpt" / f"{arm.replace(':', '_')}_seed{seed}_t{task:02d}.pt"


@torch.no_grad()
def record(B, a_sw, label, step):
    """Per run: mean slope (actual alpha), mean slope (alpha at the switch), median alpha/alpha_switch,
    median unit sd of z, CE and accuracy on B.Y (the task-21 labels)."""
    R, dev = B.R, B.device
    g = torch.zeros(R, 4, dtype=torch.float64, device=dev)
    gcf = torch.zeros(R, 4, dtype=torch.float64, device=dev)
    cnt = torch.zeros(R, 4, dtype=torch.float64, device=dev)
    s1 = [torch.zeros(R, w, dtype=torch.float64, device=dev) for w in E.WIDTHS]
    s2 = [torch.zeros(R, w, dtype=torch.float64, device=dev) for w in E.WIDTHS]
    nz = [0, 0, 0, 0]
    ce = torch.zeros(R, dtype=torch.float64, device=dev)
    hit = torch.zeros(R, dtype=torch.float64, device=dev)
    assert B.act.amp is None                          # plain Snake in both arms (a = 1)
    alphas = [B.act.alpha(l) for l in range(4)]
    for i0 in range(0, E.N_IMAGES, 50):
        out = E.forward(B.P, B.X[:, i0:i0 + 50], B.act)
        for l in range(4):
            z = out[2 * l]
            a, a0 = alphas[l], a_sw[l]
            if E.IS_CONV[l]:
                b, C, Hh, Ww = z.shape
                c = C // R
                sh = lambda t: t.reshape(1, C, 1, 1)                       # noqa: E731
                g[:, l] += (1 + torch.sin(2 * sh(a) * z)).reshape(b, R, -1).double().sum((0, 2))
                gcf[:, l] += (1 + torch.sin(2 * sh(a0) * z)).reshape(b, R, -1).double().sum((0, 2))
                cnt[:, l] += b * c * Hh * Ww
                zr = z.reshape(b, R, c, Hh * Ww).double()
                s1[l] += zr.sum((0, 3))
                s2[l] += (zr * zr).sum((0, 3))
                nz[l] += b * Hh * Ww
            else:
                _, b, n = z.shape
                g[:, l] += (1 + torch.sin(2 * a[:, None, :] * z)).double().sum((1, 2))
                gcf[:, l] += (1 + torch.sin(2 * a0[:, None, :] * z)).double().sum((1, 2))
                cnt[:, l] += b * n
                s1[l] += z.double().sum(1)
                s2[l] += (z.double() ** 2).sum(1)
                nz[l] += b
        logits = out[8]
        yb = B.Y[:, i0:i0 + 50]
        lp = torch.log_softmax(logits.double(), -1)
        ce += -lp.gather(-1, yb[..., None]).squeeze(-1).sum(1)
        hit += (logits.argmax(-1) == yb).double().sum(1)
    g, gcf = (g / cnt).cpu(), (gcf / cnt).cpu()
    runs = []
    for r, (arm, seed) in enumerate(B.slots):
        d = dict(arm=arm, seed=seed, ce=float(ce[r]) / E.N_IMAGES, acc=float(hit[r]) / E.N_IMAGES)
        for l, tag in enumerate(SITES):
            m = s1[l][r] / nz[l]
            sd = (s2[l][r] / nz[l] - m * m).clamp_min(0).sqrt()
            d[tag] = dict(slope=float(g[r, l]), slope_alpha_at_switch=float(gcf[r, l]),
                          alpha_ratio_med=float((alphas[l][r] / a_sw[l][r]).double().median()),
                          alpha_med=float(alphas[l][r].double().median()), zsd_med=float(sd.median()),
                          two_alpha_sd_med=float((2 * alphas[l][r].double() * sd).median()))
        runs.append(d)
    return dict(label=label, step=step, runs=runs)


def summarise(frame, seeds):
    out = {}
    for arm in sorted({r["arm"] for r in frame["runs"]}):
        rs = [r for r in frame["runs"] if r["arm"] == arm and r["seed"] in seeds]
        assert len(rs) == len(seeds), (arm, seeds)
        out[arm] = {tag: {k: float(np.mean([r[tag][k] for r in rs])) if k.startswith("slope")
                          else float(np.median([r[tag][k] for r in rs]))
                          for k in ("slope", "slope_alpha_at_switch", "alpha_ratio_med", "zsd_med", "two_alpha_sd_med")}
                    for tag in SITES}
        out[arm]["ce_mean"] = float(np.mean([r["ce"] for r in rs]))
        out[arm]["acc_mean"] = float(np.mean([r["acc"] for r in rs]))
    return out


def fmt10(x):
    return f"{x:.10g}" if isinstance(x, float) else str(x)


def compare_rows(rows_run, rows_now):
    """Formatted (.10g, as the run's CSV writer) comparison; returns the mismatching (slot, column)."""
    bad = []
    for (key, ref), now in zip(rows_run, rows_now):
        for k, v in now.items():
            if k not in ref:
                continue
            if fmt10(v) != ref[k]:
                bad.append(dict(slot=key, column=k, run=ref[k], replay=fmt10(v)))
    return bad


def main():
    t_start = time.time()
    device = H.setup("cuda")
    prov = json.loads((CNN / "provenance.json").read_text())
    slots = [tuple(s) for s in prov["slots"]]
    assert prov["epochs"] == EPOCHS
    paths = [ckpt_path(a, s, TASK) for a, s in slots]
    for p in paths:
        assert p.exists(), p
    B, task, ver = G.load_runs(paths, device, graph=True, epochs=EPOCHS)
    assert task == TASK and B.slots == slots
    R = B.R
    print(f"[{time.strftime('%T')}] loaded R={R} at task {task}: {ver}", flush=True)
    per_task = {(r["arm"], int(r["seed"]), int(r["task"])): r for r in G.read_rows(CNN / "per_task.csv")}

    # ---- task 21, following E.run
    B.new_labels()
    sce, sacc = E.switch_eval(B.P, B.X, B.Y, B.act)
    a_sw = [B.act.alpha(l).clone() for l in range(4)]
    frames = [record(B, a_sw, "0 updates (labels switched, before the first step)", 0)]
    ep_acc = torch.zeros(R, EPOCHS, device=device)
    B.new_order()
    B.acc_ep.zero_()
    for j in range(E.SPE):
        B.step(j)
        if j + 1 in UPD_EP1:
            frames.append(record(B, a_sw, f"{j + 1} updates", j + 1))
    ep_acc[:, 0].copy_(B.acc_ep)
    print(f"[{time.strftime('%T')}] epoch 1 done", flush=True)
    for e in range(2, EPOCHS + 1):
        B.run_epoch()
        ep_acc[:, e - 1].copy_(B.acc_ep)
        if e in REC_EP:
            frames.append(record(B, a_sw, f"{e} epochs", e * E.SPE))
            s = summarise(frames[-1], MAIN_SEEDS)
            print(f"[{time.strftime('%T')}] epoch {e}: " + "  ".join(
                f"{a} f1 {s[a]['f1']['slope']:.3f} f2 {s[a]['f2']['slope']:.3f} c2 {s[a]['c2']['slope']:.3f}"
                for a in s), flush=True)
            torch.cuda.empty_cache()

    # ---- the replay against the run's task-21 row
    ep_acc /= E.SPE
    online = ep_acc.double().mean(1)
    ev = E.evaluate(B.P, B.X, B.Y, B.act)
    curves = ep_acc.cpu().numpy()
    rows_now, rows_run = [], []
    for r, (arm, seed) in enumerate(slots):
        cur = curves[r]
        hit99 = np.nonzero(cur >= 0.99)[0]
        rows_now.append({"online_acc": float(online[r]), "switch_ce": float(sce[r]), "switch_acc": float(sacc[r]),
                         "ep_first99": int(hit99[0]) + 1 if hit99.size else -1,
                         "acc_ep1": float(cur[0]), "acc_ep10": float(cur[9]), **ev[r]})
        rows_run.append(((arm, seed), per_task[(arm, seed, TASK + 1)]))
    bad = compare_rows(rows_run, rows_now)
    n_cmp = sum(len([k for k in now if k in ref]) for (_, ref), now in zip(rows_run, rows_now))
    replay_check = dict(columns_compared=n_cmp, mismatches=len(bad), first_mismatches=bad[:10],
                        reproduces_run_task21=not bad,
                        switch_ce_match=all(fmt10(float(sce[r])) == rows_run[r][1]["switch_ce"] for r in range(R)))
    print(f"[{time.strftime('%T')}] replay vs run task 21: {n_cmp} values compared, {len(bad)} mismatches", flush=True)

    # ---- summaries and the table
    summ = {f"seeds_{MAIN_SEEDS[0]}-{MAIN_SEEDS[-1]}": [dict(label=f["label"], step=f["step"], **summarise(f, MAIN_SEEDS))
                                                      for f in frames],
            "seeds_10-19": [dict(label=f["label"], step=f["step"], **summarise(f, tuple(range(10, 20))))
                            for f in frames]}
    known = None
    if KNOWN.exists():
        kk = json.loads(KNOWN.read_text())
        known = [dict(label=r["label"], **{site: dict(slope=r["SNA"][site][0], slope_alpha_at_switch=r["SNA"][site][1])
                                          for site in SITES}) for r in kk]
    arms = sorted({a for a, s in slots})
    show = ("0 updates", "75 updates", "3 epochs", "5 epochs", "10 epochs")
    sel = [f for f in summ[f"seeds_{MAIN_SEEDS[0]}-{MAIN_SEEDS[-1]}"] if f["label"].split(" (")[0] in show]
    L = ["Mean slope (mean over all inputs of 1 + sin(2 alpha z)), task 20 -> 21 replay, mean over seeds 10-12.", "",
         "| frame | " + " | ".join(f"{a} {s}" for a in arms for s in ("c2", "f1", "f2")) + " |",
         "|---|" + "---:|" * (3 * len(arms))]
    for f in sel:
        L.append(f"| {f['label'].split(' (')[0]} | " + " | ".join(f"{f[a][s]['slope']:.3f}" for a in arms for s in ("c2", "f1", "f2")) + " |")
    L += ["", "Same trajectory, alpha held at its value at the switch (counterfactual, report only):", "",
          "| frame | " + " | ".join(f"{a} {s}" for a in arms for s in ("c2", "f1", "f2")) + " |",
          "|---|" + "---:|" * (3 * len(arms))]
    for f in sel:
        L.append(f"| {f['label'].split(' (')[0]} | " + " | ".join(f"{f[a][s]['slope_alpha_at_switch']:.3f}" for a in arms for s in ("c2", "f1", "f2")) + " |")
    L += ["", "Median alpha / alpha_at_switch (median over seeds 10-12 of the unit median):", "",
          "| frame | " + " | ".join(f"{a} {s}" for a in arms for s in ("c2", "f1", "f2")) + " |",
          "|---|" + "---:|" * (3 * len(arms))]
    for f in sel:
        L.append(f"| {f['label'].split(' (')[0]} | " + " | ".join(f"{f[a][s]['alpha_ratio_med']:.2f}" for a in arms for s in ("c2", "f1", "f2")) + " |")
    if known:
        ks = [k for k in known if k["label"] in ("切替の直前", "75 更新", "3 epoch", "5 epoch", "10 epoch")]
        L += ["", "Known (sna_cnn_cause_1009 bundle A, SNA, seeds 10-12; dist_anim/chase_cf.json):", "",
              "| frame | SNA c2 | SNA f1 | SNA f2 |", "|---|---:|---:|---:|"]
        for k in ks:
            L.append(f"| {k['label']} | {k['c2']['slope']:.3f} | {k['f1']['slope']:.3f} | {k['f2']['slope']:.3f} |")
    L += ["", f"Replay vs run (task-21 row of per_task.csv, {n_cmp} values): {len(bad)} mismatches."]
    table_md = "\n".join(L)
    out = dict(run_id="sna_grow_1017", part="cnn mechanism replay (spec §4.1, report only)",
               script="analysis/sna_grow_1017/mechanism.py", generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               checkpoint_task=TASK, replayed_task=TASK + 1, slots=[list(s) for s in slots],
               load_verify=ver, replay_check=replay_check,
               slope_definition="mean over all inputs (images x positions x channels / x units) of 1 + sin(2 alpha z); "
                                "per run, then mean over seeds",
               counterfactual="slope_alpha_at_switch: same trajectory, alpha fixed at its value right after the label "
                              "switch (before the first step)",
               summaries=summ, per_run_frames=frames, known_1009_bundleA_SNA_seeds10_12=known,
               table_md=table_md, seconds=time.time() - t_start,
               cuda_max_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(clean(out), ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    print(table_md, flush=True)
    print(f"-> {OUT}  ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
