"""band_affine_0903: 帯（0 < p_hat < 1）を秩序変数にする —— 走ゼロの再集計。

事前登録: vault `可塑性喪失/spec/帯とアフィン化_spec_0902.md`（v3・commit b8ed195）§4・§5。
新規の走なし・新規ロガーなし。読むのは既存 committed ログの `layer1_p_hat`（A〜E）と
`layer1_zmax`（G）、および 5M チェックポイント（F）。H（アフィン残差のランク）は実施しない。

窓・U・発症の定義は宿主 gate_dial_0902 / gate_dose_0830 を逐語で使う:
  U^(10)_k = タスク k-9..k の**タスク終端記録**（step % 10000 == 0）の unfit 平均、床 1e-16、
  発症 = U >= 0.05。沈下 = p_hat == 0、浮上 = p_hat == 1、帯 = その間。

    .venv/bin/python -m src.band_affine_0903 --selftest
    .venv/bin/python -m src.band_affine_0903
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, load_config
from .nets import VecMLPL

CONFIG = Path(ROOT) / "configs/gate_dial_0902.yaml"
OUTDIR = Path(ROOT) / "results/band_affine_0903"
DIAL_LOGS = Path(ROOT) / "results/gate_dial_0902/logs"
DOSE_LOGS = Path(ROOT) / "results/gate_dose_0830/logs"
CLAMP_LOGS = Path(ROOT) / "results/valley_clamp_0902/logs"
CLAMP0_LOGS = Path(ROOT) / "results/valley_clamp0_0902/logs"
EXTEND_LOGS = Path(ROOT) / "results/p3_extend_0902/logs"
DOSE_CKPTS = Path(ROOT) / "results/gate_dose_0830/ckpts"

PERIOD = 10_000
N_SEED = 10
N_PAT = 32
BOOT_SEED = 20260907
BOOT_N = 10_000
WINDOWS = {"1M": [91, 100], "5M": [491, 500]}
WINDOW_15M = [1491, 1500]
CONTROLS = ("R_1216", "LR_1216", "E_1216")
# 5M で未収束かつ CAPACITY_UNDEFINED（C7・dial_table.csv）
UNCONVERGED = ("S_b1_1216", "S_b0p3_1216", "G_b0p3_1216")
CLAMP_PAIRS = {  # clamp 腕 -> (元腕, 元腕のログ置き場, clamp 腕のログ置き場)
    "Gc_b1_1216": ("G_b1_1216", DIAL_LOGS, CLAMP_LOGS),
    "Sc_b3_1216": ("S_b3_1216", DIAL_LOGS, CLAMP_LOGS),
    "Gz_b1_1216": ("G_b1_1216", DIAL_LOGS, CLAMP0_LOGS),
    "Sz_b3_1216": ("S_b3_1216", DIAL_LOGS, CLAMP0_LOGS),
}
NMASK_ARMS = CONTROLS


# --------------------------------------------------------------------- 下拵え
def _cfg() -> dict:
    return load_config(str(CONFIG))


def _arm_table() -> list[dict]:
    """14 ダイヤル腕 + 3 対照。閾値は u_star（silu/gelu）または u_fr（elu）。"""
    cfg = _cfg()
    rows = []
    for a in cfg["arms"]:
        u_star, u_fr = a.get("u_star"), a.get("u_fr")
        thr = (float(u_star) if u_star is not None else
               float(u_fr) if u_fr is not None else float("nan"))
        rows.append(dict(arm=str(a["name"]), family=str(a["family"]), dial=float(a["dial"]),
                         is_control=0, logdir=DIAL_LOGS, shallow_thr=thr,
                         thr_kind=("u_star" if u_star is not None
                                   else "u_fr" if u_fr is not None else "none")))
    for name in CONTROLS:
        c = cfg["controls"]["arms"][name]
        u_star, u_fr = c.get("u_star"), c.get("u_fr")
        thr = float(u_fr) if (u_fr not in (None, 0.0)) else (
            float(u_star) if u_star not in (None,) else float("nan"))
        rows.append(dict(arm=name, family=str(c["family"]), dial=float(c["dial"]),
                         is_control=1, logdir=DOSE_LOGS, shallow_thr=thr,
                         thr_kind=("u_fr" if u_fr not in (None, 0.0) else "none")))
    return rows


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _window_idx(steps: np.ndarray, tasks: list[int]) -> np.ndarray:
    task = steps // PERIOD
    return np.flatnonzero((steps > 0) & (steps % PERIOD == 0)
                          & (task >= tasks[0]) & (task <= tasks[1]))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _boot_median_ci(x: np.ndarray, rng) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    draws = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    meds = np.median(x[draws], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _sign_test(x: np.ndarray) -> dict:
    from math import comb
    neg, pos = int((x < 0).sum()), int((x > 0).sum())
    n, k = neg + pos, min(int((x < 0).sum()), int((x > 0).sum()))
    p = (sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2) if n else float("nan")
    return dict(neg=neg, pos=pos, p_two_sided=min(1.0, p))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # 同順位は平均順位
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        for i in np.flatnonzero(cnt > 1):
            m = inv == i
            r[m] = r[m].mean()
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# ------------------------------------------------------------ 1 腕 1 seed の量
def seed_stats(d: dict, tasks: list[int], floor: float, thr_onset: float,
               shallow_thr: float) -> dict:
    """窓内・全記録点×全ユニットを 1 母集団として数える（seed 内の平均）。"""
    idx = _window_idx(d["step"], tasks)
    p = np.asarray(d["layer1_p_hat"], dtype=np.float64)[idx]
    sub, surf = p == 0.0, p == 1.0
    band = ~sub & ~surf
    raw = float(np.asarray(d["unfit"], dtype=np.float64)[idx].mean())
    u = max(raw, floor)
    out = dict(n_records=int(len(idx)), n_units=int(p.shape[1]),
               u=u, log10_u=float(np.log10(u)), onset=int(u >= thr_onset),
               sub_frac=float(sub.mean()), band_frac=float(band.mean()),
               surf_frac=float(surf.mean()),
               surf_among_alive=(float(surf.sum() / (~sub).sum()) if (~sub).any()
                                 else float("nan")),
               band_among_alive=(float(band.sum() / (~sub).sum()) if (~sub).any()
                                 else float("nan")),
               near_off_frac=float((np.abs(p - 1.0 / N_PAT) < 1e-6).mean()),
               near_on_frac=float((np.abs(p - 31.0 / N_PAT) < 1e-6).mean()))
    if "layer1_zmax" in d and np.isfinite(shallow_thr) and shallow_thr > 0:
        zmax = np.asarray(d["layer1_zmax"], dtype=np.float64)[idx]
        shallow = (zmax > -shallow_thr) & (zmax <= 0.0)
        out["shallow_sub_frac"] = float(shallow.mean())
        out["non_affine_frac"] = out["band_frac"] + out["shallow_sub_frac"]
    else:
        out["shallow_sub_frac"] = float("nan")
        out["non_affine_frac"] = float("nan")
    return out


def band_hist(d: dict, tasks: list[int]) -> np.ndarray:
    """帯ユニットの p_hat ヒストグラム（bin = k/32, k = 1..31）。"""
    idx = _window_idx(d["step"], tasks)
    p = np.asarray(d["layer1_p_hat"], dtype=np.float64)[idx]
    k = np.rint(p * N_PAT).astype(int)
    band = (k >= 1) & (k <= N_PAT - 1)
    return np.bincount(k[band], minlength=N_PAT)[1:N_PAT]


def _max_surf_over_history(d: dict) -> float:
    """タスク終端記録の全履歴で見た浮上率の最大（末尾窓に限らない）。"""
    steps = np.asarray(d["step"])
    sel = (steps > 0) & (steps % PERIOD == 0)
    p = np.asarray(d["layer1_p_hat"], dtype=np.float64)[sel]
    return float((p == 1.0).mean(axis=1).max()) if p.size else 0.0


def band_series(d: dict) -> list[dict]:
    """タスク境界の直前（step%P==0）と直後（step%P==1000）の帯率の時系列。"""
    steps = np.asarray(d["step"])
    p = np.asarray(d["layer1_p_hat"], dtype=np.float64)
    out = []
    for label, rem in (("pre_boundary", 0), ("post_boundary", 1000)):
        sel = np.flatnonzero((steps > 0) & (steps % PERIOD == rem))
        q = p[sel]
        band = (q > 0.0) & (q < 1.0)
        for j, i in enumerate(sel):
            out.append(dict(position=label, task=int(steps[i] // PERIOD), step=int(steps[i]),
                            band_frac=float(band[j].mean()),
                            sub_frac=float((q[j] == 0.0).mean()),
                            surf_frac=float((q[j] == 1.0).mean())))
    return out


# ------------------------------------------------------------------ F: N_mask
def n_mask_from_ckpt(cfg: dict, arm: str) -> list[dict]:
    """5M ckpt の 32 パターンを再構成し、帯ユニットの異なる発火マスク数を数える。

    窓は **final_step5000000 の 1 点**（末尾窓 491-500 ではない）。
    """
    step = int(cfg["controls"]["m_minus_checkpoint_step"])
    path = Path(ROOT) / str(cfg["controls"]["m_minus_checkpoint"]).format(arm=arm, step=step)
    if not path.exists():
        return [dict(arm=arm, status="INSUFFICIENT_DATA", path=str(path))]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    entry = dict(cfg["controls"]["arms"][arm])
    net = VecMLPL(blob["net"]["v"].shape[0], [blob["net"]["v"].shape[1]],
                  blob["net"]["W"].shape[2], torch.Generator().manual_seed(0), "cpu")
    net.load_state(blob["net"])
    net.set_activation(str(entry["activation"]),
                       float(entry["dial"]) if str(entry["activation"]) != "relu" else 1.0,
                       "alpha_exp")
    rows = []
    with torch.no_grad():
        W, b = net.Ws[0].double(), net.bs[0].double()
        flip = blob["env"]["flip_state"]
        runs, free = int(flip.shape[0]), int(W.shape[2] - flip.shape[1])
        patterns = ((torch.arange(2 ** free)[:, None] >> torch.arange(free)) & 1).to(flip.dtype)
        cur = torch.cat([flip[None].expand(patterns.shape[0], -1, -1),
                         patterns[:, None, :].expand(-1, runs, -1)], dim=2).double()
        if list(blob["centered_layers"])[0]:
            cur = cur - blob["layer_means"][0].double()[None]
        z = torch.einsum("rhd,prd->prh", W, cur) + b        # [P, R, H]
        on = (z > 0).numpy()                                 # [P, R, H]
    for s in range(on.shape[1]):
        m = on[:, s, :]                                      # [P, H]
        p_hat = m.mean(axis=0)
        band = (p_hat > 0) & (p_hat < 1)
        cols = m[:, band].T                                  # [n_band, P]
        n_uniq = int(len(np.unique(cols, axis=0))) if cols.shape[0] else 0
        rows.append(dict(arm=arm, seed=s, status="OK", window="final_step5000000",
                         n_band=int(band.sum()), n_mask=n_uniq,
                         n_mask_lt_n_band=int(n_uniq < int(band.sum()))))
    return rows


# --------------------------------------------------------------------- 本体
def analyze() -> dict:
    cfg = _cfg()
    floor = float(cfg["phase1"]["unfit_floor"])
    thr_onset = float(cfg["phase1"]["onset_threshold"])
    rng = np.random.default_rng(BOOT_SEED)
    arms = _arm_table()

    # ---- A / B / C / G -----------------------------------------------------
    seed_rows, arm_rows, hist_rows, series_rows = [], [], [], []
    per_arm_seed = {}        # arm -> window -> {key: [seed値]}
    for a in arms:
        per_arm_seed[a["arm"]] = {w: {} for w in WINDOWS}
        hist = np.zeros(N_PAT - 1, dtype=np.int64)
        ser: dict = {}
        surf_hist_max = 0.0
        for seed in range(N_SEED):
            d = _load(a["logdir"] / f"{a['arm']}_seed{seed}.npz")
            for w, tasks in WINDOWS.items():
                st = seed_stats(d, tasks, floor, thr_onset, a["shallow_thr"])
                st.update(arm=a["arm"], family=a["family"], dial=a["dial"],
                          is_control=a["is_control"], seed=seed, window=w)
                seed_rows.append(st)
                for k, v in st.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        per_arm_seed[a["arm"]][w].setdefault(k, []).append(v)
            hist += band_hist(d, WINDOWS["5M"])
            for r in band_series(d):
                ser.setdefault((r["position"], r["task"]), []).append(r)
            surf_hist_max = max(surf_hist_max, _max_surf_over_history(d))
        for (pos, task), rs in sorted(ser.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            series_rows.append(dict(arm=a["arm"], position=pos, task=task, step=rs[0]["step"],
                                    n_seed=len(rs),
                                    band_frac=float(np.median([r["band_frac"] for r in rs])),
                                    sub_frac=float(np.median([r["sub_frac"] for r in rs])),
                                    surf_frac=float(np.median([r["surf_frac"] for r in rs]))))
        tot = int(hist.sum())
        for k in range(1, N_PAT):
            hist_rows.append(dict(arm=a["arm"], family=a["family"], k=k, p_hat=k / N_PAT,
                                  count=int(hist[k - 1]),
                                  share_of_band=(hist[k - 1] / tot if tot else float("nan"))))
        row = dict(arm=a["arm"], family=a["family"], dial=a["dial"], is_control=a["is_control"],
                   thr_kind=a["thr_kind"], shallow_thr=a["shallow_thr"],
                   n_band_records=tot, surf_frac_max_over_history=surf_hist_max)
        for w in WINDOWS:
            for key in ("sub_frac", "band_frac", "surf_frac", "surf_among_alive",
                        "band_among_alive", "near_off_frac", "near_on_frac",
                        "shallow_sub_frac", "non_affine_frac", "log10_u"):
                row[f"{key}_{w}"] = float(np.nanmedian(per_arm_seed[a["arm"]][w][key]))
            row[f"n_onset_{w}"] = int(sum(per_arm_seed[a["arm"]][w]["onset"]))
        if tot:
            row["band_edge_share"] = float((hist[0] + hist[-1]) / tot)
            row["band_near_off_share"] = float(hist[0] / tot)
            row["band_near_on_share"] = float(hist[-1] / tot)
        else:
            row["band_edge_share"] = row["band_near_off_share"] = row["band_near_on_share"] = float("nan")
        arm_rows.append(row)
    A = {r["arm"]: r for r in arm_rows}

    # 15M（p3_extend_0902 のある 5 腕）
    ext_rows = []
    for arm in sorted({p.name.split("_seed")[0] for p in EXTEND_LOGS.glob("*.npz")}):
        vals = {}
        for seed in range(N_SEED):
            st = seed_stats(_load(EXTEND_LOGS / f"{arm}_seed{seed}.npz"), WINDOW_15M,
                            floor, thr_onset,
                            next((x["shallow_thr"] for x in arms if x["arm"] == arm), float("nan")))
            for k, v in st.items():
                if isinstance(v, float):
                    vals.setdefault(k, []).append(v)
        ext_rows.append(dict(arm=arm, window="15M",
                             **{k: float(np.nanmedian(v)) for k, v in vals.items()}))

    # ---- D: 谷埋め 4 腕 -----------------------------------------------------
    clamp_rows = []
    for arm, (ref, refdir, armdir) in CLAMP_PAIRS.items():
        thr = next(x["shallow_thr"] for x in arms if x["arm"] == ref)
        bc, br, uc, ur, rc = [], [], [], [], []
        for seed in range(N_SEED):
            sc = seed_stats(_load(armdir / f"{arm}_seed{seed}.npz"), WINDOWS["5M"], floor, thr_onset, thr)
            sr = seed_stats(_load(refdir / f"{ref}_seed{seed}.npz"), WINDOWS["5M"], floor, thr_onset, thr)
            sR = seed_stats(_load(DOSE_LOGS / f"R_1216_seed{seed}.npz"), WINDOWS["5M"], floor, thr_onset, float("nan"))
            for tag, st in (("clamp", sc), ("reference", sr)):
                st.update(arm=arm if tag == "clamp" else ref, seed=seed, window="5M", role=tag)
                seed_rows.append(dict(st, family="", dial=float("nan"), is_control=0))
            bc.append(sc["band_frac"]); br.append(sr["band_frac"])
            uc.append(sc["log10_u"]); ur.append(sr["log10_u"]); rc.append(sR["band_frac"])
        dband = np.array(bc) - np.array(br)
        du = np.array(uc) - np.array(ur)
        mb, lo, hi = _boot_median_ci(dband, rng)
        mu_, ulo, uhi = _boot_median_ci(du, rng)
        sg = _sign_test(dband)
        clamp_rows.append(dict(
            arm=arm, reference=ref,
            band_frac_clamp=float(np.median(bc)), band_frac_ref=float(np.median(br)),
            band_frac_relu=float(np.median(rc)),
            d_band=mb, d_band_lo=lo, d_band_hi=hi, d_band_sign=f"{sg['neg']}:{sg['pos']}",
            d_band_p=sg["p_two_sided"],
            d_log10u=mu_, d_log10u_lo=ulo, d_log10u_hi=uhi,
            band_up=int(lo > 0), same_direction=int((mb > 0) == (mu_ < 0))))

    # ---- E: 順序（Spearman）-------------------------------------------------
    order_rows = []
    all_arms = [r["arm"] for r in arm_rows]
    for variant, keep in (("all_17", all_arms),
                          ("excl_unconverged_14", [a for a in all_arms if a not in UNCONVERGED])):
        band_s = np.array([per_arm_seed[a]["5M"]["band_frac"] for a in keep])   # [n_arm, n_seed]
        sub_s = np.array([per_arm_seed[a]["5M"]["sub_frac"] for a in keep])
        u_s = np.array([per_arm_seed[a]["5M"]["log10_u"] for a in keep])
        rho_band = _spearman(np.median(band_s, 1), np.median(u_s, 1))
        rho_sub = _spearman(np.median(sub_s, 1), np.median(u_s, 1))
        draws = rng.integers(0, N_SEED, size=(BOOT_N, N_SEED))
        diffs = np.empty(BOOT_N)
        for i in range(BOOT_N):
            j = draws[i]
            db = _spearman(np.median(band_s[:, j], 1), np.median(u_s[:, j], 1))
            ds = _spearman(np.median(sub_s[:, j], 1), np.median(u_s[:, j], 1))
            diffs[i] = abs(db) - abs(ds)
        order_rows.append(dict(variant=variant, n_arms=len(keep),
                               rho_band=rho_band, rho_sub=rho_sub,
                               abs_diff=abs(rho_band) - abs(rho_sub),
                               diff_lo=float(np.percentile(diffs, 2.5)),
                               diff_hi=float(np.percentile(diffs, 97.5)),
                               band_wins=int(np.percentile(diffs, 2.5) > 0
                                             and abs(rho_band) - abs(rho_sub) >= 0.1)))

    # ---- F: N_mask ---------------------------------------------------------
    nmask_rows = [r for arm in NMASK_ARMS for r in n_mask_from_ckpt(cfg, arm)]

    # ---- 予測の判定 ---------------------------------------------------------
    R = A["R_1216"]
    p1 = R["surf_among_alive_5M"] >= 0.5 and R["band_frac_5M"] < 0.01
    e_all = next(r for r in order_rows if r["variant"] == "all_17")
    e_ex = next(r for r in order_rows if r["variant"] == "excl_unconverged_14")
    p2 = bool(e_all["band_wins"])
    unseen = [r for r in clamp_rows if r["arm"] in ("Gz_b1_1216", "Sz_b3_1216", "Sc_b3_1216")]
    p3a = all(r["band_up"] for r in unseen)
    p3b = all(r["same_direction"] for r in clamp_rows)
    leaky = [r for r in arm_rows if r["family"] == "leaky"]
    lr01 = next(r for r in leaky if r["dial"] == 0.1)
    p4 = all(lr01["band_frac_5M"] > r["band_frac_5M"] for r in leaky if r["arm"] != lr01["arm"])
    p5 = all(np.isnan(r["band_edge_share"]) or r["band_edge_share"] >= 0.5
             for r in arm_rows if r["n_band_records"] > 0)
    shallow = [r["shallow_sub_frac_5M"] for r in arm_rows
               if r["thr_kind"] == "u_star" and np.isfinite(r["shallow_sub_frac_5M"])]
    shallow_elu = [(r["arm"], r["shallow_sub_frac_5M"]) for r in arm_rows
                   if r["thr_kind"] == "u_fr" and np.isfinite(r["shallow_sub_frac_5M"])]
    p7 = all(v < 0.05 for v in shallow)
    ok = [r for r in nmask_rows if r.get("status") == "OK"]
    p6_pos = [r for r in ok if r["n_band"] > 0]
    p6 = bool(p6_pos) and all(r["n_mask_lt_n_band"] for r in p6_pos)
    # 参考（登録外の読み）: 帯ユニットが 5 個以上ある seed に限った場合
    p6_big = [r for r in ok if r["n_band"] >= 5]
    p6_big_hit = bool(p6_big) and all(r["n_mask_lt_n_band"] for r in p6_big)

    preds = [
        dict(id="P1", item="R_1216: surf_among_alive_5M >= 0.5 かつ band_frac_5M < 0.01",
             value=f"{R['surf_among_alive_5M']:.4g} / {R['band_frac_5M']:.4g}", hit=p1),
        dict(id="P2", item="|rho_band| - |rho_sub| >= 0.1 かつ CI が 0 を外す（17 腕）",
             value=f"{e_all['abs_diff']:.4g} [{e_all['diff_lo']:.4g},{e_all['diff_hi']:.4g}]", hit=p2),
        dict(id="P2(除外版)", item="同・未収束 3 腕を除く 14 腕",
             value=f"{e_ex['abs_diff']:.4g} [{e_ex['diff_lo']:.4g},{e_ex['diff_hi']:.4g}]",
             hit=bool(e_ex["band_wins"])),
        dict(id="P3'", item="未見 3 腕（Gz・Sz・Sc）の帯率が元腕より高い（CI が 0 を外す）",
             value=", ".join(f"{r['arm']}:{r['d_band']:+.4g}[{r['d_band_lo']:+.4g},{r['d_band_hi']:+.4g}]"
                             for r in unseen), hit=p3a),
        dict(id="P3''", item="帯率の差と Δlog10U の差が同じ向き（4 腕）",
             value=", ".join(f"{r['arm']}:{'同' if r['same_direction'] else '逆'}" for r in clamp_rows),
             hit=p3b),
        dict(id="P4", item="leaky 族で a=0.1 の帯率が他 4 段より高い",
             value=", ".join(f"a={r['dial']:g}:{r['band_frac_5M']:.4g}" for r in sorted(leaky, key=lambda x: -x["dial"])),
             hit=p4),
        dict(id="P5", item="帯の p_hat 分布が両端（1/32・31/32）に寄る（全腕で両端の合計 >= 0.5）",
             value=("両端 >= 0.5 の腕 "
                    + f"{sum(1 for r in arm_rows if r['n_band_records'] > 0 and r['band_edge_share'] >= 0.5)}"
                    + f"/{sum(1 for r in arm_rows if r['n_band_records'] > 0)}"
                    + f"・min={min((r['band_edge_share'] for r in arm_rows if r['n_band_records'] > 0), default=float('nan')):.4g}"),
             hit=p5),
        dict(id="P7", item="浅い沈下率（-u*/β < zmax <= 0）がどの腕でも 0.05 未満（u* のある silu/gelu 7 腕）",
             value=(f"silu/gelu(u*) max={max(shallow, default=float('nan')):.4g}"
                    f" ／ 参考 elu(u_fr): "
                    + ", ".join(f"{a}:{v:.3g}" for a, v in shallow_elu)), hit=p7),
        dict(id="P6", item="N_mask < 帯ユニット数（対照 3 腕・final_step5000000）",
             value=(f"厳密（n_band>0 の {len(p6_pos)} seed）"
                    f"{sum(r['n_mask_lt_n_band'] for r in p6_pos)}/{len(p6_pos)}"
                    f" ／ 参考（n_band>=5 の {len(p6_big)} seed）"
                    f"{sum(r['n_mask_lt_n_band'] for r in p6_big)}/{len(p6_big)}"
                    f"（{'成立' if p6_big_hit else '不成立'}）"),
             hit=p6),
    ]

    if not p1:
        verdict = "SURFACED_UNITS_ABSENT"
    elif p2 and p3a:
        verdict = "BAND_IS_THE_ORDER_PARAMETER"
    elif p2 and not p3a:
        verdict = "BAND_ORDERS_BUT_CLAMP_UNEXPLAINED"
    elif p3a:
        verdict = "CLAMP_ONLY"
    else:
        verdict = "BAND_IS_NOT_IT"

    secondary_extra = dict(P6_nband_ge5=p6_big_hit)
    return dict(verdict=verdict, arms=arm_rows, seeds=seed_rows, hist=hist_rows,
                series=series_rows, extend=ext_rows, clamp=clamp_rows, order=order_rows,
                nmask=nmask_rows, preds=preds,
                secondary=dict(P1=p1, P2=p2, P3a=p3a, P3b=p3b, P4=p4, P5=p5, P6=p6, P7=p7,
                               **secondary_extra))


# --------------------------------------------------------------------- 出力
def selftest() -> None:
    """委託先の committed 値（dial_table.csv）と窓関数・沈下率の一致を確認する。"""
    cfg = _cfg()
    floor, thr = float(cfg["phase1"]["unfit_floor"]), float(cfg["phase1"]["onset_threshold"])
    table = {}
    with open(Path(ROOT) / "results/gate_dial_0902/dial_table.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[r["arm"]] = r
    bad = 0
    for a in _arm_table():
        vals = [seed_stats(_load(a["logdir"] / f"{a['arm']}_seed{s}.npz"), WINDOWS["5M"],
                           floor, thr, a["shallow_thr"]) for s in range(N_SEED)]
        u = float(np.median([v["log10_u"] for v in vals]))
        sub = float(np.median([v["sub_frac"] for v in vals]))
        cu, cs = float(table[a["arm"]]["median_log10_U_5m"]), float(table[a["arm"]]["submerged_frac"])
        ok_u, ok_s = abs(u - cu) < 1e-9, abs(sub - cs) < 1e-9
        bad += (not ok_u) + (not ok_s)
        print(f"selftest {a['arm']:>18}: log10U {u:+.6f} vs {cu:+.6f} {'OK' if ok_u else 'MISMATCH'}"
              f" | sub_frac {sub:.4f} vs {cs:.4f} {'OK' if ok_s else 'MISMATCH'}")
        # 3 分類が尽きていること
        for v in vals:
            assert abs(v["sub_frac"] + v["band_frac"] + v["surf_frac"] - 1.0) < 1e-12
    print(f"selftest: {'ALL OK' if bad == 0 else f'{bad} MISMATCH'}（3 分類の和 = 1 も検査済み）")


def _reference_paths() -> list[Path]:
    paths = []
    for a in _arm_table():
        paths += [a["logdir"] / f"{a['arm']}_seed{s}.npz" for s in range(N_SEED)]
    for arm, (ref, refdir, armdir) in CLAMP_PAIRS.items():
        paths += [armdir / f"{arm}_seed{s}.npz" for s in range(N_SEED)]
    paths += sorted(EXTEND_LOGS.glob("*.npz"))
    paths += [DOSE_CKPTS / f"{a}_step5000000.pt" for a in NMASK_ARMS]
    return [p for p in dict.fromkeys(paths) if p.exists()]


def write_outputs(res: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTDIR / "band_table.csv", res["arms"])            # A・G
    _write_csv(OUTDIR / "seed_values.csv", res["seeds"])
    _write_csv(OUTDIR / "band_hist.csv", res["hist"])             # B
    _write_csv(OUTDIR / "band_series.csv", res["series"])         # C
    _write_csv(OUTDIR / "band_extend_15m.csv", res["extend"])
    _write_csv(OUTDIR / "clamp_contrast.csv", res["clamp"])       # D
    _write_csv(OUTDIR / "order_spearman.csv", res["order"])       # E
    _write_csv(OUTDIR / "n_mask.csv", res["nmask"])               # F
    _write_csv(OUTDIR / "predictions.csv",
               [dict(id=p["id"], item=p["item"], value=p["value"], hit=int(p["hit"]))
                for p in res["preds"]])
    _write_csv(OUTDIR / "verdict.csv", [dict(verdict=res["verdict"], **res["secondary"])])
    prov = dict(exp="band_affine_0903", git_head=_git_head(), verdict=res["verdict"],
                spec="可塑性喪失/spec/帯とアフィン化_spec_0902.md (b8ed195)",
                config_sha256=_sha(CONFIG),
                window_definition="task-end records (step%10000==0), U^(10), windows 91-100 / 491-500 / 1491-1500",
                new_runs=0, new_loggers=0, items_done="A,B,C,D,E,F,G", items_skipped="H",
                reference_logs={str(p.relative_to(ROOT)): _sha(p) for p in _reference_paths()})
    (OUTDIR / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    lines = [f"# band_affine_0903 summary", "", f"**verdict: {res['verdict']}**", "",
             "走ゼロ・新規ロガーゼロ。A〜E・G は `layer1_p_hat`（と `layer1_zmax`）、F は 5M ckpt。H は未実施。", ""]
    lines += ["## 事前予測（spec §5-1・再集計前に固定）", "", "| # | 予測 | 値 | 判定 |", "|---|---|---|---|"]
    for p in res["preds"]:
        lines.append(f"| {p['id']} | {p['item']} | {p['value']} | {'✓' if p['hit'] else '✗'} |")
    keys = ["arm", "family", "dial", "sub_frac_5M", "band_frac_5M", "surf_frac_5M",
            "surf_among_alive_5M", "band_edge_share", "shallow_sub_frac_5M",
            "non_affine_frac_5M", "surf_frac_max_over_history", "log10_u_5M"]
    lines += ["", "## A・G: 腕ごとの 3 分類（末尾窓 491-500・seed 中央値）", "",
              "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in sorted(res["arms"], key=lambda x: x["log10_u_5M"]):
        lines.append("| " + " | ".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])
                                       for k in keys) + " |")
    ek = list(res["order"][0].keys())
    lines += ["", "## E: 順序（Spearman・seed-bootstrap B=10^4・seed 20260907）", "",
              "| " + " | ".join(ek) + " |", "|" + "---|" * len(ek)]
    for r in res["order"]:
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                       for v in r.values()) + " |")
    ck = list(res["clamp"][0].keys())
    lines += ["", "## D: 谷埋め（paired・元腕は別走の committed ログ）", "",
              "| " + " | ".join(ck) + " |", "|" + "---|" * len(ck)]
    for r in res["clamp"]:
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                       for v in r.values()) + " |")
    lines += ["", "引用上の注意: condA・1 層・幅 100・用量 12.16・seed 0-9・末尾窓はタスク終端 10 点。",
              "`submerged_frac` を verdict に使わない規則（用語と記号 §1）は帯率にも適用する —— 判定は順序と介入前後の差。",
              "F の窓は final_step5000000 の 1 点であって末尾窓ではない。",
              "対照ログの sha256 は provenance.json の reference_logs。"]
    (OUTDIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    write_outputs(analyze())


if __name__ == "__main__":
    main()
