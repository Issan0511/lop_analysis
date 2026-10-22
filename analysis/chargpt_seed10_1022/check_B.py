#!/usr/bin/env python3
"""chargpt_seed10_1022 の B の検査（spec §7 の B、追補 1 の「A・B の照合の許容」、手順書 §3.2・§4）。

(1) 照合: 4 ブロック C3 seed 1000 の課題 1 末 → 課題 2 を rho10.py（--nlayer 4 --seed 1000、写し元の走と同じ引数 = 層は全部・nseq 32・
    cuda・exact）で出し直し、$CG/results/chargpt_cl_1019/quicklook/rho/out/C3/t2_a_chain.json の各項と照らす。
    許容（追補 1）: 連続量は相対 1e−8 以内、割合（符号の数え上げ）は生きた unit 1 個ぶん（1/n_live）以内。
    連続量の「相対」= 各ベクトルで max|新 − 保存| / max|保存|（スカラーは |新 − 保存| / |保存|）。要素ごとの相対も記録だけ出す。
    割合 = 層 × 計量ごとに、沈む unit（4 項の和 < 0）の割合、各項が負の割合、① と ② の符号の一致の割合（生きた unit の中と全 unit の中）、
    生きた unit の数。連続量の派生として、沈む unit の各項の割合（Σ項/Σ和 と 項/和 の中央値）も照らす。
(2) rho_split の検査（GW_vs_autograd・jvp_vs_backward・diag_direct・転置の変異）: 走らせた全状態で rho10.py の check_eval を見る。
(3) 2 ブロックの試走: C3 seed 0 の課題 1 末 → 課題 2（時間を測る）、ReLUcap seed 0 の同じ切替（cap の腕の読み込み）、
    C3 seed 0 の課題 49 末 → 課題 50 を --check つき（closure・flip の検査、記録だけ）。
出力 = $S10/results/chargpt_seed10_1022/checks/B/。
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

S10 = Path("/home/kubo/project/Nakatsuka/lop_seed10")
CG = Path("/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019")
PY = str(CG / ".venv/bin/python")
RHO10 = S10 / "analysis/chargpt_seed10_1022/rho10.py"
OUT = S10 / "results/chargpt_seed10_1022/checks/B"
SAVED = CG / "results/chargpt_cl_1019/quicklook/rho/out/C3/t2_a_chain.json"
REL = 1e-8
TERMS = ("own", "self_up", "others_same", "others_path")
VEC = TERMS + ("others_gamma",)


def run(name, args):
    out = OUT / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    env = dict(os.environ, CHARGPT_SRC=str(CG)); env.pop("CHARGPT_NLAYER", None)
    cmd = [PY, str(RHO10)] + args + ["--out", str(out)]
    t0 = time.time()
    with open(OUT / f"{name}.log", "w") as lg:
        lg.write(" ".join(cmd) + "\n"); lg.flush()
        rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"{name}: rc={rc}（{OUT / (name + '.log')}）")
    d = json.loads(out.read_text()); d["provenance"]["wall_sec_subprocess"] = time.time() - t0
    out.write_text(json.dumps(d))
    return d


def vrel(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    den = np.abs(b).max()
    r = float(np.abs(a - b).max() / den) if den > 0 else float(np.abs(a - b).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        el = np.abs(a - b) / np.abs(b)
    el = el[np.isfinite(el)]
    return r, (float(el.max()) if el.size else 0.0), int((el > REL).sum())


def fractions(L, mt):
    r = L[mt]; alive = np.array(L["alive"], bool)
    T = {k: np.array(r[k]) for k in TERMS}; S = sum(T.values())
    f = {}
    for nm, msk in (("alive", alive), ("all", np.ones_like(alive))):
        n = msk.sum()
        f[f"sink_{nm}"] = float((S[msk] < 0).mean()) if n else float("nan")
        for k in TERMS:
            f[f"neg_{k}_{nm}"] = float((T[k][msk] < 0).mean()) if n else float("nan")
        nz = msk & (T["own"] != 0) & (T["self_up"] != 0)
        f[f"sign12_{nm}"] = float((np.sign(T["own"][nz]) == np.sign(T["self_up"][nz])).mean()) if nz.sum() else float("nan")
    return f, int(alive.sum())


def shares(L, mt):
    r = L[mt]; alive = np.array(L["alive"], bool)
    T = {k: np.array(r[k]) for k in TERMS}; S = sum(T.values()); sink = alive & (S < 0)
    out = {}
    if sink.sum():
        for k in TERMS:
            out[f"agg_{k}"] = float(T[k][sink].sum() / S[sink].sum())
            out[f"med_{k}"] = float(np.median(T[k][sink] / S[sink]))
    return out


def compare(new, old):
    rows, ok = [], True
    for key in ("loss",):
        r = abs(new[key] - old[key]) / abs(old[key]); p = r <= REL; ok &= p
        rows.append(dict(item=key, kind="scalar", rel=r, pass_=p))
    for l in sorted(old["layers"]):
        Ln, Lo = new["layers"][l], old["layers"][l]
        for key in ("K_mean",):
            r = abs(Ln[key] - Lo[key]) / abs(Lo[key]); p = r <= REL; ok &= p
            rows.append(dict(item=f"L{l}:{key}", kind="scalar", rel=r, pass_=p))
        r, el, nel = vrel(Ln["sigma"], Lo["sigma"]); p = r <= REL; ok &= p
        rows.append(dict(item=f"L{l}:sigma", kind="vector", rel=r, elem_rel_max=el, n_elem_over=nel, pass_=p))
        na = int(np.array(Lo["alive"], bool).sum()); dif = int((np.array(Ln["alive"]) != np.array(Lo["alive"])).sum())
        p = dif <= 1; ok &= p
        rows.append(dict(item=f"L{l}:alive", kind="fraction", n_live=na, diff_count=dif, diff=dif / max(na, 1), tol=1 / max(na, 1), pass_=p))
        for mt in ("A", "D"):
            for k in VEC + ("top_contrib", "top_cos"):
                r, el, nel = vrel(Ln[mt][k], Lo[mt][k]); p = r <= REL; ok &= p
                rows.append(dict(item=f"L{l}:{mt}:{k}", kind="vector", rel=r, elem_rel_max=el, n_elem_over=nel, pass_=p))
            ti_n, ti_o = np.array(Ln[mt]["top_others"]), np.array(Lo[mt]["top_others"])
            rows.append(dict(item=f"L{l}:{mt}:top_others", kind="index(記録)", n_diff=int((ti_n != ti_o).sum()), n=int(ti_o.size), pass_=True))
            fn, nl = fractions(Ln, mt); fo, _ = fractions(Lo, mt)
            for k in fo:
                d = abs(fn[k] - fo[k]); p = d <= 1 / max(nl, 1) + 1e-15; ok &= p
                rows.append(dict(item=f"L{l}:{mt}:{k}", kind="fraction", new=fn[k], saved=fo[k], diff=d, tol=1 / max(nl, 1), n_live=nl, pass_=p))
            sn, so = shares(Ln, mt), shares(Lo, mt)
            if set(sn) != set(so):
                ok = False; rows.append(dict(item=f"L{l}:{mt}:shares", kind="derived", pass_=False, note="沈む unit の有無が違う"))
            for k in so:
                r = abs(sn[k] - so[k]) / abs(so[k]); p = r <= REL; ok &= p
                rows.append(dict(item=f"L{l}:{mt}:share_{k}", kind="derived", new=sn[k], saved=so[k], rel=r, pass_=p))
        for ck, v in Lo["check"].items():
            rows.append(dict(item=f"L{l}:check:{ck}", kind="check(記録)", new=Ln["check"][ck], saved=v, pass_=True))
    return ok, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-2blk", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %z"))
    ck4 = CG / "results/chargpt_cl_1019/quicklook/M/C3/seed1000/ckpts"
    # (1) 照合（写し元の走と同じ引数: run_job_gpu_exact.sh = --nseq 32 --threads 2 --device cuda --jvp exact、層は全部）
    rep = run("repro_4blk_C3_seed1000_t2_a_chain", ["--arm", "C3", "--state", str(ck4 / "task1.pt"), "--adam", str(ck4 / "task1_adam.pt"),
              "--task", "2", "--kind", "chain", "--nseq", "32", "--threads", "2", "--device", "cuda", "--jvp", "exact", "--seed", "1000", "--nlayer", "4"])
    saved = json.loads(SAVED.read_text())
    ok, rows = compare(rep, saved)
    vec = [r for r in rows if r["kind"] in ("vector", "scalar", "derived") and "rel" in r]
    fr = [r for r in rows if r["kind"] == "fraction"]
    res["repro"] = dict(all_pass=ok, n_items=len(rows), n_fail=sum(not r["pass_"] for r in rows),
                        max_rel_continuous=max(r["rel"] for r in vec), worst_continuous=max(vec, key=lambda r: r["rel"])["item"],
                        max_elem_rel=max(r.get("elem_rel_max", 0) for r in vec), n_elem_over_1e8=sum(r.get("n_elem_over", 0) for r in vec),
                        max_fraction_diff=max(r["diff"] for r in fr), min_fraction_tol=min(r["tol"] for r in fr),
                        n_top_others_diff=sum(r.get("n_diff", 0) for r in rows if r["kind"] == "index(記録)"),
                        saved_sha256=__import__("hashlib").sha256(SAVED.read_bytes()).hexdigest(),
                        elapsed_sec=rep["provenance"]["elapsed_sec"], gpu_peak_mem_bytes=rep["provenance"]["gpu_peak_mem_bytes"])
    (OUT / "repro_compare_rows.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))
    states = {"repro_4blk_C3_seed1000_t2": rep}
    # (3) 2 ブロックの試走
    if not a.skip_2blk:
        for name, arm, k, extra in (("C3_seed0_t2_a_chain", "C3", 1, []), ("ReLUcap_seed0_t2_a_chain", "ReLUcap", 1, []),
                                    ("C3_seed0_t50_a_chain_check", "C3", 49, ["--check"])):
            ck = CG / f"results/chargpt_cl_1019/M2/{arm}/seed0/ckpts"
            states[name] = run(name, ["--arm", arm, "--state", str(ck / f"task{k}.pt"), "--adam", str(ck / f"task{k}_adam.pt"), "--task", str(k + 1),
                                      "--kind", "chain", "--nseq", "32", "--threads", "2", "--device", "cuda", "--jvp", "exact", "--seed", "0",
                                      "--nlayer", "2", "--layers", "0", "1"] + extra)
    # (2) rho_split の検査（各状態）
    res["state_checks"] = {}
    for name, d in states.items():
        ce = d["check_eval"]
        res["state_checks"][name] = dict(all_pass=ce["all_pass"], n_items=len(ce["items"]), n_fail=sum(not v["pass_"] for v in ce["items"].values()),
                                         worst_identity=max(v["value"] for k, v in ce["items"].items() if "MUT_" not in k),
                                         min_mutation=min(v["value"] for k, v in ce["items"].items() if "MUT_" in k and not v.get("na")),
                                         n_mut_na=ce.get("n_na", 0),
                                         elapsed_sec=d["provenance"]["elapsed_sec"], gpu_peak_mem_bytes=d["provenance"]["gpu_peak_mem_bytes"],
                                         n_live={l: int(sum(L["alive"])) for l, L in d["layers"].items()},
                                         closure=d.get("closure"), flip_check=d.get("flip_check"))
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    res["all_pass"] = res["repro"]["all_pass"] and all(v["all_pass"] for v in res["state_checks"].values())
    (OUT / "check_B.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
