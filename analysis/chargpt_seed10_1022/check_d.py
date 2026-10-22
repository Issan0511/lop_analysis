#!/usr/bin/env python3
"""D の検査の集計（spec §7 の D・追補 1・手順書 §3.5）。短い試走（check_d_trials.sh と 2 課題の試走）の出力を読む。

  D-stream  : 試走の per_task.csv の data_sha（16 字）と provenance の tasks_info（全長）が、本走 4 ブロック
              $CG/results/chargpt_cl_1019/M/C3/seed<k> と一致するか（seed 0〜9、課題 1。seed 0 は課題 2 も）
  D-repro   : byakko の中での再現。seed 0 の 1 課題の試走と 2 課題の試走の課題 1 の online の損失がビットで同じか
  D-white-SiLU: SiLU 4 ブロック seed 1000 の課題 1 が、white-san の quicklook/M/SiLU/seed1000 とビットで同じか（差の大きさ）
  D-white-C3  : C3 4 ブロック seed 0 の課題 1 が、white-san の crosscheck_white-san/M/C3/seed0 とビットで同じか（差の大きさ）。
                byakko の本走 M/C3/seed0 との照合も（同じ台・版の違いだけ）
  D-save    : 課題末と更新 100 の保存の集合が spec §6 と同じか（来歴）。保存の仕組みの試走で、足した課題 11〜15 の
              課題末・更新 100 の checkpoint と Adam の状態が出来て、集合の外（課題 16）は出来ないか
  D-time    : 1 課題の時間
"""
import csv
import json
from pathlib import Path

import numpy as np

CG = Path("/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019")
R = CG / "results" / "chargpt_cl_1019"
O = Path("/home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/checks/D")
T = O / "trial"
SPEC_END = set(range(1, 16)) | {25} | set(range(39, 51)) | set(range(59, 71)) | set(range(89, 101))
SPEC_U100 = set(range(1, 16)) | set(range(40, 51)) | set(range(60, 71)) | set(range(90, 101))


def rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def prov(d):
    return json.loads((d / "provenance.json").read_text())


def cmp_loss(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ne = np.nonzero(a != b)[0]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(bit_equal=bool(len(ne) == 0), n_diff=int(len(ne)), n=int(len(a)), first_diff_update=int(ne[0]) if len(ne) else None,
                max_abs=float(d.max()), mean_abs=float(d.mean()), mean_a=float(a.mean()), mean_b=float(b.mean()),
                diff_of_means=float(a.astype(np.float64).mean() - b.astype(np.float64).mean()))


def main():
    res = {}
    # D-stream
    st = []
    for k in range(10):
        ref_p, ref_rows = prov(R / "M/C3" / f"seed{k}"), rows(R / "M/C3" / f"seed{k}" / "per_task.csv")
        refi = {r["task"]: r for r in ref_p["tasks_info"]}
        refr = {int(r["task"]): r for r in ref_rows}
        trials = [T / f"SiLU_seed{k}_t1"] + ([T / "SiLU_seed0_t2"] if k == 0 else [])
        for d in trials:
            if not (d / "provenance.json").exists():
                st.append(dict(seed=k, trial=d.name, missing=True))
                continue
            p, pr = prov(d), rows(d / "per_task.csv")
            for r in pr:
                t = int(r["task"])
                ti = next(x for x in p["tasks_info"] if x["task"] == t)
                st.append(dict(seed=k, trial=d.name, task=t, csv_data_sha=r["data_sha"] == refr[t]["data_sha"],
                               full_data_sha=ti["data_sha"] == refi[t]["data_sha"], rule_sha=ti["rule_sha"] == refi[t]["rule_sha"],
                               probe_sha=ti["probe_sha"] == refi[t]["probe_sha"], H_equal=ti["H"] == refi[t]["H"]))
    ok = [s for s in st if not s.get("missing") and all(s[c] for c in ("csv_data_sha", "full_data_sha", "rule_sha", "probe_sha"))]
    res["D-stream"] = dict(pass_=len(ok) == len(st) and len({s["seed"] for s in ok}) == 10, n_checked=len(st), n_ok=len(ok),
                           seeds_ok=sorted({s["seed"] for s in ok}), rows=st)
    # D-repro
    a, b = T / "SiLU_seed0_t1", T / "SiLU_seed0_t2"
    if (a / "online.npz").exists() and (b / "online.npz").exists():
        c = cmp_loss(np.load(a / "online.npz")["loss"][0], np.load(b / "online.npz")["loss"][0])
        res["D-repro"] = dict(pass_=c["bit_equal"], **c, init_sha_eq=prov(a)["init_sha"] == prov(b)["init_sha"])
    # D-white-SiLU
    w = R / "quicklook/M/SiLU/seed1000"
    d = T / "SiLU_seed1000_t1"
    if (d / "online.npz").exists():
        c = cmp_loss(np.load(d / "online.npz")["loss"][0], np.load(w / "online.npz")["loss"][0])
        pw, pd = prov(w), prov(d)
        rw, rd = rows(w / "per_task.csv")[0], rows(d / "per_task.csv")[0]
        res["D-white-SiLU"] = dict(pass_=c["bit_equal"], ref=str(w.relative_to(CG)), ref_host=pw.get("host"), ref_device=pw.get("device"),
                                   ref_git=pw.get("git_hash"), here_device=pd.get("device"), init_sha_eq=pw["init_sha"] == pd["init_sha"],
                                   data_sha_eq=pw["tasks_info"][0]["data_sha"] == pd["tasks_info"][0]["data_sha"], loss_task1=c,
                                   online_all=(float(rd["online_all"]), float(rw["online_all"])), end_metric=(float(rd["end_metric"]), float(rw["end_metric"])))
    # D-white-C3
    d = T / "C3_seed0_t1"
    if (d / "online.npz").exists():
        x = np.load(d / "online.npz")["loss"][0]
        ws, bk = R / "crosscheck_white-san/M/C3/seed0", R / "M/C3/seed0"
        cw, cb = cmp_loss(x, np.load(ws / "online.npz")["loss"][0]), cmp_loss(x, np.load(bk / "online.npz")["loss"][0])
        cwb = cmp_loss(np.load(bk / "online.npz")["loss"][0], np.load(ws / "online.npz")["loss"][0])
        res["D-white-C3"] = dict(pass_=cw["bit_equal"], vs_white_san=cw, vs_byakko_main=cb, byakko_main_vs_white_san_task1=cwb,
                                 ref_white=str(ws.relative_to(CG)), ref_byakko=str(bk.relative_to(CG)),
                                 ref_white_git=prov(ws).get("git_hash"), ref_byakko_git=prov(bk).get("git_hash"))
    # D-save
    sv = {}
    for nm in ("SiLU_seed0_t2", "SiLU_seed0_mech"):
        d = T / nm
        if not (d / "provenance.json").exists():
            continue
        p = prov(d)
        cdir = d / "ckpts" if (d / "ckpts").exists() else O.parent / "raw" / "D_trial" / nm   # 検査のあと *.pt は raw/ へ移した（.raw.pt）
        files = {f.name.replace(".raw.pt", ".pt") for f in cdir.glob("*.pt")}
        n_t = p["tasks"]
        exp = {"init.pt"} | {f"task{t}{s}.pt" for t in SPEC_END if t <= n_t for s in ("", "_adam")} \
            | {f"u100_task{t}{s}.pt" for t in SPEC_U100 if t <= n_t for s in ("", "_adam")}
        sv[nm] = dict(ckpt_tasks_eq_spec=set(p.get("ckpt_tasks", [])) == SPEC_END, u100_tasks_eq_spec=set(p.get("u100_tasks", [])) == SPEC_U100,
                      n_layer=p.get("n_layer"), decomp_ckpt=p.get("decomp_ckpt"), files_eq_expected=files == exp,
                      missing=sorted(exp - files), extra=sorted(files - exp), n_files=len(files),
                      bytes_total=sum(f.stat().st_size for f in cdir.glob("*.pt")))
    res["D-save"] = dict(pass_=bool(sv) and all(v["ckpt_tasks_eq_spec"] and v["u100_tasks_eq_spec"] and v["files_eq_expected"] and v["n_layer"] == 4
                                                for v in sv.values()) and "SiLU_seed0_mech" in sv, runs=sv)
    # D-time
    tm = {}
    for nm in ["SiLU_seed0_t2"] + [f"SiLU_seed{k}_t1" for k in range(10)] + ["SiLU_seed1000_t1", "C3_seed0_t1"]:
        d = T / nm
        if (d / "provenance.json").exists():
            p = prov(d)
            tm[nm] = dict(task_secs=[round(x, 2) for x in p["task_secs"]], wall=round(p["wall_sec"], 1))
    res["D-time"] = tm
    (O / "check_d.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in res.items():
        if isinstance(v, dict) and "pass_" in v:
            print(k, "PASS" if v["pass_"] else "FAIL")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("rows",)} for k, v in res.items()}, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
