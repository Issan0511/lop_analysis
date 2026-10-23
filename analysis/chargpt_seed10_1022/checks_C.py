#!/usr/bin/env python3
"""chargpt_seed10_1022 C の検査（手順書 §4 の C・spec §7・追補 1）。1 本ずつ GPU で走らせ、結果を checks/C/ に書く。
 (i)  開始状態（C1・C3 の S_k 末、C2 の課題 25 末）の一様 probe の L1 の死が元の layer_stats.csv（u=10000）と一致し、spec §5 の表と整合する
      → revival10.py を課題 0 個（--to_task = --from_task）で control として走らせ、compare_orig.csv の開始の行を見る。
 (ii) lnex1 の開始時の平均の分母が元と一致する（定数の検査）→ 同じく課題 0 個の lnex1（C1 の 9 seed・C3 の 10 seed）。
 (iii) control の 1 課題の再生が動き、元の online.npz・layer_stats.csv との差を出せる。ビット一致かを記録する。1 課題の時間を測る。
      ほかの腕（lnex1・nocommon・gamma03f・gamma03・ReLUcap の control）も 1 課題だけ動くことを見る。
  CHARGPT_SRC=$CG $PY checks_C.py run        # 走らせる（完走済みは飛ばす）
  CHARGPT_SRC=$CG $PY checks_C.py summarize  # checks/C/checks_C.{json,md} を書く
"""
import csv, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revival10_plan as P  # noqa: E402

S10 = HERE.parents[1]
OUT = S10 / "results/chargpt_seed10_1022/checks/C"
PY = os.environ.get("PY", str(P.CG / ".venv/bin/python"))


def plan():
    L = []
    for part in ("C1", "C2", "C3"):
        for seed, fr, to, march, arms in P.jobs(part):
            L.append(("start", part, seed, "control", fr, fr, march))
            if "lnex1" in arms:
                L.append(("start", part, seed, "lnex1", fr, fr, march))
    # 1 課題の再生
    L += [("replay1", "C2", 0, "control", 25, 26, "C3"), ("replay1", "C1", 0, "control", 50, 51, "C3"),
          ("replay1", "C1", 7, "control", 41, 42, "C3"), ("replay1", "C3", 0, "control", 25, 26, "ReLUcap"),
          ("replay1", "C1", 0, "lnex1", 50, 51, "C3"), ("replay1", "C1", 0, "nocommon", 50, 51, "C3"),
          ("replay1", "C2", 0, "gamma03f", 25, 26, "C3"), ("replay1", "C2", 0, "gamma03", 25, 26, "C3"),
          ("replay1", "C3", 0, "lnex1", 25, 26, "ReLUcap"),
          ("replay1b", "C2", 0, "control", 25, 26, "C3"), ("replay1b", "C1", 0, "lnex1", 50, 51, "C3")]   # 同じ台での再現（ビット一致か）
    return L


def odir(kind, part, seed, arm):
    return OUT / kind / part / f"seed{seed}" / arm


def run():
    env = dict(os.environ, CHARGPT_NLAYER="2", CHARGPT_SRC=str(P.CG), OMP_NUM_THREADS="2")
    for kind, part, seed, arm, fr, to, march in plan():
        o = odir(kind, part, seed, arm)
        pv = o / "provenance.json"
        if pv.exists() and json.loads(pv.read_text()).get("complete"):
            continue
        o.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(HERE / "revival10.py"), "--arm", arm, "--from_task", str(fr), "--to_task", str(to), "--seed", str(seed),
               "--src", str(P.M2 / P.SRC_ARM[part] / f"seed{seed}"), "--model_arm", march, "--out", str(o)]
        t0 = time.time()
        with open(o / "run.log", "w") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"{kind} {part} seed{seed} {arm} {fr}->{to} rc={rc} {time.time() - t0:.0f}s", flush=True)


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return None if x in (None, "", "None") else float(x)


def summarize():
    res = dict(start=[], lnex1=[], replay1=[], missing=[])
    for kind, part, seed, arm, fr, to, march in plan():
        o = odir(kind, part, seed, arm); pv = o / "provenance.json"
        if not pv.exists():
            res["missing"].append([kind, part, seed, arm]); continue
        prov = json.loads(pv.read_text()); cmp = rd(o / "compare_orig.csv")
        st = cmp[0]
        if kind == "start" and arm == "control":
            tab = (P.C1_TABLE if part == "C1" else P.C3_TABLE if part == "C3" else None)
            res["start"].append(dict(part=part, seed=seed, task=fr,
                                     uniform_L1_replay=fnum(st["uniform_L1_replay"]), uniform_L1_orig=fnum(st["uniform_L1_orig"]),
                                     uniform_L0_replay=fnum(st["uniform_L0_replay"]), uniform_L0_orig=fnum(st["uniform_L0_orig"]),
                                     chain_L1_replay=fnum(st["chain_L1_replay"]), chain_L1_orig=fnum(st["chain_L1_orig"]),
                                     spec_S=(tab[seed][1] if tab else 25), spec_death_ge09=(fnum(st["uniform_L1_replay"]) >= 0.9)))
        if kind == "start" and arm == "lnex1":
            res["lnex1"].append(dict(part=part, seed=seed, task=fr, k=prov.get("lnex1_k"), den_orig=prov.get("lnex1_den_orig_mean"),
                                     den_used=prov.get("lnex1_den_used_mean"), rel_diff=prov.get("lnex1_den_rel_diff"),
                                     start_uniform_L1_diff=fnum(st["uniform_L1_diff"])))
        if kind == "replay1":
            r1 = cmp[1]; pt = rd(o / "per_task.csv")[0]
            res["replay1"].append(dict(part=part, seed=seed, arm=arm, task=to, online_bitwise=r1.get("online_bitwise"),
                                       online_maxabs=fnum(r1.get("online_maxabs")), online_mean_diff=fnum(r1.get("online_mean_diff")),
                                       online_nonfinite=r1.get("online_nonfinite"),
                                       uniform_L1_replay=fnum(r1["uniform_L1_replay"]), uniform_L1_orig=fnum(r1["uniform_L1_orig"]),
                                       uniform_L0_replay=fnum(r1["uniform_L0_replay"]), uniform_L0_orig=fnum(r1["uniform_L0_orig"]),
                                       chain_L1_replay=fnum(r1["chain_L1_replay"]), chain_L1_orig=fnum(r1["chain_L1_orig"]),
                                       sec_task=fnum(pt["sec"]), seconds_total=prov["seconds"], gpu_mem_MB=prov.get("gpu_max_mem_alloc_MB"),
                                       freeze_max_abs_diff=prov.get("freeze_max_abs_diff"), gpu=prov["gpu"]))
    import numpy as np
    res["repro"] = []
    for kind, part, seed, arm, fr, to, march in plan():
        if kind != "replay1b":
            continue
        a_, b_ = odir("replay1", part, seed, arm), odir("replay1b", part, seed, arm)
        if (a_ / "online.npz").exists() and (b_ / "online.npz").exists():
            x, y = np.load(a_ / "online.npz")["loss"], np.load(b_ / "online.npz")["loss"]
            la, lb = rd(a_ / "layer_stats.csv"), rd(b_ / "layer_stats.csv")
            res["repro"].append(dict(part=part, seed=seed, arm=arm, online_bitwise=bool(np.array_equal(x, y)),
                                     online_maxabs=float(np.abs(x.astype(float) - y).max()), layer_stats_identical=(la == lb)))
    s = res["start"]
    v = dict(
        start_n=len(s),
        start_uniform_L1_exact=sum(x["uniform_L1_replay"] == x["uniform_L1_orig"] for x in s),
        start_uniform_L1_absdiff_max=max((abs(x["uniform_L1_replay"] - x["uniform_L1_orig"]) for x in s), default=None),
        start_uniform_L0_exact=sum(x["uniform_L0_replay"] == x["uniform_L0_orig"] for x in s),
        start_chain_L1_exact=sum(x["chain_L1_replay"] == x["chain_L1_orig"] for x in s),
        start_death_ge09=sum(x["spec_death_ge09"] for x in s),
        lnex1_n=len(res["lnex1"]), lnex1_rel_diff_max=max((x["rel_diff"] for x in res["lnex1"]), default=None),
        replay1_n=len(res["replay1"]), repro_bitwise=[r["online_bitwise"] for r in res["repro"]], missing=len(res["missing"]))
    res["verdict"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks_C.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    {"run": run, "summarize": summarize}[sys.argv[1]]()
