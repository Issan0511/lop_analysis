#!/usr/bin/env python3
"""chargpt_seed10_1022 の B の本走の駆動（spec §4、手順書 §3.2・§5）。

単位 = 腕 × seed × 課題 k+1（1 単位 = rho10.py の 1 回 = 状態 a〔課題 k 末の重みと Adam〕、chain probe 32 系列、層 L0・L1）。
  腕: C3（主）と ReLUcap（記録）。--arms で選ぶ（主の C3 を先に回し、ReLUcap は後に回す）。
  seed 0〜9、k+1 ∈ {2, 3, 5, 10, 41, 45, 50, 91, 95, 100}。
出力 = $S10/results/chargpt_seed10_1022/rho/<腕>/seed<k>/t<k+1>_a_chain.json（rho10.py が一時 file → rename で書く）。
再開: 出力があり、層 0・1 と check_eval を持つ単位は飛ばす（同じコマンドで再開できる）。腕ごとの lock で同じ腕の二重起動を止める。
完走ごとに、時間と検査の結果を rho/<腕>/provenance.json に足す（単位ごとの 1 行）。全単位の検査の集計 = rho/<腕>/checks_summary.json。
検査が落ちた単位も出力は残し、集計に FAIL として載せる（判定は check_eval.all_pass の単位だけを使う）。終了コード = 落ちた単位があれば 3。
"""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path

S10 = Path(os.environ.get("S10", "/home/kubo/project/Nakatsuka/lop_seed10"))
CG = Path(os.environ.get("CHARGPT_SRC", os.environ.get("CG", "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019")))
PY = os.environ.get("PY", str(CG / ".venv/bin/python"))
RHO10 = S10 / "analysis/chargpt_seed10_1022/rho10.py"
OUT = S10 / "results/chargpt_seed10_1022/rho"
TASKS = (2, 3, 5, 10, 41, 45, 50, 91, 95, 100)


def unit_done(p):
    try:
        d = json.loads(p.read_text())
        return set(d["layers"]) >= {"0", "1"} and "check_eval" in d
    except Exception:
        return False


def summarize(arm):
    rows, fails = [], []
    for p in sorted((OUT / arm).glob("seed*/t*_a_chain.json")):
        try:
            d = json.loads(p.read_text()); ce = d["check_eval"]
        except Exception as e:
            fails.append(dict(path=os.path.relpath(p, S10), error=repr(e))); continue
        bad = {k: v["value"] for k, v in ce["items"].items() if not v["pass_"]}
        worst_id = max((v["value"] for k, v in ce["items"].items() if "MUT_" not in k), default=float("nan"))
        min_mut = min((v["value"] for k, v in ce["items"].items() if "MUT_" in k and not v.get("na")), default=float("nan"))
        rows.append(dict(path=os.path.relpath(p, S10), all_pass=ce["all_pass"], worst_identity=worst_id, min_mutation=min_mut,
                         n_mut_na=ce.get("n_na", 0), elapsed_sec=d["provenance"]["elapsed_sec"]))
        if not ce["all_pass"]: fails.append(dict(path=os.path.relpath(p, S10), failed=bad))
    s = dict(arm=arm, n_units=len(rows), n_pass=sum(r["all_pass"] for r in rows), n_fail=len(fails),
             n_units_with_mut_na=sum(r["n_mut_na"] > 0 for r in rows),
             max_worst_identity=max((r["worst_identity"] for r in rows), default=None),
             min_min_mutation=min((r["min_mutation"] for r in rows), default=None), failures=fails, units=rows)
    (OUT / arm / "checks_summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C3"], choices=["C3", "ReLUcap"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(TASKS), help="k+1（新しい課題）")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", default=None, help="検査用: 出力の置き場を替える（本走では使わない）")
    a = ap.parse_args()
    global OUT
    if a.out_root: OUT = Path(a.out_root).resolve()
    assert set(a.tasks) <= set(TASKS), "spec §4 の 10 課題の外"
    env = dict(os.environ, CHARGPT_SRC=str(CG)); env.pop("CHARGPT_NLAYER", None)
    any_fail = False
    for arm in a.arms:
        if not a.dry_run:
            (OUT / arm).mkdir(parents=True, exist_ok=True)
            lockf = open(OUT / arm / ".lock", "w")
            try:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                sys.exit(f"{arm}: 別の駆動が動いている（{OUT / arm / '.lock'}）")
        provp = OUT / arm / "provenance.json"
        prov = json.loads(provp.read_text()) if provp.exists() else dict(arm=arm, units={})
        for seed in a.seeds:
            for t in a.tasks:
                k = t - 1
                out = OUT / arm / f"seed{seed}" / f"t{t}_a_chain.json"
                if unit_done(out):
                    continue
                ck = CG / f"results/chargpt_cl_1019/M2/{arm}/seed{seed}/ckpts"
                cmd = [PY, str(RHO10), "--arm", arm, "--state", str(ck / f"task{k}.pt"), "--adam", str(ck / f"task{k}_adam.pt"),
                       "--task", str(t), "--kind", "chain", "--nseq", "32", "--device", a.device, "--jvp", "exact",
                       "--seed", str(seed), "--nlayer", "2", "--layers", "0", "1", "--threads", str(a.threads), "--out", str(out)]
                if a.dry_run:
                    print(" ".join(cmd)); continue
                out.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time(); started = time.strftime("%Y-%m-%d %H:%M:%S %z")
                with open(out.with_suffix(".log"), "w") as lg:
                    rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
                wall = time.time() - t0
                ok = rc == 0 and unit_done(out)
                cp = json.loads(out.read_text())["check_eval"]["all_pass"] if ok else False
                prov["units"][f"seed{seed}/t{t}"] = dict(started=started, finished=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                                         wall_sec=round(wall, 1), rc=rc, written=ok, checks_all_pass=cp,
                                                         host=os.uname().nodename)
                tmp = provp.with_name(provp.name + ".tmp"); tmp.write_text(json.dumps(prov, indent=1, ensure_ascii=False)); os.replace(tmp, provp)
                print(f"{time.strftime('%F %T')} {arm} seed{seed} t{t}: rc={rc} {wall:.0f}s checks={'PASS' if cp else 'FAIL'}", flush=True)
                if not ok:
                    any_fail = True
                    print(f"  失敗: log = {out.with_suffix('.log')}", flush=True)
        if not a.dry_run:
            s = summarize(arm); any_fail |= s["n_fail"] > 0
            print(f"{arm}: 単位 {s['n_units']}、検査 PASS {s['n_pass']}、FAIL {s['n_fail']}", flush=True)
    sys.exit(3 if any_fail else 0)


if __name__ == "__main__":
    main()
