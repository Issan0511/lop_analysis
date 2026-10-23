#!/usr/bin/env python3
"""chargpt_seed10_1022 の D の本走の駆動（手順書 §3.5・§5）: seed 0〜9 を 1 本ずつ順に（GPU 1 本）。

- 完走した seed（provenance.json があり、tasks = 100・n_layer = 4・fresh = 10/25/50/100・ckpt の移しが済み）は飛ばす。
- 途中で止まった seed の出力は消さずに <out>.partial_<時刻> へ名前を変えて、その seed を頭から回し直す
  （元の道具は課題の途中からの再開を持たない。CSV と npz は走の終わりにまとめて書かれる）。
- 完走したら、checkpoint（*.pt）を共通の約束の規則で raw/ に移し、名前に .raw. を入れる:
    <S10>/results/chargpt_seed10_1022/silu4/raw/SiLU/seed<k>/<元の名前の .pt を .raw.pt に>
  元の名前・置き場・sha256・大きさを <out>/ckpt_manifest.json に書く（学習と測定には触らない後処理）。
- かかった時間は provenance.json の wall_sec・task_secs（元の道具）に加え、drive_wall_sec を足す。

使い方: $PY drive_silu4.py [--seeds 0 1 ...] [--dry]
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CG = Path(os.environ.get("CHARGPT_SRC", "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019"))
S10 = Path("/home/kubo/project/Nakatsuka/lop_seed10")
PY = CG / ".venv" / "bin" / "python"
WRAP = S10 / "analysis" / "chargpt_seed10_1022" / "run_silu4.py"
ROOT = S10 / "results" / "chargpt_seed10_1022" / "silu4"
FRESH = [10, 25, 50, 100]


def out_dir(k):
    return ROOT / "SiLU" / f"seed{k}"


def raw_dir(k):
    return ROOT / "raw" / "SiLU" / f"seed{k}"


def cmd(k):
    return [str(PY), str(WRAP), "run", "--box", "M", "--arm", "SiLU", "--seed", str(k), "--tasks", "100",
            "--n_per_task", "10000", "--alpha", "0.01", "--fresh_tasks", *map(str, FRESH), "--out", str(out_dir(k))]


def env():
    e = dict(os.environ, CHARGPT_SRC=str(CG), CHARGPT_NLAYER="4", CHARGPT_DECOMP_CKPT="1")
    return e


def run_complete(k):
    pv = out_dir(k) / "provenance.json"
    if not pv.exists():
        return False
    d = json.loads(pv.read_text())
    return (d.get("tasks") == 100 and d.get("n_layer") == 4 and d.get("arm") == "SiLU" and d.get("seed") == k
            and d.get("fresh_tasks") == FRESH and d.get("n_per_task") == 10000 and "wrapper_blob" in d)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def move_ckpts(k):
    src = out_dir(k) / "ckpts"
    man_p = out_dir(k) / "ckpt_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else dict(files=[])
    dst = raw_dir(k)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.pt")) if src.exists() else []:
        q = dst / (p.name[:-3] + ".raw.pt")
        man["files"].append(dict(orig=f"ckpts/{p.name}", raw=str(q.relative_to(S10)), sha256=sha256(p), bytes=p.stat().st_size))
        shutil.move(str(p), str(q))
    if src.exists() and not any(src.iterdir()):
        src.rmdir()
    man.update(rule="lop_seed10_work/_common.md: *.pt は raw/ に置き名前に .raw. を入れる", n=len(man["files"]))
    man_p.write_text(json.dumps(man, indent=1))
    return man["n"]


def done(k):
    return run_complete(k) and (out_dir(k) / "ckpt_manifest.json").exists() and not (out_dir(k) / "ckpts").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    for k in a.seeds:
        if done(k):
            print(f"seed{k}: 完走済み、飛ばす", flush=True)
            continue
        if run_complete(k):                                    # 学習は済み、ckpt の移しだけ残っている
            print(f"seed{k}: 学習は済み、ckpt を移す {move_ckpts(k)} 本", flush=True)
            continue
        if a.dry:
            print(f"seed{k}: 回す: {' '.join(cmd(k))}", flush=True)
            continue
        o = out_dir(k)
        if o.exists():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            o.rename(o.with_name(o.name + f".partial_{ts}"))
            print(f"seed{k}: 途中の出力を {o.name}.partial_{ts} に退けた", flush=True)
        t0 = time.time()
        with open(ROOT / "logs" / f"SiLU_seed{k}.log", "a") as log:
            log.write(f"# {datetime.datetime.now().isoformat()} {' '.join(cmd(k))}\n")
            log.flush()
            rc = subprocess.call(cmd(k), env=env(), stdout=log, stderr=subprocess.STDOUT)
        if rc != 0 or not run_complete(k):
            raise SystemExit(f"seed{k}: 失敗（rc {rc}）。{ROOT}/logs/SiLU_seed{k}.log を見る。同じコマンドで再開できる")
        pv = o / "provenance.json"
        d = json.loads(pv.read_text())
        d.update(drive_wall_sec=time.time() - t0, drive=str(Path(__file__).resolve()))
        pv.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        n = move_ckpts(k)
        print(f"seed{k}: 完走 {time.time() - t0:.0f} 秒、ckpt {n} 本を raw に移した", flush=True)


if __name__ == "__main__":
    main()
