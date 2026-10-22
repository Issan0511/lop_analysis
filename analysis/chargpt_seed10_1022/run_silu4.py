#!/usr/bin/env python3
"""chargpt_seed10_1022 の D（spec §6・追補 1、手順書 §3.5）: 4 ブロック SiLU の新しい学習の薄い wrapper。

写し元（書き換えない・import して使うだけ）:
  $CG/src/chargpt_cl_1019_kune.py  git blob 272e75548fef16d364c464327c8e44d034b9cbb2（$CG HEAD 033679e0）
  $CG/src/chargpt_cl_1019.py       git blob cc524f85d5f2d9813a272abb6b9476b065f1c17b
変えた所は 1 つだけ: 保存の課題の集合に spec §6 の課題を足す。
  課題末（C.CKPT_TASKS）       : 元 = {1,10,25,50} ∪ 1〜10 ∪ 39〜50 ∪ 89〜100（kune の run_wrap）に 11〜15・59〜70 を足す
  更新 100（K.DECOMP_TASKS）   : 元 = 1〜10 ∪ 40〜50 ∪ 90〜100 に 11〜15・60〜70 を足す
  → 課題末 = 1〜15・25・39〜50・59〜70・89〜100、更新 100 = 1〜15・40〜50・60〜70・90〜100（spec §6 のとおり）。
Adam の状態は kune の既存の仕組み（課題末は torch.save の差し替え、更新 100 は run_task_freeze）でそのまま付く。
学習・測定・乱数・腕の定義には触らない。

走らせ方（手順書 §3.5）:
  CHARGPT_SRC=$CG CHARGPT_NLAYER=4 CHARGPT_DECOMP_CKPT=1 $PY run_silu4.py run --box M --arm SiLU --seed <k> \
      --tasks 100 --n_per_task 10000 --alpha 0.01 --fresh_tasks 10 25 50 100 --out <out>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

CG = Path(os.environ.get("CHARGPT_SRC", "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019"))

# spec §6 の保存の課題（走る前に固定。ここを変えない）
SPEC_END = set(range(1, 16)) | {25} | set(range(39, 51)) | set(range(59, 71)) | set(range(89, 101))
SPEC_U100 = set(range(1, 16)) | set(range(40, 51)) | set(range(60, 71)) | set(range(90, 101))
ADD_END = set(range(11, 16)) | set(range(59, 71))
ADD_U100 = set(range(11, 16)) | set(range(60, 71))
KUNE_BLOB = "272e75548fef16d364c464327c8e44d034b9cbb2"
BASE_BLOB = "cc524f85d5f2d9813a272abb6b9476b065f1c17b"


def git_blob(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def setup():
    if os.environ.get("CHARGPT_NLAYER") != "4":
        raise SystemExit("run_silu4.py: CHARGPT_NLAYER=4 が要る（spec §6 の 4 ブロックの箱）")
    if os.environ.get("CHARGPT_DECOMP_CKPT") != "1":
        raise SystemExit("run_silu4.py: CHARGPT_DECOMP_CKPT=1 が要る（spec §6 の保存）")
    kb, bb = git_blob(CG / "src" / "chargpt_cl_1019_kune.py"), git_blob(CG / "src" / "chargpt_cl_1019.py")
    if (kb, bb) != (KUNE_BLOB, BASE_BLOB):
        raise SystemExit(f"run_silu4.py: 写し元の版が違う kune {kb} base {bb}")
    sys.path.insert(0, str(CG / "src"))
    import chargpt_cl_1019_kune as K  # noqa: E402
    C = K.C
    assert C.L == 4, C.L
    assert "SiLU" in C.ARMS and C.ARMS["SiLU"].get("act") == "silu" and not C.ARMS["SiLU"].get("cap"), C.ARMS.get("SiLU")
    C.CKPT_TASKS = tuple(sorted(set(C.CKPT_TASKS) | ADD_END))   # kune の run_wrap がこれに 1〜10・39〜50・89〜100 を足す
    K.DECOMP_TASKS = set(K.DECOMP_TASKS) | ADD_U100             # run_task_freeze は K の大域名を走るたびに引く
    end_final = set(C.CKPT_TASKS) | set(range(1, 11)) | set(range(39, 51)) | set(range(89, 101))
    assert end_final == SPEC_END, sorted(end_final ^ SPEC_END)
    assert K.DECOMP_TASKS == SPEC_U100, sorted(K.DECOMP_TASKS ^ SPEC_U100)
    return K, C, kb, bb


def out_arg(argv):
    return Path(argv[argv.index("--out") + 1]) if "--out" in argv else None


def main():
    K, C, kb, bb = setup()
    C.main()
    out = out_arg(sys.argv)
    pv = out / "provenance.json" if out else None
    if pv is not None and pv.exists():                         # 何を足したかを来歴に残す
        d = json.loads(pv.read_text())
        assert sorted(d["ckpt_tasks"]) == sorted(SPEC_END), "課題末の保存の集合が spec §6 と違う"
        d.update(wrapper=str(Path(__file__).resolve()), wrapper_blob=git_blob(Path(__file__).resolve()),
                 kune_blob=kb, base_blob=bb, u100_tasks=sorted(K.DECOMP_TASKS),
                 spec="lop_seed10/specs/spec_chargpt_seed10_1022.md §6・追補 1")
        pv.write_text(json.dumps(d, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
