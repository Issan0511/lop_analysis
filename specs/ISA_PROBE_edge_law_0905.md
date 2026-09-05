# 別マシンで走らせてよいかの判定（35 秒）— `edge_law_0905`

この走の登録検査 `S-null` / `S-null-E` / `S-mirror` は、**このマシン（`white-san`・i7-14700K）で作った committed ログとのバイト比較**である。`S-mirror` は**命題 1 の確証的統計量**なので、バイトが 1 つでもずれれば命題 1 は判定不能になる。**しかもこの 3 つは走の中で出る検査なので、620 CPU 分を使い切ったあとで判明する。**

したがって**別マシンを使う前に必ずこのプローブを通す**こと。所要 35 秒。

## 0. 参照側（`white-san`・作成済み）

```
機械           white-san / Intel(R) Core(TM) i7-14700K
ISA            AVX2（AVX-512 なし）・avx_vnni あり・L3 33 MiB
torch          2.13.0+cu130      cpu_capability = AVX2
numpy          2.5.1   python 3.12.3   pyyaml 6.0.3
OMP_NUM_THREADS=1     git_head = dad3bc7
腕/steps       LRnull_1216 / 100,000
```

参照ハッシュは `reference.json`（10 seed × `state_hash_final` ＋ 数値列 63 本の sha256 先頭 16 桁）。
場所: このセッションのスクラッチパッド `isa_probe/reference.json`。**リポジトリに置く場合は `results/_isa_probe_edge_law_0905/reference.json`。**

## 1. なぜ CPU の型番が効くのか（実測）

`ATEN_CPU_CAPABILITY` でディスパッチ経路を切り替えると、この網が使う演算の**バイトが全部変わる**:

| 演算 | DEFAULT | AVX2 |
|---|---|---|
| `einsum` | `ff6bd246…` | `ef321345…` |
| `elu` | `0c2a3f61…` | `680748b9…` |
| `tanh` | `e89eab32…` | `7fdb2146…` |
| `softplus` | `5cd8d85d…` | `68ec31af…` |

さらに `MKL_VERBOSE=1` で、毎ステップの `torch.einsum("rhd,rd->rh", W, x)` が **Intel MKL の `SGEMM_BATCH`** に落ち、**`CNR:OFF`（数値再現性オフ）で Intel AVX2 + DL Boost カーネル**を選んでいることを確認済み。つまり:

- **AVX-512 を持つ CPU** → torch が AVX512 経路を選ぶ → 不一致（GCP の Intel 系は全部これ）
- **AMD** → MKL のベンダ判定が別経路 → 不一致（GCP の c2d/t2d）
- **Alder/Raptor Lake の consumer 部品**（AVX2 止まり＋`avx_vnni`）→ **一致の見込みがあるのはここだけ**

`i9-13900KF` は Raptor Lake なのでこの条件を満たす。ただし **L3 が 33 → 36 MiB と違う**ので、MKL のブロッキングが変わらない保証はない（行列が 100×20 と極小なので効かないはず、という程度）。**だから測る。**

## 2. 走らせる側の手順

```bash
# (a) コードを揃える（dad3bc7 以降）
git -C <repo> fetch && git -C <repo> checkout dad3bc7   # または最新の main

# (b) 環境を揃える。**同じ wheel を使う**こと（+cpu wheel は MKL のリンクが違いうる）
python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0+cu130 numpy==2.5.1 pyyaml==6.0.3
#   RTX 4090 + ドライバ 560 で cu130 wheel は「import できる」だけでよい（GPU は使わない。
#   登録 config が gpu: false。torch.cuda.is_available() が False でも一切問題ない）

# (c) 自分の環境を確認
grep -c avx512 /proc/cpuinfo          # 0 であること
grep -c avx_vnni /proc/cpuinfo        # 1 以上であること
OMP_NUM_THREADS=1 .venv/bin/python -c \
  "import torch,numpy,sys;print(sys.version.split()[0],torch.__version__,numpy.__version__,torch.backends.cpu.get_cpu_capability())"
#   期待: 3.12.3 2.13.0+cu130 2.5.1 AVX2

# (d) プローブ本体（35 秒）
cd <repo>
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 \
  --arm LRnull_1216 --steps 100000 --outdir /tmp/isa_probe

# (e) 参照と突き合わせ
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python - <<'PY'
import numpy as np, glob, json, hashlib
ref = json.load(open("results/_isa_probe_edge_law_0905/reference.json"))   # 置いた場所に合わせる
bad = []
for f in sorted(glob.glob("/tmp/isa_probe/logs/*.npz")):
    seed = f.split("seed")[-1].split(".")[0]
    z = np.load(f, allow_pickle=True)
    r = ref["seeds"][seed]
    if str(z["state_hash_final"]) != r["state_hash_final"]:
        bad.append((seed, "state_hash_final"))
    for k, h in r["col_sha"].items():
        got = hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()[:16]
        if got != h:
            bad.append((seed, k))
print("MISMATCH" if bad else "MATCH — このマシンで本走してよい")
for b in bad[:20]:
    print("   ", b)
PY
```

## 3. 判定

- **MATCH**: そのマシンで本走してよい。`S-null` / `S-null-E` / `S-mirror` は committed 参照に対してそのまま成立する。
- **MISMATCH**: **許容誤差でごまかさない。** 選択肢は 2 つだけ。
  1. `white-san` で走らせる。
  2. そのマシンを新しい正本にする — つまり**参照腕も同じマシンで作り直す**（`src/p3_runs_0902.py --exp extend --arm LR_1216 --steps 5000000` と `--arm E_1216`）。ただしこれは `gate_dose_0830 → gate_dial_0902 → p3_extend_0902` と続いてきた**バイト連鎖を切る**ので、結果ノートに逸脱として明記し、`S-null` が証明する内容が「committed 宿主出力の再現」から「同一マシン上の宿主出力の再現」へ**弱くなる**ことを書く。

## 4. MATCH だった場合、その機械で何が良くなるか

| | white-san | i9-13900KF 機 |
|---|---|---|
| 実コア | 20（8P+12E） | **24（8P+16E）** |
| メモリ | 30 GiB | **62 GiB** |
| 30 腕同時投入（42 GiB 必要） | **不可**（段階投入が要る） | **可** |
| 壁時計 | 段階投入で 2.5〜3 h | **臨界パスのみ ≈ 50 分** |

**今夜の律速はメモリだった**ので、62 GiB は素直に効く。`--launch-plan --parallel 30` で一括投入できる。

GPU は**使わない**（`gpu: false`。GPU は縮約順が変わって bit 一致検査が壊れる。そもそもバッチ 1・20→100→1 の網では CPU の方が 2.5 倍速い）。

## 5. 参照ログの扱い

`S-null` / `S-mirror` / 登録判定は committed 参照ログを読むが、これは `.gitignore` されていて `white-san` にしかない（`p3_extend_0902/logs` 2.6 GB ＋ `gate_dial_0902/logs` 2.2 GB）。

**推奨の分担**: 別マシンは**訓練だけ**、`--tail-extract` で絞って持ち帰り、**検査と登録判定は `white-san`** で回す。こうすれば参照ログを転送する必要がなく、登録された数値がすべて同じ numpy ビルドの上で計算される。
