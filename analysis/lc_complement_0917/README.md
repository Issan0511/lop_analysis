# LC complement implementation

登録仕様は [spec_lc_complement_0917.md](../../specs/spec_lc_complement_0917.md)。
このディレクトリの実装完了と、本走の実施・完了は別。

## 環境と検査

repo root から、既存環境（torch/numpy/pandas入り）で実行する。
追加依存は不要。CPUはwhite-sanの同じtorch・AVX2。checkpointの出所照合を弱めて
別CPUで再利用しない。データは `data/mnist/` に既存MNISTの4つのgzを配置する。
worktreeならこのディレクトリを唯一のcloneのdata/mnistへsymlinkしてよい。

```bash
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
"$PY" -m unittest analysis.lc_complement_0917.checks -v
"$PY" -m src.lc_complement_0917 c1-ee --seed 0 --smoke --out /tmp/lc_c1_smoke
"$PY" -m src.lc_complement_0917 c2 --seed 0 --smoke --out /tmp/lc_c2_smoke
"$PY" -m src.lc_complement_0917 c3 --seed 0 --arm S36 --smoke --out /tmp/lc_c3_smoke
```

C2 smokeは1epochの人工的な短走なので3000更新の衝撃やK_revを作動させない。
その境界はunit testで全6000更新について検査する。C3 smokeは2task×1epoch。
空でない出力フォルダを上書きしない。同じ名前で再実行する場合は出力先を変える。

## 本走

コードとspecをcommit/pushしてから実行。以下は**順次実行**で、勝手にGPUやCPUの並列数を増やさない。
実行時間: C1 CPUは再解析のみ、C1 GPUは150taskの再生、C2は240継続task、
C3は2250万更新。GPU再生とC3を含む本走は長時間を要する。

```bash
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=/home/issan/Projects/obsidian-research-data/lc_complement_0917/main
for s in {0..9}; do
  "$PY" -m src.lc_complement_0917 c1-ee --seed "$s" --out "$OUT/c1/s$s" || break
done
"$PY" -m analysis.lc_complement_0917.verdict c1 --input "$OUT/c1" --out results/lc_complement_0917/c1

"$PY" -m src.lc_complement_0917 c1-gpu --out "$OUT/c1_gpu"
"$PY" -m analysis.lc_complement_0917.verdict c1-gpu --input "$OUT/c1_gpu" --out results/lc_complement_0917/c1_gpu

for s in {0..9}; do
  "$PY" -m src.lc_complement_0917 c2 --seed "$s" --out "$OUT/c2/s$s" || break
done
"$PY" -m analysis.lc_complement_0917.verdict c2 --input "$OUT/c2" --out results/lc_complement_0917/c2

for arm in E1 S36 C36 E36 E1_lr1e3; do
  for s in {0..2}; do
    "$PY" -m src.lc_complement_0917 c3 --arm "$arm" --seed "$s" --out "$OUT/c3/$arm/s$s" || break 2
  done
done
"$PY" -m analysis.lc_complement_0917.verdict c3 --input "$OUT/c3" --out results/lc_complement_0917/c3

"$PY" -m src.lc_complement_0917 c4 --out results/lc_complement_0917/c4
```

デフォルトのC1/C2入力は `obsidian-research-data/resp_ee_0917/results/resp_ee_0917/runs/`。
CSV/provenanceの正本はclone内の `results/resp_ee_0917/runs/`。
`--archive` と `--reference` で指定可能。ただしcheckpointの絶対backupパスとhashは
commit済みmanifestに記載されたものに一致する必要がある。

## 出力と制限

- `provenance.json`: 開始時のコード・spec・環境。正常終了だけCOMPLETE。
- `rows.csv`: C1は分岐時の層別、C2はbranch×arm×継続task、C3はtask別。
- `units.npz` / `curves.npz` / `checkpoint.pt`: 生配列。gitに入れない。
- `verdict.csv`, `verdict.json`, `summary.md`: 全seed揃った場合だけ集計。
- C3 checkpointは事故時の状態保存用。自動再開機能はない。
- C2の「CIが0を含む」は同等性の検定ではない。C4は回復時間の下限のみ。
- GPUの同一軌道は本走中に既存onlineとz平均の全行一致で検証する。
  他GPUでの一致や150task全再生を短い実装検査の合格から主張しない。

本走終了後に小さい集計結果と生データmanifestをcommit/pushし、mainへ統合・worktreeを削除する。
