# results/mlp_center_x10_1020

spec: `specs/spec_mlp_center_x10_1020.md`。5+1 CIFAR MLP（`cifar5p1_mlp_0920` の箱）を課題あたり 7,800 更新で、R・CH・CHB0 の 3 腕、seed 0〜9、std、lr 1e−4。

## 既存の 0920 の 10 倍走に何が残っているか（2026-10-20 に調べた）

- `results/cifar5p1_mlp_0920/` の 10 倍走は `R_std_x10_7800x7800`・`CH_std_x10_7800x7800`・`KKT1_std_x10_7800x7800`（hard・easy とも 7,800）と、`CH_std_x10_7800x780`・`KKT1_std_x10_7800x780`（hard だけ 7,800）の 5 本。seed はどれも 0〜9。CHB0 は無い。
- 各ディレクトリには `per_task.csv`・`fresh_control.csv`・`provenance.json` しか無い。
- `per_task.csv` の unit 統計（`zbar`・`zsd`・`mob`・`w_norm` など）は課題の訓練画像の上で取った unit の中央値で、全 unit の配列は無い。課題末の重み・checkpoint・課題の中の学習曲線も無い。
- fresh 対照（課題 29 を初期値から 7,800 更新）はある。
- よって 3 腕とも診断つきで回し直した。旧走の軌道を再現するのは R だけで、CH は扉 C の平均画像（GPU の float32 の和）の丸めが違うため別の実現になる（spec §3、checks の S-reuse-early・S-door-repro）。

## 中身

| パス | 内容 |
|---|---|
| `<腕>_std_x10_7800x7800/per_task.csv`・`fresh_control.csv` | 0920 のエンジンがそのまま書く行 |
| `<腕>_std_x10_7800x7800/diag_units.npz` | unit ごとの配列（課題, slot, unit）。probe（test の 1,200 枚）と課題の訓練画像の上の z̄・σ・開く数、実効階数、bias、行ノルム、扉 H の m、20 更新ごとの online の累積、fresh の累積 |
| `<腕>_std_x10_7800x7800/diag_per_task.csv` | (seed, 課題) ごとの要約: 最初の 100・780 更新と課題全体の online、σ の中央値、p⁺ の 6 区分の割合、実効階数 |
| `checks/checks.json` | 本走前の検査（`src/mlp_center_x10_1020_checks.py`） |
| `smoke/` | seed 100・3 課題・CHB0 の試走 |
| `production.log`・`exit_code.txt` | 本走のログと終了コード（`analysis/mlp_center_x10_1020/production.sh`） |
| `verdict.json`・`summary.md`・`paired_tests.csv` | 判定（`analysis/mlp_center_x10_1020/verdict.py`） |
