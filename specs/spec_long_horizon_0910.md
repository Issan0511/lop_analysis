# 長い地平線での leaky と ELU の分岐 spec 0910
状態: 検証中 / 更新: 2026-09-10
親: [[駆動源問題_0909]] §10 / [[幅の規制は可塑性を買うか_0910]] / [[ELUの幅成長と沈下_結果_0909]]
run id: `long_horizon_0910`

## 0. 問い（Issa・2026-09-10 夜）
「leaky は LoP しない気がする（幅が増えているだけで斉次的だから）。ELU はだいぶ悪くなりそう。」
120 タスクまでの既発表データ（`elu_growth_0909`）では両者の精度低下はまだ同程度（t20→t120: leaky −2.9 pt、ELU1 −2.2 pt）。ただし leaky は死亡ユニット 0/100 で傾きが減速（−2.4 → −1.8 pt/100task）、ELU1 は死亡 ~10/100・飽和対 0.59 で傾きが加速（−1.3 → −2.0）。**分岐は 120 タスクでは未決。地平線を 400 に伸ばして決める。**

## 1. 走
`LR`（leaky 0.1）・`ELU1`・`SNA` × seed 0..2、init から t1–400、介入なし（参照軌道のみ）。同じ init・ストリーム（`H.init_params`・`H.stream`）。毎タスク終端で `width_sink_clamp_0909` と同じ測定（zbar/sigma/between/pos_frac/star_sd/cnorm/w2col/acc/ce_probe、per-unit npz）に加えて **`dead_hard`** = 全プローブ標本でゲート φ′ < 1e−6 のユニット数（float32 の実質的な死）と **`dead_soft`** = φ′ < 0.05 のユニット数。
G1: t1–120 の per-unit ‖W̃ᵢ‖ が `elu_growth_0909` と maxabs ≤ 1e−10。
コスト: 1 腕seed ≈ 400 タスク × 0.6 s ≈ 4 分。9 本・2 プロセスで ≈ 20 分。

## 2. 判定（事前登録・3 seed 一致でのみラベル）
窓: **late = t301–400**、参照 = t20–120。精度は 100 タスク窓のタスク終端精度の中央値。
- **A. 分岐**: d = acc_late(LR) − acc_late(ELU1)（pt、seed 対応）。3 seed とも d ≥ 1.0 → `ELU_DIVERGES`；3 seed とも |d| < 0.5 → `NO_DIVERGENCE`；他 `PARTIAL`。
- **B. leaky の頭打ち**: LR の late 窓の精度傾き（pt/100task）。3 seed とも > −0.5 → `LEAKY_PLATEAUS`；3 seed とも < −1.0 → `LEAKY_KEEPS_FALLING`；他 `PARTIAL`。
- **C. ELU の加速**: ELU1 の late 傾きが t20–120 の傾きより 3 seed とも負に大きい → `ELU_ACCELERATES`；3 seed とも小さい → `ELU_DECELERATES`；他 `PARTIAL`。
- **D. 死の蓄積**: ELU1 の `dead_hard`(t400) が 3 seed とも ≥ 20 → `DEATH_ACCUMULATES`；3 seed とも ≤ 10 → `DEATH_SATURATES`；他 `PARTIAL`。LR の `dead_hard` は構成上 0 でなければならない（検査）。
- 副（ラベル無し）: SNA の同量、ELU1 の acc 低下と dead_hard/sat の対応、leaky の acc 低下と 1/‖W̃‖ の対応。

## 3. 事前予測（記名・走る前）
- **Issa**: A `ELU_DIVERGES`、B `LEAKY_PLATEAUS`（leaky は斉次なので幅の成長は尺度で吸収され LoP しない。ELU は非斉次で深いユニットが死ぬ）。
- **Claude**: A `ELU_DIVERGES`（1〜2 pt・D は `DEATH_ACCUMULATES`）。B は **`PARTIAL`**: leaky は死なないが、‖W̃‖ ∝ √t で Adam の相対歩幅が 1/√t で縮み、softmax も鋭くなるので**緩く落ち続ける**（late 傾き ≈ −1 pt/100task、t20–120 の −2.4 の 0.45 倍）。分岐点は **B**: Issa は頭打ち、Claude は緩い低下。C は `ELU_ACCELERATES`。
- 反証の形: B が `LEAKY_PLATEAUS` なら Claude の「尺度で緩く落ちる」が外れ。A が `NO_DIVERGENCE` なら両者外れ。

## 4. 検算
G1（上）・`dead_hard`(LR) ≡ 0・測定非侵襲（測定込みで G1 を通す）・変異対照（init +1e−3 で G1 が落ちる）。
