# 第1層の増幅器は Adam か — RL-CIFAR 層別キメラの ε 介入（le_eps_cifar_0922）

書いた: Claude, 2026-09-22 13:00. 依頼: Issa「やってほしいです」（12:5x、下の仮説の検証）。
親: `specs/spec_layer_chimera_cifar_0921.md`（S-A）。エンジン・データ・登録判定の規則は親のまま。

## 0. 問い

S-A で LE（leaky→ELU, raw）は第2層が t2–t5 に死に、同じ時刻に第1層の正側ユニットが
a₁ rms 4 → 88、‖µ₂‖ 48 → 847 まで伸びた。LL の第1層は同じ leaky で 4–5 のまま。
「第2層の死への反応」とだけ書いて機構は未解明だった。

Issa の仮説（12:4x）: 第2層の活性が −1 に張り付き W₂ が µ₂ に反平行に偏った状態は、
符号を反転すれば正の重みと 1−e^{−z} の飽和活性化と同値で、要求された第2層の修正
δh を届けるには δz = δh·e^{|z₂|} が要る。その指数的な要求が第1層に伝わって伸ばす。

Claude の位置づけ（12:5x、スナップショットの事後観察 10 seed 中央値）:
- 逆伝播はこの倍率を掛けない。割る（第1層に届く勾配は e^{z₂} 倍に減衰）。
  戻すのは Adam の正規化で、勾配が e^{−20} でも歩幅は lr のまま。
- 実測: t2→t3 の ΔW₁ rms は 0.140（LL の健康時 0.15–0.18 と同じ）なのに、
  e^{z₂} > 1e−8 の (unit, image) 対は 31% → 0.2%（83 枚・32 unit）→ 6 枚 → 0。
- 向き: ΔW₁ の行と x̄ の |cos| は 0.06 → 0.38 → 0.70 → 0.85（LL は 0.03）。
  raw CIFAR は 1 枚ずつが x̄ に似る（cos 中央値 0.91）ので、少数画像で決まる一歩は ±x̄ になる。
- −∇W₁L（課題開始時）と ΔW₁ の cos は死後 0.22–0.36、健康時 0.04–0.11。
- 止まるのは微分の厳密な 0（t7–8 で ΔW₁ rms = 0.000）。W₂ は t4 から凍結（行ノルム 2.562 不変）、
  以後の z̄₂ の沈み −231 → −564 は全部 ‖µ₂‖ × 凍った反平行（2.56 × 847 × (−0.25) ≈ −542）。

仮説の検証可能な形: **第1層の伸びは、消えかけた勾配を Adam が lr の歩幅に戻して作る。**
√v ≪ ε の領域では Adam は SGD（実効 lr = lr/ε）になり、e^{z₂} 倍の勾配は e^{z₂} 倍の歩幅しか作れない。
よって ε を上げれば、第2層の死後の第1層の伸びは消える。

## 1. 介入

Adam の ε（`p -= lr·m̂/(√v̂ + ε)`）を 1e−8 から上げる。他は親と同一（lr 1e−3, β 0.9/0.999, WD 0, raw, R=10）。

ε は算術で決める（`/tmp/.../le_demand.py`, 3 seed, 75 minibatch of 16 の per-parameter 勾配 rms = √v の代理）:

| 状態 | W₁ の √v 中央値 | 90% | 最大 |
|---|---|---|---|
| 健康・課題開始付近（LE t1/t2, LL t3/t5） | 5e−4 〜 3e−3 | 3e−3 〜 1e−2 | 0.01 〜 0.04 |
| 覚え終わった課題末（LE t1 s1/s2, LL t1） | 3e−6 〜 1e−5 | 1e−5 〜 5e−5 | 1e−4 |
| LE 死後 t3 末 | 5e−7 〜 4e−5 | 6e−6 〜 3e−4 | 7e−5 〜 1e−3 |
| LE 死後 t4 末 | ≤ 6e−9 | ≤ 8e−8 | ≤ 1e−6 |

- **ε = 1e−4**: t4 以降は歩幅 ≥ 100 分の 1、t3 末は中央値で 4 分の 1。健康な課題開始の歩幅は 5% 減。
- **ε = 1e−3**: t3 末も中央値で 30 分の 1。健康な課題開始の歩幅は 3 割減（覚え終わりの微調整は 10 分の 1）。

健康側への影響は LL を同じ ε で回して測る（対照）。ε=1e−8 の腕は S-A の LE/LL（同じ seed・同じエンジン、S1 で bit 一致を確認）をそのまま使い、再走しない。

## 2. 腕

| 腕 | セル | ε | 走 |
|---|---|---|---|
| LE_e8 | LE | 1e−8 | S-A の `results/layer_chimera_cifar_0921/LE/`（再走なし） |
| LE_e4 | LE | 1e−4 | 本走 |
| LE_e3 | LE | 1e−3 | 本走 |
| LL_e8 | LL | 1e−8 | S-A（再走なし） |
| LL_e4 | LL | 1e−4 | 本走 |
| LL_e3 | LL | 1e−3 | 本走 |

seed 0–9、raw、50 課題、R=10 の束で 1 腕 1 ジョブ、4 ジョブ並列。

## 3. 測るもの

親の `measure.py` の列（a_rms, mu_norm, zbar_med, p_pos, w_row_med, cos_med）に加え、
スナップショットから（`analysis/le_eps_cifar_0922/tail.py`, 事後観察と同じ定義）:
- `open_frac`: e^{z₂} > 1e−8 の (unit, image) 対の割合。`n_img`, `n_unit`: それを持つ画像数・unit 数。
- `dcos`: ΔW₁(t−1→t) の行と x̄ の |cos| の中央値。`drms`: ΔW₁ の rms。
- `dcos_grad`: −∇W₁L（t−1 の状態、課題 t のラベル、full batch）と ΔW₁ の cos。

すべて seed ごとに出し、判定は seed 中央値と seed 対の符号検定（親 §5: 勝ち = 負け ≤ 1 かつ p < 0.05）。

## 4. 登録判定

記法: r_t = a_rms₁(t5) / a_rms₁(t2)（同じ腕の同じ seed）。S-A の実測: LE 5.64–12.71（中央値 9.14）、LL 0.78–0.87。
境 2.2 = √(0.87 × 5.64)（両帯の最も近い端の幾何平均）。

**Q1（主）** 第1層の伸びは消えるか（LE_e3 の r）:
- `AMPLIFIER_IS_ADAM`: r の中央値 < 2.2 かつ 10/10 seed で r < 5.64。
- `REDUCED`: 中央値 < 5.64 だが上を満たさない。
- `NOT_ADAM`: それ以外。
LE_e4 は同じ規則で副ラベル（`_e4` を付す）。単調性の記録: 中央値 r が e8 > e4 > e3 なら `MONOTONE`。

**Q2** ‖µ₂‖ と z̄₂ の沈み（LE_e3, t5）: seed 対で LE_e8 と比べる。
- ‖µ₂‖(t5) が 10/10 で LE_e8 より小さく、中央値が LE_e8 の t2 の値（48）の 2 倍未満 → `MU2_STOPPED`。
- z̄₂(t5) の中央値が LE_e8 の t2 の値（−30）の 2 倍より浅い（> −60）→ `SINK_STOPPED`。
  どちらか片方なら `PARTIAL`、両方外れれば `SINK_CONTINUES`。

**Q3** 向き（LE_e3, t2→t3 の dcos）: 中央値 < 0.15（健康帯 0.03–0.06 と S-A の 0.38 の幾何平均）→ `INCOHERENT`、
それ以外 `STILL_ALIGNED`。

**Q4** 救命（LE_e3）: 親 §5 の窓（t31–50 online）と死活（窓 < 0.5）。
- 窓の中央値 ≥ 0.5 → `RESCUED`；0.2 ≤ 窓 < 0.5 → `PARTIAL_RESCUE`；窓 < 0.2 → `NOT_RESCUED`（S-A の LE は 0.100）。
- 補助: t50 の open_frac（S-A LE は 0）。

**Q5** 健康側の税（LL_e3, LL_e4）: 窓の seed 対差（LL_eX − LL_e8; S-A LL 0.720–0.733）。
- 10/10 で |差| < 0.05 → `NO_TAX`；中央値の差 > −0.15 → `SMALL_TAX`；それ以下 → `LARGE_TAX`。
LARGE_TAX なら Q1 の `AMPLIFIER_IS_ADAM` は「健康側も止まる ε での結果」と注記する（主張の限定）。

## 5. 予測（走の前に埋める）

| # | 述語 | Claude | Issa |
|---|---|---|---|
| P1 | Q1 = AMPLIFIER_IS_ADAM（e3） | 0.85 | |
| P2 | Q1_e4 ∈ {AMPLIFIER_IS_ADAM, REDUCED} | 0.8 | |
| P3 | MONOTONE | 0.85 | |
| P4 | Q2 = MU2_STOPPED かつ SINK_STOPPED（e3） | 0.75 | |
| P5 | Q3 = INCOHERENT（e3） | 0.7 | |
| P6 | Q4 = NOT_RESCUED（e3） | 0.7 | |
| P7 | Q4 ∈ {PARTIAL_RESCUE, RESCUED}（e3） | 0.3 | |
| P8 | Q5(e3) ∈ {NO_TAX, SMALL_TAX} | 0.7 | |
| P9 | Q5(e4) = NO_TAX | 0.85 | |

P6/P7 の理由: t2 末で z̄₂ は −30 だが対の 31% はまだ開いている。ラチェット（‖µ₂‖ の伸び → 凍った W₂ 経由で残りが沈む）が止まれば開いた対が残り、部分的に学べる余地はある。ただし W₂ 自身の反平行が t2 で既にあるので、救命までは行かないと読む。

Issa の列は空けてある。解析（§8）の前に埋めればそのまま登録扱い、埋まらなければ Claude の列のみで採点する。

## 6. 検査（本走前）

- **S1 S-off**: `--eps 1e-8` で LE を 3 課題、S-A の `results/layer_chimera_cifar_0921/LE/per_task.csv` の t1–t3 と全列文字列一致、スナップショット bit 一致（obsidian-research-data の S-A 退避先と比較）。
- **S2 S-reach**: `--eps 1e-3` の t1 行が S1 と一致しない（ε が更新に届いている）。加えて手計算: 勾配 g ≪ ε の 1 パラメータで 1 歩が lr·g·(1/(1−β₁))/ε の float32 に一致（ステップ関数を R=1 の eager で 1 回呼ぶ）。
- **S3 S-resume**: `--eps` は ckpt の meta に入り、別の ε で resume すると拒否される。
- **S4 provenance**: `adam_eps` が provenance.json に入る。
- **S5 cost**: 1 課題の壁時計 × 50 で 4 並列の見積り（S-A の実測 2.0–2.4 h/セル）。

## 7. 手順

1. `src/layer_chimera_cifar_0921.py` に `eps` 引数（既定 1e−8、`--eps`）。meta・provenance に追加。
2. `analysis/le_eps_cifar_0922/`: `launch.py`（親の fork、`--eps`）、`tail.py`（§3）、`report.py`（§4）。
3. §6 → `results/le_eps_cifar_0922/checks.json`。
4. 本走 4 ジョブ → `results/le_eps_cifar_0922/<arm>/`。
5. measure（親の `measure.py --src`）→ tail → report → `summary.md`。
6. CLAUDE.md §4（退避・main・worktree 削除）。

## 8. 記録

- 13:00 spec commit（57c7b77）。12:5x に Issa「やってほしいです」。Issa の予測列は未記入のまま走に入った（Claude の列のみで採点）。
- 12:54 検査: S1 rows/snapshots bit 一致（30 行・40 スナップショット）、S2a 10/10 slot の t1 行が変わる、S2b replica 3.7e−8 対 変異 0.029、S3 拒否、S4 adam_eps 記録。`results/le_eps_cifar_0922/checks.json`。
- 12:54–14:19 本走 4 腕（4 並列、各 82–83 分、3 ms/step）。
- 14:2x tail → report: **Q1 REDUCED（e3 3.19, e4 4.52, e8 9.14; MONOTONE）**, Q2 SINK_CONTINUES, Q3 INCOHERENT, Q4 NOT_RESCUED, Q5 NO_TAX（e3, e4）。Claude 7/9（P1, P4 外れ）。読みは `results/le_eps_cifar_0922/summary.md` 事後の読み。
