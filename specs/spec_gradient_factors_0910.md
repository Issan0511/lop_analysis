# spec_gradient_factors_0910 — 生勾配をゲート因子と誤差因子に分け、ゲートだけを動かす

## 0. 位置づけ

[[step_persistence_0910]] で、幅の成長速度の腕差が

```
活性化 φ → 【未解明】 → ‖∇W1‖ → ρ → D² → ‖W̃‖ の速度
```

の左端 1 箇所に絞れた（右 3 本は測定済み: ρ ∝ ‖∇W1‖^−0.81 が 264 点で残差 sd 0.082、D² = S²·ρ で S² は腕間 CV 5.1%）。

事後の粗い見積り（z を正規と仮定した RMS φ′）は Spearman(RMS φ′, ‖∇W1‖) = +0.881、Spearman(RMS φ′, ρ) = −0.952 と強いが、3 つの穴がある:

1. leaky 族の内部順序が合わない（RMS φ′ は LR03 > LR001 > LR、実測は LR03 > LR > LR001）
2. log-log の傾きが **0.51** で、ゲートだけなら 1 のはず → **誤差側が半分打ち消している**
3. RMS φ′ は z̄・σ から計算しており、その z̄・σ は成長の結果 → **ループ**

私は φ′ の汎関数で既に 2 回外している（φ″ の曲率で `gate_scale_invariance_0909` の 4 判定、φ′ の変化量 gcorr で `step_persistence_0910` V4）。3 回目なので事後の相関では採らない。**A で厳密に分解し、B でゲートだけを外から動かす。**

## 1. A — 生勾配の厳密分解

第 1 層の勾配は行ごとに次の 4 因子へ**厳密に**分かれる。バッチ $X\in\mathbb R^{B\times784}$、$K=XX^{\mathsf T}$、$e=\partial L/\partial a_1\in\mathbb R^{B\times100}$（autograd）、$p=\varphi'_{\text{bwd}}(z_1)$、$d=e\odot p=\partial L/\partial z_1$ として、ユニット $i$ について

$$\lVert\nabla W_{1,i}\rVert^2 = d_i^{\mathsf T}K\,d_i = g2_i\cdot e2_i\cdot \xi_i\cdot R_i$$

- $g2_i=\frac1B\sum_s p_{s,i}^2$ — **ゲート因子**（活性化の形が直接入る唯一の項）
- $e2_i=\sum_s e_{s,i}^2$ — **誤差因子**（層 2 から降りてくる誤差の強さ）
- $\xi_i=\lVert d_i\rVert^2/(g2_i\,e2_i)$ — ゲートと誤差の**相関因子**（独立なら 1）
- $R_i=\hat d_i^{\mathsf T}K\hat d_i$ — 入力との**整列因子**（$\hat d=d/\lVert d\rVert$）

$\log$ を取れば $\log\lVert\nabla W_1\rVert^2$ の腕間分散が 4 項へ加法分解できる。寄与率
$f_\bullet = \mathrm{Cov}(\log\bullet,\ \log\lVert\nabla W_1\rVert^2)/\mathrm{Var}(\log\lVert\nabla W_1\rVert^2)$
は厳密に和が 1。

あわせて**プローブ上の $\langle\varphi'^2\rangle$ を実測**し（正規仮定を外す）、§0 の見積りと比較する。

## 2. B — ゲートだけを動かす介入

前向きを leaky $a_f=0.1$ に固定したまま、**逆伝播のゲートだけ** $\varphi'=\text{leaky}'(a_b)$ に差し替える腕を作る（カスタム autograd Function）。前向き関数はどの腕でも完全に同一なので、同じ $W$ に対する $z$ 分布も同一。動くのは勾配だけ。

| 腕 | $a_f$ | $a_b$ | 狙う RMS $\varphi'_{\text{bwd}}$（$p^+\!\approx\!0.18$） |
|---|---|---|---|
| `BL001` | 0.1 | 0.01 | 0.425 |
| `BL010` | 0.1 | 0.1 | 0.434 — **`LR` と厳密に同一でなければならない** |
| `BL050` | 0.1 | 0.5 | 0.620 |
| `BL100` | 0.1 | 1.0 | 1.000 |

`BL010` が `LR` とビット一致することが、カスタム Function が何も壊していないことの検査になる。

## 3. 腕・seed・走

参照 8 腕 `LR` `LR001` `LR03` `SN02` `SN06` `SN15` `ELU1` `LIN`（= `step_persistence_0910` と同一）＋ 介入 4 腕 = **12 腕 × 3 seed = 36 走 × 120 タスク**。測定は TRACK タスク（20, 30, …, 120）の全 625 ステップ。判定は t = 100, 120。同 seed の全腕は初期値・置換列・抽出・バッチ順を共有。host は編集しない。

## 4. 事前登録の予測と判定

### V1 `GRADIENT_LOCUS` — 腕差はどの因子に乗るか
参照 8 腕で $f_{g2}, f_{e2}, f_\xi, f_R$ を出す。
- `GATE_WITH_COMPENSATION` … $f_{g2}\ge0.6$ かつ $f_{e2}\le0$
- `GATE` … $f_{g2}\ge0.6$ かつ $f_{e2}>0$
- `ERROR` … $f_{e2}\ge0.6$
- `SPLIT` … その他

**予測: `GATE_WITH_COMPENSATION`。** 根拠: 見積りの傾き 0.51 は「ゲートが押し上げ、誤差が半分押し戻す」と読める。ゲートが大きい腕ほど層 2 がよく当たり誤差が小さいはず。

### V2 `GAUSSIAN_OK` — §0 の正規仮定は公正だったか
実測 $\langle\varphi'^2\rangle^{1/2}$ 対 正規見積りを 8 腕で比較。
- `OK` … Spearman ≥ 0.9 かつ最大相対誤差 ≤ 25%
- `BIASED` … その他

**予測: `BIASED`。** 根拠: Snake の見積りは $e^{-2\alpha^2\sigma^2}$ を含み、$2\alpha^2\sigma^2$ が 0.04〜25 と桁で動く。正規からのわずかなズレが効く。順序は残るが値は外れると見る。

### V3 `INTERVENTION` — ゲートだけ上げれば連鎖は動くか（**本題**）
4 つの `BL` 腕（t = 100, 120）で 3 条件:
(i) Spearman(RMS $\varphi'_{\text{bwd}}$, ‖∇W1‖) ≥ 0.8、(ii) Spearman(‖∇W1‖, ρ) ≤ −0.8、(iii) ‖W̃‖(t120) が RMS $\varphi'_{\text{bwd}}$ に対して単調減。
- `GATE_CAUSAL` … 3 つとも成立
- `PARTIAL` … (i) のみ成立
- `NO_EFFECT` … (i) が不成立

**予測: `GATE_CAUSAL`。**

### V4 `ON_THE_LINE` — 介入腕は既存の直線に乗るか
参照 8 腕で引いた $\log\rho = \alpha + \beta\log\lVert\nabla W_1\rVert$ に `BL` 腕を当て、残差を見る。
- `ON` … 最大 |残差| ≤ 0.25（参照の残差 sd 0.082 の約 3 倍）
- `OFF` … その他

**予測: `ON`。**

### V5 `Z_HELD` — ループは実際に切れたか（**限界の自己申告**）
`BL` 腕間の σ の相対ばらつきと z̄ の振れ幅。
- `HELD` … σ の幅 ≤ 25% かつ |Δz̄| ≤ 1.0
- `DRIFTED` … その他

**予測: `DRIFTED`。** 逆伝播を変えれば学習されるものが変わり、z̄・σ は結局ずれる。**ずれた場合 V3 は交絡を含む**ので、V5 の実測値を V3 の解釈に必ず添えて報告する。前向き関数と初期値が同一であることまでしか保証できない。

## 5. 検査（すべて変異対照つき）

| # | 検査 | 合格 | 変異対照 |
|---|---|---|---|
| G1 | `LR` が既存 committed checkpoint 42 点を再現 | params maxabs **0.0** | init に +1e−3 → ≥1e−4 |
| B0 | `BL010` が `LR` と全 120 タスクで一致 | params maxabs **0.0** | `BL050` と比較 → ≥1e−3 |
| C1 | $\lVert\nabla W_{1,i}\rVert^2 = d_i^{\mathsf T}Kd_i$（autograd の ∂L/∂W1 と照合） | 相対 ≤1e−12 | $K$ を転置前で作る → ≥1e−3 |
| C2 | $g2\cdot e2\cdot \xi\cdot R = \lVert\nabla W_{1,i}\rVert^2$ | 相対 ≤1e−12 | $\xi$ を 1 に固定 → ≥1e−2 |
| C3 | $d = e\odot\varphi'_{\text{bwd}}$ が autograd の ∂L/∂z1 と一致 | ≤1e−14 | $\varphi'_{\text{fwd}}$ を使う → `BL` 腕で ≥1e−3 |
| C4 | `graw` が `step_persistence_0910` と一致（参照 8 腕・同 seed） | **0.0** | 別タスクと比較 → ≥1e−2 |
| C5 | 測定の非侵襲（logging on/off の終端 params） | **0.0** | measure 時 +1e−9 → >0 |
| C6 | $\varphi'$ が autograd と一致（全腕・前向き／逆向き別々に） | ≤1e−14 | 別腕の $\varphi'$ → ≥0.5 |

C3 の変異対照は `BL` 腕でのみ意味を持つ（参照腕では $\varphi'_{\text{fwd}}=\varphi'_{\text{bwd}}$ なので恒真）。参照腕では別腕の $\varphi'$ を使う形に置き換える。

## 6. 限界（事前に宣言）

- V5 が `DRIFTED` なら V3 はループを完全には切れていない。前向き関数と初期値の同一性までしか保証されない。
- `BL100`（逆伝播が線形）は前向きと逆向きが不整合な最適化なので、損失が単調に下がらない可能性がある。下がらなければ V3 (iii) の解釈は保留する。
- $f_\bullet$ の分解は 8 腕 = 8 点。$\xi$ と $R$ は中間量で、機構的な意味づけは事後になる。
- 判定は t = 100, 120 のみ。
- SGD 対照は無い。
- 本 spec は「何がゲートを決めるか」には答えない。$g2$ は $z$ 分布と $\varphi$ の両方に依存し、$z$ 分布は成長の結果である。
