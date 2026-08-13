# methods_sde_0813: 既存手法 (Leaky ReLU / S&P / CBP) の SDE 分解 — 結果サマリ

2026-08-13 実行。270 runs / 36 groups、batch=1、1M step、lr=0.01、seed 5本。
数値は @1M step の seed 平均 (lop_metrics)。事前登録予測 (仕様 §3) との対照。

## 主要判定

### 1. Leaky ReLU: 「吸収壁の遮断」ではなく「連続減衰」 — 対抗仮説が勝つ

neg_gate_frac (ReLU 換算 dead 相当; P(pre>0) < 0.05 のユニット割合):

| | none | leaky0.01 | leaky0.1 | leaky0.3 |
|---|---|---|---|---|
| A w5 | 0.80 | **1.00** | 0.72 | 0.20 |
| A w100 | 0.968 | **0.978** | 0.822 | 0.266 |

- 先生予測「任意の α > 0 で dead 消失 (α 非依存の遮断)」は**棄却**。α=0.01 は 1M step
  経っても基準と同じ割合のユニットが負側に滞在し続ける (時系列でも none と重なる)。
- 復帰の速さは α に**単調** (α=0.3 でようやく大半が復帰)。eval_loss も同順で
  A_w100: 0.94 → 0.83 → 0.064 → 0.000。**復帰レートが α に比例する連続減衰**の描像。
- dead_frac (|a| 基準) は全 leaky アームで厳密に 0 — 仕様 §2(d)③ の予想どおり
  指標として無効。主判定を neg_gate_frac に置いた設計が機能した。
- wcos_mean の α 単調減少 (ゲートロック仮説の間接検証) は B_w100 c=2 で弱く出る
  (0.179 → 0.178 → 0.175 → 0.171) が A では非単調 — 明確な支持ではない。
- **副作用**: leaky は発散を増やす (B_w100 lr=0.01: leaky0.3 で c=0 の 5/5、c=2 の 1/5 が
  NaN 化、leaky0.1 でも 1 run)。負側にも勾配が流れる分、実効的な更新量が増えるため。

### 2. S&P perturb-only: Path A/B 逆向き効果の非自明予言が的中

B_w100 c=2 の dead_frac: none 0.112 → **perturb-only 0.742** (c=0: 0.008 → 0.176)。
時系列では perturb-only だけが単調増加し続ける。

- 仕様の副作用予測「diffusion 増は Path A (吸収壁到達) を促進しうる」が**そのまま観測された**。
  等方ノイズ注入は整列 (Path B) を押し戻す一方で dead 化 (Path A) を悪化させる —
  統一 SDE モデルの「ノイズは A に毒、B に薬」という予言の直接検証。
- 一方 wcos_mean は低下せず微増 (0.179→0.255; ただし dead 0.74 と交絡)。eff_rank_W の
  回復も見えず (20.4→19.4)。**「diffusion 床で整列を押し戻す」側の予測は支持されず**。
- eval_loss は悪化 (0.044→0.145)。perturb 単独は害の方が大きい。

### 3. S&P shrink-only: drift 抑制ではなく「W の低ランク化」が支配

- A_w100: eff_rank_W 15.1 → **3.47**、top1_frac 0.25 → **0.68**、wcos_mean 0.29 → **0.037**。
  B_w100 でも eff_rank_W 20.4 → 17.6、top1_frac 増。
- 解釈: 毎ステップの等方収縮が勾配の来ない方向を 0 に潰し、W が少数方向に集中する。
  wcos の激減は「操舵権限回復」(v2 §5(b)) とも「発火率経由の drift 鈍化」とも整合するが、
  eff_rank_W の激減と top1_frac の増加は**むしろ低ランク化の促進**であり、
  「S&P が Path B を救う」という単純な図式には乗らない。dead_frac もほぼ不変 (0.93)。
- 予告どおり 2 説の分離は本実験ではできない (射影半径スイープの管轄)。

### 4. S&P 両方 (標準形): 相殺で両 Path とも改善

A_w5 dead 0.80→0.08、A_w100 0.968→0.380、eval_loss 0.94→0.18 (A_w100)。
shrink 単独の低ランク化 (eff_rank_W 3.5) も perturb 単独の dead 増も、併用では消える
(eff_rank_W 15.6 ≈ 基準、dead は基準以下)。**S&P の効能は分子と分母の合わせ技**で、
どちらか片方では逆効果になりうる — D* = σ²/(2λ) の形と整合的。

### 5. CBP: reset (ジャンプ項) は両 Path を同時に抑える最強アーム

- A_w100 (rho=1e-4): dead 0.968 → **0.034**、wcos 0.291 → 0.190、eval_loss 0.937 → **0.005**。
  rho=1e-5 でも dead 0.370 / eval_loss 0.0004。A_w5 でも同様 (0.80 → 0.08)。
- snr_drift は倍増 (0.15→0.30): 再初期化ユニットに生きた勾配が流れ続ける。
- 報告上の位置づけは仕様の注記どおり「drift/diffusion 修正系 (Leaky, S&P) + reset 系 (CBP)」
  の二分で書く (CBP は SDE の連続項ではなく離散ジャンプ項)。

## 補足

- 条件B (整列系) は w5/w100 とも dead がほぼ出ない (c=2 の none で 0.11 が最大) ため、
  Path A の判定は条件A が担う設計どおりの結果。
- 条件B の wcos_mean は手法間の差が小さい (0.17–0.26)。1M step・batch=1 では
  整列への介入効果は Path A への効果より一桁弱い。
- 凍結測定 (freeze) は run_freeze: false でスキップ。snr_drift は lop_metrics 内の
  eval バッチ計測 (仕様 §2(d)①)。
- 実行: 36 グループを CPU 並列 (w5: 1スレ×4 / w100: 4スレ×6、計28コア) で約17分。
  GPU は batch=1 のカーネル起動律速で CPU 4スレより遅く不採用。

## 図

`figures/fig_ms_<metric>.png` (時系列、method 色分け・c 線種) /
`fig_ms_bar_<metric>.png` (最終値 method 横断バー、seed 平均 ± SD)。
metric ∈ {dead_frac, neg_gate_frac, wcos_mean, eff_rank_W, top1_frac, snr_drift,
trC_W, eff_rank, eval_loss}。
