# sink_roots_0930 追補 5 — 導出の二巡目の依頼の優先 3〜5 のうちエンジンで回せるもの（登録・走る前）

書いた: Claude（fork2）、2026-09-30。依頼と予言の原文は `specs/sink_roots_0930_round2_requests.md`（本体の `run_requests_round2.md` の写し）。予言は §3 に原文のまま写す。

この追補に入れるのは、エンジン（`src/sink_roots_mnist_0930.py`、9a63e4e で --bias-floor を足した。既定の計算は不変を確認）か v4 の箱だけで回せる 5 件。2 層の再生の 4 件（RR-1 replay_factors・RR-2 gate_frozen_backward・RR-3 center_input_full・RR-4 refit_2layer）、G6_cap_shrink_ledger（導出役の replay_geom.py）、K_sweep_label_luck（「有利」の定義が要る）、RR4_twolayer_caps（第 2 層の上限をエンジンに足す）は、この後に別に登録する。

## 1. 走

- **G4_silu_gelu_cap_scale**（優先 3）: SiLU・GELU × 行ノルムの上限 ×0.5・×2（課題 5 から、`--wcap-from 5 --wcap-scale`）× seed 0〜2 × 30 課題（12 走）
- **bias_shift_depth**（優先 3）: SiLU・GELU × seed 0〜2 × 30 課題。課題 5 から各課題の頭で、z の中央値（全 1,200 枚）が −8 より深い unit の bias を、中央値が −8 になるまで上げる（下げない。W と Adam は触らない。`--bias-floor -8 --bias-floor-from 5`）（6 走）
- **G5_age_erase**（優先 4）: ELU・leaky 0.1 × seed 0〜2 × 40 課題。課題 5〜40 の毎課題の頭の状態を保存（`--snap-before 5 … 40`）。f_k の帯ごとの計算は本体の側（依頼の「本側で出す」）（6 走）
- **cap_plus_absphi_GELU**（優先 5）: GELU × seed 0〜2 × 30 課題。行ノルムの上限と逆伝播の |φ′|（bwabs）を、どちらも課題 2 から（`--wcap-from 2 --bwmode abs --bw-from 200`。bwabs は R3 と同じく各課題の 201 更新目から）（3 走）
- **R2e_valley_long_eps**（優先 5）: v4 の箱（`src/sink_roots_round4/valley/alpha_ladder_1layer.py`）、GELU × ε1 {1e−6, 1e−12} × seed 0・1 × 1,000 課題、z の snapshot は 100 課題ごと（4 走）

## 2. 集計

- G4・bias_shift_depth・cap_plus_absphi: `analysis/sink_roots_0930/r4_readouts.py` の量（課題末の k・上端/s・m/s・m・s・正答率・被覆 0・全閉、Δm・Δs、上限に張り付いた割合と効き始めの課題）に、被覆 0 の入力の課題末の正答率（correct_end）を足す
- R2e: 追補 2 の §7 と同じ（200 課題ごとの全閉・μ₊・μ₋・帯・壁）に、閉じた期間の KM（始まりの時期 100 課題ごと × 壁（上端 ≤ −6.75）に触れたか）と、年齢 100 以上の危険率（事象 30 以上の区間だけ）を足す
- G5: 状態を置くだけ（`obsidian-research-data/sink_roots_0930/mnist/G5age_<act>_s<seed>/state_before_tXXX.npz`）

## 3. 予言（依頼の原文のまま）

- G4_silu_gelu_cap_scale: 課題 30 の k の中央値はどの腕も 0〜5。SiLU: ×0.5 は m −3〜−5・正答率 ≥ 0.95。×2 は課題 17〜20 で上限が効き、m −15〜−22・正答率 0.35〜0.60。GELU: ×0.5 は m −3〜−5・正答率 ≥ 0.8。×2 は上限がほとんど効かず、正答率 0.28〜0.36。SiLU ×2 の正答率 ≥ 0.85 なら、帯の中の入力の数は変数ではない
- bias_shift_depth: [候補・par の腕からの類推] SiLU の課題 30 の正答率は 0.50〜0.80、全閉 ≤0.10（確信 50%）。GELU は 0.30〜0.60（45%）。SiLU が 0.9 以上なら深さだけで足り、0.4 以下なら深さは担い手でない
- G5_age_erase: ELU で Σ で重みを付けた危険率は、年齢 1 で 0.15〜0.35、年齢 10 以降 ≤ 0.08。帯ごとの危険率は λ の 0.3〜0.6 乗で増える。leaky は年齢 1 で 0.3 以上で、ELU より大きい。年齢 1 と年齢 10 の差が 0.03 未満なら、先生の 2 項則とは別の型
- cap_plus_absphi_GELU: 課題 30 の正答率 ≥0.85、全閉 ≤0.10（確信 50%）
- R2e_valley_long_eps: (a) 1e−6 の全閉（801〜1000）≥0.87（55%）。(b) 1e−12 の全閉（801〜1000）は 151〜200 より 0.04 以上高い（55%。導出役の頭打ちの予言と逆なので、この走で分かれる）。(c) 壁に触れない遠出でも、始まりが遅いほど S_C(10) が高い（60%）

## 4. 走らせ方

`analysis/sink_roots_0930/launch_round5.py`（同じ仕組み、合計 10 本の枠を共有）。追補 4 の優先 1・2 がすべて起動してから起動する。
