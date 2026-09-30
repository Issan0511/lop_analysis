# sink_roots_0930 追補 3 — 導出の一巡目の依頼の残り（R2p・R3p・R1p）（登録・走る前）

書いた: Claude（fork2）、2026-09-30。依頼: 本体セッション「ELUが沈降しない事件」の `run_requests_round1.md`（写しは `specs/sink_roots_0930_round1_requests.md`）の「その後」の 5 件のうち、設計が依頼の文面で決まる 3 件。R4_late_sign_boxes と condA_balance_check は箱と量の定義を本体に確かめてから別に登録する。予言は依頼の原文のまま §4 に写す。

## 1. R2p_branch_depth（同じ状態からの枝を seed と活性化に広げる）

- 本体の `adam/derive/branch_probe.py`（`adam_window_probe.py` を import）をそのまま `src/sink_roots_round3/adam/` に写して回す。1 層 RL-MNIST、ELU・GELU・SiLU・leaky 0.1 × seed 0〜2 × 切替 3・10・20、腕は既定の 9 本（base・β₂ 0.9・0.99・0.9999・v×10・v×0.1・SGD 0.01・0.1・long 16,000 更新）。
- 集計 `analysis/sink_roots_0930/r3_branch_depth.py`: 切替時に開いた unit で、深さ = 腕の課題全体（4,000 更新、long は 16,000）の軌道（10 更新刻み）の最小の中央値、転回点 = 最小の時刻の中央値、戻り÷深さ = 中央値(末 − 最小) ÷ (−深さ)。同じ切替の base との比。
- 判定: ELU は全腕・全切替・全 seed の深さの比が [0.8, 1.25] に入る割合。GELU・SiLU は β₂ 0.9 の転回点の比 ≥ 2 の数と、戻り÷深さが base より小さい数。

## 2. R3p_confidence（深さを決めるのは p_old か、logit の散らばりか、課題の長さか）

- エンジン `src/sink_roots_mnist_0930.py` に `--stop`（8eec266、既定の計算は不変を確認）: 課題の当てはめが目標に届いたら課題を切り上げる。500 更新以降 25 更新ごとに全 1,200 枚で判定、上限 `--T 16000`。目標は (i) 今のラベルの p の平均（= 次の切替の p_old）0.26・0.43・0.57、(ii) logit の散らばり（probe の logit_sd と同じ定義）1・2.4・7。ELU・SiLU・leaky × seed 0〜2 × 30 課題（54 走）。名前 `R3p_<pold|lsd><目標>_<act>_s<seed>`。
- 集計 `analysis/sink_roots_0930/r3_confidence.py`: 課題 5〜30 の各課題で、切替時に開いた unit の「課題全体の probe の最小 − 切替時」の中央値を深さとし、切替時の p_old・logit_sd と前の課題の長さと並べる。
- 判定（依頼の予言の読み）: (1) ELU の散らばりの腕で −深さ/散らばり が [1.6, 2.3]、深さが散らばり 1・2.4・7 で約 −1.6〜−2.3・−3.8〜−5.5・−11〜−16。(2) 活性化ごとに課題単位で全 6 腕をまとめ、log(−深さ) の log(散らばり) への R² が log(p_old) への R² より大きい。(3) p_old の腕の中（p_old をそろえた中）で log(−深さ) と log(散らばり) の相関が正。

## 3. R1p_cifar_path_ledger（RL-CIFAR ELU/std の崩壊までの窓で、交差の経路の配分）

- 新しい script `src/sink_roots_round3/cifar_path_ledger.py`: rlcifar_mlp_battle_0918 の箱（ELU/std、seed の 1,200 枚、課題ごとの乱数ラベル、30,000 更新/課題、Adam 1e−3）を R = 1 の CPU で、エンジンの eager の 1 歩と同じ式で回す。seed 0〜4 × 課題 1〜3。100 更新ごとに W₂・b₂・μ₂（1,200 枚の第 1 層の出力の平均）・m₂・k₂ を保存。各課題の末で 6 つの重みを R7 の A 腕（同じ箱の自然な走）の snapshot とビット一致で照合する（初期値は一致を確認済み）。
- 集計 `analysis/sink_roots_0930/r3_path_ledger.py`: unit ごと・窓ごと（各課題と課題 1〜3 の通し）に端点の帳簿（自分・上流・交差）と経路の帳簿（100 更新刻み、恒等式の閉包を確認）、交差のうち自分へ行く分の割合、f・g（端点の向きへの射影の時間の形）と ∫g df（左リーマン和）、前半と後半の増分の cos（W と μ）。合成の例（f = t、g = t^0.3）で割合と ∫g df が 0.767 で一致することを確認済み。
- 判定: W と μ の両方の cos が 0.8 以上の unit-窓で、|自分の割合 − ∫g df| ≤ 0.2 の割合（窓の種類ごと）。配分の向きは記述だけ。

## 4. 予言（依頼の原文のまま）

- R2p_branch_depth: ELU の深さは全腕で基準の 0.8〜1.25 倍。GELU・SiLU は β₂ 0.9 で転回点が 2 倍以上遅れ、戻り÷深さは基準より小さい（全課題の走と同じ向き）
- R3p_confidence: 深さは p_old ではなく logit の散らばりに従う。ELU で深さ ÷ 散らばり = 1.6〜2.3（散らばり 1・2.4・7 に深さ約 −1.6〜−2.3・−3.8〜−5.5・−11〜−16）。(i) の腕で p_old をそろえても、散らばりが違えば深さは散らばりの側に動く
- R1p_cifar_path_ledger: 配分の向きは予言しない（RL-MNIST の課題 1〜3 で、自分の取り分は −0.05〜0.86）。予言は式の側に置く。前半と後半の増分の向きの cos が 0.8 以上の課題では、経路の自分の取り分は ∫g df と ±0.2 で合う

## 5. 走らせ方

`analysis/sink_roots_0930/launch_round3.py`（追補 2 と同じ仕組み: 自分の走の合計 12 本の枠、SIGSTOP で止めた走は数えない、停止は `results/sink_roots_0930/STOP_round3`）。順は R1p（5 本、長い）→ R2p（12 本）→ R3p（54 本）。
