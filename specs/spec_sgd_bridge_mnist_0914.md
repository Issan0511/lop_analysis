# sgd_bridge_mnist_0914 spec：箱 B の optimizer だけを SGD に替える（SGD 橋）

状態: **事前登録（実装前・走る前）** / 更新: 2026-09-14 JST / 起草: Claude（Opus 5）
親: `specs/spec_snake_phase_mnist_0914.md`（aa78461）§9-2 で後段に回した「段階 S（SGD 橋）」を、範囲を絞って独立の実験として登録する / vault [[Snakeの位相をMNISTへ移す_結果_0914]]
発端: Issa（2026-09-14 午前）「peak は重みを縮めましたか？」→ Claude「MNIST・Adam では縮まず normal より育った。CondA とは optimizer・損失・深さが同時に違う」→ Issa「SGD 橋いりそうですね」→ 範囲は「α=0.6 の 5 腕」を選択
run id: `sgd_bridge_mnist_0914` / worktree `wt/sgd_bridge_mnist_0914` / branch `claude/sgd_bridge_mnist_0914`（origin/main cbac3ee から）

> 数値の等級: **登録** = results/*/summary.md・verdict.csv から写したもの。**事後** = 起草時に既存ログから計算したもの。

## 0. 問い

snake_phase_mnist_0914（Adam）では、peak（P06）は normal（N06）より隠れ層の中心化ノルム ‖W̃ᵢ‖ が大きく育ち（CT1 層 1 ΔlogN +0.120、`GROWTH_ENHANCED`、登録）、初期値より縮むユニットは層 1 で 0%（登録 REPORT）だった。CondA（1 層・MSE・SGD）では peak の自由重み成長が normal より小さく（log 差 −0.09877、`DIRECTIONAL_PHASE_SUPPRESSION`、登録）、peak の 70.8% の個体が縮んだ。

**同じ箱 B・同じ初期値・同じデータ列のまま optimizer だけを plain SGD に替えると、(i) peak − normal の成長差の符号は CondA 側へ動くか、(ii) 初期値より縮む多数と育つ少数への分岐が現れるか。** あわせて時間劣化（E1–E3）を 0914 と同じ規則で出す。

この実験で分けられるのは optimizer（Adam → SGD、および更新数 625 → 2,500）の効果だけで、損失（CE 対 MSE）・入力・深さは CondA と違ったまま。

## 1. 箱

箱 B（0914 と同一: 784–100–100–10・CE・batch 16・1 タスク 10,000 枚・正則化なし・t1–120・CPU float32・1 スレッド・宿主のストリームと初期化）。**optimizer だけを替える**: plain SGD（`p -= lr·g`）、lr 0.02、1 タスク = 同じ 625 batch 順を 4 周（2,500 更新）。これは `pmnist_boundary_host_0908.run_one(optimizer='sgd', epochs=4)`（`src/pmnist_0905.py` とバイト一致）の更新規則と同一で、pmnist_adapt_0905 の SGD 箱と同じ lr・epoch（その箱は 200 タスク・CUDA だったので bit の錨にはしない。錨は §6 S16 の run_one との一致）。lr は腕ごとに調整しない。

## 2. 腕（5）と seed

0914 と同じ定義（`src/snake_phase_acts_0914.py`）、init map なし: **N06**（normal α=.6）、**P06**（peak）、**V06**（valley）、**LIN**、**LR**（leaky a=.1）。seed 0–9（0914 と同じ seed = 同じ初期 W とタスク列）。10 seed × 5 腕 = 50 走。走る順は seed 昇順・腕交互。

## 3. 記録

0914 §5 と同じ（rows.csv・units.npz・snapshot t20/t60/t120・provenance）。fresh probe も同じタスク（1・16–20・101–120）で、その腕の θ₀ から **SGD で 2,500 更新**。学習ループは 0914 の `src/snake_phase_mnist_0914.py` を import し、Adam の 1 タスク関数だけを SGD の同じ形の関数に差し替えて使う（ファイルは編集しない。差し替えは provenance に記録）。

## 4. 主 endpoint と判定

検定・CI・Holm・n_min・分解能マージンの計算は 0914 §6–§7 と同じ（正確な符号反転検定、反転 CI、族内 Holm で m 固定、n_min(m) = ⌈log₂(2m/.05)⌉）。n = 10。

### 4.1 G：SGD 下の成長差（CT1-SGD）

ΔlogN_ℓ = ½ log meanᵢ,t∈101..120 cnormᵢ²（θ）− 同（N06）、層 ℓ = 1, 2。対比 C1 P06−N06、C2 V06−N06。**1 族 m=4**（2 対比 × 2 層）。ラベル（0914 §6.7 と同じ）: `GROWTH_SUPPRESSED`（Holm 有意・負）／`GROWTH_ENHANCED`（有意・正）／`GROWTH_BELOW_CONDA_SCALE`（(1−2·.05/4) 反転 CI ⊂ [−0.09877, +0.09877]）／`INCONCLUSIVE`。有意でも CI がマージン内なら接尾 `_BELOW_CONDA_SCALE`。

### 4.2 X：optimizer × 位相の交互作用（確認的な見出し）

seed k ごとに X_k = ΔlogN_ℓ,k(SGD) − ΔlogN_ℓ,k(Adam)。Adam 側は 0914 の同じ seed の値（`results/sgd_bridge_mnist_0914/adam_reference.json`、本 spec と同じ commit で固定。0914 の units.npz から算出し sha256 付き）。同じ seed で初期 W とタスク列が同一なので対になる。対比 C1・C2 × 層 1・2、**1 族 m=4**。ラベル: `X_NEGATIVE`（Holm 有意・負 = SGD で peak/valley の成長差が Adam より小さい向き）／`X_POSITIVE`／`X_EQUIVALENT`（(1−2·.05/4) 反転 CI ⊂ [−0.09877, +0.09877]）／`INCONCLUSIVE`。修飾 `SIGN_REVERSED`: Adam の登録ラベル（C1 層 1 `GROWTH_ENHANCED` など）と SGD の 4.1 のラベルが逆向きに有意。

**見出しは X のラベル**。「optimizer が peak の成長の向きを決める」と書けるのは、X_NEGATIVE かつ SIGN_REVERSED（SGD で GROWTH_SUPPRESSED）のときだけ。X_NEGATIVE だけなら「SGD では peak の余分な成長が小さくなる」まで。

### 4.3 F：個体の分岐（腕 × 層ごと）

seed ごとに f = strict な縮小（‖W̃ᵢ(t120)‖² < ‖W̃ᵢ(0)‖²、許容幅なし・CondA の `n2[-1] < n2[0]` と同じ）のユニット割合。μ̂・σ̂ = そのseed の log(cnormᵢ(t120)/cnormᵢ(0)) のユニット間の平均と SD（ddof 1）、単峰の正規近似の予測 f₀ = Φ(−μ̂/σ̂)、超過 e = f − f₀。
- `FORK_PRESENT`: f の反転 95% CI の下端 > 1/100、かつ e の反転 95% CI の下端 > 0。
- `FORK_ABSENT`: f の反転 95% CI の上端 < 1/100。
- `FORK_INCONCLUSIVE`: それ以外。
1/100 は 100 ユニット層の分解能（1 seed あたり 1 ユニット）。CondA の比較値（REPORT）: peak q0 70.8%・normal q0 27.1%（`zero_attraction_learning_0913` の ALL 行、登録の事後集計）。

### 4.4 E：時間劣化・水準・fresh gap（副次）

0914 §6.1–§6.4 と同じ定義・ラベル・総合クラス。族 FS = {C1 P06−N06, C2 V06−N06}、m=2。可検定性は D_pair(N06, LIN)（TESTABLE_FIXED）。分解能マージンは基準対 N06−LIN・LR−LIN から。E の窓は同じ（base t16–30、late t101–120）。

### 4.5 REPORT（ラベルなし）

腕ごとの D・A_late・Gap・logN1/logN2 の late 値と初期値からの倍率、f の seed 平均、log 成長比 SD、Adam 走の同じ腕との差。M1 の Φ（C1・C2、層 1・2、両窓）。

## 5. 段階とゲート（全自動）

1. **G1 事前登録**: 本 spec・config・adam_reference.json を commit・push。vault に写しと現在地 1 行。
2. **検査**（§6）all_pass・変異検出、実装を commit・push。
3. **安定性パイロット**: seed 100–102 × 5 腕 × t1–40（`results/_smoke_sgd_bridge_mnist_0914/pilot/`、endpoint に入れない）。どれかの腕で非有限（DIVERGED）→ lr を 0.01 に下げた箱として追補に記録し 1 回だけ再パイロット。再パイロットでも発散 → `SGD_UNSTABLE` で本走を止めて報告。
4. **資源**: P06 seed 0 の 120 タスク 1 走で RSS_peak と壁時計を測る。
5. **本走** 50 走（0914 の launch と同じメモリの式・見張り・追補 1 の swap 条件）。
6. **判定** → summary.md・verdict.csv → vault 結果ノート → 生データ退避 → main 統合・片付け。

## 6. 検査（変異対照つき）

| # | 検査 | 変異対照 |
|---|---|---|
| S16 | **SGD の橋**: 新ループ（N06・LR・LIN、seed 0、3 タスク、extras なし）の各タスク終端 acc が `H.run_one(arm∈{SN06,LR,LIN}, optimizer='sgd', epochs=4, lr=.02, device=cpu)` の acc と完全一致（文字列） | epochs=3／lr=0.0201 → 不一致を検出 |
| S6s | **fresh probe（SGD）**: t1 の fresh が逐次 task 1 の終状態を bit 再現 | 開始点を task 1 終状態にする → 不一致 |
| S7s | **非侵襲（SGD）**: 3 タスクで extras あり・なしの毎タスクのパラメータ sha256 が同一 | probe が batch 生成器から 1 回引く → t3 以降で相違 |
| S8s | **ストリーム**: 各走の provenance の stream_sha256 が 0914 の同じ seed の走と一致 | seed を 1 ずらした比較 → 不一致を検出 |
| S14s | **判定コード**: 合成データで 4.1–4.3 の全ラベル・SIGN_REVERSED が期待どおり | 腕の入れ替え（符号反転）／Adam 参照の seed 対を崩す（CI が広がる）／FORK の strict < を ≤ 0.95·c0 に替える（境界ユニットの数え方が変わる） |
| S15s | **launch の自己検査**（0914 と同じ見張りの検査） | 0914 と同じ変異 |

## 7. 事前予測（Claude・走る前）

| # | 予測 | 確率 | 反証 |
|---|---|---|---|
| Q1 | S16 の橋が完全一致 | .90 | 不一致 |
| Q2 | lr 0.02 のパイロットで 5 腕とも発散しない | .70 | どれかが DIVERGED |
| Q3 | TESTABLE_FIXED（N06−LIN の D_pair CI_lo > 0）が SGD でも成立 | .70 | 不成立 |
| Q4 | 見出し X の C1 層 1 は X_NEGATIVE（SGD では peak の余分な成長が Adam より小さい） | .60 | X_POSITIVE か X_EQUIVALENT |
| Q5 | SGD の C1 層 1 は GROWTH_SUPPRESSED（CondA と同じ向き・SIGN_REVERSED） | .35 | それ以外 |
| Q6 | 5 腕とも層 1 で FORK_ABSENT（CE・784 入力・SGD でも CondA 型の分岐は出ない） | .55 | どれかが FORK_PRESENT |
| Q7 | 層 2 で少なくとも 1 腕が FORK_PRESENT か FORK_INCONCLUSIVE | .40 | 全腕 FORK_ABSENT |
| Q8 | FS 族の C1（P06−N06）の E1 は 0914（Adam）と同じ LESS_DECLINE | .45 | それ以外 |

**Issa**: 未記入（記入を待たずに登録する。書く場合は addendum で区別）。

## 8. 引用上の限定

- SGD の 2,500 更新/タスク は Adam の 625 と更新数も違う（pmnist_adapt_0905 の SGD 箱の規則をそのまま使ったため）。X は「optimizer と更新数を合わせて替えた効果」。
- CondA との残りの違い（MSE 対 CE、32 点支持の 5 自由入力 対 MNIST 784 入力、1 層 対 2 層、α=1 対 0.6）は分けない。
- n=10。腕間の幅の差は W 病理の証拠ではない（0914 と同じ）。
