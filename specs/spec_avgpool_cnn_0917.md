# avgpool_cnn_0917 — RL-CIFAR の CNN の max-pool を avg-pool に替えて、SNA の劣化と谷越えが消えるかを見る

書いた時刻: 2026-09-17 13:51（**実装の前・本走の前**）。Claude（Opus 5）。
経緯: swish_battle_0917 の途中の会話で、Issa「max-pool が悪いのか」「MLP と似てる挙動の CNN どれだと思う？」。
Claude が「max-pool を抜いた CNN（avg-pool かストライド付き conv）」と答え、切り分けの小さい方として avg-pool 版を提案した。Issa「おすすめの段取りで」。
親: `rlcifar_cnn_0908`（参照の max-pool 腕）、`swish_battle_0917`（同日・走行中。その Swish 系 CNN の結果はこの spec を書いた時点で未読）。

## 0. 一行

RL-CIFAR の CNN の max-pool 2 か所を avg-pool に替え、`SNA`（適応 c=0.6）と `SN3`（固定 α=3）を 10 seed ずつ回す。
**(A)** SNA の conv2 のチャネルが φ′ の谷（2αz̄ = −π/2）を越えなくなるか、**(B)** 順位が MLP の向き（SNA > SN3）になるかを決める。

## 1. なぜ（事後の観察。n=10、参照は `rlcifar_cnn_0908`・`swish_battle_0917`）

- 窓（t31–50）の順位は CNN と MLP で逆。CNN: SN3 0.964 > SNAc3 0.962 > SN06 0.958 > SNA 0.946。MLP: SNA 0.986 > SNAc3 0.981 > SN06 0.950 > SN3 0.900。
- CNN の SNA は c1・c2 のチャネル平均が谷を越えていく（c2 の 2α_med·z̄: t10 −1.12 → t50 −2.42。越えたチャネルの割合 t10 0.09 → t50 0.66）。
  fc1・fc2 は MLP のユニットと同じく谷の手前で止まる（−1.1〜−1.4）。WD 版（`SNA_l2`、λ=1e−4）は c2 が越えず（0.01）、劣化もしない。
- max-pool は (i) 各窓で勝った位置にしか勾配を流さない（宿主の `mob_pool` はその位置のゲート）、(ii) それ自体が折れ線の非線形（maxout と同じ形）。
  avg-pool はどちらも持たない（勾配は 4 か所に 1/4 ずつ、演算は線形）。avg-pool の直後の conv は、2×2 ごとに同じ値を持つカーネルのストライド 2 の conv と同じ計算。
- 反対側の事実:
  - max-pool の勝ち位置のゲートが低いのは、劣化しない `SNAc3` でも同じ（c2 で t10–30 に 0.61–0.66。SNA の t30 は 0.66）。
  - SNA の t1→t50 の下がり幅 0.038 のうち 0.012 は、c2 の谷越えがまだ 0.09 の t1→t10 に起きる（同じ時期に fc が谷に入る）。
  - 前半は max-pool では説明できない可能性がある。
- この spec は max-pool の関与だけを問う。fc の谷の効果、c の値の効果、ストライド付き conv は答えない（§8）。

## 2. 箱と変更点

- 宿主 `src/rlcifar_cnn_0908.py` を無改変で import し、`forward_cnn` だけを実行時に差し替える。
  差し替える関数は宿主の `forward_cnn` と一行ずつ同じで、2 か所の `F.max_pool2d(·, 2, 2)` を引数の pool 関数に置き換えただけのもの。
  宿主の学習ループ・`evaluate_cnn`・`preact_hist` はすべて `forward_cnn` をモジュールの名前で呼ぶので、差し替えは全経路に効く（S-wiring で確認）。
- pool ∈ {max, avg}。avg = `F.avg_pool2d(·, 2, 2)`（パディングなし、2×2 の平均）。**max は検査（S-reuse）専用**で、本走は avg だけ。
- それ以外は宿主と同一: CIFAR-10 の 1200 枚（seed ごとに固定）、50 タスク、400 epoch（30,000 step）、batch 16、Adam 1e−3（moment はタスクをまたいで保持）、
  init・ラベル列・バッチ順は seed だけで決まる、seed 0–9。SNA は c=0.6・β=0.01・clip [0.005, 3]・チャネル単位（宿主の既定）。
- 装置: GPU（参照と同じ RTX 5060 Ti・torch 2.13.0+cu130）。
- `evaluate_cnn` の `mob_pool_*` は宿主のまま（max-pool の勝ち位置でのゲート）。avg 版ではこれは勾配に効くゲートではない（効くのは全位置の平均 = `mob_*`）。比較のために残す。
- 出力: `results/avgpool_cnn_0917/<腕>/seed<k>/`（per_task.csv・provenance.json・hist/）。

## 3. 腕

| 腕 | 活性化 | pool | 本数 |
|---|---|---|---|
| `SNA_avg` | 宿主の `SNA` | avg | 10 |
| `SN3_avg` | 宿主の `SN3` | avg | 10 |

参照（max-pool、再計算しない）: `rlcifar_cnn_0908` の `SNA`・`SN3`・`SN06`・`LR`・`R`、`swish_battle_0917` の `SNAc3`。
起動順は seed ごとに `SNA_avg` → `SN3_avg`。

## 4. 検査（`src/avgpool_cnn_0917_checks.py`。本走の前に all_pass を commit する）

- **S-reuse**（GPU）: pool=max の差し替え forward を通した宿主 `run_one` が、保存済みの参照行（`SNA`・`SN3`、seed 0、task 1、400 epoch）を csv の 10 桁で全列再現する。
  検出力: 同じ行が参照の seed 1 の行とは食い違う。
  これで差し替え関数が宿主と同じ計算であること（SNA の層ごとの α の当て方を含む）が保証され、avg 版との違いは pool 関数だけになる。
- **S-pool**（単体）: avg-pool の出力が 2×2 の手計算の平均と一致し（1e−6）、逆伝播で 4 か所すべてにちょうど 1/4 ずつ勾配が流れる。
  変異: max-pool では 4 か所中 3 か所の勾配が 0 になり、この検査が落ちる。
- **S-wiring**: 短い avg 走（2 epoch × 2 タスク）で、
  - 学習・評価・hist の forward がすべて差し替え関数を通る（呼び出し元を数える）。
  - `F.max_pool2d` を呼ぶのは `evaluate_cnn` の mob_pool だけ。
  - 捕捉した最終重みで `nn.AvgPool2d` を使った独立 forward を組むと、記録された memo_acc と zbar_c2 が再現する（`SN3` と `SNA`。SNA は捕捉した α を使う）。
  - 変異: 独立 forward を `nn.MaxPool2d` にすると zbar_c2 が食い違う。
- **S-switch**: pool=avg の task 1 の行は、pool=max の行（= 参照）と学習列（online_acc・memo_acc・w_norm_*）で食い違う。
- 走の開始時に git の状態（HEAD と src/・analysis/ の未 commit 変更の有無）を記録する。
  本走中に src/・analysis/ を編集するときは、STOP ファイルで起動を止めてから commit する（swish_battle_0917 の教訓）。

## 5. 読み出し（登録）

窓 = タスク 31–50 の online_acc（seed ごとの平均）。早期 = タスク 1–10、後期 = タスク 41–50、低下 = 早期 − 後期。
対は seed 対応の符号検定（宿主の `sign`、両側・正確）。「勝ち」= 負けが 1 seed 以下かつ p<0.05。

- **A（谷越え）**: seed ごとに、t50 の c2 の 16 チャネルのうち 2·α_med_c2(t50)·m_c2[t50, ch] < −π/2 の割合を取る
  （α_med_c2 は per_task の中央値、m_c2 は hist のチャネル平均）。
  10 seed の平均が 0.20 未満なら `NEAR_SIDE`、そうでなければ `CROSSES`。参照の max-pool `SNA` は同じ式で 0.66。
- **B（順位）**: `SNA_avg` − `SN3_avg`（窓）。`SNA_avg` の勝ちなら `MLP_ORDER`、`SN3_avg` の勝ちなら `CNN_ORDER`、どちらでもなければ `TIE`。
- 副（登録）:
  - **C_SNA**: `SNA_avg` − `SNA`（窓。同じ画像・ラベル・init）。`AVG_BETTER` / `AVG_WORSE` / `TIE`。
  - **C_SN3**: `SN3_avg` − `SN3`。同上。
  - **D**: `SNA_avg` の低下が参照 `SNA` の低下（+0.0284）の半分未満なら `DROP_HALVED`、そうでなければ `DROP_KEPT`。
- 記述（判定なし）: サイト別の座席 2·α_med·z̄ と越えたチャネルの割合の時間経過（4 サイト）、mob・mob_pool、W、memo_acc、崩壊（online<0.5）、発散。
- 発散した走は宿主の規則どおり打ち切り、その seed は窓が欠けるので対から外す（数を報告する）。

## 6. 予測

### Claude（Opus 5）

13:23 に（avg-pool の走が存在しない時点で）会話と状態メモに書いたもの。原文のまま:

- P_avg1: SNA_avg c2 stays on the near side of the valley (far-side channel fraction at t50 < 20%, 2·alpha_med·zbar per channel from m_c2): 55%
- P_avg2: SNA_avg beats SN3_avg in the window (>=9/10 seeds, p<0.05), i.e. the MLP ordering: 40%

13:51 に追加（実装の前）:

- P_avg3: D = `DROP_HALVED`: 50%
- P_avg4: C_SN3 = `AVG_WORSE`（max-pool の非線形を失うので SN3 は下がる）: 45%
- P_avg5: 両腕とも、seed 平均の memo_acc が全タスクで 0.99 以上: 75%
- P_avg6: `SNA_avg` の fc1・fc2 は窓で谷の手前に座る（seed と t31–50 で平均した 2·α_med·z̄ が −π/2 と 0 の間。max-pool 版と同じ）: 70%
- 条件つき: A = `NEAR_SIDE` なら C_SNA = `AVG_BETTER`: 65%

### Issa

（空欄。avg 腕の結果を読む前に書かれた場合だけ、時刻つきで追記する）

## 7. 実行計画

- GPU は swish_battle_0917 の CNN（残り約 30 本）と共有する。検査が通ったら swish の launcher に STOP を置き、走っている分が終わって空いた枠から avg 腕を起動する（2 つの launcher の GPU 本数の合計を 5 本までに抑える）。
- avg 20 本（約 7 時間）が終わったら STOP を消して swish を再開する。swish の途中結果はこの間も読まない。

## 8. 答えないこと

ストライド付き conv の CNN、c のサイト分割、fc の谷の効果、Swish と pool の組み合わせ、WD との組み合わせ、avg-pool が他の活性化（LR など）に与える影響（`LR_avg` は回さない）。

## 追補 1（2026-09-17 14:05、本走の前）: 検査で見た値の開示

- 検査は 14:03 に all_pass（`results/_checks_avgpool_cnn_0917/checks.json`、bf0d301 の上で実行）。
- S-switch の中身として、avg 腕の **task 1 の値**を見た: online_acc は SNA_avg 0.9710（max 版 0.9790）、SN3_avg 0.9624（max 版 0.9724）。memo_acc はどれも 1.0。
  S-wiring の 2 epoch × 2 タスクの値（学習前に近い）も見ている。登録した判定（t31–50 の窓、t50 の谷越え）にかかわる値はまだ見ていない。
- Claude の予測（§6）はこれより前に書いたもの。Issa の予測がこの後に書かれる場合は、上の値を見られる状態で書かれたことを併記する。
- 起動の実際: swish の launcher に STOP を置いたのは 13:59:09。その直前（13:58 前後）に swish の seed 4 が起動しており、それは最後まで回す。avg 腕は GPU の空き（合計 5 本）から順に起動する。
