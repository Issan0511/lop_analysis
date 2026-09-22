# 2 ラベル交互（ABAB）で第1層の W とその幅はどう動くか — RL-CIFAR の交互ラベル走（altlabels_cifar_0923）

書いた: Claude, 2026-09-23 00:40（草案）→ 01:2x（v2: Codex gpt-6-astra の批評を入れて確定。批評の全文は `obsidian-research-data/altlabels_cifar_0923/spec_review/`）。
依頼: Issa「実際に動かしましょう！ 眠いので spec はあなたと codex に書かせて」（00:2x）。
親: `specs/spec_rlcifar_mlp_battle_0918.md`（エンジン・データ・1,200 枚・30,000 更新/課題・Adam 状態の持ち越し）。
理論の前置き: vault `理論/2ラベル交互ABAB_WとV幅の予測_理論解析_0923`（Codex、走る前の予測。IID からは一意に決まらず 3 択）。

## 0. 問い

IID（毎課題 fresh な乱数ラベル）の 50 課題では、第1層のノルム ‖W1‖² は t に線形（指数 0.99）、幅 σ は t^0.36 で σ* ≈ 92 に向かう。帯別帳簿では top（PC1–10）は毎課題 29% 書いて 28% 消す釣り合い、mid2（PC101–439）は 4% 書いて 2% 消す過渡。IID の帳簿は負の増分相関と侵食を示すが、一定のラグ 1 相関だけでは累積量を再現できず、帳簿から推定した f は因果的な復元係数とは限らない（理論ノート §10）。

ラベルを 2 つ（A, B）に固定して交互に出すと何が変わるか。理論解析の答えは「決まらない」で、遅い帯（mid2 以下）の 50 課題内の成長は 3 択:
- **減る**: 増分がラベルの差 H(y_t − y_{t−1}) の形なら、A→B と B→A で打ち消す。
- **IID と同じ**: 増分の大半が「fresh」（写像 H 自体が課題ごとに変わる分）なら、交互でも同じ速さで伸びる（成長比 1）。
- **IID より速い**: 増分が現在のラベルの関数 G(y_t) なら、G(A), G(B), G(A), … が足し算になる。

OU-F・周期的書き込み・残差書き込み・収束後成長を競合する条件付きモデルとして比べる。加えて、A の再訪で当てはめが速いか（同じ checkpoint からの fresh C との対比）、幅が固定解側（20 前後）に来るか IID 側（82）か、連続学習（AAAA）だけで IID の端点にどこまで届くかを同じ走で見る。**どの判定も 50 課題内の分類で、有界性や漸近次数の判定ではない。**

## 1. 設計

エンジンは親の `src/rlcifar_mlp_battle_0918.py` の `run()`（R 本の lockstep、CUDA graph、Adam 状態と歩数計 tc は課題をまたいで持ち越し）。入力は親の `standardize()` と同一の固定 channel 定数（平均 (0.4914, 0.4822, 0.4465)・SD (0.2470, 0.2435, 0.2616)）で正規化し、選んだ 1,200 枚から再推定しない（μ⊥ 方向が残る）。変更は次の 4 点で、既定値では親と bit 一致（S1）。

1. **ラベルの日程 `--schedule`**: `iid`（既定、親のまま）/ `abab` / `aaaa` / `abc`。seed の乱数流 `rlc_labels` から A = 1 回目の draw、B = 2 回目、C = 3 回目を日程が要る数だけ取り（abab 2・aaaa 1・abc 3）、以後は draw しない。課題 t のラベルは abab: t 奇数 → A・偶数 → B、aaaa: 常に A、abc: t mod 3 = 1 → A・2 → B・0 → C。バッチ順の流 `rlc_batch` は親と同じく課題ごとに進める（再訪でもミニバッチは新しい）。A/B/C・seed・クラス数・ラベル間一致率（A–B, A–C, B–C）を `labels.npz` に保存。**abab の課題 1・2 は同じ配置（R=10, std）の iid の課題 1・2 と bit 一致する**（S1b）。
2. **軌跡 `--hit-every 100`**: 100 更新ごと（と各課題の更新 0 で 1 回）に 1,200 枚の eval 前向き（no_grad、`train=False`、act の状態を触らない）で、スロットごとに step・正解数（整数）・CE・余裕の中央値・n1 = ‖W1‖²・n2・n3・σ_med（z1 の画像方向 SD の 100 ユニット中央値）を `trace/<arm>_<cond>_seed<s>.npz` に積む。`hit99`/`hit999` は正解数 ≥ 1,188 / ≥ 1,199（= 0.999 × 1,200 の切り上げ）に最初に達した更新数（更新 100 から有効、−1 = 未到達）を per_task.csv に足し、`hit999 + 500` 時点と以後の最小の正解数も残す。CUDA graph の外で行い、graph の静的テンソルに触らない。
3. **課題ごとの完全 checkpoint `--keep-ckpts`**: `ckpts/t<NN>.pt`（P, m, v, tc, `rlc_batch`/`rlc_labels` の状態、act の状態（Snake の V）、alive、meta）。R=10 で 1 課題 ≈ 38 MB、50 課題 ≈ 1.9 GB/腕。課題末は epoch の境界なので permutation の途中状態は無い。同じ配置で eager 再生すれば bit 一致で局面別（衝撃・当てはめ・静穏・一撃）の帳簿を後から取れる（親の S-graph: graph = eager bit 一致）。
4. **停止則 `--stop 0.999,500`**（stop 連結）: **再生エンジン `rl_width_posthoc_0922/replay2/width_replay.py` の規則を再現する**: epoch の頭で permutation を引き、内側ループの頭で `step ≥ stop_at` を検査する（停止が epoch 境界に一致すると次の permutation を 1 回引いて捨てる）。100 更新ごとの評価で最初に正解数 ≥ 1,199 になった更新 s から `stop_at = min(s + 500, 30,000)`。次の課題は新しい epoch から。tc は数え続ける。**R = 1 のときだけ許す**（lockstep の tc はスロット共通）。eager。

5. **fork**（別 stage `fork`）: 主走の `ckpts/t<NN>.pt` から、**主走と同じ配置（seed 0–9 × std、R=10、graph）**で P, m, v, tc, `rlc_batch` の状態, act の状態を復元し、ラベルだけ変えて 1 課題を回す束を 3 本出す: **A**（`labels.npz` の A。t が偶数なら主走の課題 t+1 と同一 → S4）、**B**（直前に終えた B の継続対照）、**C**（fresh: `rlc_labels_fork_t<t>` 流の 1 回目の draw。(seed, t) → C の対応は固定）。スロットごとに軌跡から hit999 を取り、`hit999 + 500` の時点でそのスロットのスナップショットを書く（凍結はしない。束は全スロットが hit999 + 500 を過ぎるか 30,000 で終わる）。fork 点: LR_abab の t ∈ {2, 10, 20, 30, 40, 48}（**登録は t = 48**、t = 2 は初期の錨、他は副）。LR_iid の t = 48 から {next（= 主走の課題 49 のラベル）, C} も出す（fresh 同士の到達時間のばらつきの対照）。記録: `forks.csv`（seed, t, branch, hit99, hit999, stop_step, 束の長さ, 停止時の正解数, 親 ckpt の sha256）と停止時・束の終わりのスナップショット。

## 2. 腕

| 腕 | 活性化 | cond | 日程 | seed | R | 課題 | 停止 | 走 |
|---|---|---|---|---|---|---|---|---|
| **LR_iid** | leaky 0.1 | std | iid | 0–9 | 10 | 50 | なし | GPU。**同じ配置の対照（新走）**。trace・keep-ckpts |
| **LR_abab** | leaky | std | abab | 0–9 | 10 | 50 | なし | GPU。trace・keep-ckpts |
| LR_aaaa | leaky | std | aaaa | 0–9 | 10 | 50 | なし | GPU。連続 150 万更新（課題の境界は目盛りだけ） |
| LR_abc | leaky | std | abc | 0–9 | 10 | **12** | なし | GPU。ラグ 3 の記述的診断 |
| SNA_abab | Snake 適応 α | std | abab | 0–9 | 10 | 50 | なし | GPU |
| LR_abab_fork | leaky | std | fork | 0–9 | 10 | 1 | 0.999,500 | GPU、主走の後。§1.5 |
| LR_iid_fork | leaky | std | fork | 0–9 | 10 | 1 | 0.999,500 | GPU、t = 48 だけ |
| LR_abab_stop | leaky | std | abab | 0–4 | 1 | 50 | 0.999,500 | **GPU eager R=1**、seed 直列（`chain` stage。CPU では 1 画像の argmax の反転で課題長が 500 ずれて E-stop15 と一致しない: S5b） |
| LR_iid_stop | leaky | std | iid | 0–4 | 1 | 50 | 0.999,500 | 同上 |
| （外部参照）IID R=20 | leaky / Snake | raw+std | iid | 0–9 | 20 | 50 | なし | 親の走（`obsidian-research-data/rlcifar_mlp_battle_0918/`）。配置が違うので bit 一致は仮定せず、差は S1d で記録 |

GPU 5 ジョブ並列（S7 の実測: trace 込みで R=10 単独 0.82 ms/step、3 本共有で 2.62 → 5 並列の見積り 4.37 ms/step ≈ 1.8 時間）。stop 連結は GPU の R=1 eager で seed 1 本 1.7 分、2 腕で 17 分（主走と並走）。fork は 20 束 × ≤ 30,000 更新（実際は最遅スロットの hit999 + 500 まで）。主走の保存 ≈ 13 GB。AAAA_stop と SNA の stop は無し（Codex: 人工境界の残りを捨てる意味が無い／leaky が固まってから）。

## 3. 測るもの（`analysis/altlabels_cifar_0923/`）

スナップショット W1（t00–t50）から、帯別帳簿 `obsidian-research-data/rl_ledger_posthoc_0922/band_ledger.py` の基底（seed ごとの入力 PCA、top/mid1/mid2/low/μ⊥/comp）で。**LR_iid（新走）も同じコードで同じ seed から計算する**（親の per_task_bands.csv は写さない）。親の `replay_stack()`/`replay()` は IID の draw 順でラベルを作り直すので、abab/aaaa/abc の腕には使わず `labels.npz` を使う。

- `ledger.csv`（arm, seed, t, band）: N_b, d_b, e_b（ΔN = d + e を検算）, V_b（λ 重み。4 帯の和 = 全体、μ⊥ と comp は 0）, σ_med（numpy の中央値 = 中央 2 点の平均）, H = mean σ²/median σ², hit99, hit999, tc。
- `cycle.csv`（arm, seed, j, band）: 周期 j = (2j−1, 2j) の中心 M_j = (W_{2j−1} + W_{2j})/2、直径 D_j = (W_{2j} − W_{2j−1})/2、E_{b,j} = [N_b(2j−1) + N_b(2j)]/2 = ‖M_j‖²_b + ‖D_j‖²_b（検算）、隣接周期の中心の変位、同ラベル復帰 ‖W_t − W_{t−2}‖²_b、a_t = ⟨W_t, W_{t−2}⟩_b/‖W_{t−2}‖²_b と残差 ‖W_t − a_t W_{t−2}‖²_b、top のコサイン cos(W_t, W_{t−2})。ABC は t−3 で。
- `lags.csv`（arm, seed, band, window ∈ {early t5–14, late t26–50, all t2–50}）: 帯別増分 Δ_t = W_t − W_{t−1} のラグ k = 1..6 の相関を 2 通り（コサインの算術平均、エネルギー重み付き相関）、両方の増分が窓内にある対だけ。累積比 r_b（t6–50、`r_top` はこの窓に固定）、q_b(t0) を ABAB では A（t0 = 19 → 49）と B（t0 = 20 → 50）に分けて。C₂ − (C₁ + C₃)/2 も出す。
- `growth.csv`（arm, seed, band）: **主統計量 = 完全な AB 周期の平均エネルギー E_{b,j}（j = 14..25 = 課題 27–50）を x_j = 2j − 0.5 に回帰した傾き**。IID も同じ集約・同じ窓。同 seed の LR_iid との比 ρ_b。副: t26–50 の課題ごとの OLS 傾き（周期 2 の一定振動はこの窓の OLS を偏らせない: Σ(t−38)(−1)^t = 0）、log-log 指数 β（N_b、t6–50、残差付き。旧帳簿の expo は √N の t5–50 なので混ぜない）、N_b(25)・N_b(50)、σ_med(25)・σ_med(50)。分母 ≤ 0 は明示して除外。
- `forks.csv`（§1.5）: 打ち切り: hit999 = −1 は比に使わず 30,000 で右打ち切り。到達した腕との大小が確定する対だけ符号を数える。τ = min(H, 30,000) も記録するが τ_A/τ_C = 1 を「両方成功」と読まない。
- `stop.csv`: stop 腕の per_task（steps, hit999, tc, N_b, σ）。課題数と累積更新数の両軸。
- `trace` から: 局面（衝撃 1–200 / 当てはめ 201–hit999 / hit999+500 まで / 残り）ごとの Δn1。一撃の有無（親の規則: 3,000 更新以降で ‖u‖² > 1e−3 が連続、に加え hit999 以降の超過）は後日の eager 再生で（§1.3）。

**seed の扱い**: 各 seed を独立な解析単位として seed 内統計量を先に出す。中央値で分類し、各区間に入る seed の本数を併記する（三分岐は「同じ区間」を数える）。欠測・発散・未到達・分母非正は区別して明示し、黙って落とさない。**8/10 は整合性の旗で、5% の検定ではない**（片側符号検定 p = 0.055、9/10 で 0.011。各 seed が 8 割で予測の側に出るとき ≥ 8/10 の検出力は 68%）。「同じ」の主張は同等性の議論が要り、差が出ないことでは足りない。

## 4. 登録判定

境界は理論ノート §11 の競合予測の中点（0.5/1.5 と 1,800 は運用上の目安で、モデル由来ではない）。名前は測ったものだけを言う。

**Q1（主）mid2 の 50 課題内の成長**（LR_abab 対 LR_iid、周期平均の傾き比 ρ_mid2、seed 中央値）:
- ρ < 0.5 → `REDUCED_GROWTH`（0 と 1 の中点）
- 0.5 ≤ ρ ≤ 1.5 → `IID_SCALE_GROWTH`
- ρ > 1.5 → `FASTER_GROWTH`（運用上の 5 割増し）
副（同じ規則、登録外の記録）: low・N_all・comp（comp は絶対傾き 2.1/課題 = 0 と fresh 漏れ 4.2 の中点も併記。LR_iid の後期の実測は約 3.9）。β_all > 1.5 は記述的な旗 `EXP_GT_1.5`。周期の中心の移動と直径の増大を分けて報告する。

**Q2 周期性**（LR_abab, late, 副）: mid2 の C₂（コサイン平均）≥ +0.077（OU-F −0.005 と周期混合 +0.16 の中点 0.0774）→ `PERIODIC`、それ以外 `NOT_PERIODIC`。副: low C₂ ≥ +0.059、top C₂ ≥ +0.13。正の C₂ だけでは動径の持続と区別できないので C₂ − (C₁ + C₃)/2 と偶奇の型（ラグ 6 まで）を併記。

**Q3 top の正味変位**（LR_abab, 副）: r_top（t6–50）< 0.0636（厳密復帰 1/45 と IID 0.105 の中点）→ `LOW_NET_DISPLACEMENT`、以上 → `HIGH_NET_DISPLACEMENT`。Q3b（記録）: late の A→A（t 奇数、t−2）の top コサインの中央値が (c_IID + 1)/2 以上 → `HIGH_DIRECTIONAL_RETURN`。c_IID = LR_iid の同 seed・同窓・同帯のラグ 2 コサイン（ABAB を見る前に固定）。「同じ解」とは呼ばない: a_t と残差、振幅の変化を併記する。

**Q4（主）再訪の節約**（LR_abab_fork, **t = 48**、A 対 C、10 seed）:
- A が C より早く hit999 に達する seed が ≥ 8/10 → `SAVINGS`（整合性の旗）、未満 → `SAVINGS_NOT_ESTABLISHED`（「無い」とは言わない）
- `SAVINGS` かつ hit999_A/hit999_C の seed 中央値 < 0.5 → `STRONG_SAVINGS`（観測値か打ち切り上限で確定する場合だけ）
- 加えて hit999_A の中央値 ≤ 1,800（fresh 網 1,767 の次の評価点）かつ A < C → `FAST_REVISIT`
副: t = 2, 10, 20, 30, 40 の同じ量の時系列、B 継続対照（hit は 100 で stop 600 になるはず）、LR_iid の next 対 C（fresh 同士のばらつき）。fork 点を独立標本として合算しない。
頑健性: hit999（正解数 ≥ 1,199）は 1 画像で数百更新動く刃の上の統計量（検査の smoke: seed 0 課題 1 は更新 1,700–1,900 で 1,199 のまま、2,000 で 1,200。R=1 の再生は 2,000、R=10 は 1,700 と報告する）。登録は hit999 のまま、`hit_full`（1,200/1,200 に最初に達する更新）と hit99 でも同じ判定を出して併記する。

**Q5 幅**（LR_abab, t50, 副）: σ_med の seed 中央値 < 60.1（固定復帰 20.3 と IID 82.5 の分散の中点 60.08）→ `LOWER_WIDTH_SIDE`、以上 → `IID_WIDTH_SIDE`。記録: t25 を 47.7（64.3 は外挿の代理）で、V_top(50)/V_top,iid(50)（境界なし）、mid1 の分散。

**Q6 AAAA**（LR_aaaa, t50, 記録）: N_all(50)/N_all,iid(50)（同 seed）の中央値 ≥ 0.80（Adam だけの帰無の下端 1.09 と PMNIST 移植 0.52 の中点）→ `CONTINUOUS_FIT_GE_80PCT_IID`、未満 → `CONTINUOUS_FIT_LT_80PCT_IID`。150 万更新の連続単一ラベル学習で、Adam の単独十分性やラベル切替の必要性はこれだけから結論しない。時間軸は累積更新数。

**Q7 ABC**（LR_abc, 12 課題, 窓 t4–12, 記録）: mid2 の (C₁, C₂, C₃) を混合 (−0.08, −0.08, +0.16) と純復帰 (−0.5, −0.5, +1) に照らす型の篩: C₃ ≥ +0.08 かつ C₂ < +0.08 → `PERIOD3_PATTERN`、C₃ ≥ 0.08 かつ C₂ ≥ 0.08 → `GENERIC_PERSISTENCE`、他 `NONE`。C₃ − C₂ も出す。

**Q8 Snake**（SNA_abab, t50, 副）: σ_med < 29.4（固定再利用 20.7 と IID 平衡 36 の分散の中点 29.36）→ `LOWER_SNAKE_WIDTH`、以上 → `IID_SNAKE_WIDTH`。副: ρ_mid2 を Q1 の規則で（対照は外部参照 SNA R=20 なので配置差あり、記録扱い）。

**Q9 stop 連結**（LR_abab_stop 対 LR_iid_stop、seed 0–4、記録）: 「各課題で同じ到達基準を使う学習方策」の比較。N_all(50) の比（中央値。0.5・1.5 は運用上の目安）と、周期平均の傾き比、累積更新数の軸での比較を報告。full との差を「収束後局面の効果」と呼ぶには同一 checkpoint からの hit999+500 対 30,000 の比較が要るので、ここでは呼ばない。

## 5. 予測（走の前に固定。Codex 列は spec_review §5、† は草案の述語（t26–50 OLS・4 fork 点の連言）への値を転用）

| # | 述語 | Claude | Codex | Issa |
|---|---|---|---|---|
| P1 | Q1 = IID_SCALE_GROWTH | 0.50 | 0.50 | |
| P2 | Q1 = FASTER_GROWTH | 0.30 | 0.15† | |
| P3 | Q1 = REDUCED_GROWTH | 0.20 | 0.35† | |
| P4 | EXP_GT_1.5（β_all > 1.5） | 0.15 | 0.15 | |
| P5 | Q2 = PERIODIC（mid2 C₂ ≥ 0.077） | 0.60 | 0.85 | |
| P6 | Q3 = LOW_NET_DISPLACEMENT（r_top < 0.0636） | 0.70 | 0.70 | |
| P7 | Q3b = HIGH_DIRECTIONAL_RETURN | 0.40 | 0.75 | |
| P8 | Q4 = SAVINGS（t48、A < C が ≥ 8/10） | 0.75 | 0.75 | |
| P9 | Q4 = STRONG_SAVINGS | 0.35 | 0.25† | |
| P10 | Q5 = LOWER_WIDTH_SIDE（σ(50) < 60.1） | 0.45 | 0.50 | |
| P11 | Q6 = CONTINUOUS_FIT_GE_80PCT_IID | 0.45 | 0.45 | |
| P12 | Q7 = PERIOD3_PATTERN（12 課題） | 0.50 | 0.80† | |
| P13 | Q8 = LOWER_SNAKE_WIDTH | 0.35 | 0.55 | |
| P14 | Q9 の N 比 < 0.5 | 0.30 | 0.40 | |
| P15 | Q9 の N 比 ∈ [0.5, 1.5] | 0.50 | 0.55 | |
| P16 | r_top < 0.0636 かつ N_top(49)/N_top(25) > 1.10（向きは戻り振幅は伸びる） | 0.50 | 0.55 | |
| P17 | comp の周期平均の傾き > 2.1/課題 | 0.60 | 0.60 | |

Claude の理由: 遅い帯の「fresh」な部分は写像 H が課題ごとに変わる分で、交互でも変わり続けるので IID_SCALE が本命。A の再訪では B の解の上に A の残滓が溶接されているので当てはめは速い（SAVINGS）が、fresh 網の 1,767 までは行かない。top は D 成分が打ち消して r は小さくなるが、解の向きは 2 課題で変わる。幅は top が有界でも mid1 が伸びて 57–70 に来るので境界上。AAAA は Adam の drift だけで IID の端点に届く帰無が t² なので、半分の確信で ≥ 0.8。Codex の理由は spec_review §5。Issa の列は空けてある（就寝中）。埋まらなければ Claude・Codex の列で採点する。

## 6. 検査（本走前。`results/altlabels_cifar_0923/checks.json`。行の一致だけで通す逃げ道は無し）

- **S1 bit 一致**: (a) 親の未改変スクリプト（読み取り専用クローン `proj_004_drift/src/rlcifar_mlp_battle_0918.py`、`--seeds 0-9 --conds std --tasks 2`）対 改変エンジンの既定値、同じ R=10 std 配置・同じ device・graph: 行（(seed, cond, task) で揃える）・スナップショット・ckpt の P/m/v/tc が一致。(b) 改変エンジンの `abab` 課題 1–2 対 `iid` 課題 1–2、同配置: 一致。(c) trace あり対なし: LR と SNA で軌跡が一致。(d) 外部参照 R=20 のスナップショット対 新 R=10: 最大絶対差を記録（合否なし）。(a)–(c) が bit 一致しなければ止めて報告。
- **S2 日程と乱数流**: `labels.npz` と走の debug から abab の課題 3 = 課題 1・4 = 2、aaaa は全課題 = 1、abc の課題 4 = 1 が厳密一致、abab の課題 3 ≠ iid の 3 回目の draw。ミニバッチの index 列が同 seed の iid と一致（A/B/C の先取りが iid の流を進めない）。resume がラベル辞書を復元する。
- **S3 評価の読み取り専用**: 1 回の trace 評価の前後で P, m, v, tc, Snake の V, 乱数流の状態, 累積器が一致し、次の更新が評価なしの走と一致（= S1c）。
- **S4 fork の同一性**: LR_abab `ckpts/t02.pt` からの A 束が主走の課題 3 を再現: 100 更新ごとの trace 行（正解数・n1・n2・n3・σ_med）が束の終わりまで一致、スロットごとの hit999 が一致。最初のミニバッチ index も一致。
- **S5 stop 連結の外部一致**: LR_iid_stop seed 0 の課題 1–15（CPU、8 スレッド）の (steps, hit999) が `E-stop15_taskends.csv` の (steps, stop_eval) と 15 課題すべて一致（課題 1 は 2,500/2,000）。n1 の相対差 ≤ 1e−6（超えれば記録して報告）。permutation の消費規則（§1.4）を含む。
- **S6 歩数計と再開**: full 走で tc = 30,000·t、stop 走で tc = Σ steps、fork の tc は ckpt の値から数える。resume は schedule・stop・hit の設定が違えば拒否。provenance に schedule, hit_every, keep_ckpts, stop, labels の sha256, run_id, parent, permutation 規則, fork の親 ckpt の sha256。
- **S7 帳簿の恒等式と合成検査**（解析側、走の後でよい）: ΔN = d + e、帯の和 = 全体、V の 4 帯の和、μ⊥/comp の分散 0、固定 A/B 端点で同ラベル変位 0・r_{6:50} = 1/45、伸びる 2 周期で r が小さく振幅が伸びる合成例、ゼロ増分のコサインは未定義。
- **S8 費用と保存**: trace を有効にして 5 並列の 1 課題の壁時計 × 50。ckpt は clone してから保存（R のスライスが元テンソルを掴まない）、原子的に書く。

## 7. 手順

1. `src/rlcifar_mlp_battle_0918.py` に §1 の 1–4（既定値で親と同一）。`src/altlabels_cifar_0923.py` に CLI（run / fork）と fork の実装。
2. `analysis/altlabels_cifar_0923/`: `launch.py`（le_eps の fork、GPU 5 ジョブ）、`plan_stop.json`（CPU 2 腕）、`checks.py`（§6 S1–S6, S8）、`ledger.py`（§3）、`report.py`（§4）。`band_ledger.py` は退避先から analysis/ に写す（sha256 を記録）。
3. §6 → checks.json。spec を commit し sha256 を §8 に記す（予測の固定）。
4. 本走: GPU 5 腕（launch.py）。同時に CPU で stop 2 腕。主走が終わったら fork 20 束。
5. ledger → report → `summary.md`（Q1–Q9、P1–P17 の採点。seed ごとの表付き）。
6. 退避（ckpts・snap・trace・labels・forks を `obsidian-research-data/altlabels_cifar_0923/`）、main に入れ、worktree を消す（CLAUDE.md §4）。

## 8. 記録

- 00:40 草案。01:1x Codex（gpt-6-astra、xhigh、read-only、4,245 トークン相当の返答 347 行）の批評: 設計の欠陥 8 件（R=20 対 R=10・「有界」の名の付け過ぎ・Q4 の連言・AAAA_stop・stop の乱数消費規則 ほか）、閾値の検算（0.077/0.059/0.13/0.0636/60.1/47.7/29.4 は算術どおり、0.5/1.5/1,800/0.80 は運用値、top V の 0.5 は導出不能 → 登録外）、S 検査の作り直し、予測列、削る順（ABC50 → AAAA_stop → Snake → AAAA の seed → fork 点 → stop の seed）。v2 に反映: IID の同配置対照を追加、判定名を測ったものに、Q1 を周期平均の傾きに、Q4 を t48 の対に、fork を同配置の束に、stop を再生規則に、ABC を 12 課題に、AAAA_stop を削除。
- 01:2x v2 確定。実装は Opus（同時進行、修正点は 01:1x に送付）。Codex には実装後の diff と checks.json の独立レビューをさせる（本走と並走。致命的な指摘が出たら止めて走り直す）。
- 01:5x Opus 実装完了（commit b96e800–891fca2）。checks.json 20/20: S1a 親の未改変スクリプトと bit 一致（行・30 スナップショット・ckpt の P/m/v/tc）、S1b/c、S1d 外部参照 R=20 との最大絶対差 0.63（同配置対照 LR_iid が必要だった）、S2a–f、S3a、S4a（A 束が主走の課題 3 を 260 行の trace で再現）、S4b、S5（E-stop15 と 15 課題厳密一致、n1 相対差 2e−16）、S6、S7。実装上の逸脱: stop 連結は CUDA R=1（§2）、`chain` stage 追加、`tc` は provenance と trace に（per_task.csv には stop のときだけ）。**予測固定前に見えてしまった観測**: 検査 S4 の smoke fork（t = 2）で A の hit999 1,000–2,000・fresh C 1,700–2,700・B 継続は 100（全 10 seed）。P8/P9 は変えていない（登録は t = 48）。
- spec を commit して sha256 を `results/altlabels_cifar_0923/PREDICTIONS.sha256` に記録してから起動。
