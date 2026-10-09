# cnn_drive_verify_1009 — Codex の CNN 駆動源理論を実軌道（RL-CIFAR CNN の課題末チェックポイント）で数値的に検証する

状態: **登録（この commit）。計算コード・検査・本計算はこの commit より後。**

作成: 2026-10-09 23:10 JST / 起草・実装: Claude（Opus 5.5、検証担当サブエージェント）/ 起点 `origin/main=2060e20e`。
run: `cnn_drive_verify_1009` / branch: `claude/cnn_drive_verify_1009` / worktree: `wt/cnn_drive_verify_1009`。
親: 親セッション（Fable）経由の Issa の依頼「CNN の駆動源理論は Codex がかなり詰めたが、Claude も数値的に検証してほしい」。
理論の出典（読むだけ・書かない）: `wt/cnn_drive_1009/results/cnn_drive_1009/summary.md`（24 項目）、`analysis/cnn_drive_1009/proof.md` と付録、`verify*.py`。
状態の出典（読むだけ）: `wt/sna_cnn_cause_1009/results/sna_cnn_cause_1009/A/`（束 A: SNA・SNAc3・CV06FC3・CV3FC06 × seed 10–19、50 課題 × 400 epoch、走行中）。

## 0. 一行

Codex の定理はすべて「構成した状態」で検算されており、**実際に学習した CNN の軌道上で前提や結論が成り立つか**は測られていない（summary.md 50 行目）。課題末のチェックポイント（重み・Adam の m/v・step・Snake の α）から、(a) 新しい一様ラベルの期待勾配が各チャネルの平均前活性 z̄_j を押す向き（full と literal self）、(b) 保存された Adam 状態からの次課題の最初の更新（と最初の数更新）で SGD の向きと Adam の向きが逆転する割合、(c) 長期の z̄_j の変化と、課題内の押し（最初の部分）と戻り（残り）の帳簿、を測る。

## 1. 既知情報（登録前に見たもの。独立な予測成功として数えない）

- Codex の文書（summary.md 全部、proof.md 全部、`full_ntk_positive.md`・`ce_trained_state.md`・`adam_state.md`・`adam_stationary_b16.md`・`adam_tenclass_equilibrium.md`・`adam_constant_rate_boundary.md`、`verify_full_ntk_positive.py`・`verify_ce_full_architecture.py`・`verify_ce_different_classes.py`・`verify_adam_full_architecture.py`）。vault の `駆動源問題_総まとめ_1007.md`・`CNN駆動源_有限強度の符号保証_1009.md`。
- `sna_cnn_cause_1009/A/per_task.csv` の**チャネル中央値**（zbar・seat・past_valley・alpha_med・zsd、c1・c2、課題 1・2・3・5・10・15・20・25 の seed 中央値）を一度表示した。例: zbar_c1 は 4 腕とも課題とともに単調に下がる（SNA −0.040 → −0.263、SNAc3 −0.072 → −0.309、CV06FC3 −0.065 → −0.554、CV3FC06 −0.082 → −0.248、課題 1 → 25）。zbar_c2 は SNA で −0.65（t1）→ −1.23（t5–10）→ −0.99（t25）と下がってから戻り、SNAc3・CV3FC06 は seat_c2 が −1.6〜−3.7 で谷（−π/2）の下にいるチャネルが過半。**§7 の (c) の予測はこれを見たうえで書いた**（チャネルごとの量・押しと戻りの分解・勾配の量は一切計算していない）。
- チェックポイントの形式（キー・形状、tc = 30000 × 課題数）だけを確認した。値は読んでいない。
- MLP 側の既知（vault 総まとめ §5・G13a）: Snake 族の段 1 の押しの符号は sign(cos θ̄)（θ̄ = 2α z̄、谷 θ̄ = −π/2）。3 層 RL-CIFAR MLP の KKT1・適応 Snake で、実際の切替の押しの交点は θ̄ −1.28〜−1.42。§7 の (a) の予測の根拠に使う。

## 2. 固定する箱（読むだけ）

- 網: conv5×5 3→16（pad 2）→ Snake → maxpool2 → conv5×5 16→16（pad 2）→ Snake → maxpool2 → fc 1024→100 → Snake → fc 100→100 → Snake → fc 100→10。全 bias あり。conv → φ → pool の順（宿主 `rlcifar_cnn_0908`）。
- 活性化は**適応 Snake**: φ(z) = z + sin²(αz)/α、φ′(z) = 1 + sin(2αz) ∈ [0, 2]。α = clip(c/√V, 0.005, 3)（V はバッチ分散の EMA、β 0.01、勾配は流れないバッファ）。腕ごとの c は SNA (0.6,0.6,0.6,0.6)・SNAc3 (3,3,3,3)・CV06FC3 (0.6,0.6,3,3)・CV3FC06 (3,3,0.6,0.6)（c1 c2 f1 f2）。
- 1200 枚（seed ごとの固定部分集合、[0,1]、正規化なし）。毎課題、画像ごとに独立な一様乱数ラベル（`torch.randint(10)`）。batch 16（毎 epoch の randperm を 16 ずつ）、400 epoch = 30000 更新/課題。CE は batch 平均。Adam lr 1e−3、β (0.9, 0.999)、ε 1e−8、bias 補正は全体の step 数 tc（課題をまたいで累積）。WD・clip なし。
- チェックポイント `ckpt/<arm>_seed<s>_t<tt>.pt`: 課題 t の末の P（10 テンソル）・m・v・tc・活性化の状態（V・ada・fixA・cval）。t ∈ {1, 2, 3, 5, 10, 20, 30}（t30 は 23:30 ごろ）。

**ReLU との違い（報告で毎回明示する）**: Codex の定理の大半は ReLU（正斉次・gate の微分 0・φ(z<0)=0）を仮定する。ここで測るのは Snake の網である。G の恒等式・literal self・Adam の分解・容量の定義は活性化によらず定義できるので「定義の上では」同じ量を測れるが、Codex の**十分条件の成立／不成立**と**符号の結論**は ReLU の網について述べたものであり、Snake の実測で支持・不支持を言うのは「同じ量が Snake の実軌道で何をしているか」までである。ReLU/leaky の CNN で測るには別の走が要る（§8）。

## 3. Codex の記号 → 我々のテンソル（操作可能な定義）

曖昧な箇所は Codex の verify*.py の実装を正とした（下の「出典」欄）。

| Codex | 意味 | ここでの計算 | 出典 |
|---|---|---|---|
| m_c（第一 Conv の mean） | チャネル c の全画像・全畳み込み位置の平均前活性 | z̄¹_j = mean_{n<1200, s∈32×32} z1[n,j,s]（zero-pad 込みの patch）。W1[j]・b1[j] のアフィン関数 | proof §1、`verify_full_ntk_positive.py` の `get_u`・`verify_ce_full_architecture.py` の `m=z[layer][:,0].mean()` |
| 第二 Conv の mean（Codex は §11 で「深層 mean」として扱う） | 同上（16×16 位置） | z̄²_j = mean z2[n,j,s]。W1・b1・W2[j]・b2[j] と c1 の α（バッファ）に依存 | `verify_ce_full_architecture.py`（layer 2） |
| u = ∇_θ m | 平均前活性の全パラメータ勾配（Euclid 計量） | autograd。c1 は W1[j] に平均 augmented patch・b1[j] に 1、c2 は W1・b1・W2[j]・b2[j] に非零。fc の成分は 0 | 同上 |
| R_n = D_θ f_n[u] | logits の u 方向 JVP（10 成分） | 画像 n・クラス c ごとの conv パラメータのヤコビアン J_conv[n,c,:]（7,632 座標）と u の内積 | `ce_trained_state.md` §2 |
| G = E_new⟨∇m, ∇L_new⟩ | 新しい一様ラベルの期待 CE 勾配と ∇m の内積。**G > 0 が沈める側**（SGD で E[Δm] = −ηG、c1 は厳密） | ラベルが画像ごとに iid 一様なので期待は解析的に厳密: E_y[L] = L_u = mean_n[logsumexp f_n − mean_c f_nc]、G = ⟨u, ∇L_u⟩。恒等式 G = (1/N)Σ_n R_n·(p_n − 1/C) で照合 | `ce_trained_state.md` (1)、`verify_ce_full_architecture.py` の `uniform_loss` |
| full | 全パラメータ・全チャネルを残した網の G | 上の G | |
| literal self（第一 Conv） | **他の第一 Conv チャネルを網から除き forward し直す**（head の refit なし、他の値は full と同じ） | W1 → W1[j:j+1]、b1 → b1[j:j+1]、W2 → W2[:, j:j+1]（入力チャネル列を除く）。b2・fc・head は同じ値。α も同じ値（c1 は α_j、他の層は full と同じ α）。**「ヤコビアンの該当列だけ残す」ではない**（Codex の `selfpars`・`self_pp` がこの切り出し） | `verify_full_ntk_positive.py` の `selfpars`、`verify_ce_different_classes.py` の `self_pp`、`full_ntk_positive.md` §1 |
| literal self（第二 Conv）（**Claude の延長**。Codex は第一 Conv だけ定義） | 他の第二 Conv チャネルを除く | W2 → W2[j:j+1]、b2 → b2[j:j+1]、fc1 の入力列を channel j の 64 列（flatten の 64j:64j+64）に限る。c1 は全部残す | — |
| 条件 (3) | 各画像・全クラス対で (f_c − f_d)(R_c − R_d) ≥ 0 | 画像 × チャネルごとに判定 | `ce_trained_state.md` (3) |
| 条件 (5) | top クラス t の logit 差 Δ > log 9 かつ γ = min_{c≠t}(R_t − R_c) > 0 | 同上 | 同 (5) |
| 条件 (8) | a(p_t − 1/C) > (b − a)B（a, b = top−loser の R 差の最小・最大、B = Σ_{loser, p>1/C}(p − 1/C)） | 同上 | 同 (8) |
| Adam の 1 歩 | A_j = κ(β₁m_j⁻ + (1−β₁)g_j)/(√(β₂v_j⁻ + (1−β₂)g_j²) + ε̃)、κ = √(1−β₂^t)/(1−β₁^t)、ε̃ = ε√(1−β₂^t) | 宿主の式どおり（m/c1、√(v/c2)+1e−8、t = tc+1）。Δθ = −lr·A | `adam_state.md` §2 |
| SGD の向き | E[Δm] = −ηG | 符号だけ比べる（η は任意の正） | 同 §6 |
| 凍結参照（parameters は境界に固定、moments だけ課題の更新を進める） | Codex §21 の参照 | J_conv を固定し、課題のラベルを固定・毎 epoch の randperm で batch を組み、Adam の m/v を step ごとに進める。Σ_s u·A_s を記録 | proof §21、`adam_task_reuse.md` |
| 容量 Φ_λ（任意 (d)） | ½ log det(I + K/λ)、K = JJᵀ（全 raw パラメータ）、方向微分 ½ tr((λI+K)⁻¹K′)、K′ = J′Jᵀ + JJ′ᵀ、J′ = D_uJ | 部分集合の画像（N_sub 枚 × 10 出力）で J と J′ を作る。λ = 1（Codex の検算と同じ） | `verify_full_ntk_positive.py` の `calc` |

精度: (a)(b) の量は **float64**（GPU）で計算する。訓練は float32（cudnn TF32 既定）だったので、(c) の再走だけはエンジンそのもの（float32）で行う。

## 4. 測るもの

対象: 4 腕 × seed 10–19 × t ∈ {1, 5, 10, 20}（t30 が出揃えば 30 も）。(c) の連続差のために t ∈ {1, 2, 3} も読む。計算量が許さなければ seed を 10–14 に減らし、減らしたことを結果に書く。

### (a) 向き: full と literal self

チャネル j（c1 の 16 + c2 の 16）ごとに:
1. G_full_j = ⟨u_j, ∇L_u⟩（autograd）と (1/N)Σ_n R_n·(p_n − 1/C)（J_conv から）の両方。
2. G_self_j = ⟨u_j, ∇L_u^self⟩（literal self の網で forward し直した L_u の勾配）。
3. 画像ごとの条件 (3)・(5)・(8) の成立割合、画像ごとの項 g_n = R_n·(p_n − 1/C) > 0 の割合。
4. 副次: seat_j = 2α_j z̄_j、zsd_j。
集計: sign(G_full) と sign(G_self) の一致率（チャネル × seed × 課題、c1・c2 別、腕別・課題別）、偶然の一致の基準 p_f p_s + (1−p_f)(1−p_s)（seed ごと）との差、沈める側（G > 0）の割合、seat との関係。

### (b) Adam の反転

1. **最初の 1 更新**: チェックポイントの m・v・tc から、次課題の最初の batch（1200 枚から一様に 16 枚、ラベル iid 一様）での Adam 更新 Δθ と Δm_j = u_j·Δθ（c1 は厳密、c2 は一次。c2 の非線形の残りは抜き取りで forward し直して照合）。ラベルと batch の期待を **J_conv を使ったモンテカルロ（K = 65,536 標本）**で取り、MC の SE を付ける。SGD の向きは −G_full（厳密）。反転 = 符号が逆。実際に起きた 1 更新（`labels_at(seed, t+1)` と g_batch の次の randperm の先頭 16 枚）も記録する。
2. **最初の数更新（凍結参照）**: パラメータを固定し、課題のラベルを 1 回引いて課題内で使い回し、毎 epoch の randperm で batch を作って m/v を進める。累積 Σ_{s≤S} u_j·A_s を S ∈ {1, 10, 75, 750, 3000} で記録（ラベル L = 32 本 ＋ 実際のラベル・実際の batch 順）。S = 3000（40 epoch）で v は定常に近い（時定数 1000）。期待の符号を SGD の −G と比べる（Codex の主張 9・12・24 の対象である「固定状態の定常応答」に当たる）。
3. 分解（報告）: 古い m の寄与（m⁻ = 0 にした場合との差）、分母の大きさ（√v̂ と (1−β₂)^{1/2}|g| の比）。

### (c) 長期の沈降と押し・戻りの帳簿

1. 課題末のチャネルごとの z̄_j を、初期化（t = 0、宿主の init と V = 1）と各チェックポイントで計算。初期化からの正味の変化 z̄_j(t) − z̄_j(0) の符号（Codex の「最終 mean は strict に低下」の実軌道版）。
2. 1 課題の変化: t1→t2・t2→t3 はチェックポイントの差。
3. **再走**: チェックポイント t から課題 t+1 をエンジン（`E.Bundle`、宿主の更新式・CUDA graph、float32）でそのまま再走し、z̄_j を step 1・10・75（1 epoch）・750・7500・30000 で記録。t ∈ {1, 2, 5, 10, 20}（＋30）。t1→t2・t2→t3 の再走は保存済みの t02・t03 と比べ、再走の忠実度（z̄_j の差）を帯にする。
4. 帳簿: 押し = 最初の 1 epoch の Δz̄_j、戻り = 残りの Δz̄_j（和が課題の正味）。押しの符号と (b) の予測（SGD・Adam）の符号の一致、戻りが押しと逆符号の割合、|戻り| < |押し| の割合。
5. per_task.csv の zbar_c1・zbar_c2（チャネル中央値）の課題ごとの差の符号の割合（全課題、seed ごと）。

### (d) 任意: 容量の方向微分（full と literal self）

時間が許せば、部分集合（例: 4 腕 × 2 seed × t ∈ {1, 20}、N_sub = 32 枚）で ½ tr((I+K)⁻¹K′) を full と self で計算し、符号の一致と、G（CE）との符号の一致を見る（Codex の主張 3・11・17）。予測は置かない（報告のみ）。

## 5. 検査（本計算の前に全部通す）

- **K1 有限差分**: 自動微分の full の方向微分 ⟨u_j, ∇L_u⟩ が、中心差分 [L_u(θ+hu_j) − L_u(θ−hu_j)]/(2h)（float64、h を 3 通り）と相対 1e−4 以内で一致（c1・c2 の数チャネル、2 チェックポイント）。self も同様。
- **K2 恒等式**: autograd の G と (1/N)Σ R·(p − 1/C) の一致（float64 の丸めの範囲）。この差の最大値をそのチェックポイントの数値の床とし、|G| がこれ以下のチャネルは符号を「不定」とする。
- **K3 J_conv**: 画像ごと・クラスごとの conv ヤコビアン（逆伝播 10 回 + unfold の積）を、単一画像の autograd 勾配（抜き取り 20 組）と照合。
- **K4 self の定義**: c1 の self を「切り出し」で作った値と、「他チャネルの W1・b1 を 0 にして forward」した値が一致すること（Snake は φ(0) = 0 なので一致するはず。一致しなければ切り出しを正とする）。
- **K5 Adam の式**: J_conv から作った 1 更新が、エンジンの `step` を 1 回呼んだ実際の更新（同じ batch・ラベル）と一致すること（conv パラメータ全座標、相対 1e−5 以内。float32 と float64 の差）。
- **K6 期待の厳密性**: K = 65,536 の MC の SGD 平均 E[u·g] が厳密な G と SE の範囲で一致すること（MC の偏りの検査）。
- **K7 再走の忠実度**: t1→t2・t2→t3 の再走と保存済み t02・t03 の z̄_j の差（報告。§6 の帯に使う）。

## 6. 判定の帯（雑音から導く）

- (a) の G は厳密量。符号が「不定」なのは |G| ≤ K2 の数値の床のときだけ。割合は seed ごとに出し、10 seed の平均 ± SD と t 分布（自由度 9）の 95% 区間で書く。「偶然より多く一致」は、一致率 − 偶然の基準 の seed 平均の 95% 区間の下端 > 0。
- (b) の MC 期待は SE 付き。チャネルの判定: 「Adam が SGD と同じ向き」= 符号が同じで |E| > z·SE、「反転」= 符号が逆で |E| > z·SE、それ以外は「不定」。z は Bonferroni（全チャネル × チェックポイントの件数 M について片側 0.025/M）から決める（M ≈ 6,000 なら z ≈ 4.2）。点推定の反転率も併記する。
- (c) のチェックポイントの差は厳密量。再走の量は K7 の忠実度の差（チャネルごとの |再走 − 保存| の最大）を帯とし、|Δ| がそれ以下なら「不定」。

## 7. Claude の予測（計算の前に登録。確率は Claude の主観）

(a) 向き
- **A1**（p 0.65）: 課題 1 末の c1 で、沈める側（G_full > 0）のチャネルは 4 腕をまとめて 70% 以上。
- **A2**（p 0.55）: c1 で、seat > −π/2 のチャネルの沈める側の割合は、seat < −π/2 のチャネルの割合より 0.3 以上大きい（MLP の Snake の交点 −1.3〜−1.4 がおおむね移る）。
- **A3**（p 0.40）: c2 でも A2 と同じ差（0.3 以上）が出る。
- **A4**（p 0.60）: full と literal self の符号一致率は c1・c2 とも 0.75 未満（self の網は 1 チャネルしか下流に渡さないので予測クラス自体が変わり、一致は弱い）。偶然の基準より有意に多いのは c1 で p 0.50。
- **A5**（p 0.90）: Codex の十分条件 (5) または (8) が 1200 枚**全部**で成り立つのは、G_full > 0 のチャネル × チェックポイントの 1% 以下（条件は実軌道では証明書としてほぼ使えない）。画像ごとに見ると、(5) を満たす画像の割合のチャネル中央値は 0.3 以下（p 0.60）。
- **A6**（p 0.90）: 全クラス対の順位条件 (3) を満たす画像は全体の 1% 以下。

(b) Adam
- **B1**（p 0.45）: 最初の 1 更新で有意に反転する（§6 の帯）チャネルの割合は c1 で 2〜25%。c2 の割合は c1 より大きい（p 0.60）。
- **B2**（p 0.70）: 課題末では v⁻ ≪ (1−β₂)g² なので、最初の 1 更新は符号 SGD に近く、conv の座標の 90% 以上で |Δθ_k| が lr·(1−β₁)/√(1−β₂) ≈ 3.16·lr の ±20% に入る。
- **B3**（p 0.55）: 凍結参照の S = 3000 での反転率は、c1 で最初の 1 更新の反転率以下。

(c) 長期と帳簿
- **C1**（p 0.65）: 初期化から t = 20 までの正味で z̄_j が下がった c1 チャネルは、どの腕でも 75% 以上。c2 も同じ（p 0.70）。
- **C2**（p 0.70）: t1→t2・t2→t3 の 1 課題で z̄_j が下がる c1 チャネルは全体で 60% 以上。
- **C3**（p 0.55）: 後期の 1 課題（再走 10→11・20→21）で z̄_j が下がる c1 チャネルの割合は 0.40〜0.75（下がる向きはあるが拮抗に近い）。
- **C4**（p 0.50）: 押し（最初の 1 epoch の実際の Δz̄_j）の符号は −G_full の符号と c1 の 60% 以上で一致。戻り（残り）が押しと逆符号のチャネルは 50% 以上（p 0.55）。
- **C5**（p 0.45）: 課題末の −G_full_j と次の課題の実際の Δz̄_j の順位相関（c1、全体）が 0.2 より大きい。

予測の採点は結果の節で 1 件ずつ行う（外れも残す）。

## 8. 範囲と、ReLU/leaky の CNN で測るなら

- ここで「支持／不支持」と言えるのは、Codex の主張のうち**定義が活性化によらないもの**（G の恒等式、条件 (3)(5)(8) の成立、Adam の反転の有無、長期の mean の正味の変化）を Snake の実軌道で測った範囲まで。ReLU を仮定した十分条件の結論そのもの（正斉次から出る K′ の正定値など）は、Snake の測定では支持も不支持もしない（「測れない」に入れる）。
- ReLU/leaky の CNN で同じものを測るには、宿主の `R`・`LR` 腕で課題末のチェックポイント（P・m・v・tc）を保存する走が要る。R は課題 3 で死ぬ（宿主の既知）ので、R は t ∈ {1, 2} と死の前後、LR は t ∈ {1, 2, 3, 5, 10, 20} 程度が要る（seed 10 本）。これは phase 2 で親が判断する（今は回さない）。

## 9. 実行の約束

- GPU は共有（束 A・束 D が走行中）。`nvidia-smi` を見てから回し、1 プロセス ≤ 3 GB・一度に 1 本。他のプロセスは殺さない（kill は自分の PID だけ）。
- コード: `src/cnn_drive_verify_1009.py`（計算）、`analysis/cnn_drive_verify_1009/`（集計）。結果: `results/cnn_drive_verify_1009/`。sna_cnn_cause_1009 のモジュールは `sys.path` に `wt/sna_cnn_cause_1009` を足して import（コピーしない）。
- 中間結果が出た時点で `results/cnn_drive_verify_1009/summary.md` に表を書いて commit する。push は `claude/cnn_drive_verify_1009` へ。main には入れない。
