# l2_wall_0916 — 第2層の µ₂ 壁: 崩壊層はどう円錐へ入るか、Adam の ε と LayerNorm はそれをどう変えるか

- 親: vault `可塑性喪失/論点/横断/明度チャネルρとµの壁_0916.md` §6.3（第2層の ρ は未検証）・`可塑性喪失/主張/中心主張v10作業リスト_0916.md` H3（多層で誰が運ぶか）
- 起案・実装・予測: Claude（2026-09-16）。Issa の指示: 閉じていない 6 項目のうち再走で閉じるものを閉じる／「よしなに LN を配置してください」（LN の配置は Claude に一任）
- 状態: **走行前に登録（この commit）**。Part A は観察（判定ラベルなし）、Part B・C は介入（§5 のラベルを走行前に固定）
- worktree `wt/l2_wall_0916`・branch `claude/l2_wall_0916`

## 0. 登録前に見たもの（事後・この spec の問いはこれらを見てから立てた）

1. `elu_environment_0913/checkpoint.pt`（task 50・RL・固定 1200 枚）で第2層の壁を µ₂ 方向 $e_2=\mu_2/\|\mu_2\|$ で測った。$c_2=\|\mu_2\|/\mathrm{sd}(e_2^\top a_1)$、$\kappa_2=(\|\mu_2\|-\min_x e_2^\top a_1)/\mathrm{sd}$、$\rho_\mu=q_2^2\mathrm{Var}(e_2^\top a_1)/\sigma_2^2$。
   - 崩壊した ELU 第2層: $d_2$ −3.30/−3.51/−3.45、$c_2$ 3.36/3.52/3.48、$\kappa_2$ 2.14/2.38/2.20、$\rho_\mu$ 中央 0.97/1.00/0.98、$b_2/\sigma_2$ 0.003–0.004、$\min e_2^\top a_1$ +64〜+79、$U_2<0$ 0.97–1.00、$\sigma_2^2$ の 96–100% が µ 方向。
   - leaky 第2層: $d_2$ −2.0〜−2.2、$\rho_\mu$ 0.32–0.41、$c_2>\kappa_2$。
   - 全1方向 $S_2=\mathbf 1^\top a_1$ で書いた近似 $m_2\bar S_2+b_2$ は残差 45–120%。第2層の壁の座標は全1ではなく µ₂。
2. `elu_reserve_0913` の PM checkpoint（t20/t100）: ELU 第2層 $d_2$ −0.71 → −1.6〜−1.7、$\rho_\mu$ 0.03 → 0.12–0.19、$U_2<0$ は t100 で 0.16–0.19。$c_2>\kappa_2$ は 12/12。
3. `wt/mucap_el_0916` の ref 腕（RL・ELU→leaky・150 タスク・本走は判定前で cap 腕は読んでいない）: 第2層 $d_2$ −1.9（t20）→ −1.1〜−1.3（t150）。タスク端点のスカラー帳簿は t50→150 で上流（$\|\mu_2\|$ の伸び）が主。密な点で刻むと t1→10 の交差項が −13〜−20 に膨らみ、スカラー分解は $e_2$ の回転で悪条件。
4. `layer_chimera_rl_0914/units.npz` の第2層 $d_2$ 中央値: RL EE −0.92（t5）/ −2.43（t10）/ −3.44（t20）、LE −0.13 / −1.65 / −2.90、t50 −3.13。`relu_gelu_silu_rl_ext150_0914`: 中央 $d_2<-3$ の初回は RL ELU1 t11–14、RL SILU t22/t25（s0, s1）、PM GELU t64/t102/t67、PM SILU t78/t67（s0, s2）。
5. 同じ task 50 checkpoint の表現に LayerNorm（LN）をかけた幾何（学習はしていない）: 全1方向の和は厳密に 0。µ 方向は崩壊した ELU 第2層で $c$ 130–159・$\kappa$ 6.0–10.9（片側 3/3）、PM 第2層で $c$ 8.5–16.4（片側 6/6）、RL の leaky 第2層は $\cos(\mu_2,\mathbf 1)=-0.97\sim-0.78$ で LN 後 $c$ 2.72–4.31・$\kappa$ 3.38–3.94（片側 2/3）。ReLU(LN(z₁)) の µ 方向 $c$ 11–65。活性化前 LN をその層に置くと各サンプルで 37% 以上のユニットが正。
6. Lyle et al. (arXiv:2402.18762) は LN を各非線形性の前に置き、前活性の分布シフトへの対処として扱う。

## 1. 問い

- **A1（円錐への入り方）**: 崩壊する第2層で、$\rho_\mu\to1$・$d_2\to-c_2$ は、ゲートの崩壊・W₂ の Adam が ε 域に入ること・$U_2<0$ に対していつ起きるか。$d_2,\rho_\mu,\sigma_2$ の変化を、W₂ 自身（W₂ と b₂）・入力の平均 µ₂・入力の共分散 Σ₂ の 3 因子に Shapley で割る（代数的な帰属で、因果ではない）。
- **A2（早期のベクトル帳簿）**: $\Delta\bar z_2=(\Delta W_2)\mu_2+W_2\Delta\mu_2+\Delta W_2\Delta\mu_2+\Delta b_2$ をベクトルで密な点ごとに閉じる。上流項は伸び $\Delta\|\mu_2\|$ と回転 $\Delta e_2$ に対称差分で割る。
- **A3（LE の壁定数）**: leaky→ELU の第2層で $c_2,\kappa_2,\min e_2^\top a_1$。
- **A4（PM GELU/SiLU の崩壊層）**: $d_2\approx-3$ は釘付け（$\rho_\mu\approx1$・$b_2/\sigma_2\approx0$）か、b 経路か。
- **A5（釘付けと ε ブレーキの順序）**: 崩壊層で W₂ が ε 域に入った後、$\bar z_2$ を動かすのは自己更新か上流か。
- **B（ε 介入）**: Adam の ε を替えると、崩壊層の $d_2$ は変わるか（ε が深さを決めるか）。
- **C（LN 介入）**: LN は µ₂ の壁を消すか。崩壊が残るならその経路は µ 方向の自己更新か、実効 bias か。

## 2. 箱と腕

### Part A — 既存 2 走の bit 一致再走（観察）

- `layer_chimera_rl_0914`: 24 モデル（seed 0–2 × RL/PM × EE/EL/LE/LL）・50 タスク。
- `relu_gelu_silu_rl_ext150_0914`: 30 モデル（seed 0–2 × RL/PM × LR/ELU1/R/GELU/SILU）・150 タスク。
- エンジン・乱数ストリーム・ループは親のもの（親モジュールを import）。診断は CUDA graph のブロックの間で状態を**読むだけ**。
- **bit 一致ゲート（S5）**: 各タスク終端で親の `evaluate()` の全出力が commit 済みの `units.npz`・`rows.csv` と max|Δ|=0。一致しなければ「同じ箱の別軌道」として報告する（出力は書いてから判定する）。

### Part B・C — 変種エンジン（層別キメラの箱・同じストリーム）

層別キメラの箱（784–100–100–10・1200 枚固定・batch 16・6000 更新/タスク・Adam lr 1e−3・50 タスク）で、1 本のエンジンに次の 132 モデルを積む。

| 腕 | 内容 | 組 |
|---|---|---|
| `none` | LN なし・ε 1e−8（Part B の ε 対照を兼ねる） | EE/EL/LE/LL × RL/PM × seed 0–2 |
| `inLN` | 各隠れ層の線形変換の入力に LN（x と a₁）・affine なし | 同上 |
| `inLN_a` | 同上。a₁ 側の LN だけ affine（γ, β を学習）。画素側は affine なし | 同上 |
| `preLN` | 各隠れ非線形性の直前に LN（h_l = LN(z_l)）・affine なし | 同上 |
| `preLN_a` | 同上・affine あり（Lyle らの配置） | 同上 |
| `eps1e-6` | LN なし・Adam ε = 1e−6 | EE/LE × RL × seed 0–2 |
| `eps1e-30` | LN なし・Adam ε = 1e−30（ブレーキを実質外す） | 同上 |

- LN(t) = (t − mean_j t_j) / sqrt(var_j t + 1e−5)（サンプルごと・特徴軸）。affine は γ⊙LN + β（初期 γ=1, β=0）で、モデル本体と同じ Adam（同じ ε）で学習。affine なしの腕の γ, β は勾配 0 で動かない。読み出し層の入力 a₂ には何もしない。b₁, b₂ は親と同じく残す。
- 逆伝播は親と同じ手書き。LN の逆伝播は $\partial L/\partial t=(g-\bar g-n\,\overline{g\odot n})/s$。
- `none` の演算列は親と同じ（`where` で選ぶだけ）。バッチ数が 24 → 132 に変わるので、親との bit 一致は保証しない（S2 で M=24 の同型エンジンを別に作り、親と bit 一致することを確かめて、差をバッチ数だけに帰す）。
- **配置の理由**: (i) `inLN` は貼られた読みの「線形層の入力」で、β の有無で b 経路を切り分ける。(ii) `preLN` は先行研究の配置で、β が無ければ層全体の吸収は構造上起きない。(iii) LE と LL は µ₂ がほぼ −1 方向の場合で、LN が壁の大半を消すはず。(iv) PM の第1層は `inLN` で明度チャネル（段階0の壁）を失う。

## 3. 診断（全 Part 共通・float64・状態を読むだけ）

- 時点: 各タスクの開始（重みは前タスク終端のまま・入力とラベルだけ新しい）と終了。層別キメラの箱と変種エンジンでは t ≤ 20 で 75・375・1500・3000 更新も。ext150 は開始と終了だけ。
- 各隠れ層 l、各ユニット（固定 1200 枚上）: 線形出力 z の $\bar z,\sigma,U=\max_x z,p^+,d=\bar z/\sigma,K$、非線形性の入力 h（`preLN` では LN 後、他は z と同じ）の同じ量と平均・RMS $\varphi'$・$|\varphi'|<0.05$ と $<10^{-8}$ と $<0$ の割合、µ 基底の $q=w^\top e,\ \|v\|$、行平均 $m$・$\|\widetilde w\|$・$\|w\|$・$b$、分散の 3 項 $q^2\mathrm{Var}(e^\top u)$・$\mathrm{Var}(v^\top u)$（直接計算）・$2q\,\mathrm{Cov}$ と $\rho_\mu$。
- 各モデル: 各層の入力 u（第1層は x または LN(x)、第2層は a₁ または LN(a₁) の affine 後）の $\|\mu\|$・$\mathrm{sd}(e^\top u)$・min・max・$c$・$\kappa$・$\cos(\mu,\mathbf 1)$・全1方向の和の平均と sd、第2層入力の共分散の trace・上位 3 固有値・最上位固有ベクトルと e₂ の cos、a₁ の最大・最小・平均が −0.9 未満の座標の割合、h₂ で正のユニットの割合のサンプル最小・平均・0 のサンプル割合。`inLN_a` では $W_2\beta$（実効 bias の β 分）。
- Adam（W_l の行 + b_l、ユニットごと）: $\hat m$ の RMS、$\sqrt{\hat v}+\varepsilon$ の RMS、$\sqrt{\hat v}$ の最小、$\sqrt{\hat v}<\varepsilon$ の割合（ε 域）、実際の歩幅 $\hat m/(\sqrt{\hat v}+\varepsilon)$ の RMS、ε を除いた歩幅の RMS。
- 帳簿（連続する 2 時点・ユニットごと）: 第2層は上の 4 項と閉包残差、上流項の伸び・回転、$W_2\beta$ の変化、W₂ の行の変化ノルム。第1層も 4 項。Shapley: $d_2,\rho_\mu$ は 3 因子（(W₂,b₂)・µ₂・Σ₂）、$\sigma_2$ は 2 因子（W₂・Σ₂）。
- 生データ（ユニット別・帳簿・タスク終端の W₂）は `results/l2_wall_0916/logs/`（git 外）に置き、終了後 `obsidian-research-data/l2_wall_0916/` へ退避して `backup_manifest.json` を commit する。

## 4. 検査（許容はすべて演算の算術から導く・変異は許容の 10 倍で判定）

ε₆₄ = 2.2e−16、N = 1200、d_in = 784 または 100。

| 検査 | 内容 | 許容 | 変異対照 |
|---|---|---|---|
| S1 手書き逆伝播 | 変種モデル 5 腕 × ELU/leaky の勾配を float64 autograd と照合（γ≠1, β≠0 で affine 経路を通す） | $10^4 ε_{64}\max|g|$ | LN 逆伝播の $n\overline{gn}$ 項を落とす／γ を掛け忘れる／in と pre のマスクを入れ替える |
| S2 親との同型 | M=24 の `none` だけの変種エンジンが親エンジンと forward・1 更新で bit 一致。M=132 との差は報告 | 0 | 1 モデルの W₁ を 1 ulp ずらす |
| S3 ε の適用 | 1 更新の実更新がモデルごとの ε の式と bit 一致 | 0 | 全モデル ε=1e−8 で計算（効いた座標数 > 0 を要求） |
| S4 恒等式 | $\bar z=qr+b$（第1・2層）、分散 3 項の和 = $\sigma^2$、$W_2\beta$ の分解、帳簿の閉包、伸び+回転 = 上流、Shapley の角が実測の $d_2,\rho_\mu,\sigma_2$ と一致 | $\bar z$: $(d_{in}+N+16)ε_{64}Z$、分散: $4(d_{in}+N+16)ε_{64}Z^2$、閉包: その 2 時点の和、角: $8(N+d_{in}^2+16)ε_{64}(|f|+1)$（$Z$ はユニットの $\max|z|+|b|+|qr|$） | 一様軸で q を計算／a₁ をモデル間でずらす／共分散項を落とす／交差項を落とす／µ₂ を float32 で計算／伸びを旧 r だけで計算／新しい角に旧 Σ₂ |
| S5 bit 一致（Part A） | 各タスク終端の親 `evaluate()` 出力 = commit 済み | 0 | t と t−1 を照合（差 > 0 を要求） |
| S6 float64 と float32 | 再計算した $\bar z$ と commit 済み `zmean` の差 ≤ $(d_{in}+N+16)ε_{32}\max|z|$ | 左 | 層を取り違える |
| S7 費用 | 短い走で peak RSS と 1 タスクの時間 | 報告 | — |

## 5. 集計と判定

床の規則は親と同じ（t41–50 の online 精度平均が F + 3s/√10 以下なら床。F・s はその窓の多数派ラベル割合）。腕のラベルは 3 seed 一致のときだけ立て、割れたら `SPLIT`。

### Part A（ラベルなし）
崩壊モデルごとに次の初回時点を出す: 中央ユニットの $\sqrt{\rho_\mu}\ge\kappa_2/c_2$、中央 $d_2\le-c_2+0.1c_2$、中央 ε 域割合 > 0.5、平均 φ′ < 0.05、$U_2<0$ の割合 > 0.5。窓 t1–5・6–10・11–20・21–50（ext150 は 21–50・51–100・101–150）で帳簿と Shapley を合計する。

### Part B（RL の EE と LE を別々に）
各モデルの $D=$ t41–50 平均の（中央ユニット $d_2$ / $c_2$）。同じ seed の `none` との差 $\Delta D$。
- `PIN`: 両 ε 腕・3 seed のすべてで $|\Delta D|\le\delta$。
- `BRAKE`: 両 ε 腕・3 seed のすべてで $|\Delta D|>\delta$、かつ ε を大きくした腕と小さくした腕で符号が逆。
- それ以外は `MIXED`。
- δ は `none` 自身の揺れから取る: `none` 3 seed の t41–50 における $d_2/c_2$ のタスク間 sd の最大値の 3 倍。

### Part C（RL の 4 問）
- **C1（EE・`inLN`）**: 床の上なら `RESCUED`。床なら帳簿（t1→t50 の全区間の和・ユニットごと）で $S_\mu=\sum(\text{自己}+\text{上流}+\text{交差})$ と $S_b=\sum\Delta b_2$ を比べ、中央ユニットで $|S_b|>|S_\mu|$ なら `B_PATH`、そうでなければ `MU_PATH`。
- **C2（LE・`inLN`）**: `RESCUED` / `COLLAPSED`。
- **C3（EE・`preLN`）**: `RESCUED` / `COLLAPSED`。
- **C4（EE・`preLN_a`）**: `RESCUED`。床なら t50 の中央 β₂ < 0 かつ h₂ の中央ユニットの $\bar h_2<0$ で `COLLAPSED_VIA_BETA`、それ以外は `COLLAPSED_OTHER`。
- 報告のみ: PM・EL・LL の全腕、`inLN_a` の実効 bias、各腕の $\rho_\mu$ と $c,\kappa$ の時間変化。

## 6. 予測（走行前・記名）

### Claude
- **P-A1**: RL EE で、中央 $\sqrt{\rho_\mu}\ge\kappa_2/c_2$ の初回は、中央 ε 域割合 > 0.5 の初回より遅くない（3/3）。60%
- **P-A2**: RL EE の t1→t20 で、$\Delta d_2$ の Shapley は W₂ 自身が 3 因子で最大（3/3）。55%
- **P-A3**: RL EE の t1→t10 で、帳簿の合計は |自己| > |上流|、|交差| < 0.1 |Δz̄₂|（3/3）。65%
- **P-A4**: RL EE で ε 域に入ってから t50 まで、|上流| > |自己|（3/3）。70%
- **P-A5**: RL LE の t50 で $\min e_2^\top a_1>0$ かつ |中央 $d_2+c_2$| < 0.2（3/3）。60%
- **P-A6**: ext150 PM GELU の t150 で中央 $\rho_\mu\ge0.8$ かつ中央 $|b_2|/\sigma_2\le0.3$（3/3）。55%
- **P-A7**: ext150 RL SILU の崩壊 2 seed も P-A6 と同じ（釘付け）。55%
- **P-B（主）**: EE `PIN`・LE `PIN`。50%（EE）・45%（LE）
- **P-B1**: ε=1e−30 は `none` より t50 の中央 $|\bar z_2|$ が大きい（EE・LE とも 3/3）。85%
- **P-B2**: ε=1e−6 は `none` より小さい（同）。75%
- **P-B3**: ε 腕はすべて床（12/12）。85%
- **P-C1（主）**: EE `inLN` は `MU_PATH`。50%（`RESCUED` 20%、`B_PATH` 30%）
- **P-C1b**: EE `inLN` の t41–50 の中央 $d_2$ は同じ seed の `none` の $-c_2$ より深い（3/3）。55%
- **P-C2**: LE `inLN` は `RESCUED`。45%
- **P-C3**: EE `preLN` は `RESCUED`。70%
- **P-C4**: EE `preLN_a` は `RESCUED`。60%（`COLLAPSED_VIA_BETA` 30%）

### Issa が貼った別チャットの読み（µ 方向に読み替えて記録）
入力 LN で µ 方向の高 ρ ユニットが消え、吸収線越えは実効 bias の負の大きさで並ぶ。leaky と ELU も GELU と同じ b 経路で死ぬ。→ C1 `B_PATH`、C2 `COLLAPSED`。

### Issa
空欄（LN の配置は Claude に一任）。

## 7. 保存・費用

- commit: `results/l2_wall_0916/{taskend_*.csv, summary_*.csv.gz, events.csv, windows.csv, verdict.csv, checks.json, report.md, provenance_*.json, backup_manifest.json}`
- git 外: `results/l2_wall_0916/logs/*.npz`
- 費用: 層別キメラの箱は 24 モデル 137 秒・ext150 は 30 モデル 580 秒（親の実測）。診断を足した実測は S7。GPU 1 本で Part A（2 走）→ Part B・C の順に直列。並走中の `mucap_el_0916`（CPU 6 本）には触れない。

## 8. 限定

- Part A は観察。帳簿と Shapley は代数的な帰属で、項を除いた反実仮想ではない。
- 3 seed・1 箱。問いは §0 を見てから立てた。
- Part C の LN は 1 つの実装（eps 1e−5・画素側 affine なし・b 残し）。LN 一般の性質とは書かない。
- ε=1e−30 は float32 の非正規化数域に入る。非有限になったモデルはそのモデルだけ記録して止める。
