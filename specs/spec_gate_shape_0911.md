# ゲート被覆か非線形性か spec 0911
状態: **未実行（事前登録）** / 更新: 2026-09-11 / 起草: Claude（Issa 就寝中・Issa の予測欄は未記入）
親: [[幅の規制は可塑性を買うか_0910]] §3-1（主軸 B・c_act の正体） / [[中心主張v9草案_0910]] §3-5・§10「§3-5 の検定」 / 発端: Issa の問い（9/10 夜: ELU は浅い負側で働く・「負側にいる」と「そこで動けない」は別・Snake は位相）
run id: `gate_shape_0911`（本走・13 腕・t1–120）＋ `gate_wclamp_0911`（副走・Snake 梯子と leak 梯子の wclamp 継続 t21–100）
実装: lop_analysis `collective_kick_0908` / `specs/spec_gate_shape_0911.md`（spec 単独 commit を実行前に記す）/ `src/gate_shape_0911.py`・`src/gate_wclamp_0911.py`・`src/gate_shape_report_0911.py`
姉妹 spec: [[低ゲート個体の滞留_spec_0911]]（本走の出力を使う解析）・[[個体別3分表_spec_0911]]・[[高ゲート帯の幅ダイヤル_spec_0911]]

## 0. 何を決めるか

V9 は loss ≈ c_act · g(‖W̃‖) と書いた。腕の中では g（幅）が損失を担い（介入で 75–90%）、腕の間の差は係数 c_act（活性化の形）が担う。**c_act を決める量は未同定**（主軸 B）。候補が 2 つある。

- **(a) 低ゲート被覆**（Issa・9/10 夜）: 前活性分布のうち高ゲート帯に乗る質量が多い腕ほど可塑性を保つ。ELU の浅い負側（φ′ = e^z）は leaky の 0.1 より強く、それが深部の弱さを補う、という読み。さらに「低ゲートの個体が固定メンバーか交代か」で意味が変わる。
- **(b) 支持上の実効非線形性**（V9 §3-5 の未検定仮説）: 支持の上で解決される非線形性が小さい腕（SN02・LIN の「ほぼ線形」）ほど c_act が小さい。

両者は同じ per-(unit, input) の φ′ から計算でき、**Snake α 梯子で分離できる**。α は (b) を単調に動かすが、(a) は SN02（ユニット丸ごと低ゲートで固定）と SN15（全ユニットが入力ごとに振動）で質が違う。

事後の材料（未登録・9/10 夜の手元集計・Gaussian 近似の E[φ′]・`elu_growth_0909` の per-unit 列）: ELU の集団ゲート中央値は t20 0.24 → t120 0.047 で t60 に leaky（0.107）を下回るのに、損失は ELU ≤ leaky。登録済みの `dclamp` は p⁺ を 0.235 → 0.389 に上げても ρ 0.03–0.09。**これらは (a) に不利だが、(a) の本体（per-input の被覆と腕間の順序）は測られていない。** 本 spec で測る。

## 1. 走

箱 B（pmnist 784→100→100→10・Adam lr 1e−3・batch 16・625 更新/タスク・CPU 1 スレッド・`H.setup('cpu')`）。init・置換列・抽出・バッチ順は seed で決まり腕に依らない（committed の host をそのまま使い、編集しない）。

**本走 `gate_shape_0911`**: 13 腕 × seed 0–2 = 39 走、t1–120、無介入。

| 腕 | 活性化 | G1 の錨（committed） |
|---|---|---|
| R | ReLU | `leak_ladder_force_posthoc_0910`（6 配列 × t1–100） |
| LR03 / LR / LR001 | leaky a = 0.3 / 0.1 / 0.01 | LR: `elu_growth_0909`（3 配列 × t1–120）、LR03・LR001: `leak_ladder_force_posthoc_0910` |
| ELU1 / ELU03 | ELU α = 1 / 0.3 | `elu_growth_0909` |
| SN02 / SN06 / SN15 | Snake α = 0.2 / 0.6 / 1.5 | `gate_scale_invariance_0909`（cnorm_i × t20…120 の 11 点） |
| SNA | 適応 α Snake（c=0.6・β=0.01） | `elu_growth_0909` |
| LIN | 恒等 | `linear_growth_0910`（3 配列 × 11 点） |
| GELU / SILU | 谷型（`valley_acts_0910`） | `long_horizon_acts_0910`（6 配列 × t1–120） |

**副走 `gate_wclamp_0911`**: SN02 / SN06 / SN15 / LR03 / LR001 × seed 0–2 = 15 走。t1–20 を共有前置きとして走り、t20 状態から `ref` と `wclamp`（各ユニットの ‖W̃ᵢ‖ を t20 の値へ毎更新射影。行平均・bias・W2・W3・Adam は自由。[[幅か沈下か_クランプ介入_結果_0909]] と同一の committed 実装 `clamp_horizon_0910.apply_clamp`）の 2 分岐を t21–100。
LR / ELU1 / SNA の wclamp は `width_sink_clamp_0909`（t21–100）、R / GELU / SILU は `clamp_horizon_acts_0910`（t21–400 のうち t21–100）の committed 出力を引く。LIN は wclamp 無し。

**測定（本走・毎タスク終端）**: 固定プローブ 512 枚（seed ごと・committed と同じ `boundary_probe` 系列）を**現タスクの置換**で通し、第 1 層の φ′(z_i(x)) の 512×100 行列 G を取る。ユニット i ごとに

- ḡᵢ = mean_x φ′、vᵢ = Var_x φ′（母分散）、offᵢ = P_x[φ′ < 0.25]、offabsᵢ = P_x[|φ′| < 0.25]、off10ᵢ = P_x[φ′ < 0.1]、off50ᵢ = P_x[φ′ < 0.5]、q10ᵢ = φ′ の 10% 点
- zcurᵢ・sdcurᵢ（現置換での前活性の平均・sd）
- gcorr（タスク開始時と終端の G の Pearson 相関、同じプローブ・同じ置換）
- 既存の測定（`zbar_i`・`sd_i`・`cnorm_i`・`m_i`・`bias_i`・`star_i`、8 参照置換平均・committed の `measure`）と test 精度（1 万枚）

φ′ は解析式（autograd 対照つき）。GELU・SiLU は**符号付き**のまま（負のゲートは「逆向きに動く」ので、off は符号付きを主・|φ′| 版を副）。θ = 0.25 は [[PermutedMNIST_追加診断_0905]] §11 の探針と同じ。

## 2. 量の定義（登録）

窓: **base = t16–20**、**late = t101–120**（本走）、**late61 = t61–100**（wclamp 分割）。窓内はタスク終端の中央値、seed は中央値で束ねる（3 seed の値も表に出す）。

- 損失 **L = acc_base − acc_late** [pt]
- 腕ごとの候補量（late 窓の中央値・ユニット集約は下記）
  - **(a) Cov** = mean_i offᵢ = P_{i,x}[φ′ < 0.25]
  - **(a′) Cov_chronic** = (1/100) Σ_{i∈chronic} offᵢ。chronic = t101–120 の **すべて**でゲート下位 10%（ḡᵢ の順位）に入っていたユニット（[[低ゲート個体の滞留_spec_0911]] と共通の定義）
  - **(b1) NL_x** = mean_i vᵢ（支持内のゲート分散）
  - **(b2) NL_u** = Var_i ḡᵢ（実効ゲートのユニット間分散）
  - **(c) gcorr**（タスク内のゲート安定性）
  - 対照: Ḡ = mean_i ḡᵢ、N = ‖W̃ᵢ‖ のユニット平均（幅）
- **残差損失 R_c = acc_base − acc_wclamp(late61)**（幅を止めても残る損失 = c_act が担う分・11 腕）。あわせて L61 = acc_base − acc_ref(late61)、ρ_w = 1 − R_c / L61 を出す。

## 3. 判定（事前登録）

候補 Q ∈ {Cov, Cov_chronic, NL_x, NL_u, gcorr} × 目標 T ∈ {L（13 腕）, R_c（11 腕）} について、腕の seed 中央値の Spearman ρ(Q, T):

| ρ | ラベル |
|---|---|
| ≥ +0.6 | `<Q>_ORDERS_<T>` |
| +0.2 < ρ < +0.6 | `<Q>_PARTIAL_<T>` |
| ‖ρ‖ ≤ 0.2 | `<Q>_FAILS_<T>` |
| −0.6 < ρ < −0.2 | `<Q>_PARTIAL_INVERTED_<T>` |
| ≤ −0.6 | `<Q>_INVERTED_<T>` |

向きは「Q が大きいほど損失が大きい」を +。n=13 の Spearman は |ρ| ≥ 0.56 で p<0.05（両側）、n=11 は 0.62。帯 0.6 はこれに合わせた。

- **A4 勝者**: R_c に対する |ρ| が最大の候補が次点より 0.2 以上大きければ `WINNER_<Q>`、そうでなければ `NO_WINNER`。
- **A5 決定的な対（SN02 対 SN06・seed 対）**: 各候補について sign(Q_SN06 − Q_SN02) = sign(L_SN06 − L_SN02) が 3/3 seed で成り立てば「その候補は対を説明する」。ラベル `PAIR_<候補の集合>`（空なら `PAIR_NONE`）。この対は幅がほぼ同じ（6.0 対 6.9）で損失が 3 pt 違う既知の対（[[中心主張v9草案_0910]] §3-5）。
- **A6 幅の分離**: Spearman(N, L) を併記。登録走では +0.07 だった。0.3 を超えたら「幅が腕間でも効く」と書き、R_c 版を主に読む。
- **可検定性**: 発散した腕seed は `DIVERGED` として落とす。L の符号は問わない（順位統計）。

## 4. 進捗ゲートと変異対照

- **G1（再現）**: `ref` の per-unit 配列が各腕の錨と **maxabs ≤ 1e−10**、比較件数が期待件数（錨の配列数 × 重なるタスク数）に一致（件数ガード）。副走の `ref` も同じ錨。副走の `wclamp` は錨無し（新規）。
- **G1 変異対照**: 初期値 W1[0,0] += 1e−3 で最初の錨点（大半は t1、gate_scale・linear_growth の錨は t20）の再現が 1e−4 を超えて外れる。**測定の非侵襲**: 測定時に W1 += 1e−9 する対照で G1 が外れる（本走の G1 が 0.0 なら測定は軌道を動かしていない）。
- **G2（φ′）**: 全活性化で φ′ が autograd と 1e−6 以内、別の活性化を当てる対照が 1e−3 超。
- **G3（ゲート帳簿）**: Var_{i,x} φ′ = mean_i vᵢ + Var_i ḡᵢ が 1e−9 以内（母分散）。vᵢ を 1.01 倍する対照で外れる。
- **G4（wclamp の実効性・副走）**: `c3_cnorm_rel` ≤ 1e−6・`c3_m_absdiff` ≤ float32 の証明可能床・`clamp_tail_absdiff` = 0（committed の検査）と、skip / rowscale / tail の変異対照（committed と同じ）が各 100 倍で発火。
- 時間上限 900 s / 走（本走）、1200 s / 走（副走）。

## 5. 事前予測（記名・走る前）

**Claude**:
- Cov: **`Cov_PARTIAL_L`**（+0.3〜0.5）。LIN・LR03（a=0.3 は θ=0.25 の上なので被覆 0）・SN02 が低被覆・低損失、GELU/SiLU が高被覆・高損失で正に寄るが、ELU（被覆は leaky 以上・損失は以下）と Snake 内（SN02 ≈ SN06 の被覆で 3 pt）で崩れる。R_c でも同じ帯。
- Cov_chronic: **`FAILS`**（固定の芯を持つ ELU・R・GELU が損失の両端に散る）。
- NL_x: **`FAILS` または `INVERTED`**（Snake 族が最大の支持内分散で最小級の損失、GELU が小さい分散で最大の損失）。
- NL_u: **`FAILS`**（SN15 は全ユニット ḡ≈1 で分散最小だが損失は SN02 より大きい）。
- gcorr: **`FAILS`**。
- A4 **`NO_WINNER`**。A5 **`PAIR_NL_x`**（SN06 > SN02 は NL_x だけが同符号。被覆は同程度と読む）。A6: Spearman(N, L) は 0.3 未満。
- 総括の予想: **c_act を静的なゲート形状の 1 量では並べられない。** 並ぶ量が出れば V9 §10 の更新条件を満たす。

**Issa**: 未記入（就寝中）。起床後、結果を読む前に書く欄:
> Cov: ／ Cov_chronic: ／ NL_x: ／ NL_u: ／ gcorr: ／ A4: ／ A5:

## 6. 副測定（ラベル無し・結果前に列挙）

θ = 0.1 / 0.5 の被覆、|φ′| 版の被覆、Ḡ、q10 の分布、per-unit の ḡᵢ・vᵢ 散布、gcorr の時間推移、腕ごとの late 窓の z̄・σ・N・p⁺・dead_hard、Snake の 2αz̄ᵢ と αWᵢ（[[低ゲート個体の滞留_spec_0911]] へ）。A1–A3 の散布図。

## 7. 限界（結果前に明記）

- t120（登録走の t400 より短い）。プローブ 512 枚・現置換。
- θ = 0.25 の 1 点が主（0.1・0.5 は副）。
- R_c の窓は t61–100 で、L の窓 t101–120 と違う（LR/ELU1/SNA の committed wclamp が t100 で終わるため）。
- LIN の R_c 無し。GELU/SiLU の錨は同 spec 系（transport_holes 副走 A）で、独立の登録走ではない。
- 13 腕の順位相関は n が小さく、帯の境界（0.6）を跨ぐ結果は「決めない」と読む。
