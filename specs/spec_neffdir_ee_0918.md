# 微分の担い手を直接動かす（H7）— respdyn の残りは担い手の数で決まるか

状態: **下書き（未登録）**。設計・実装・較正・速い検査は Claude（2026-09-18 未明）。**登録から後は Codex が §11 の手順で行う**（Issa「全部 Codex に渡す」）。登録は、§11 の手順 3 で本ファイルの状態行を「事前登録」に替えて commit・push したときに成立する。
作成: 2026-09-18 / 起草: Claude / 依頼: Issa「やりましょう」（vault `主張/中心主張v10作業リスト_0917` の H7）
前の走: `specs/spec_escneff_ee_0917.md`（seed 10–19・主 `REMAINDER_REDUCED_BY_HOLD`・`NEFF_TRACKS`・`SLOPE_POSITIVE`）、`specs/spec_escape_ee_0917.md`（seed 0–9）
実装: `src/neffdir_ee_0918.py`・`analysis/neffdir_ee_0918/{checks.py,verdict.py}`（`launch.sh` は未作成・§11）
branch: `claude/neffdir_ee_0918`（worktree `~/Projects/claude/wt/neffdir_ee_0918`）

## 0. 一行と問い

escneff_ee_0917 では、移植先（健康な t2 の網。第2層に、崩壊していく t10 の網の動く場を当てる）の成長を止めると、課題中の $\bar n_{\rm eff}$（第2層の訓練の微分を運ぶ (ユニット, 画像) の実効数）が減り、応答の床の上の残り $R$ がそれに比例して減った（seed 間 r +0.92）。**ただし、動かしたのは成長で、$\bar n_{\rm eff}$ は学習の結果も含む。** 本走では、成長とは別に担い手そのものを動かし、学習能力 E（継続 1 タスク目の online 精度）がそれに従うかを問う。

- **Q1（主）**: 成長を止めた網（c12）に、担い手を外から足すと、E は上がるか。
- **Q2**: その上がり方は、動かした $\bar n_{\rm eff}$ に比例するか（$R=\beta\,\bar n_{\rm eff}$、$\beta$ は seed ごとに基準の腕から）。
- **Q3**: 担い手を落とすと、E は下がるか。成長を止めない網と止めた網で、課題全体と前半（最初の 1500 更新）で見る。
- **Q4（補償）**: 担い手を落とされた網は、自分の担い手を増やして取り戻すか。
- **Q5**: 毎更新引き直すマスク（同じ雑音で、担い手は失われない）は、固定マスクと同じだけ E を下げるか。
- **Q6**: 厳密な 0 だけを消す（F.elu の微分。$\bar n_{\rm eff}$ はほぼ動かない）と、E は変わるか。

## 1. なぜ・何を知っているか（REPORT_ONLY）

### 1.1 escneff_ee_0917（seed 10–19）の腕平均（登録済みの走・既知）

| 腕 | E（継続 1） | $\bar n_{\rm eff}$ | $R/\bar n_{\rm eff}$（床 0.2195 からの傾き） |
|---|---|---|---|
| N2r（自然） | 0.7128 | 0.4476 | 1.1（飽和） |
| S2_10r（固定した場） | 0.4730 | 0.0254 | 10.0 |
| S2_10r_c12 | 0.3896 | 0.0180 | 9.4 |
| **S2dyn_10r**（動く場） | 0.3475 | 0.0108 | 11.9 |
| S2dyn_c1 | 0.3234 | 0.0087 | 12.0 |
| S2dyn_c2 | 0.3289 | 0.0108 | 10.1 |
| **S2dyn_c12** | 0.2961 | 0.0075 | 10.3 |
| S2dyn_c12b | 0.2935 | 0.0070 | 10.6 |
| S2dyn_frz（凍結） | 0.2195 | 0.0031 | 0（外れ） |
| S2u30r（床） | 0.2195 | 0 | — |

$\bar n_{\rm eff}\lesssim0.03$ の範囲では、凍結した腕を除いて $R\approx\beta\,\bar n_{\rm eff}$、$\beta\approx10\sim12$ で並ぶ（腕平均・事後）。seed 間の $\Delta R/\Delta n$ の傾きは 9.3（seed 10–19）、10.0（20 seed・事後）。**本 spec の比例の予測（§4）は、この読みを seed ごとの基準の腕で較正して使う。**

### 1.2 較正（seed 44・登録外・2026-09-18 未明・Claude）

設計の大きさを決めるために、登録外の seed 44 で腕を回した。**読んだのは $\bar n_{\rm eff}$・0 の割合・登り・重みのノルムだけで、E（online_acc・early_acc）は読んでいない。** 出力は `results/_checks_neffdir_ee_0918/calib_s44/`（E の列を含む。§11 の手順 6 の予測の前に開かないこと）。

| 腕 | $\bar n_{\rm eff}$(S) 課題全体 | 前半（s ≤ 1500） | 開始時 | 網自身の担い手 $\bar n_{\rm eff}$(T) | 0 の割合(S) | 登り | ‖w1‖ | ‖w2‖ |
|---|---|---|---|---|---|---|---|---|
| S2dyn_10r | 0.00480 | 0.00231 | 0.00486 | 0.00480 | 0.976 | +0.97 | 6.35 | 2.51 |
| S2dyn_m50 | 0.00492 | 0.00126 | 0.00255 | **0.00988** | 0.985 | +1.05 | **7.56** | 2.53 |
| S2dyn_m25 | **0.00881** | 0.00084 | 0.00147 | **0.03463** | 0.985 | +1.78 | **8.65** | 2.53 |
| S2dyn_s50 | 0.00340 | 0.00225 | 0.00486 | 0.00340 | 0.981 | −0.03 | 5.67 | 2.43 |
| S2dyn_a02 | 0.01470 | 0.01144 | 0.02281 | 0.01470 | 0.946 | +8.09 | 7.03 | 3.04 |
| S2dyn_felu | 0.00500 | 0.00279 | 0.00508 | 0.00464 | **0.221** | +1.24 | 6.35 | 2.51 |
| S2dyn_c12 | 0.00433 | 0.00239 | 0.00486 | 0.00433 | 0.976 | +1.75 | 4.51 | 1.98 |
| S2dyn_c12_m50 | 0.00360 | 0.00126 | 0.00255 | 0.00697 | 0.986 | +2.47 | 4.51 | 1.98 |
| S2dyn_c12_a02 | 0.01124 | 0.01106 | 0.02281 | 0.01124 | 0.959 | +7.56 | 4.51 | 1.99 |

読み（設計に使った・1 seed）:

1. **固定マスクは課題の中で取り戻される（補償）。** m50 の $\bar n_{\rm eff}$(S) は開始時に半分だが、網自身の担い手（T）が倍になり、課題全体では基準と同じになった。m25 は基準を超えた。取り戻しは主に後半で起き（探針の時系列: m50 は s ≤ 1500 で基準の約 0.55 倍、s = 3750 以降で基準を超える）、成長の上限（c12）の下でも一部起きた（c12_m50 は課題全体で 0.83 倍）。→ **落とす腕は「課題全体」と「前半」の両方で読み、補償そのものを量として登録する。**
2. **足す腕は成長を止めても効く。** c12_a02 は $\bar n_{\rm eff}$ を 2.6 倍にし、上限は保たれた（‖w1‖・‖w2‖ が c12 と同じ）。→ **主判定は「成長を止めた網に足す」（c12_a02 対 c12）。**
3. **F.elu は 0 の割合だけを大きく動かし（0.98 → 0.22）、$\bar n_{\rm eff}$ はほとんど動かさない。** 前向きの値は宿主の ELU と、切り捨ての境（z = −16.6355）でだけ 1 刻み（2⁻²⁴）違う（検査 S1c）。

### 1.3 $\bar n_{\rm eff}$ の定義（構造的 $\bar n_{\rm eff}$・S）

継続タスク中の 750 更新ごと（s = 0, 750, …, 6000）に、1200 枚で網自身の第2層前活性 $z_2$ に場 $d$ を足した $z_2+d$ の**その腕が訓練に使う微分** $g$ を計算する（宿主の expm1 の係数。F.elu の腕は F.elu の係数。固定マスクの腕は、マスクで落とした (ユニット, 画像) を 0 にする。毎更新引き直すマスクは、どの対も同じ頻度で残るので掛けない）。ユニットごとに $(\sum_x g)^2/(N\sum_x g^2)$（全画像で 0 のユニットは 0）、ユニット平均を取り、9 点で平均したものが $\bar n_{\rm eff}$(S)。前半は s = 0, 750, 1500 の 3 点。**マスクも F.elu も無い腕では respdyn・escneff の $\bar n_{\rm eff}$ と同じ値**（検査 S1a）。網自身の担い手 $\bar n_{\rm eff}$(T) は、マスクを掛けず宿主の係数で数えた同じ量（respdyn の定義そのもの）。

## 2. 設計

### 2.1 箱（escneff_ee_0917 と同一）

Random Label MNIST（1200 枚固定）、784–100–100–10、両隠れ層 ELU(α=1)、Adam lr 1e−3（β 0.9/0.999、ε 1e−8）、batch 16、6000 更新/タスク、float32、CPU 1 スレッド、`flush_denormal`。

### 2.2 seed と腕

- **seed 30–39**（どの走でも使っていない）。1 seed = ref の自然軌道 task 1–12（escape の prefix。t2 と t10 を保存）→ t2 の網から 12 腕 × 2 タスク（Adam 初期化・t2 のラベル列と順序列）。動く場は escape・escneff と同じ（影 = N10 の自然な継続）。
- 12 腕（`src/neffdir_ee_0918.py` の `ARMS`）:

| 腕 | 場 | 保持 | 担い手の操作 | 役割 |
|---|---|---|---|---|
| S2dyn_10r | 動く | なし | なし | 基準（成長あり） |
| S2dyn_m50 | 動く | なし | 固定マスク q = 0.5 | 落とす |
| S2dyn_m25 | 動く | なし | 固定マスク q = 0.25（m50 の部分集合） | 落とす・深く |
| S2dyn_s50 | 動く | なし | 毎更新マスク q = 0.5 | 雑音の対照 |
| S2dyn_a02 | 動く | なし | 2% を持ち上げる | 足す |
| S2dyn_felu | 動く | なし | F.elu の係数 | 厳密な 0 を消す |
| S2dyn_c12 | 動く | c12 | なし | 基準（成長なし） |
| S2dyn_c12_m50 | 動く | c12 | 固定マスク q = 0.5（m50 と同じ表） | 落とす・成長なし |
| **S2dyn_c12_a02** | 動く | c12 | 2% を持ち上げる（a02 と同じ表） | **足す・成長なし（主）** |
| S2u30r | 一様 −30 | なし | なし | 応答の床 |
| N2r | — | なし | なし | 自然な継続 |
| N2r_m50 | — | なし | 固定マスク q = 0.5 | 健康な網で落とす（報告） |

- 検査だけの 3 腕（`CHECK_ARMS`）: S2dyn_m100（q = 1）・S2dyn_s100（q = 1）・S2dyn_a00（r = 0）。どれも S2dyn_10r と bit 一致しなければならない（検査 S1b）。

### 2.3 担い手の操作

前向きは resp_ee の固定補正つきの式 $a_2=\phi(z_2^0)+[\phi_2(z_2+d)-\phi_2(z_2^0+d)]$ のままで、分岐点の logits は全腕で自然な t2 の網と bit 一致する（検査）。

- **固定マスク q**: seed ごとの一様乱数表 $U\in[0,1)^{1200\times100}$（`MASK_SALT = 918001` + seed の生成器、先に $U$、次に $V$）で、$U<q$ の対だけが微分を通す。通した対は $1/q$ 倍（勾配の期待値は変えない）。実装は $u_{\rm d}+s\,(u-u_{\rm d})$、$s=\mathbb 1[U<q]/q$、$u=\phi_2(z_2+d)$、$u_{\rm d}$ はその detach。**前向きの値は $u$ と bit 一致し、逆向きだけが $s$ 倍になる**（検査 S1b）。q を変えても同じ $U$ を使うので、m25 の残す対は m50 の残す対の部分集合。
- **毎更新マスク q**: 勾配を取る前向きのたびに、専用の生成器（`STEP_SALT = 918002` + seed）から $(16\times100)$ の一様乱数を引き、同じ式で掛ける。宿主のラベル列・順序列には触れない。勾配を取らない前向き（探針・要約）では引かない。preflight の 75 回も引く（1 タスク目 75 + 6000 回、2 タスク目 +6000 回）。
- **持ち上げ r**: 一様乱数表 $V<r$ の対（seed あたり約 2400 対）について、影の更新ごとに場を $d=\max_x z_2^{(\rm 影)}(x,u)-z_2^{(\rm 分岐)}(x,u)$ にする（float64 で計算して float32 に）。その対は、影のユニットの最上位の画像と同じ深さに、宿主自身の動きを足した位置に来る。影のユニットが全画像で零点より下なら、持ち上げても担い手は増えない（seed 44 では 92% のユニットが持ち上げ可能）。錨の $\phi_2(z_2^0+d)$ も同じ場で計算するので、分岐点の logits は変わらない。r = 0 は S2dyn_10r と bit 一致（検査）。
- **F.elu**: 括弧の $\phi_2$ を `F.elu` にする。訓練の微分は $z\le0$ で F.elu の kernel の $e^z$（float32・flush で $z<-87.34$ のときだけ 0）、$z>0$ で 1。前向きは宿主の ELU と、宿主の expm1 が切り捨てる点でだけ 2⁻²⁴ 違う。探針の係数は F.elu の autograd そのもの（torch.exp とは一部の z で 1 ulp 違うため）。
- **c12**: escape の保持（`escape_ee_0917.Hold("c12")`: 第1層の $|q_i|$・$\|v_i\|$ と第2層の行ノルムに、t2 の値で上限）。

## 3. 測る量（`arms.csv`・腕 × 継続タスク）

escape の列（`online_acc`・`climb2`・`w1norm_end`・`w2norm_end`・保持の外れ・書き込み行数・影と分岐点の照合など）に加えて、`nbar_neffS2`（§1.3 の S）、`nbar_neffT2`（T）、`probe_zeroS2_mean`、`early_acc`（更新 0–1499 の online 精度）、`nbar_neffS2_early`・`nbar_neffT2_early`（s ≤ 1500 の 3 点）、`neffS2_start`・`neffT2_start`、`keep_share`（固定マスクは表の割合、毎更新マスクは引いた割合の累積）、`step_draws`、`lift_share`・`lift_units_alive`・`field0_top_ok`、`field0_equal_fixed`（持ち上げの腕では持ち上げない対について）。探針の時系列は `traj.csv`。

## 4. endpoint

継続 1 タスク目。$E$ = `online_acc`、$F=E(\text{S2u30r})$、$R(a)=E(a)-F$、$n(a)$ = `nbar_neffS2`、$T(a)$ = `nbar_neffT2`。前半は `early_acc`・`nbar_neffS2_early` で同じ式（床も前半の値）。

- 変化 (tag, 基準, 腕, 向き $\sigma$): add_h = (c12, c12_a02, +)、add = (10r, a02, +)、drop_h = (c12, c12_m50, −)、drop = (10r, m50, −)、m25 = (10r, m25, −)、前半の drop_e = (10r, m50, −)・drop_h_e = (c12, c12_m50, −)・add_h_e = (c12, c12_a02, +)。
  - $\Delta R=\sigma\,[R(\text{腕})-R(\text{基準})]$、$\Delta n=\sigma\,[n(\text{腕})-n(\text{基準})]$、$\kappa=n(\text{腕})/n(\text{基準})$（seed ごと）。
  - **不足 $\varphi=\sigma\,[\kappa R(\text{基準})-R(\text{腕})]$**: 比例の予測 $R(\text{基準})(\kappa-1)$ に対して、観測した変化がどれだけ足りないか（正 = 予測より小さい）。
- 補償: comp = $T(\text{m50})-T(\text{10r})$、comp_h = $T(\text{c12\_m50})-T(\text{c12})$。
- 構造: struct = $E(\text{s50})-E(\text{m50})$。報告 dR_step = $R(\text{10r})-R(\text{s50})$。
- F.elu: d_felu = $E(\text{felu})-E(\text{10r})$、操作の確認 dzero_felu = 0 の割合(S)(10r) − (felu)、報告 dn_felu。
- 並び: seed ごとに {m25, m50, 10r, a02} の $n$ と $E$ の Spearman 順位相関。
- 成長（escneff の再現）: dR_growth = $R(\text{10r})-R(\text{c12})$。報告 I_nat = $E(\text{N2r})-E(\text{N2r\_m50})$。

## 5. 判定

### 5.1 帯

seed 内の差の t 区間（自由度 n−1）。主の $\Delta R_{\rm add\_h}$ は 97.5%、他は 95%。符号は区間が 0 を含まなければ ±、含めば 0。

### 5.2 適用条件

- **(A)** seed ごとに: 12 腕 × 2 タスクがそろい、E・S・T・0 の割合・前半の値が有限。全腕で分岐点の logits が一致（全体と最初の minibatch）。prefix が 12 タスクで、記録（resp_ee）と矛盾しない（記録が無いのは可）。記録のある計算（S2dyn_10r・S2dyn_c12・S2u30r・N2r）が resp_ee・respdyn・escape の記録と矛盾しない（registered seed には記録が無いので空欄）。動く場の全腕で影 = N10、$d(0)$ = 固定した場（持ち上げの腕は持ち上げない対で）。c12 の 3 腕が上限の外れ ≤ 1e−4 で、各タスクで上限が書く。マスクの割合が |割合 − q| ≤ 0.01（120000 対の二項分布の 6 SD を超える幅）。持ち上げの割合が |割合 − 0.02| ≤ 0.003（同 6 SD）で、`field0_top_ok`。有効 seed が 8 未満なら INAPPLICABLE。
- **(B)** $R(\text{10r})$ と $R(\text{c12})$ の 95% 区間の下端 > 0。満たさなければ NOT_REPRODUCED。
- **(C)** 各変化の $\Delta n$ の 95% 区間の下端 > 0（F.elu は dzero_felu > 0）。満たさなければ、その問いは NOT_MANIPULATED。

### 5.3 ラベル（`analysis/neffdir_ee_0918/verdict.py`）

| 問い | 量 | ラベル |
|---|---|---|
| **primary**（97.5%） | $\Delta R_{\rm add\_h}$ | ADD_H_MOVES_E（+）/ ADD_H_NO_EFFECT（0）/ ADD_H_REVERSED（−）/ NOT_MANIPULATED / NOT_REPRODUCED / INAPPLICABLE |
| prop_add_h・prop_add・prop_drop_h・prop_drop・prop_drop_early・prop_drop_h_early | $\varphi$ | 区間が (−M, +M) に収まれば PROPORTIONAL、下端 > 0 なら SUBPROPORTIONAL、上端 < 0 なら SUPRAPROPORTIONAL、それ以外 PROPORTION_UNRESOLVED |
| add・drop_h・drop・m25・drop_early・drop_h_early | $\Delta R$ | 〈名〉_MOVES_E / 〈名〉_NO_EFFECT / 〈名〉_REVERSED（+ / 0 / −） |
| compensation・compensation_h | comp・comp_h | COMPENSATES / NO_COMPENSATION / RAW_FALLS（_H つき） |
| structure | struct | STRUCTURE_MATTERS / STRUCT_EQUAL / STEP_WORSE |
| felu | d_felu | 区間が (−M, +M) に収まれば FELU_SAME、下端 > 0 なら FELU_HIGHER、上端 < 0 なら FELU_LOWER、それ以外 FELU_UNRESOLVED |
| ladder | 順位相関 | 10 seed の符号で正確な両側符号検定、p < 0.05 で NEFF_TRACKS / NEFF_OPPOSES、他は NEFF_UNRESOLVED |
| growth | dR_growth | GROWTH_REMAINDER / GROWTH_NO_REMAINDER / GROWTH_REVERSED |

**M = 0.03**。導出: escneff の $R_{\rm none}$ = 0.128 で、「E が担い手に応じない」（目印の模型）なら q = 0.5 の落としで $\varphi=(1-\kappa)R_{\rm none}\approx0.064$、比例の模型なら 0。その中点を切り下げた。前半の窓では $R$ が小さいかもしれず、目印の模型の $\varphi$ が 2M を下回るときは、前半の比例ラベルは判別力を持たない（報告で $R_{\rm none\_e}$ と $\kappa$ を併記する）。

### 5.4 読み方（登録）

- **ADD_H_MOVES_E** → 成長を止めた網でも、担い手を外から足せば学習能力が上がる。escneff の「成長 → 担い手 → 残り」の後半（担い手 → 残り）が、成長と別に介入で支持される。**さらに prop_add_h が PROPORTIONAL** なら、上がり方は $\bar n_{\rm eff}$ の比例の予測どおり（$\bar n_{\rm eff}$ が残りの量を決める）。SUBPROPORTIONAL なら、$\bar n_{\rm eff}$ は効くが escneff の傾きほどではない（成長そのものの寄与が残る）。
- **ADD_H_NO_EFFECT** → 担い手を足しても学べない。$\bar n_{\rm eff}$ は、この箱では成長の結果に付いてくる目印で、原因とは言えない（add も NO_EFFECT なら一層強い）。
- **落とす側**: drop・drop_h が MOVES_E なら、担い手を減らすと下がる（必要側）。NOT_MANIPULATED は「課題の中で取り戻された」を意味し、compensation のラベルと併せて読む（COMPENSATES なら「学習が担い手を作る」向きの矢印がある）。前半の窓は補償の前なので、drop_early・drop_h_early を必要側の主な読みにする。
- **structure**: STRUCT_EQUAL で drop が MOVES_E なら、落とした効果は雑音と区別できない（担い手の数の効果とは言わない）。
- **felu**: FELU_SAME なら、この移植で学べなさを作っているのは厳密な 0 の数ではなく、微分を運ぶ実効数。FELU_HIGHER なら、小さい 0 でない微分も効く（$\bar n_{\rm eff}$ の大きさの重みづけは足りない）。
- **ladder**: NEFF_TRACKS は、補償で $\bar n_{\rm eff}$ が予想外の向きに動いた腕（m25）も含めて、E が $\bar n_{\rm eff}$ の順に並ぶこと。
- 範囲: ELU→ELU・乱数ラベル・1200 枚・6000 更新・t2 の網・t10 の動く場・Adam 初期化・継続 2 タスク・t2 の値での保持。持ち上げは人工的な場の変更（固定補正つき）で、足した担い手はユニットの最上位の画像の深さに置いたもの。

## 6. 検査（登録前に開発実行。許容は導出から）

`python3 analysis/neffdir_ee_0918/checks.py` → `results/neffdir_ee_0918/checks.json`。各検査は本物のコードで通り、列挙した変異（対象ファイルの 1 か所の置換）で必ず落ちなければならない。検査は E を出力しない。

| 検査 | 要求 | 変異 | 開発実行（2026-09-18 未明・Claude） |
|---|---|---|---|
| S1a 記録（seed 0・80 epoch） | S2dyn_10r・S2dyn_c12・S2u30r・N2r がこの runner を通して respdyn・escape・resp_ee の記録と状態 hash が一致（2 タスク）、分岐点の logits、影と $d(0)$、c12 の保持と書き込み、`nbar_neffT2` が escape の記録と完全一致し `nbar_neffS2` もそれに等しい、provenance | 4 | **未実行**（約 20 分） |
| S1b 担い手（seed 45・8 epoch） | q = 1・r = 0 の 3 腕が S2dyn_10r と bit 一致、全腕の logits・影・場、マスクの割合と入れ子と引いた回数、持ち上げの割合と位置（2 eps32 (\|top\|+\|z\|) 以内）、c12 の 2 腕の保持、前半の列、実 minibatch で autograd の係数 = G·s·φ₂′（固定マスク・F.elu・恒等・健康な網のマスク）、探針の独立な再計算 | 8 | **未実行**（約 15 分） |
| S1c 係数（格子） | F.elu と宿主 ELU の autograd の係数が runner の係数と完全一致、F.elu の係数が exp の 2 ulp 以内、0 になる範囲（−16.64・−87.34）、φ₂ の値（F.elu は宿主と 2⁻²⁴ 以内） | 2 | pass・2/2 検出 |
| S8 判定 | 合成 seed 30–39 の 40 場面でラベルと推定値（1e−9）、t 分位点（公表値）、SEEDS・MARGIN・DYN | 22 | pass・22/22 検出 |
| S9 CLI | seed 45・2 epoch で provenance（実験名・spec・flush・乱数表の hash・code hash）、定数（seed 30–39・12 腕・検査 3 腕・塩・前半 1500） | 6 | pass・6/6 検出 |
| S-cost | seed 40–42（登録外）で 3 プロセス同時・腕 3 本（N2r・S2dyn_10r・S2dyn_m50）、exit 0・MemAvailable の 80% で 3 スロット以上 | — | 未実行 |

## 7. 事前予測

### 7.1 Claude（2026-09-18 未明・seed 30–39 のどの腕も回す前・seed 44 の E は読んでいない）

| 項目 | 予測 |
|---|---|
| (A)・(B) | 満たす 95%・90% |
| (C) add_h・add | 満たす 95%・95% |
| (C) drop・drop_h（課題全体） | 満たさない（補償）55%・満たす 55% |
| (C) drop_e・drop_h_e（前半） | 満たす 90% |
| **primary** | **ADD_H_MOVES_E 70%** / NO_EFFECT 20% / REVERSED 3% / その他 7% |
| $\Delta R_{\rm add\_h}$ | +0.05（0 〜 +0.12）。比例の予測は +0.10 前後（κ ≈ 2.5、$R_{\rm c12}$ ≈ 0.07） |
| prop_add_h | SUBPROPORTIONAL 45% / PROPORTIONAL 30% / UNRESOLVED 20% / SUPRA 5% |
| add | ADD_MOVES_E 80% |
| drop_early・drop_h_early | MOVES_E 45%・40%（前半の窓は短く、E の差が小さい） |
| compensation・compensation_h | COMPENSATES 85%・70% |
| structure | STRUCTURE_MATTERS 35% / STRUCT_EQUAL 50% / STEP_WORSE 15% |
| felu | FELU_SAME 75% / FELU_HIGHER 20% |
| ladder | NEFF_TRACKS 55%（m25 で $n$ と E が逆に並ぶ可能性） |
| growth | GROWTH_REMAINDER 90%（escneff の再現） |

### 7.2 Codex（任意・§11 の手順 6 の前に記入）

### 7.3 Issa（本走の前に、§11 の手順 6 で選択式に記入）

## 8. 費用と実行

- 1 seed = prefix（約 1 分）+ 動く場の 9 腕（1 腕 2 タスクで 1〜2.5 分。混み具合による）+ 3 腕（約 20 秒ずつ）で 12〜25 分。ピーク RSS 約 1.0 GiB。最大 3 並列で約 1 時間（S-cost で見積もる）。
- 起動は、登録 commit を runner の `PREREG_COMMIT` に書き込み、全検査を 1 回通して all_pass になり、Issa の予測を記録してから。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- `results/neffdir_ee_0918/runs/s<seed>/`（seed 30–39。`prefix.csv`・`arms.csv`・`traj.csv`・`provenance.json` は git、`units.npz` は退避）。判定: `analysis/neffdir_ee_0918/verdict.py` → `results/neffdir_ee_0918/{summary.md,verdict.csv,paired.csv,provenance.json}`。
- `units.npz`・`results/_checks_neffdir_ee_0918/`（較正 `calib_s44/` を含む）は `~/Projects/obsidian-research-data/neffdir_ee_0918/` へ退避し、`results/neffdir_ee_0918/backup_manifest.json` を commit。main へ統合し、worktree と branch を消す。

## 10. 実行記録（本走後に追記。計画は変えない）

（未実行）

## 11. 引き継ぎ（Codex へ・2026-09-18）

Issa の指示で、登録から後を Codex が行う。**この branch と worktree は Claude が作ったが、引き継ぎにより Codex が使い、§4 の片付け（worktree の削除・branch の削除）まで行ってよい。**

### 11.1 引き継ぎ時点の状態

- worktree `~/Projects/claude/wt/neffdir_ee_0918`（branch `claude/neffdir_ee_0918`、origin/main `961f530` から）。`data` は `~/Projects/claude/proj_004_drift/data` への symlink（`.git/info/exclude` 済み・commit しない）。
- commit 済み（WIP）: `src/neffdir_ee_0918.py`・`analysis/neffdir_ee_0918/{checks.py,verdict.py}`・本 spec。
- git 外: `results/_checks_neffdir_ee_0918/`（開発実行の出力 `checks_partial.json`・`s9_cli/`、較正 `calib_s44/`）。
- 検査: S1c・S8・S9 は pass（§6）。**S1a・S1b・S-cost は未実行**。`launch.sh` は未作成。
- マシン: 共有のデスクトップ（white-san・28 コア・30 GiB）。他のセッションの走（lc_complement・lc_elu_lr・swish_battle など）が動いている。

### 11.2 手順

1. **読む**: 本 spec、`src/neffdir_ee_0918.py`、`analysis/neffdir_ee_0918/{verdict.py,checks.py}`。仕組みは import 先の `src/escape_ee_0917.py`（Hold・prefix）、`src/swap_ee_0917.py`（CapStepper・CapDynField・build）、`src/respdyn_ee_0917.py`（Stepper・DynField・probe）、`src/resp_ee_0917.py`（run_prefix・固定補正つきの前向き・unit_point・preflight）。前の走の文書は `specs/spec_escneff_ee_0917.md`・`specs/spec_escape_ee_0917.md`・`results/escneff_ee_0917/summary.md`。規則は `~/Projects/claude/CLAUDE.md`。
2. **遅い検査を開発実行する**: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/neffdir_ee_0918/checks.py --only S1a,S1b`（出力は `results/_checks_neffdir_ee_0918/checks_partial.json`）。落ちたら runner・verdict・checks を直す。**要求を緩めて通さない。** 要求が間違っていたら導出を書いて直し、§6 に記録する。変異はすべて検出されなければならない。runner や verdict を直したら、S1c・S8・S9 も通し直す。
3. **登録**: 本 spec の状態行を「**事前登録**（本ファイルの commit が登録。seed 30–39 では、どの腕もまだ走っていない）」に替え、§6 の開発実行の列を更新し、spec・src・analysis を commit して push する。この commit が登録。その hash を `src/neffdir_ee_0918.py` の `PREREG_COMMIT` に書き、commit・push する。
4. **全検査**: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/neffdir_ee_0918/checks.py`（S-cost を含む・約 45 分）→ `results/neffdir_ee_0918/checks.json` が `all_pass: true`。S-cost が落ちたら（空きメモリ不足など）、他の走が減るのを待つ。
5. **launch.sh**: `analysis/escneff_ee_0917/launch.sh` を写して `analysis/neffdir_ee_0918/launch.sh` を作る（OUT・MOD・seed 30–39・`MAX_SLOTS` の既定 3・checks.json の all_pass・`PREREG_COMMIT`・未 commit なし・push 済みの確認・flock・デスクトップ用に 6 GiB を残すメモリの門・STOP ファイル・終わった seed は飛ばす）。commit・push する。
6. **予測**: 本走の前に、Issa に選択式で聞く（少なくとも primary・prop_add_h・compensation・felu）。§7.3 に日付つきで記録する（Issa の確率は付けない）。Codex の予測を書くなら §7.2 に。commit・push する。
7. **本走**: `setsid nohup bash analysis/neffdir_ee_0918/launch.sh > results/neffdir_ee_0918/logs/launch.log 2>&1 &`。走っている間は `src/`・`analysis/` を編集しない（provenance が dirty になる）。どうしても必要なら、先に `results/neffdir_ee_0918/STOP` を置いて新しい起動を止める。
8. **判定**: 10 seed が終わったら `python3 analysis/neffdir_ee_0918/verdict.py`。verdict.py を使わず、`arms.csv` から $\Delta R_{\rm add\_h}$・$\varphi_{\rm add\_h}$・$\Delta R_{\rm drop}$・comp を独立に計算し直して一致を確かめる。§10 に実行記録（日時・commit・並列数・seed ごとの時間・記録の照合・判定・独立検算）を追記する。**判定スクリプトは本走後に変更しない**（変更が要るなら事後として別に書く）。結果（per-seed の csv と provenance、summary.md・verdict.csv・paired.csv・provenance.json・checks.json・logs）を commit・push する。
9. **退避**（CLAUDE.md §4-1）: `git status --porcelain --ignored --untracked-files=all` で git 外のファイル（`units.npz`・`results/_checks_neffdir_ee_0918/` など。`__pycache__` と `data` の symlink は除く）を `~/Projects/obsidian-research-data/neffdir_ee_0918/` へ同じ相対パスで移し、`results/neffdir_ee_0918/backup_manifest.json`（source・backup・bytes・sha256）を commit・push する。
10. **main へ統合して片付け**（§4-2・4-3）: `git fetch origin && git merge origin/main && git push origin HEAD:main` → `git -C ~/Projects/claude/proj_004_drift merge-base --is-ancestor claude/neffdir_ee_0918 origin/main && echo OK` → `worktree remove` → `branch -D` → `push origin --delete claude/neffdir_ee_0918`。push 済みの commit は書き換えない。git stash は使わない（共有）。
11. **vault**（`~/Projects/obsidian-research`・共有）: 結果ノートを `可塑性喪失/測定/` に作り（名前の例: `担い手を直接動かす_n_effの因果の向き_結果_0918`）、`可塑性喪失/ハブ/走の履歴`・`可塑性喪失/現在地`（直近の走の表）・`可塑性喪失/主張/中心主張v10作業リスト_0917`（H7 の行）・escneff の結果ノート（`逃げを塞ぐ移植を未使用seedでn_effを操作の確認にして回す_結果_0917`）に 1 行ずつリンクを足す。編集前に `git log` で並走を確かめ、自分のファイルだけを commit する。予測の当否（Claude・Codex・Issa）を結果ノートに書く。

### 11.3 注意

- **E を予測の前に見ない。** 検査は E を出力しない。較正の `calib_s44/arms.csv` には E の列があるが、Claude は読んでいない。予測の記録（手順 6）より前に開かないこと。
- `torch.set_flush_denormal(True)` と 1 スレッドは runner の `main()` と checks が設定する。python は `/usr/bin/python3`（escneff と同じ）。
- 記録との照合（S1a）は、このマシンの float32 kernel に依存する（別のマシンでは合わないことがある）。
- 補償（§1.2）は 1 seed の観察。10 seed で落とす腕の (C) が満たされないのは、設計の失敗ではなく結果として扱う（§5.4）。
