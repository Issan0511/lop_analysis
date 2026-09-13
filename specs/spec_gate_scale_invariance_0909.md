# ゲートの尺度不変性と幅成長の指数 spec 0909
状態: 検証中 / 更新: 2026-09-09
親: [[W増大メカニズム_0909]] §D-3（$h$ から指数への定量対応は未導出）
run id: `gate_scale_invariance_0909`

## 0. 問い
`elu_growth_0909` で第1層中心化重みの後半指数が $|h|$ の順に並んだ（LR 0.47 > ELU03 0.44 > ELU1 0.43 > SNA 0.38、3 seed 重なりなし）。事後の追加測定（seed0・4 腕）で、**指数と完全に同順に並ぶのは「注入が $N$ について伸びる速さ」$\beta$**（$D^2\propto N^\beta$）であり、タスク内の方向持続因子ではないことが分かった:

| 腕 | $\beta$ | 持続因子 | $\mathbb E|z\varphi''|$ | 指数 |
|---|---:|---:|---:|---:|
| LR | 0.542 | 52.5 | 0.000 | 0.47 |
| ELU03 | 0.453 | 46.7 | 0.025 | 0.44 |
| ELU1 | 0.375 | 51.8 | 0.077 | 0.43 |
| SNA | 0.236 | 30.4 | 0.769 | 0.38 |

持続因子は LR 52.5 対 ELU1 51.8 でほぼ同じなのに指数が違うので、**持続では説明できない**（この案は棄却済み）。

**作業仮説**: 正斉次性を微分すると $\varphi'(\kappa z)=\varphi'(z)$、すなわち $h\equiv0\iff\varphi''\equiv0\iff$**ゲートが $z$ のスケールに不変**。ゲートが尺度不変なら、重みが太っても勾配の構造が相似に拡大するので Adam 正規化後の注入が追随でき（$\beta$ 大）、侵食（$\propto N$）に負けにくく指数が 1/2 に近い。尺度不変性が壊れると注入が置いていかれ、劣拡散になる。

**問い**: 指数を決めているのは**ゲートの尺度不変性の破れ**か、それとも活性化ごとの別の性質か。

## 1. 腕と学習
`elu_growth_0909` と同一の枠組み: Permuted MNIST 784→100→100→10、CE、batch16、625 更新/タスク × **120 タスク**、Adam lr=.001 β=(.9,.999) eps=1e−8、正則化なし。`H.init_params`/`H.stream` は腕を取らない（S-init）ので**全腕が同一の初期値・置換列・層化抽出・バッチ順を共有する**。seed 0..2。

**群1 — 尺度不変（$\varphi''\equiv0$、$h\equiv0$）**: `R`（ReLU）、`LR03`（leaky a=0.3）、`LR001`（leaky a=0.01）。既存 `LR`（a=0.1）を参照として再走する。
**群2 — 曲率を単調に振る（固定 $\alpha$ Snake、$\varphi''=2\alpha\cos(2\alpha z)$）**: `SN02`（α=0.2）、`SN06`（α=0.6）、`SN15`（α=1.5）。
**再走参照**: `LR`・`ELU1`（α=1）。`ELU1` は群1・群2 の外側の対照。

計 9 腕 × 3 seed = 27 走。CPU 1 腕seed 約 60 秒（持続因子の追跡込み）、最大 2 process。
実装は新モジュールに duck-typed で置き、`pmnist_boundary_host_0908.py` は編集しない（既存 provenance の host sha256 を固定するため）。ReLU と固定 α Snake は既存 `H.ARMS` にあるのでそれを使う（`R`、`SN02`、`SN06`）。`SN15`・`LR03`・`LR001` は `H.Activation` を直接構成する。

## 2. 測定
各タスク終端で `params` から厳密に: ユニット別 $\lVert\widetilde W_i\rVert$、行平均、bias。10 タスクごとに追加で: タスク内 625 歩の $\sum\lVert\text{step}\rVert^2$（持続因子用）、固定 probe（seed ごと test512・現タスク perm）上の $\mathbb E|z\varphi''(z)|$・$\mathbb E|\varphi''|$・$\varphi'$ の CV・$\sigma_z$・$\bar z$、および固定訓練 4096 標本での $\langle\nabla_{W_1}L,W_1\rangle$・$\langle\nabla_{W_2}L,W_2\rangle$・$E$・$\sum(\partial L/\partial a_1)h(z)$。タスク終端の test 精度。
毎タスク: $\Delta\lVert\widetilde W\rVert^2$ の注入 $\lVert\Delta\widetilde W\rVert^2$ と侵食 $2\langle\widetilde W,\Delta\widetilde W\rangle$ への分解、$\cos$。

## 3. 判定（事前登録）
seed ごとに算出、**3/3 一致のときだけラベル**。CI・有意差は出さない。窓は固定（$\beta$ と指数はいずれも t21–40 と t101–120 の 2 窓、または t60–120 の回帰）。
- $\beta$: $\ln D^2$ を $\ln N$ に回帰（2 窓の平均値を結ぶ 2 点。`elu_growth_0909` の事後測定と同じ定義）
- 指数 $p_{\rm late}$: $\ln\lVert\widetilde W\rVert$ を $\ln t$ に回帰（t60–120）。既測: LR 0.465/0.471/0.476、ELU1 0.437/0.430/0.419、SNA 0.372/0.379/0.386
- 曲率 $\kappa_g:=\mathbb E|z\varphi''(z)|$（t101–120 平均）

**V1 群1（尺度不変腕）が leaky と同じ位置に来るか**
`SCALE_INVARIANCE_DECIDES`: `R`・`LR03`・`LR001` の 3 腕とも $p_{\rm late}$ が **[0.44, 0.52]**（LR の帯）かつ $\beta$ が **[0.45, 0.65]**、を 3 seed とも満たす。
`SCALE_INVARIANCE_INSUFFICIENT`: いずれかの腕が 3 seed とも $p_{\rm late}<0.42$ または $\beta<0.40$。
それ以外 `V1_PARTIAL`。

**V2 群2（曲率ダイヤル）で単調か**
`CURVATURE_MONOTONE`: $\kappa_g$ の順（SN02 < SN06 < SN15 を実測で確認したうえで）に対し $\beta$ と $p_{\rm late}$ がともに**狭義単調減少**、を 3 seed とも満たす。
`CURVATURE_NONMONOTONE`: いずれかの seed で順序が崩れる → `V2_PARTIAL`。$\kappa_g$ 自体が単調にならなければ `V2_VOID`（ダイヤルが効いていない）とし、$\alpha$ と $\kappa_g$ の実測を報告する。

**V3 $\beta$ と指数の対応（全 9 腕）**
`BETA_PREDICTS_EXPONENT`: 9 腕の seed 中央値で $\beta$ と $p_{\rm late}$ の Spearman 順位相関が 3 seed とも $\ge0.85$。
`BETA_UNRELATED`: 3 seed とも $\le0.4$。それ以外 `V3_PARTIAL`。

**V4 持続因子が説明しないことの再確認**
`PERSISTENCE_NOT_THE_MECHANISM`: 持続因子と $p_{\rm late}$ の Spearman が 3 seed とも $\beta$ とのそれより **0.2 以上小さい**。

**進捗ゲート G1**: 再走した `LR` が既存 committed checkpoint 42 点（`boundary_groups_0908` の task1・task100、`boundary_tasks20_40_0909` の task21–40 終端、`boundary_groups_0908` の task101–120 終端）を **params maxabs 0** で再現。かつ群1 の 3 腕すべてで $|E|/$スケール < 1e−6（$h\equiv0$ の帰結）。**1 つでも外れたら全腕を無効として停止。**

## 4. 事前予測（記名・参考値）
- **Claude**: `SCALE_INVARIANCE_DECIDES`・`CURVATURE_MONOTONE`・`BETA_PREDICTS_EXPONENT`・`PERSISTENCE_NOT_THE_MECHANISM`。とくに **leaky a を 0.01〜0.3 で 30 倍振っても指数は 0.46–0.48 から動かない**と予測する（ゲートの値は変わるが尺度不変性は保たれるため）。$\kappa_g$ は SN02 < SN06 < SN15 の順に増え、$\beta$ は SN02 > SN06 > SN15。
- **反証の形**: `LR001`（a=0.01、ゲートが極端に非対称）の指数が LR から外れれば、効いているのは尺度不変性ではなくゲートの値そのもの。SN の 3 腕で $\beta$ が単調にならなければ、曲率は指数を決める量ではない。
- Claude の的中は通算 2.5/10、Issa は 9.5/10。**Issa の予測が入れば併記して両論で採点する。**
- なお $\beta=0.54$（leaky の実測）の**値そのもの**は本走では導出しない。Adam が完全正規化なら 0、完全相似拡大なら 2 で、中間である理由は未解明のまま残る。

## 5. 検算と変異対照
すべての検査に変異対照を付け、対照が実際に検査を落とすことをアサートする。
1. **G1 の厳密再現**（§3）。対照: 初期値 W1[0,0] に +1e−3 で maxabs > 1e−4。
2. **恒等式 $E=\sum(\partial L/\partial a_1)h(z)$**（全腕。分母は $\lVert\nabla W_1\rVert\lVert W_1\rVert+\lVert\nabla b_1\rVert\lVert b_1\rVert+\lVert\nabla W_2\rVert\lVert W_2\rVert$、許容 1e−6）。対照: 別の活性化の $h$ で同じ分母 > 1e−4。
3. **群1 で $h\equiv0$**（$\varphi''\equiv0$ の帰結、同じ分母で <1e−6）。対照: 検査 2 の対照。
4. **$\varphi'$・$\varphi''$ が autograd と一致**（各腕・1e−6）。対照: 活性化パラメータを 1.01 倍した式で > 1e−3。
5. **$\Delta\lVert\widetilde W\rVert^2$ の分解恒等式**（1e−10）。対照: 符号を反転した式。
6. **測定が学習を変えない**（測定込みと測定無しで task120 の params が差 0）。対照: 測定側で +1e−9。
失敗は隠さず停止する。

## 6. 集計・保存・運用
seed ごとに 120 点の時系列、3 seed 並記。median(A)+median(B)≠median(A+B) なので分解恒等式は seed 単位で検算する。
CPU 1 thread・最大 2 process。1 腕seed 300 秒上限。
spec とコードを実行前に commit/push。CSV・verdict・summary・図を commit、ユニット別 npz は SHA 照合付きで `obsidian-research-data/gate_scale_invariance_0909` へ。
**限界**: $\beta$ は 2 窓 2 点の回帰で、曲線形状を仮定している。120 タスクは推定天井（leaky $N^*\approx13$）より手前で、天井の有無は本走では決まらない。$\kappa_g$ は活性化の関数形と $z$ 分布の積なので、腕間で $z$ 分布が違えば同じ $\alpha$ でも $\kappa_g$ は違う——**ダイヤルは $\alpha$ ではなく実測 $\kappa_g$ で並べる**。精度・LoP への接続は本走では扱わない。元 GPU 軌道ではなく CPU 学習。
