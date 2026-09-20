# initgeom_cifar_0920 — 入力平均の比から初期片側率を予測する（A5）

状態: **起案・推奨設計とCodex予測の記録まで。未実装・未計時・未実行。Issaの設計採用・予測・GO待ち。訓練は行わない。**

2026-09-20 / Codex / 起点main `989a24b`。run `initgeom_cifar_0920`、branch `codex/initgeom_cifar_0920`、worktree `wt/initgeom_cifar_0920`。親: vault `背骨CIFAR_S4S5_A3-A6_spec起案プロンプト_0920` A5、`中心主張v11作業リスト_0920` A5、`中心主張v11草案_0920` §1。宿主: [CIFAR battle spec](spec_rlcifar_mlp_battle_0918.md)、`src/pmnist_0905.py:init_params`、`src/pmnist_rlcifar_0907.py`。

## 0. 一行

**未使用のCIFAR初期状態20seedで、r=||mu||/sqrt(tr Sigma)から片側unit率を予測する登録近似式が、有限標本の予測帯と両立するかを測る。** 初期配置の予測であり、訓練後のLoPや初回学習性能を予言したとはしない。

## 1. 既知情報と起案文の訂正

既知seed0–9の事後値として、親ノートはraw片側率25.3%±2.5%、近似式22.3%、stdほぼ0を報告する。これらを見て式と条件を選んだことを開示する。この走では新規seed20–39のW、前活性、f、r、r2はまだ読んでいない。

起案時、repoのCIFAR/relu_doors関連provenance/config/input_manifest/checksのseedメタデータ、およびspec/src/analysisの明示的な20–39表記を検索し、対象seedの既使用は見つからなかった。全世界で未使用の保証ではなく、ローカル記録内の確認。実装時にも入力/初期化streamの既使用をmetadataだけで再確認し、重複が見つかったら結果を見る前に別seedを追補登録する。黙って置き換えない。

起案文を次のように修正する。

1. 2.33はPhi^-1(.99)の丸め。主式は **q=Phi^-1(.99)=2.3263478740408408** を用い、丸めた2.33版はREPORT_ONLY。結果に合わせてqをfitしない。
2. 宿主のWもbiasも一様初期化で、biasは0ではない。rだけの式はbiasを無視する近似。**主はbiasを含む本来の網**、同じWでbiasだけ0とした評価と、bias分散を入れた近似を副に保存する。
3. CIFARの画像方向共分散は等方でなく、投影分布も正規とは限らない。式を恒等式・分布によらない定理として扱わない。
4. 20seedの**中央値**の帯は単一seedの二項SDを20で割って作れない。二項モデルと順序統計量から予測帯を構成する。
5. 「平均を0にすれば片側率0」は検査の恒真ではない。中心化しても極端に歪んだ分布なら片側率は高くできる。Cのf=0は測定する予測であってassertにしない。
6. 画素を並べ替えるならWの対応する列も並べ替える。Xだけの画素置換で同じWのfが不変とは限らない。画像行の並べ替えはr/fを変えない。
7. `relu_doors`のCはsubsetの平均画像を引く操作。stdは固定channel別mean/sdで、Cと同じrとしない。

## 2. データ・初期化・数値の箱

seed20–39の20seed。各seedの `RC.subset_idx(seed)` によるCIFAR-10訓練1200枚、u8/255とチャンネル順は宿主どおり。ネットワーク3072–100–100–10、両隠れ層ReLU。`H.init_params(seed, cpu, DIMS)`を使い、W_lとb_lはそれぞれU(−1/sqrt(fan_in),+1/sqrt(fan_in))、W1のSDは1/sqrt(3*3072)。初期化role/seed/dtype/draw順は変更せず、条件間で同一の全Pを使う。学習・optimizer・新ラベル生成は不要。

**幾何の主測定はCPU float64**。宿主float32で生成された入力/W/bをコピーし、指定した入力変換と前向きをfloat64で再評価する。native float32学習の軌道再現ではなく、同じ初期パラメータの幾何を測る。丸め近傍の符号だけで判定が動く場合を誤差区間で開示する。CPU/BLAS2thread、1プロセス、seedを逐次処理。初期化分布と実現値の同一性は独立照合する。

第1層はz1=X W1^T+b1、第2層はa1=ReLU(z1)、z2=a1 W2^T+b2。層別に測定し、第2層で単一層の式を「同じだから正しい」としない。

## 3. 条件

raw Xは宿主のu8/255 float32をfloat64へ変換。stdは宿主の固定channel mean=(.4914,.4822,.4465)、sd=(.2470,.2435,.2616)によるfloat32変換をコピーする。mu_rawはrawの1200画像float64平均、X0=X−mu_raw。

| 条件 | 入力 |
|---|---|
| raw | X |
| std | 宿主standardize(X) |
| C | X0（float64中心化、丸め残差を記録） |
| gamma025 | X0+.25 mu_raw |
| gamma050 | X0+.50 mu_raw |
| gamma075 | X0+.75 mu_raw |
| gamma100 | rawへの明示alias |
| gamma150 | X0+1.50 mu_raw |
| gamma200 | X0+2.00 mu_raw |

9表示行、raw/gamma100を重複計上しない**8独立な条件定義**。統計的に条件同士が独立とは仮定しない。同じseed・W・画像を対応させる。ダイヤルは共分散を保ち、実数ではr_gamma=gamma*r_raw。丸め残差は共分散/中心化の検査で上界を確認する。rawをX0+muの再加算値へ置き換えない。

追加の訓練1task、別活性化、seed選別、gamma探索は行わない。

## 4. 登録読み出し・近似の意味

層への入力をH（第1層X、第2層a1）とし、N=1200、d=3072または100。

```
mu = mean_images H
trSigma = mean_images ||H-mu||^2       # Nで割る母分散。N-1ではない
r = ||mu|| / sqrt(trSigma)
p(r) = 2*Phi(-q/r) = erfc(q/(sqrt(2)*r))
```

r=0ならp=0。trSigma=0かつ||mu||>0ならr=inf,p=1としてDEGENERATE_INPUTを併記。両方0はUNDEFINED_INPUTであり、seedを除いて標本数を減らさずその条件の科学ラベルをUNDEFINEDとする。

各unitは正・負・厳密0の画像数を整数で数える。起案文のmin(P(z>0),P(z<0))<.01は、0が多いと全0unitも片側に数えてしまう。主定義は **N_positive>0.99NまたはN_negative>0.99N**、すなわち1200枚中1189枚以上が同じ厳密符号。ゼロが無い場合は原定義と同値。各層f_s=片側unit数/100。全0unit数、正/負の別、旧min定義もREPORT_ONLY。

zの符号は独立float64内積の演算上界を使い、誤差区間が0を跨ぐ画素は符号不確実として記録。確実な同符号数からf_lower、可能な同符号数からf_upperを作り、科学ラベルはこの区間を使う。曖昧なunitを分母100から除外しない。表示用の実計算f、z平均/SD、正負/0率も残す。

近似の導出: もし各unitの画像方向zが正規で、投影分散w^T Sigma wをsigma_w^2 trSigmaで置き換えられ、biasを無視できれば、unit間のm=w^T muをN(0,sigma_w^2||mu||^2)と近似してP(|m|/sd_z>q)=2Phi(−q/r)。実際には一様W、非等方Sigma、有限画像、ランダムbias、投影分散の変動がある。これらを検証前に隠さない。

各seedの予測p_sは自身の入力統計から作る。第1層は入力統計→prediction manifestを先に保存してからWの応答fを計算する。第2層は測ったa1からr2を作る**条件付き予測**で、a1やr2自体を事前に予言したとはしない。

副診断:

- 同じ全Wで各層biasを0とした前向き（第2層ではb1も0とした別a1でr2を再計算）。主を置き換えない。
- bias分散を含めた近似 `r_bias=sqrt(||mu||^2+1)/sqrt(trSigma)`。宿主ではVar(b)=Var(w_j)なので+1が出る。Gaussian近似のままであり、結果に合わせた補正ではない。
- unit別の実際のm=w^Tmu+b、sd=sqrt(w^T Sigma w)から作るGaussian proxy `1[abs(m)>q*sd]` とf。これは測ったWと分布幅を使う記述で、rだけの予測とは区別。
- 入力の||mu||、rms、trSigma、r、bias/投影SD、unit別m/sd、0分母率。共分散の非等方性はeffective rank `(trSigma)^2/tr(Sigma^2)` で報告し、予測式のfitには使わない。

## 5. 登録判定

### 5.1 中央値の予測帯

主の第1層は8条件を1 familyとしてalpha_c=.05/8（Bonferroni）。各条件のp_sを固定し、帰無モデルK_s~Binomial(100,p_s)、seed間独立。近似式が正しければunitは条件付きでこの確率を持つ、という**モデルの予測帯**であり、画像1200枚を独立Bernoulliとは置かない。観測の対象は20seedの中央2順序統計量の平均、M=(K_(10)+K_(11))/200。

二項CDF F_s(k)から、N_k=sum_s 1[K_s<=k]のPoisson-binomial分布をDPで計算する。`P(K_(j)<=k)=P(N_k>=j)`、j=10,11。G_jの分位をq_j(a)=min{k:G_j(k)>=a}とする。

```
L = (q_10(alpha_c/4)+q_11(alpha_c/4))/200
U = (q_10(1-alpha_c/4)+q_11(1-alpha_c/4))/200
```

各順序統計量の左右の外れ確率をalpha_c/4で抑え、union boundで中央値のcoverage>=1−alpha_c。離散性で保守的な帯になる。SD=0でも正しく点区間になる。単純な正規SEやbootstrapへの事後切替なし。pが極端な場合はlog-domain/厳密なp=0,1の分岐で安定に計算する。

同様に観測のf_lower/f_upperのseed中央値からM_lower/M_upperを作る。

| 観測の位置 | 条件別ラベル |
|---|---|
| [M_lower,M_upper]が[L,U]に含まれる（等号を含む） | PREDICTED |
| M_lower>U | OFF_HIGH |
| M_upper<L | OFF_LOW |
| 数値区間が帯内外をまたぐ | NUMERIC_UNRESOLVED |
| 入力未定義・欠測/非有限/検査不合格 | UNDEFINED / INCOMPLETE / DIVERGED / CHECK_FAILED |

8条件全てPREDICTEDならL1_ALL_COMPATIBLE、いずれかOFFならL1_MODEL_MISS（外れた条件/向きを全て示す）、それ以外L1_UNRESOLVED。**PREDICTEDはこの予測帯と両立した意味で、近似式の証明や同等性の検定ではない。** 同じraw/gamma100を二重の成功に数えない。

第2層はa1に条件付けた同じ8条件の副family、同じalpha_cで別表に出す。L2_CONDITIONAL_ALL_COMPATIBLE / L2_CONDITIONAL_MODEL_MISS / L2_UNRESOLVED。第1層との総合的な「16条件95%同時保証」は称さない。bias0・r_bias・Gaussian proxyはREPORT_ONLYで、主の外れを置換しない。

### 5.2 ダイヤル（登録副）

gamma=.25,.5,.75,1,1.5,2の第1層seed中央値を比較。全ての隣接差が非負ならMONOTONE_SAMPLE_MEDIANS、負の差があればNONMONOTONE_SAMPLE_MEDIANS、符号不確実なunitで成否が変わるならNUMERIC_UNRESOLVED。これは標本の順序の記述で、無検定の母集団単調性主張ではない。

全5隣接対のseed内f差と平均を示し、Student-t df19、各99%両側区間（5対Bonferroni）を副に併記する。差のSD=0は点区間、端点0は非検出。unitを独立seedとして水増ししない。第2層の順序もREPORT_ONLY。厳密な単調性を実装検査にして走を落とさない。

## 6. 実装前予測・裁定

Codex、2026-09-20（このspecのcommit時刻が正本）、新規seedの入力統計・W・f・r2、実装/検査/計時を見る前。§5の明確化した判定に対する予測:

| 命題 | 確率 |
|---|---:|
| rawの第1層がPREDICTED | .75 |
| stdとCの第1層がともにPREDICTED | .90 |
| 第1層8条件がL1_ALL_COMPATIBLE | .55 |
| 第1層ダイヤルMONOTONE_SAMPLE_MEDIANS | .90 |
| 第2層8条件がL2_CONDITIONAL_ALL_COMPATIBLE | .35 |

rawの既知の小さなずれ、非等方性、一様初期化・biasの無視から、全条件の定量的一致は単調性より不確かと予想する。全命題をbinary Brierで採点し、未定義/失敗/数値未解決は理由付き未採点。重なりのある命題を独立な証拠数とはしない。起案セッションのClaude予測（raw/std/C .8、単調 .9、第2層 .5）は設計が異なる参考値で、本採点へ混ぜない。

**Issaの予測: 未記入。** 内容への同意でもよい。確率はCodex本人の値として保持し、本人の確率に転記しない。

| 裁定点 | 推奨 |
|---|---|
| 主対象 | 宿主のランダムbiasを含む実初期状態 |
| biasを省いた式の限界 | 主でそのまま検証、bias0/r_biasを副に開示 |
| 定数・片側 | 正確なq99、1189/1200以上が同じ厳密符号 |
| 評価精度 | 初期float32状態をfloat64で幾何評価 |
| 判定 | 順序統計量の保守的予測帯、L1と条件付きL2を分離 |
| 任意の訓練追加 | 行わない |

## 7. 必須検査（全て未実施）

| ID | 独立の参照 | 検出する変異 |
|---|---|---|
| A5-source-seed | 宿主のdata/subset/init stream・全P hash、条件間共通 | seedずれ、再初期化、biasを0、Gaussian初期化 |
| A5-r | math.fsumの独立二段中心化・trSigma・r、定数/零入力 | N-1、||mu||二乗、rと1/r取り違え |
| A5-dial | 対応画像の中心化差/共分散とgamma*rの誤差上界、raw alias一致 | std混同、X全体をgamma倍、gamma100再計算/二重計上 |
| A5-sign | 整数カウント1188/1189、正負/厳密0、全0、符号区間を合成例で確認 | <=閾値、0を正へ、曖昧unit除外 |
| A5-normal | 解析的な正規母集団のm/sd→片側条件、独立erfc/分位計算 | q=1.96、tail係数2欠落、r_biasを主へ混入 |
| A5-forward | H.forwardによる同じP/Xのdouble forwardと独立unit内積、native32との差上界 | 第2層でXを使用、b1/b2取り違え、r2をr1で代用 |
| A5-permute | 行置換、X列とW列の同時置換、不正なXだけの置換を別扱い | Xだけで不変をassert、labelや列の対応崩し |
| A5-C-counterexample | 平均0だが1199正/1負の決定的入力で片側になる例 | 中心化f=0をhardcode |
| A5-band | 小標本・少数unitの全列挙とDP/CDF/順序分位を照合、p=0/1、等号 | 平均SEを中央値に使用、raw重複、family補正欠落 |
| A5-verdict | 全ラベル/数値跨ぎ/欠測/非有限/単調tie・逆転を独立fixtureで検査 | 外れ条件隠し、20未満seedで成功、単調性を必須検査へ転用 |
| A5-resume-manifest | seed境界の中断再開、明示8条件×20seed×2層、source/input一致 | 重複/別run混入、partial report、先行出力上書き |
| A5-cost | 検査seed100–101で全処理の時間・RSS・保存量 | 初期化だけの計時、全条件を同時RAM保持 |

全必須IDと変異一覧をcollectorで固定し、欠けたPASSを拒否。検査seedと本seedを混ぜない。算術許容は長さd/Nと実際の絶対和から導くgamma_n、sqrt/division/erfcの数値評価誤差を用い、観測fに合う固定幅を足さない。正規分布の有限乱数標本は必ず合うfixtureにせず、解析解/決定的fixtureの検査と分ける。確率的な検査失敗を引き直しで消さない。

## 8. 実行計画・費用・中断

採用記録→spec commit/push→実装・検査seed→実装commit/push→本seedの入力統計/prediction manifest→初期化/第1層応答→第2層の条件付き予測/応答→全件完了→report/予測採点。今回の納品はspecまで。

学習0、CPU1プロセス・2thread、20seed×8条件。1seedずつ処理して巨大な全seed画像配列を保持しない。所要時間は分単位を想定するが未計時。正確な費用はA5-cost後に記録。GPU不要。1task学習の任意追加は今回の範囲外。

`results/initgeom_cifar_0920/STOP`はseed境界でatomic保存。入力統計・全P・処理済み条件/seed・hashを持ち、同一source/inputで再開する。初期化順はseed別独立streamなので再開でdrawをずらさない。provenance_startに起動時git/dirty/source/input/env/UTC/JST/PID、endを別保存。ログ `initgeom_cifar_0920.log`。長い場合の状態確認は10分間隔、完了後に一度だけ科学reportを開く。

## 9. 開示と解釈

式は既知seedの事後観察で選んだ近似で、今回の結果から選んだものではない。未使用seedの入力統計に条件付ける第1層予測と、測定a1に条件付ける第2層予測を区別する。OFFは近似の誤差を示すが、「入力平均が無関係」を意味しない。PREDICTEDも平均だけで全配置や学習性能が決まる証明ではない。

片側は正/負の両向きで、ReLUの死だけではない。負側片側化と正側の線形化を別に示す。今回の対象は初期化のみ、誤差修正・Adamの成長・訓練後のLoPは測らない。独立監査なし。

## 10. 成果物・片付け

予定: `analysis/initgeom_cifar_0920/`、`results/initgeom_cifar_0920/`にsummary/verdict、prediction_manifest、per_seed/per_unit/input_stats、layer2_conditional、dial/bias_diagnostics、checks/predictions/provenance、図PNG/PDF。明示したseed/条件/層だけを集計する。

git外の入力抜粋・初期P・z配列・ログ・検査attemptは `/home/issan/Projects/obsidian-research-data/initgeom_cifar_0920/` へサイズ・SHA256を照合して退避し、backup_manifest.jsonをcommitする。共有CIFAR本体・symlink・既存resultsは移さない。CLAUDE.md §4どおりmainへmerge/push・到達確認・自分のworktree/branch削除。spec起案もmainへ統合して片付け、採用後に最新mainから同じrun名で作り直す。
