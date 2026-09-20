# cap_cifar_ee_0920 — CIFAR・ELU/std の重みの成長を層別に止める（S5）

状態: **設計・予測の登録案。Issa は「S4→S5」の順を採用（2026-09-20）。推奨設計の裁定・Issa の予測・実装 GO は未記入。実装・検査・本走は未実施。**

作成: 2026-09-20 / 起草: Codex / 起点: `origin/main=273a6bc`（S4のspec起案まで。S4の実験結果はまだ無い）。
run: `cap_cifar_ee_0920` / branch: `codex/cap_cifar_ee_0920` / worktree: `wt/cap_cifar_ee_0920`。
親: vault `背骨CIFAR_S4S5_A3-A6_spec起案プロンプト_0920` S5・A6後の追補、`中心主張v11作業リスト_0920` S5。
型: [l2cap_ee](spec_l2cap_ee_0917.md)、前段 [S4](spec_resp_cifar_ee_0920.md)、帳簿 [A6](spec_cifar_ledger_0920.md)。以下を推奨案として具体化し、採用後の変更は実装・結果を見る前に別commitで固定する。

## 0. 一行

**ELU/std の第1層・第2層の重みの行ノルムをtask1終端値以下に保つ介入で、第2層の応答と50タスク内の再学習能力を保てるかを問う。** 主比較はcap12−ref。cap1・cap2で層別の寄与を調べ、cap12_bfixで隠れ層biasの移動が残る場合を比較する。

## 1. 出所・既知情報と修正点

- vault のV10整理、V11草案／作業リスト、起案プロンプト、運用ルール・引用禁止を参照。collaboration notice/changesに未確認の先生側変更なし。共通前置きに従い今回はspecのcommitまで。
- [A6結果](../results/cifar_ledger_0920/summary.md)（完了 `d7e09b9`）ではELU/stdの先行低応答層は第2層10/10。主窓の上流16.5%・自己23.0%・交差約60%、R2=MIXED。上流項中のmu2の伸びは約90.7%。固定補助窓の上流はt00→t10約73%、t10→t50約99%。これらは既知軌道の再解析であり、単一の因果的な運び手を決める比ではない。
- **A6だけでcap1/cap2を落とさず、4介入を保つ。** t1以後の介入なので、t00→t1の形成原因をこの走で検定したとはしない。t1に既にある損傷を上限だけで元に戻す処理ではない。
- MNISTのl2capでcap12が救済された結果は既知。CIFARへの移行では、入力・活性化実装・更新予算・cap1の定義が異なる。旧結果は動機と予測に使い、効果量や操作検査の帯を転用しない。
- **stdはsubsetを厳密に零平均化しない。** 入力平均が小さいことを「mu1の方向が定義できない」とは書かない。今回は入力平均方向の分解を使わず、W1全行の上限を選ぶ。従ってMNISTのq/v別上限と同じ介入ではない。
- native `F.elu`・autogradを用いる。旧expm1の床や手書きdphiを訓練微分として流用しない。起案文の「E2=e^z」は正側の微分1と実装上の丸めも含めたnative g_trainで置き換える。
- 起案文のbfix「勾配0」は親l2capの定義と違う。本案は **Adam更新後にb1/b2をt1値へ射影し、momentは通常どおり更新**する親の方法を採る。値の固定と勾配0を混同しない。
- S4の実装・介入結果は本起案では見ていない。S4の結果で本specの腕・半径・窓・予測を選び直さない。

## 2. 箱・状態・上限

### 2.1 データと宿主

- [battle engine](../src/rlcifar_mlp_battle_0918.py) を基にしたRandom Label CIFAR-10。1200枚固定、3072–100–100–10、両隠れ層ELU(α=1)、biasあり。入力はstdのみ、u8/255後に宿主のchannel別mean/sdを適用する。
- Adam lr=1e-3、β=(0.9,0.999)、ε=1e-8、WDなし、batch16、400epoch×75=30,000更新/タスク、**50タスク固定**。毎タスク一様10クラスラベル、毎epochは1200枚の順列。momentと時刻はtask境界を越えて保持する。
- **seed0–9、1腕につきR=10、slotはseed昇順**。5腕を一つのstackに束ねない。seedの故障や未完了でRを縮めない。既知自然seedへの介入であり、未使用seedでの外部再現ではない。
- 同じCUDAマシン、float32、宿主の`baddbmm`・Adam演算順・graph・H.setupを継承。TF32 off、決定論設定・subnormal/flushを記録。CPU/BLAS2thread、集計と検査の独立参照はfloat64。
- 宿主のsubset/init/ラベル/順列streamをそのまま使う。全腕で入力、画像index、全タスクのラベルと順列のhashが一致する。診断と射影は学習RNGを進めない。検査/計時はseed100–109のR10と合成入力。

### 2.2 S5自身の接頭部とref

**S5で新しい無介入R10のtask1を走らせる。** 終端P/m/v/tc、各seedのgenerator state、入力・ラベル・順列hashを保存し、5腕を独立cloneしてtask2–50を継続する。task1行は共通の接頭部を参照する。

S4のt1/t10 checkpointや旧R20バトルをrefの代用にしない。S4と同じR10でも新規refを走らせ、同じ宿主との独立照合で資格を確認する。S4との自然行の一致は報告のみ。capの半径は **このS5のtask1** から取る。

### 2.3 行ノルム上限

各seed・各unitについて `r_l,i = norm_float32(W_l,i at end of task1)` を一度だけ計算し、値とhashを保存。task2の最初のAdam更新直後から、対象行のノルムがrを超える場合だけ

`w_i ← w_i * (r_i / norm(w_i))`

とする。方向を保つ球への射影で、上限以下の行はbitを変えず、小さい行を膨らませない。ノルムは行の全入力次元に対するL2ノルム。W1は3072次元、W2は100次元。ゼロ行・ゼロ半径は明示分岐し0除算しない。半径をtaskごと・seed全体・unit全体で平均し直さない。

順序は **宿主の全Adam更新→W1射影→W2射影→bfixの書き戻し**。WのAdam momentは射影しない。行ごとに発火回数、除去量、最大ノルム超過を記録する。W3/b3は全腕で自由。

半径と比・乗算の丸めを無視して「常に実数ノルム≤r」とassertしない。独立float64 normで検査する許容は演算誤差から導く。例えば長さdの非零行、u=2^-24、γ=γ_(2d+2)では、normの下方誤差を1−γ、比と乗算を各1+uで抑えた保守的上界

`norm64(w_after) ≤ r * (1+u)^2/(1−γ) + sqrt(d)*η32`

を使える（η32は実測flush仕様を反映した絶対丸め上界）。checker自身のfloat64誤差も足す。全更新について超過max/countをGPU側で蓄積してtask末に保存し、検査用短縮走では各更新を独立に確認する。実装がnormをfloat64で作る別案へ変わるなら、その演算と上界を本走前commitで固定する。

### 2.4 bias固定

cap12_bfixは各Adam更新の後に **b1/b2だけをtask1終端のbitへ書き戻す**。b3は自由。b1/b2の生勾配・m/vは通常どおり計算し、値だけを固定する。これらのmomentがWの更新へ流入しないことも検査する。grad=0・optimizerから除外する別実装には黙って変更しない。

値が同じtask1の前後に対する差は常に0だが、介入の発火は書き戻し直前の候補値とb_fixedの差から測る。固定後の差0だけで「bfixが働いた」とする空虚な検査をしない。

## 3. 腕（ref＋4介入、各10seed）

| 腕 | W1行ノルム上限 | W2行ノルム上限 | b1/b2 | 格 |
|---|---|---|---|---|
| ref | なし | なし | 自由 | 共通参照 |
| cap1 | t1値 | なし | 自由 | 層別の副比較 |
| cap2 | なし | t1値 | 自由 | 層別の副比較 |
| **cap12** | t1値 | t1値 | 自由 | **主比較** |
| cap12_bfix | t1値 | t1値 | t1値へ毎step書き戻す | 残るbias経路の副比較 |

refを含む2×2で交互作用を報告する。cap12_bfixはb1/b2を同時に固定するため、両者のどちらが原因かはこの比較だけでは決められない。raw、WD、半径探索、t1以前への遡及介入、20taskへの短縮は含めない。

## 4. 登録する読み出し

### 4.1 主窓・主量

主窓 **W=t31–50（20task）**。A_s,a(t)=online、G_s,a(t)=第2層のnative局所微分の全画像→全unit算術平均。

- `E1_s,a = mean_W A_s,a(t) − A_s(1)`。
- `E2_s,a = mean_W G_s,a(t) − G_s(1)`。
- 対応効果 `δ_j,s,a = Ej_s,a − Ej_s,ref`。初期値は共通だが、解析では共通hashを確認してから引く。主はcap12のE1/E2。

native微分は、その時点の元F.eluに対するautogradで測る。診断dphiも別列に残す。全画像B=1200の評価でGを定義し、訓練B=16との差は操作検査で示す。

### 4.2 全taskの診断と帳簿（主ラベルの代用にしない）

- online、memo、CE、最多クラス率F_s(t)、両層G・Q=`count(abs(g)<1e-6)/(1200*100)`・厳密0率・全画像厳密0のunit率。Qは整数pair数から算出。
- 両層unit別zの平均/SD、W行ノルム、bias、mu2とnorm、cos(w2,mu2)、第2層入力の平均/変動の比。ゼロ分母/方向未定義は別記して残す。
- [A6実装](../analysis/cifar_ledger_0920/ledger.py)の旧状態による4項分解 `Δm2 = ΔW2*mu_old + W2_old*Δmu2 + ΔW2*Δmu2 + Δb2`、上流と自己のnorm/方向対称分解をfloat64で計算。直接mean zとの差はnative演算の誤差として分ける。閉包誤差・上界を毎区間保存。
- 帳簿はt1→t10、t1→t50、t30→t50の **総和と1task当たり率** を全seedで報告。unit算術平均を主表示し中央値もREPORT_ONLY。寄与率は符号付きで、正味沈下が数値誤差から分離できない場合はundefined。A6のR2を新しい主判定へ流用しない。
- capの発火行数・除去量・norm/r、bfixの候補変位、b1/b2の軌跡。t1→t50のb2平均変化とmu2変化を別々に示す。
- t2–10、t11–30、t31–50のA/G、t1に対する初期低下、最後の10taskだけの値はREPORT_ONLY。主窓を結果に応じて替えない。

## 5. 登録する判定

### 5.1 完全性・操作・自然現象

1. 5腕×10seed×50taskが揃い、同一入力/ラベル/順列/接頭部で有限、全必須検査がPASSであること。欠落は `INCOMPLETE`、誤実装は `CHECK_FAILED`、非有限はseed/arm/taskとともに `DIVERGED`。欠落seedを除いてnを縮めず、該当比較の科学ラベルを出さない。他の出力は保存する。同一設定の正しい再開は可。
2. 対象ノルムが誤差上界を越えないこと、bfix値が固定されることは必須。**発火0は装置故障と同義にしない。** 各対象層でt2–50累積発火が0のseedは `CAP_NOT_ENGAGED` として数える。副腕の対応を保存し、その上限の作用が同定できないことを明記する。毎taskの発火>0を強制しない。
3. refの主窓で下記AT_FLOORが **9/10以上**、かつrefのE2の95%対応前後差区間の上端<0を自然現象の条件Bとする。満たさなければ全科学ラベル `NOT_REPRODUCED`。9は多数確率≤1/2に対する片側正確二項の名目5%で最小のk: P(Bin(10,1/2)≥8)=0.0546875、≥9=0.0107421875。seed独立性を仮定した操作的な再現条件で、母集団一般化の保証ではない。

### 5.2 偶然水準の操作的な帯

各seedのtaskごとの **最良の定数予測器** はF_s(t)=max_c(n_s,t,c/1200)。同じラベルなので全腕共通。Wの平均Fbar_sと標本SD sF_sから

`U_s = Fbar_s + t_(19,0.95) * sF_s/sqrt(20)`

を作り、`mean_W A_s,a ≤ U_s` を **AT_FLOOR** とする。sF=0ならU=Fbar、境界の等号は床。起案文の固定3倍を、登録した片側名目5%のt分位点に置き換えた。50→20taskに短縮してこの窓を変えない。

この帯はラベル構成のtask間変動を尺度にした **操作的な床**。学習中のonlineが二項独立であるとは仮定せず、30,000回の更新を独立標本数にもせず、モデルが定数予測器と統計的に同等だと証明する検定とはしない。各seedのU・A・差を全て残す。

腕の床状態は10/10床ならFLOORED、0/10床ならALIVE、残りはSPLIT。refの適用条件9/10と介入腕の全seed状態を混同しない。

### 5.3 対応差の区間と主ラベル

cap12のE1/E2でn=10、df=9、標本SD sを用い

`mean(δ) ± t_(9,1−0.05/(2×2))*s/sqrt(10)`

（2比較Bonferroni、各97.5%両側区間）。95%も併記。下端>0は+、上端<0は−、他は0。SD=0は同一点区間と `DEGENERATE_SD` を併記し、tで0除算しない。端点=0は非検出。cap1/cap2/cap12_bfixのE1/E2は95%区間による **副ラベル** とし、主の多重性制御の範囲に入れない。

| 腕の床状態 | E1符号 | E2符号 | ラベル |
|---|---|---|---|
| FLOORED | 任意 | 任意 | COLLAPSED |
| SPLIT | 任意 | 任意 | SPLIT |
| ALIVE | + | + | RESCUED |
| ALIVE | + | 0 / − | RESCUED_FUNCTION_ONLY |
| ALIVE | 0 / − | 任意 | ALIVE_UNRESOLVED |

主結論はcap12のこの表。非検出は同等性ではない。floor上に少し残るだけの腕を「能力を完全保持」とせず、Aの絶対水準とt1からの低下を併記する。`CAP_NOT_ENGAGED` がある場合は発火seed数と結果をセットで示す。

### 5.4 初期学習への悪影響フラグ

介入後最初のtask2で、各seedのref正解率R_sと定数床F_s(2)の中点 `H_s=(R_s+F_s(2))/2` を取る。これはrefと床への二乗距離が等しくなる点で、任意の正解率差を置いたものではない。

全seedでR_s>F_s(2)が成り立つ場合、介入A_s,a(2)<H_sのseedが **6/10以上（厳密多数）** なら `IMPAIRED_EARLY`。いずれかで参照の向きが成立しない場合は `EARLY_FLAG_NOT_ASSESSABLE` とし、正常扱いしない。等号はフラグなし。

フラグは主ラベルを置換せず併記する。IMPAIRED_EARLYなら、成長抑制が初期から学習を損なう費用を隠して「可塑性を保った」とは書かない。起案文の「IMPAIRED」を救済/崩壊と排他的なラベルにしない。

### 5.5 時間と交互作用（副）

`T_half`はt≥2で `(A(t)−0.1)/(A(1)−0.1)<1/2` となる最初のtask。0.1は一様10クラスの水準、1/2は初回学習の余剰の半減というendpointの定義である。A(1)≤0.1はundefined、50まで超過しなければ右打切り。

各腕とrefをseed内で比較する。片方のみ打切りならその腕は観測された他方より遅い。両方打切りは順序不明として符号検定に入れず、同じ観測時刻もtieとして除く。比較可能な符号に両側正確二項検定、p<0.05でLATER/EARLIER、他はNO_TIMING_DIFF。比較可能0件はTIMING_UNDETERMINED。これは副解析であり、主判定の欠測seed除外ではない。

E1/E2の交互作用はseed別 `δ_cap12−δ_cap1−δ_cap2`（=Δcap12−Δcap1−Δcap2+Δref）。95%対応区間でREPORT_ONLY。単独腕の非検出を「両方必要」と言い換えない。

### 5.6 読みの上限

- cap12 RESCUEDかつ初期悪影響なしなら、この半径・予算・50taskで成長への介入が機能と応答を保った例とする。輸送の変化は帳簿と別に示し、E1/E2だけから媒介は言わない。
- cap12_bfixがより良ければ、両隠れ層biasの固定を足す効果。b1/b2の片方、biasが唯一の原因、永久治癒までは示さない。
- cap12_bfixでも落ちれば、固定したノルム上限とbiasでは十分でない。Wの方向・特徴構造等を候補として列記できるが原因確定はしない。
- S4とS5がともに支持的でも、capが応答を介して救うことを同一網への交換で測ったわけではない。自然な全媒介率や全箱への一般化は未検証のまま残す。

## 6. 実装前の予測と裁定欄

### Codex（2026-09-20・S4/S5の新規実装・probe・結果より前）

適用条件B成立は0.90。B成立・検査完了を条件とした予測:

| 項目 | 予測内容 | 確率 |
|---|---|---:|
| 主cap12 | RESCUED | 0.65 |
| cap1副ラベル | COLLAPSED | 0.65 |
| cap2副ラベル | SPLITまたはCOLLAPSED | 0.75 |
| cap12_bfix副ラベル | RESCUED | 0.80 |
| cap1の時間 | LATER | 0.65 |
| cap12のb2 | mean_seed(mean_unit[b2(t50)−b2(t1)])<0 | 0.75 |
| cap12の初期悪影響 | IMPAIRED_EARLYではない（ASSESSABLEの場合） | 0.70 |

各命題を真偽とbinary Brierで評価。NOT_REPRODUCED/検査失敗/undefinedは除外せず理由付きの未採点として残す。cap1単独の悪化とcap12救済を見込む理由は既知MNISTの両経路とA6の交差項だが、同じ結果を保証しない。

Claudeの起案予測はcap12救済0.60、cap1崩壊かつ遅延0.60、cap2分裂または崩壊0.70、cap12でもb2沈下0.60。床・窓・bfixの仕様を明確化した本案とは同一の予測として数えない。

**Issaの予測**: ［未記入］。A6の採用をS5へ自動で移さない。内容への同意でも記入を満たせるが、確率をCodexからIssaへ転記しない。

| 設計の裁定 | 推奨案 | 採用記録 |
|---|---|---|
| cap1の定義 | W1全行ノルム、S5自身のt1値 | 未記入 |
| 期間と範囲 | 50task・主窓31–50・stdのみ | 未記入 |
| bfix | 更新後にb1/b2を書き戻し、momentは保持 | 未記入 |
| 床と完全性 | t分位の操作的帯、全10seedを固定 | 未記入 |
| 実装・本走GO | S4→S5の順、予測記録後 | 未記入 |

推奨案をまとめて採用する回答で足りる。S4の結果を受けて設計を変える場合は、現在の予測を残し「S4の結果を見た改訂」として実装前に新しい追補をcommitする。黙って現在の登録を上書きしない。

## 7. 検査（すべて未実施）

全必須ID/変異の期待リストを持つcollectorで、未実行・空・別sourceのPASSを受け入れない。

| 検査 | 独立の参照・要求 | 検出する変異 |
|---|---|---|
| S-host/off | 無改変宿主とrefがR10・2task×400epochsでP/m/v/tc・RNG・全登録行までbit一致 | lr/β補正時刻、norm経路残留、初期化/標準化変更 |
| S-radius | 保存t1のWから独立再計算した行別rが一致。上限以下/等しい/超過/ゼロ行で期待写像 | 初期値半径、行平均半径、層取り違え、小さい行も膨張 |
| S-project | GPU写像と独立float64射影の差が演算上界内、非発火行はbit不変、方向保持、全更新の上界監視 | 更新前射影、W3射影、発火しても書かない、momentも射影 |
| S-strength | 検査fixtureで×1000の半径が真に全step非発火ならrefとbit一致。×0.5は保存した試行更新が上限を越えるfixtureで射影参照と照合 | 尺度を無視、半径上書き。単なる「どこか違う」検査にしない |
| S-bfix | 各更新後b1/b2がt1値とbit一致、W3/b3自由。書き戻し前候補・生勾配・m/vを独立更新で照合 | b2だけ固定、b3も固定、初期biasへ戻す、勾配0にする |
| S-branch/RNG | 独立cloneからの全5腕のtask1状態hash、後続全ラベル/順列が一致。診断前後RNG不変 | state alias、腕順依存、refだけ別乱数 |
| S-derivative | 元F.eluのautogradと保存gが一致、深い負側/0/subnormalを含める | expm1/output+1、dphiを無検証で代入 |
| S-ledger | A6と同じ4項・norm/方向分解の独立float64恒等式、誤差伝播で閉包 | cross/bias落とし、W_newで評価、他seedのmu |
| S-graph | eager/graphが同R10で全状態bit一致、warmupが半径・発火カウンタ・bfix状態も戻す | counter/RNG/momentの巻戻し漏れ |
| S-resume/STOP | epoch境界・t1→t2・介入後からの再開が無中断と一致 | r/b_fixed欠落、m/v欠落、acc欠落、別commit入力 |
| S-verdict | 独立合成fixtureで全ラベル、floor等号、SD0、9/10 gate、IMPAIRED、打切り、交互作用を確認 | 窓30–49、固定0.1床、対応差を崩す、Bonferroni省略、欠測seed縮小 |
| S-manifest/CLI | arm/task/seedを明示列挙、重複・欠損・偽完了marker・hash不一致を拒否、CLIと関数の一致 | broad glob混入、20task既定、provenance後書き |
| S-cost | 検査seedでref/cap12_bfixの時間・RAM/VRAM・ログ伸びを計測。科学判定ではない | 不十分な測定は費用未確定として扱う |

S-projectの合格幅はu・γ_n・絶対量から計算し、観測残差に合わせた相対許容を後付けしない。操作が働かない状態だけでS-off/発火を試さない。全検査で対応する誤実装を実際に通して落とす。失敗は実測/期待/上界を出力してから停止。

## 8. 実行計画

採用commit→実装→合成/検査seedの検査→実装commit/push→S4完了後にS5の新規task1→5腕→完全性確認→report。予定入口は `src/cap_cifar_ee_0920.py`、`analysis/cap_cifar_ee_0920/{checks.py,report.py,launch.sh}`。現時点ではこれらを作っていない。

- 学習量は共通task1＋5腕×49task=246task相当、7,380,000 stack更新（R10）。射影offのrefを含む。5本のtask1を再実行する実装なら250task相当と記録し、どちらも同じ学習仕様を満たす。
- 本走はGPU1プロセス、ref→cap1→cap2→cap12→cap12_bfixの直列。共有GPU lock `/tmp/lop_analysis_gpu.lock` と研究ジョブ占有を確認する。途中成績で腕順・実行有無を変えない。
- 暫定費用は起案文のR10で35分/50taskから **約3時間＋検査・帳簿**。作業リストの「走1時間」をそのまま使わない。正式見積りはGO後S-costで更新。今回は計時・GPU起動なし。
- 全画像zを全taskでRAM保持せず、必要な縮約・状態をdiskへ保存。S-costのpeak RSS/VRAMと実際の空き容量を照合し、desktopの余裕を確保する。CPU/BLAS2thread、GPU腕の並列実行なし。
- `results/cap_cifar_ee_0920/STOP` でepoch境界にatomic checkpoint。P/m/v/tc、r1/r2、b_fixed、全RNG、task/epoch、acc・発火集計・履歴を含める。入力/spec/実装/環境/slotを照合して再開し、20task完了を50task結果として扱わない。
- 起動時 `provenance_start.json` にgit_hash、dirty diff/hash、spec/approval、入力SHA、argv、実装・環境、UTC/JST/PID。終了時記録は別ファイル。ログ名は `cap_cifar_ee_0920_<arm>.log`。PIDとログ初行を確認する。
- 長時間の状態確認は5分以上間隔（短い検査と異常を除く）。途中は生存・資源・完全性だけ。reportの入力は明示`--src`、スモーク先は `results/_smoke_cap_cifar_ee_0920/`。不完全な主結果を読む用途にreportを使わない。

## 9. 開示・設計の限界

独立監査なし。今回は静的なspec確認だけで、数値検査のPASSは無い。対象・t1半径・50taskは既知軌道と先行MNISTを見て選んだ介入。比率から因果を断定せず、結果を受けて半径・bias規則・seedを最適化しない。

上限はノルムの成長を止めるが、方向・Adam履歴・自由なbias・第3層・特徴の学習を止めない。cap12_bfixでもWの方向は自由。成功/失敗の射程はこの予算と閾値に限る。S4からの順序は実施順であり、支持的なS4結果だけを条件にS5を報告する選別ではない。

## 10. 出力・片付け

`results/cap_cifar_ee_0920/` に summary.md、verdict.csv/json、paired.csv、per_seed.csv、per_task.csv、timing.csv、secondary.csv、ledger_summary.csv、checks.json、provenance_start/end.json、input_manifest.json。radii/bias基準、unit/ledger配列、checkpoint、ログはschema/hash付きでgit外に保存する。

`CLAUDE.md` §4どおりgit外出力（pycache以外）を `/home/issan/Projects/obsidian-research-data/cap_cifar_ee_0920/` へ退避し、source/backup/bytes/SHA256のmanifestをcommitする。mainへmerge/push・到達確認後に自分のworktree/branchを削除。元バトル、A6/S4、共有data、他セッションのworktreeには触らない。

**今回のspec納品**: 実装・生データなし。文書commitをmainへ統合して片付け、GO後に同じrun名で最新mainからworktreeを作り直す。
