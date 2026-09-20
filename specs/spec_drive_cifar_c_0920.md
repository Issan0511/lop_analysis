# drive_cifar_c_0920 — 課題間の第2層駆動をAdamの実更新で測る（A2）

状態: **実装・検査を許可（末尾追補）。検査結果はresults/drive_cifar_c_0920/implementation.jsonを正本とする。本走未実行・本走GO待ち。Issa本人の予測は未記入。**

2026-09-20 / Codex / 起点main `6bf41b6`。run `drive_cifar_c_0920`、branch `codex/drive_cifar_c_0920`、worktree `wt/drive_cifar_c_0920`。親はvault `背骨CIFAR_S4S5_A3-A6_spec起案プロンプト_0920` A2追補、`中心主張v11作業リスト_0920` A2、`駆動源問題_0909` §12.1–12.7。型: [erosion_race](spec_erosion_race_0919.md)、[relu_doors](spec_relu_doors_0919.md)。

## 0. 一行

**入力中心化C腕が課題を切り替えた後、第2層自身のAdam更新に対する沈降の十分条件が、上昇の十分条件より多く成立するかを測る。** 上流も更新される実際の全移動と区別し、境界付近・課題内の継続・上向き/下向きの量を保存する。観測であり、介入による原因同定ではない。

## 1. 既知情報と、起案文から直す点

既知のerosion_race M1はraw・第1層・課題1の更新1–750でReLU 5/5、GELU 4/5がCONF_NEGATIVE_DOMINANT。課題間・第2層へ外挿しない。C腕は初回を学んだ後に第2層で機能を失うという既存列の事後読みに基づいて選んだ。S4/S5が既に完了していることも既知だが、両実験はELU/stdで、この走はReLU/C。結果を同じ軌道として扱わない。

起案時にrepoのコード・既存spec・vaultの導出・運用ルール/引用禁止を確認した。新しいA2の条件成立頻度・窓の較正・タイミングprobeは一切読んでいない。独立監査なし。

重要な修正:

1. **Cはseedごとの平均画像を引く操作で、stdとは異なる。** `relu_doors.slot_inputs(..., center=True)` は `x-x.mean(0)`。実数でmu1=0、実装では丸め残差を持つ。「stdとCのrは共に0.121」はこの実装には当てはめない。mu1を消してもb1更新や特徴の変化は残るので、第1層の全駆動が消えたとはしない。
2. **§12.5の保証対象は入力平均を固定した対象層自身の移動。** 第2層の全移動には上流入力の変化が加わる。自己移動の証明と全移動の符号を同一の検査にしない。
3. **T>Rの無情報率を1/2と置かない。** R>=0なのでTが対称でもP(T>R)は通常1/2以下。時系列・unit間の依存もある。条件の下向き/上向きの対称な版を比較し、独立標本単位はseedとする。
4. **5seedの過半だけでは名目5%の符号検定にならない。** 片側Bin(5,1/2)でP(K>=4)=6/32、P(K=5)=1/32。主の検出条件は5/5とする。4/5を成功に繰り上げない。
5. **R10の計算形を保持する。** 5seedを取り出してR5で学習しない。0–9を同じ順で走らせ、主推論は事前固定した0–4、5–9は境界窓の較正専用とする。

## 2. 箱・データ・状態

無改変の `src/relu_doors_0919.py` C腕を宿主とする。raw CIFAR-10から各seed固定1200枚、入力のみ中心化、3072–100–100–10、両隠れ層ReLU、biasあり。Adam lr=1e-3、beta=(.9,.999)、eps=1e-8、WD・追加射影なし。batch16、400epoch×75=30,000更新/課題、t1–t5固定。momentと時刻は課題境界で継続。画像・初期化・ラベル・順列の宿主streamを維持する。

本走はseed0–9のR10、昇順slot。新しい初期状態から5taskを再走し、過去のt50 checkpointを巻き戻さない。主対象はseed0–4×task2–5×100unit×30,000更新で、全12,000,000 unit×更新/seed。初回t1は状態形成・検査用として保存するが主分母には入れない。seed5–9を主検定へ追加しない。

同一CUDA機、float32・native ReLU/autograd・宿主のbaddbmm/Adam演算順、TF32 off、宿主setup、CPU/BLAS2thread。観測はfloat64と誤差区間、学習のP/m/v/RNG/順列には書き込まない。画像probe64枚に縮めず、各更新の評価平均mu2は**固定1200画像の現在のa1**から取る。

宿主資格は検査seed100–109でR10・2task×400epochのP/m/v/tc・全RNG・宿主既存列のbit一致を必須とする。過去Cのseed0–9・task1–2の保存P/既存列との一致も別に報告する。過去結果と不一致なら原因を記録し、旧軌道の再利用とは呼ばない。無改変宿主と観測付き新規走の一致が確認できない限り本走しない。

## 3. 腕と評価する要求

学習腕はCのみ。SGD・label変更・conf除去・moment resetは行わない。SGD形は同じ観測点から計算するREPORT_ONLY。

task t>=2の各画像について、oは前課題t−1の実際のラベル、nは現在課題tのラベル。学習が進んでもoをargmaxや現在ラベルへ差し替えない。旧/新ラベルが同じ画像も除外せず、d=0として残す。

同一更新前状態・学習batchで、h_s=a1(x_s)、z_is=z2(x_s)、J_ci=W3_ci（現在の最終読み出し）、p_s=softmax(logit_s)。

```
d_is = J_{o_s,i} - J_{n_s,i}
e_is = sum_c p_sc J_ci - J_{o_s,i}
u_is(new) = d_is + e_is
epsilon_os = 1 - p_{s,o_s}
L_is = max_{c != o_s} abs(J_ci - J_{o_s,i})
abs(e_is) <= epsilon_os * L_is
```

uは一標本損失の活性微分。batch/stackの平均係数を二重に入れない。ReLU微分は実際の学習forwardのzに対するautograd（z=0では0）。B16の学習用h・z・logitと、B1200の評価平均をそれぞれ保存・縮約する。

## 4. 読み出しと十分条件

### 4.1 Adam版（主）

theta_i=(w2_i,b2_i)、k=(mu2_old,1)、h̃_s=(h_s,1)。M_prevは更新前の実際の第2層一次moment、V_newは今回のnative勾配を入れた実際の二次moment。全履歴の時刻qを使い、c_q=1−beta1^q、D_q=diag(1/(sqrt(Vhat_new)+eps))。

```
K^D_is = k^T D_i h̃_s
T^D_i = mean_batch[K^D_is * phi'(z_is) * d_is]
R^D_i = mean_batch[abs(K^D_is * phi'(z_is)) * epsilon_os * L_is]
H^D_i = k^T D_i M_prev_i
Qminus_i = beta1*H^D_i + (1-beta1)*(T^D_i-R^D_i)
Qplus_i  = beta1*H^D_i + (1-beta1)*(T^D_i+R^D_i)
```

実数の式ではQminus>0なら自己移動S_i<0、Qplus<0ならS_i>0。実装では次節の演算誤差を差し引き、**符号まで保証できる**ものをCERT_DOWN/CERT_UPとする。どちらでもない場合はUNRESOLVED。同じunit×更新で両方成立したらCHECK_FAILED。

旧課題の残差上界が時間とともに大きくなり、条件が成立しなくなることは科学的な結果。旧課題に高確信という近似を課題末まで持ち越さない。Dをconf/label/旧新それぞれで計算し直したものを実Adamの加法成分とは呼ばない。

### 4.2 自己移動・上流・全移動

P_old/P_newはnative学習の実際の更新前後、mu_old/mu_newはそれぞれ全1200画像で計算する。

```
S_i = (w_new-w_old)^T mu_old + (b_new-b_old)
U_i = w_new^T (mu_new-mu_old)
Delta_m_i = S_i + U_i
          = w_new^T mu_new + b_new - w_old^T mu_old - b_old
```

SをM2xの符号保証の相手とする。直接native mean(z2_new)−mean(z2_old)との差はB1200の行列演算丸めとして別に閉包を検査する。S<0でもUが勝ってDelta_m>0なら理論違反ではない。さらにSの負の上界がUの正の上界を上回るときだけCERT_TOTAL_DOWNを報告する（副）。全移動の正負は条件成立/不成立を問わず保存する。

### 4.3 誤差・境界

計測値・条件・閉包は成分wiseのBall(value,error)で評価。u=2^-24/2^-53、gamma_n=nu/(1−nu)を使い、内積長101・batch16・評価1200・読み出し10の和積誤差、abs/max、softmax/CE backwardのnative再構成誤差、EMA、実分母のsqrt/除算、bias補正係数、実際のfloat32パラメータ減算の丸めを伝播する。非正規化数は実測flush仕様とabsolute rounding boundを含める。

具体的にはnative autograd勾配g_actualと再構成g_decompの差を独立に検査し、その**測った差を隠さず誤差項として射影**する。実Adam候補変位と−lr/c*D*M_newの差も保存し、Sに必要な誤差幅へ足す。`abs(residual)<=bound`だけを自己参照で作らず、変異検査は観測S・native gを固定して理論側だけ変える。チェッカーのfloat64内積誤差も加算する。

CERT_DOWNは負の自己変位上界がチェッカー誤差を含めても厳密に0未満、CERT_UPは正の下界が厳密に0超。等号・0を跨ぐ区間はUNRESOLVED。数学的なQminus>0/Qplus<0の素の件数、数値的に証明できなかった件数も併記する。曖昧な件を主分母から落とさない。量ごとの実装可能な誤差伝播式・独立参照を検査コードのcommitで確定し、本走の残差に合わせて帯を広げない。

### 4.4 conf/label/historyとSGD（副）

CE=L_conf+L_label、L_conf=mean(logsumexp(logit)−pi^T logit)、L_label=−mean((onehot(n)−pi)^T logit)、pi_c=.1。課題開始momentをM_startとして、s更新後M_hist=beta1^s M_start、M_confは0起点のconf勾配EMA、M_label=M_actual−M_hist−M_conf。生のCE勾配とAdamが正本。3成分に共通の実V_newと全履歴bias correctionを用い、S_conf+S_label+S_histとS_actualの加法閉包・実パラメータ丸めを別記する。

§12.4のK=h^T mu+1、T/Rも同じ点で保存するがSGD仮想量と明記する。d/e分解とconf/label分解は異なる分解なので名前・CSV列を混ぜない。課題1のhistory=0だけで検査を済ませない。

## 5. 登録判定

### M1x（唯一の主科学判定）

主窓はtask2先頭〜task5末、全unit・全更新。seed sごとにp_down_s=N_CERT_DOWN/N、p_up_s=N_CERT_UP/N、D_s=p_down_s−p_up_s。N=12,000,000。task平均を取る場合も全taskが同じ長さで同値であることを検査する。各p、曖昧率、task/unit別分布も全て保存。中央値への集計変更をしない。

帰無はseed間独立かつD_sの正負に方向の優勢がないこと。非零Dの片側正確符号検定で、正の個数k、非零数n、p=sum_{j=k}^n C(n,j)/2^n。

| 状態 | ラベル |
|---|---|
| 全5seed D>0（p=1/32<.05） | CROSS_TASK_SINK_CONDITION_HOLDS |
| 全5seed D<0 | NOT_SUPPORTED（逆方向が揃ったという記述。独立の両側5%検定とは称さない） |
| その他、tieや両条件0を含む | UNRESOLVED |

条件成立頻度そのものは全5seedで報告する。HOLDSは**証明可能な方向の下向き優勢**であり、過半の全更新が証明された・沈降量の過半を説明した・正味沈降が証明されたという意味ではない。これが起案文の1/2帯を置き換える推奨案。5seedによる検出力の制限を結果に残す。

### M2x（保証と実装の整合）

CERT_DOWNにおけるS<0とCERT_UPにおけるS>0を独立再計算で照合。証明マージンが数値誤差を越えた反例が1件でもあれば、全状態/画像id/旧新ラベル/全項/境界を先に保存してCHECK_FAILED、本走・解釈を停止する。違反0ならCERTIFICATE_CONSISTENT、証明件数0ならNO_CERTIFIABLE_EVENTS（PASSの成功率100%としない）。全移動Delta_mの一致率はREPORT_ONLY。

### M3x（境界と後続、副）

200更新などの窓を転用しない。seed5–9だけのtask2–5における、epochごとの平均全移動（全unit平均、全seed/task等重み、75更新で割った1更新当たり値）を400点v_eに縮約する。この較正データだけを先に開く。

一定モデルと1変化点の2水準モデルを全break b=1..399で最小二乗。BIC0=400*log(RSS0/400)+1*log400、BIC1=400*log(RSS1/400)+3*log400（2水準＋break）。同率なら単純モデル、break同率なら最小b。全一定/RSS両方0は単純モデル、RSS1のみ0は2水準。2水準のBICが低く、先頭水準<0かつ先頭水準<後続水準ならboundary=更新1..75b、later=75b+1..30,000を採る。その他はWINDOW_NOT_IDENTIFIED。

これは自己相関を含む系列の**操作的な分割規則**であり、BIC差を有意性や真の切替時定数の検定としない。コード/規則は本走前固定、得られた窓・較正input hashは主seedの条件頻度を開く前に `window_calibration.json` としてcommitする。主M1xは窓選択に依存せず常に全task2–5。

窓が定義された場合、主seedごとに4taskそれぞれのp_down(boundary)−p_down(later)を等重み平均し、同じ片側正確符号検定で5/5正ならBOUNDARY_ENRICHED、5/5負ならLATER_ENRICHED（記述）、他はUNRESOLVED。後続区間を「定常状態」とは呼ばない。分割不能ならM3xは未判定のまま残し、主M1xの窓を動かさない。

### M4xと完全性

S/U/Delta_mの正部分・負部分を別々に積算し、総和・1更新当たり率・conf/label/history成分・平均とunit分布を各task、全主窓、定義できた境界/後続に保存する。上流の動きや条件不成立の輸送を捨てない。t1終了・各task終了のonline、memo、G2、zero/dead、mean z/sdはREPORT_ONLY。

欠落seed/task/必須列/偽markerはINCOMPLETE、非有限はDIVERGED。主seedを減らさず全科学判定を抑止する。操作/資格検査FAILはCHECK_FAILED。適切に同一状態を復元した再開は可。少ない証明件数や不支持の結果は中断理由にしない。

## 6. 実装前予測・採用欄

Codex、2026-09-20（記録時刻はこのspecのcommitが正本）、新規A2実装/観測/計時より前。§5の改訂案を採った場合の条件付き予測:

| 項目 | 内容 | 確率 |
|---|---|---:|
| M1x | CROSS_TASK_SINK_CONDITION_HOLDS | .55 |
| M1x | UNRESOLVED | .40 |
| M1x | NOT_SUPPORTED | .05 |
| 境界窓の同定 | WINDOW_NOT_IDENTIFIED以外 | .65 |
| M3x（窓が定義された場合） | BOUNDARY_ENRICHED | .75 |
| confの自己輸送（副） | 主窓S_confのunit/更新総和が全5seedで負 | .65 |

旧課題の残差上界が課題内で増えるので、条件が常時成立するとは予想しない。M1xはmulticlass Brier、その他はbinary Brier。適用不能/検査失敗/M3x窓未定義は理由付き未採点。証明の正しさを実証の的中数に足さない。

**Issa予測: 未記入。** 内容への同意で記録できるが、Codexの確率を本人の確率として転記しない。今回の「A2 A5お願いします」は本起案の依頼として扱い、S4/S5の設計採用をこの新規設計に転用しない。

| 裁定点 | 推奨 |
|---|---|
| 条件 | Adam版を主、SGD版を副 |
| 保証の相手 | 自己移動S。上流/全移動は別集計 |
| 主判定 | p_down−p_up、5seedの片側正確符号検定 |
| 束ねと較正 | R10維持、0–4主、5–9窓較正、主は全t2–5 |
| 実装・本走 | 上の修正案と予測を採用後に開始 |

## 7. 必須検査（未実施）

collectorは期待ID/変異一覧と実source hashを持ち、未実行・空・別sourceのPASSを拒否する。

| ID | 独立の照合 | 必ず落とす変異 |
|---|---|---|
| A2-host | 無改変C宿主とR10・2task×400epochの全状態/乱数/行bit一致 | R5、観測によるRNG消費、Adam演算順変更 |
| A2-input | 全1200画像の中心化・mu1丸め、全label/order hash、旧ラベル保持 | stdをC扱い、argmaxを旧ラベル、別task/seed |
| A2-CE | float64の独立CE微分、u=d+eと残差上界、native実gとの閉包 | batch平均二重、conf/label入替、dの符号逆 |
| A2-Adam | 実分母/履歴/bias correction、独立scalar更新、history逆転例 | 現勾配のみ、成分別V、taskで時刻reset |
| A2-self-total | 同じ固定画像の前後PからS/U/全移動、native平均との差の誤差 | U省略、W_oldをUに使用、別seed mu |
| A2-certificate | 保証成立/不成立/厳密境界/丸め0を独立fixtureで照合 | 誤差を無視、Qplusをdownに使用、K=1 |
| A2-nonvacuous | mu2非零・W2更新と上流更新がともに非零のfixture | 理論側だけmu交換、観測Sを固定し閉包FAIL |
| A2-confhist | t2以降の3成分EMA和とactualM、共有実VによるS和 | historyなし、confをlabel扱い、moment上書き |
| A2-graph-resume | eager/graph・warmup全巻戻し・epoch再開一致 | 観測counter/旧ラベル/moment/acc欠落 |
| A2-window | 合成1/2水準・同率・定数・逆向き、較正のみ使用 | 主seedでwindow変化、200更新固定 |
| A2-verdict | 全主ラベル・5/5と4/5・tie・全0・未定義・欠測を独立列挙 | unit/stepを独立標本、5seed未満で成功、M2空分母PASS |
| A2-manifest | run/seed/task/source/input/hash/marker明示照合 | 途中report、別run混入、偽完了 |
| A2-cost | 検査seedでt1と崩落後のt2以降、全1200観測を含む時間/RSS/VRAM/I/O | batch16だけの計時、観測を間引いた見積り |

順列変異は、画像・旧新labelの対応を意図的に壊して検知する。理論側と観測側を同時に同じ誤りへ置き換える空虚な一致検査は禁止。

## 8. 実行計画・費用

採用記録→spec commit/push→実装・検査→実装commit/push→本走R10×5task→完全性確認→較正seedだけの窓生成/commit→主report→予測裁定。今回の納品はspecまで。前段S4/S5の実装・本走GOをA2へ拡張しない。

学習はR10の150,000 stack更新、合計1,500,000 model更新。起案文の5seed×5task=750,000は主seed分だけで、R10維持の計算量ではない。毎更新のB1200前後forwardはB16訓練より重く、旧「30分」を確定費用にしない。正式な全体費用はA2-cost後に記録し、その値と空きRAM/VRAM/diskを照合する。

GPU1プロセス・共有lock `/tmp/lop_analysis_gpu.lock`。他ジョブと同時に走らせない。100unit×全更新の全テンソルをCPUへ転送せず、各epochの件数・符号別総量・閉包max・証明違反数をGPU縮約。主/較正seedを別ファイルへ出す。数値反例は最初の完全状態を必ず保存。各taskの最初と最後のepochを再検算可能な固定監査区間として保存（選択は結果非依存）。

`results/drive_cifar_c_0920/STOP`でepoch境界にatomic checkpointし、P/m/v/tc、全RNG、旧新ラベル、観測EMA/件数/acc、進行位置を復元する。provenance_startは起動時のgit/dirty/source/data/env/UTC/JST/PID、endは別記録。ログ名 `drive_cifar_c_0920.log`。長時間状態確認は10分間隔、途中科学成績は開かない。

## 9. 解釈の範囲

十分条件の成立は原因の除去実験ではない。自己移動・全移動・重みのノルム成長・機能的LoPを分ける。confは定義だけで侵食、labelは定義だけで回復とはしない。C腕/既知seed/5task/ReLUの観測で、未使用seedや他活性化への一般化は未検証。証明可能な頻度が少なければそのまま報告し、条件を緩めない。

## 10. 出力・片付け

予定: `analysis/drive_cifar_c_0920/`、`src/drive_cifar_c_0920.py`、`results/drive_cifar_c_0920/`にsummary.md、verdict.json/csv、per_seed/per_task/per_epoch、window_calibration、certificate/closure/transport/conf_history集計、predictions、checks、provenance/input_manifest。raw/checkpoint/ログ/全検査attemptはgit外、退避先 `/home/issan/Projects/obsidian-research-data/drive_cifar_c_0920/` にサイズ・SHA256付きmanifestで保存。

CLAUDE.md §4どおりmainへmerge/pushし、到達確認後に自分のworktree/branchだけ削除。今回のspec起案もmainへ統合して片付け、GO後に同名runのworktreeを最新mainから作り直す。既存C/S4/S5/A6・共有data・他セッションのファイルは変更しない。

## 実装追補（2026-09-20、科学seed観測前）

Issa「とりあえず実装して」によりA2・A5の実装と検査seedによる検証を開始する。本走GO・Issa本人の数値予測は未記録。本追補は実装許可であり、本走の開始記録ではない。

宿主を読むとReLUは`torch.clamp(z,min=0)`であり、native backwardは厳密なz=0でも微分1を返す。§3の「z=0では0」は宿主の挙動を誤記していた。**宿主不変を優先してnative clampの微分（z>=0）を使う**。ゼロでの挙動を独立autograd fixtureで検査する。`relu_doors`の診断用dphi（z>0）との違いも隠さない。

誤差伝播は`analysis/drive_cifar_c_0920/numerics.py`を正本とする。実測gradient defectはCE再構成の独立gamma128(float32)境界を、moment defectはgamma4、実パラメータ差とdouble Adam候補の差はgamma12境界を通過してから条件の誤差幅へ射影する。float64の内積・和・平均の境界を別途足す。flushされた非正規化数を含む絶対項にはfloat32 tinyを用いる。入力平均は各更新前後のnative全1200枚a1のdouble平均。主条件はfloat32保存状態の実自己変位に対するもの。

固定監査区間は最初/最後epochの開始時全状態・全順列・終了時全状態を保存し、全75更新を再現可能にする。毎epochのatomic checkpointを正本とし、未checkpointのshardは再開時に再計算する。失敗時はepochの再現可能fixtureを先に保存し、eager再走で最初の失敗更新の前後全状態と計測項を保存する。

実装検査の追補: p_oldがdoubleで1に丸められても他クラスの正の質量が残る境界を合成fixtureで固定。epsilon=1−p_oldの絶対誤差をgamma64(float64)×Lとして残差上界の誤差へ伝播する（大きさは観測残差からfitしない）。検査seed t1/e205の最初の失敗fixtureも保存し、修正前の検査をPASSに書き換えない。
