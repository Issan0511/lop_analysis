# resp_cifar_ee_0920 — CIFAR・ELU/std の第2層の応答を交換する（S4）

状態: **設計・予測の登録案。Issa は「S4→S5」の順を採用（2026-09-20）。以下の推奨設計の裁定・Issa の予測・実装 GO は未記入。実装・検査・本走は未実施。**

作成: 2026-09-20 / 起草: Codex / 起点: `origin/main=d7e09b91a952e00ac02bceb9d144d2678caa50d1`。
run: `resp_cifar_ee_0920` / branch: `codex/resp_cifar_ee_0920` / worktree: `wt/resp_cifar_ee_0920`。
親: vault `背骨CIFAR_S4S5_A3-A6_spec起案プロンプト_0920` S4 と A6 後の追補、`中心主張v11作業リスト_0920` S4。
型: [resp_ee](spec_resp_ee_0917.md)、[relu_doors](spec_relu_doors_0919.md)。判断待ちでも読めるよう、推奨案を一つの実行可能な設計として記す。採用後の変更は実装・結果を読む前に別 commit で固定する。

## 0. 一行

**ELU/std の自然な網の初期出力を揃え、第2層に健康時／崩壊後の応答の場を交換すると、次タスクの学習能力が両方向に変わるかを問う。** 主は復元 P1 と沈降 P2。S4 は応答への介入、S5 は重みの成長への介入であり、両者の成功だけで同じ経路の完全な媒介を証明したとはしない。

## 1. 出所・既知情報と起案文の訂正

- vault の現在地・V10 の整理・V11 草案／作業リスト・運用ルール・引用禁止、起案プロンプトの共通前置きを参照した。2026-09-20 の collaboration notice/changes に未確認の先生側変更はなかった。他セッションの worktree は変更しない。
- [A6 summary](../results/cifar_ledger_0920/summary.md)（結果 `ec1a0dd`、完了 `d7e09b9`）: 既知の R=20 の ELU/std は L2_FIRST 10/10、Q 初回超過の中央値 t2、online<0.5 は t3。第2層・ELU/std の選択に使った既知情報であり、独立な予測成功として数えない。A6 は既知軌道の再解析で、独立監査なし。
- [resp_ee spec](spec_resp_ee_0917.md) の固定場、初期出力一致、Adam 対照、両方向の比較を継承する。MNIST の成功は既知。CIFAR の介入結果は存在せず、本起案で新しい接頭部・probe は走らせていない。
- 宿主 [battle engine](../src/rlcifar_mlp_battle_0918.py) の `ELU.phi` は `F.elu(z,1.0)`、訓練は `torch.autograd.grad`。`Act.dphi` は診断用。**前向きの関数を変更して autograd で微分する**。手書き dphi だけを交換しない。
- MNIST の `expm1` の床を移植しない。`F.elu` の深い負側、subnormal、厳密零は実行環境で検査・記録する。起案文の「−87 付近」を零点の固定定数にしない。小さい非零微分での停止を、結果を測る前から本箱の性質と断定しない。
- Q は全画像×unit の `abs(g_train)<1e-6` の割合。宿主 `dead_frac`（画像に関する最大応答で unit を分類）と分母が違う。**健康／崩壊の資格は次タスクの機能的な差で確認し、Q の超過を LoP の発生と同一視しない。**
- 旧 R=20 と新 R=10 の不一致は合格条件ではない。一致／不一致を報告するだけで、新しい自然継続の資格は同じ R=10・同じ演算の再現で決める。

## 2. 箱・分岐・介入

### 2.1 固定する箱

- Random Label CIFAR-10（強化学習ではない）。seed ごとに1200枚を固定。宿主の subset・初期化・`rlc_labels`・`rlc_batch` の seed 別 stream を継承。各タスクは独立一様な10クラスラベル、各 epoch は1200枚の順列。
- 3072–100–100–10、両隠れ層 ELU(α=1)、出力線形、bias あり。入力は `std` のみ: u8/255 の各 channel に宿主の mean=(0.4914,0.4822,0.4465)、sd=(0.2470,0.2435,0.2616) を適用。各 subset の平均を厳密に0にする処理ではない。
- Adam lr=1e-3、β=(0.9,0.999)、ε=1e-8、WDなし、batch16、400 epoch×75=30,000更新/タスク。online は更新前の全 minibatch 正解率の平均、memo はタスク末に同じラベルで全1200枚を評価した正解率。
- **本走 seed=0–9、1腕につきR=10、slot順はseed昇順。** 腕を同じ stack に足さず、途中で slot を抜かない。既知の自然軌道と同じseedを使う対応介入であり、未使用seedの再現ではない。
- 元マシンの CUDA・float32・`baddbmm`・要素ごとの Adam・CUDA graph を継承。TF32 off、宿主 `H.setup` と同じ決定論設定。CPU/BLASは2 thread。torch/CUDA/cuBLAS/GPU、flush 設定、演算形状を起動時に保存。診断の和・統計・誤差計算はfloat64。
- 検査・計時はseed100–109のR=10と合成入力。本走0–9の介入を検査用に先に走らせない。既知A6状態の読み取り検査は既知データの再利用と明記する。

### 2.2 自然な接頭部と分岐点

**健康候補 t_h=1末、崩壊候補 t_c=10末を全seed共通に固定**。成績でt2や別seedへ選び直さない。接頭部は **t11まで** 新規に走らせる。t2・t11は自然継続 N1・N10 の照合先で、介入は各分岐から1タスクのみ。

保存点t1・t2・t10・t11に、全P、Adam m/v/tc、全seedの次のラベル・標本順generator state、slot順、画像index、入力hash、現在のラベル、完了task、状態hashを保存する。donor用の両層zはfloat32で新規に再計算し、旧snapshotのfloat16 zを用いない。

全腕は分岐状態の独立cloneから始める。同じ分岐の腕は次タスクのラベルと全順列を共有する。t1からはtask2、t10からはtask11を学ぶ。P1/P2はそれぞれ同じ課題内の対応差であり、P1とP2の絶対量の差を同じ課題での非対称性とは読まない。

自然腕N1/N10の終端P/m/v/tc/RNG・行は、それぞれ接頭部t2/t11とbit一致を要求する。これは移植腕の結果を読む前の操作検査。接頭部・場の作成・診断は学習用RNGを進めない。

### 2.3 画像id上の固定場と出力一致

同じseed・同じunit indexを対応させ、donorの全1200枚での前活性を z_src、受け手の分岐時のものを z_br とする。学習前に

`d_l[id,i] = float32(float64(z_src[id,i]) − float64(z_br[id,i]))`

を作って固定する。一様腕は `d=-Δ`。受け手の凍結P0から、**現在のminibatchと同じR・B・演算で** z0 を毎回再計算し、対象層のみ

`a = phi(z0) + [phi(z+d) − phi(z0+d)]`

とする。凍結項・dに勾配を通さず、更新するPからの z にだけ通す。分岐時は z=z0 なので括弧は厳密0、特徴・logit・損失が自然腕とbit一致する。対象層の局所微分は **native F.elu の g_train(z+d)**。非対象層はその腕の現在の前活性で微分する。P0は更新もaliasもしない。

全画像評価B=1200と訓練B=16は行列積の丸めが違い得る。場はB=1200の表に固定し、訓練では画像idで引く。分岐の凍結forwardを全画像キャッシュで代用しない。donorとの微分一致は下の算術誤差で検査し、初期出力のbit一致と混同しない。符号つき0もbit検査に含め、保存が必要なら値のbitだけを保つ処理にし、対象の局所微分を0にする分岐を入れない。

**場の誤差上界**: 独立checkerがmanifestで固定したdonor/recipientからd_refを作り、保存dと全要素bit一致を先に確認する。unit・画像ごとに `q=z_br,eval + float64(d_ref) − z_src,eval` をfloat64で計算する。B=16でのshifted引数の誤差は、`|q| + |z_br,mb−z_br,eval| + u32*(|z_br,mb|+|d_ref|) + η32` 以下（u32=2^-24、η32は実測flush仕様で消え得る絶対量の上界。float64のq計算の丸め上界も加える）。donorのB=16を比較するときは `|z_src,mb−z_src,eval|` も足す。F.eluの導関数 `exp(min(z,0))` はLipschitz定数1なので、この引数誤差に独立に呼び出したnative kernelとfloat64参照の各丸め誤差を足してgの上界とする。壊した場・引数・勾配の実測残差から許容を作らず、wrong-id/sign/seedの変異を受け入れないことを確認する。

これは固定した人工的な関数変更。初期出力を揃えるが、その後の特徴は変わる。自然な崩壊の全媒介率、純粋な最適化前処理、時々刻々のdonor追随とは主張しない。Δ=0は数学的恒等だが括弧の丸めで長い軌道がbit一致するとは要求しない。**無介入は宿主forwardへ直接dispatch**し、そちらで軌道のbit一致を検査する。

## 3. 腕（12腕×10seed×1タスク）

`r`はAdamの全m/v/tcを0へ戻す。P、画像・ラベル・順列stream、凍結P0は変えない。

| 腕 | 受け手の末task | 場・対象層 | Adam | 用途 |
|---|---:|---|---|---|
| N1 / N10 | 1 / 10 | 無介入 | 継続 | 自然対照・再開照合 |
| N1r / N10r | 1 / 10 | 無介入 | reset | reset単独の対照 |
| **R1_10** | 10 | donor t1・第2層 | 継続 | **P1** |
| R1_10r | 10 | donor t1・第2層 | reset | 復元の履歴依存・副 |
| **S1_10r** | 1 | donor t10・第2層 | reset | **P2** |
| S1_10 | 1 | donor t10・第2層 | 継続 | 沈降の履歴依存・副 |
| S1u5r / S1u10r / S1u20r | 1 | 一様−5 / −10 / −20・第2層 | reset | 固定した大きさの階段・副 |
| S1_L1_10r | 1 | donor t10・第1層 | reset | P2と同じ方式の層対照・副 |

P1は崩壊時のoptimizerも保って場だけを戻す。P2は、過去のmomentによる移動が低応答化を迂回することを分けるためreset同士で比較する。**P1/P2で履歴の条件が異なる**ことを結果に併記する。全腕でW1/b1/W2/b2/W3/b3は更新する。

Δは先行移植の試験値から選んだ介入強度で、判定閾値でもCIFARから導いたε境界でもない。値を見て段を足さない。raw、両層同時交換、動く場、ε変更、cap併用は本specの対象外。

## 4. 登録する読み出し

- 主E: 各継続taskのonline。memo、CE、最多クラス率F=max_c(n_c/1200)も記録する。
- 両層のnative g_train: 画像平均→unit算術平均G、全pairの `abs(g)<1e-6` 率Q、厳密0率、全画像で厳密0のunit率、診断dphiとの差。割合は整数pair数から計算し、ちょうど半数を浮動小数平均で超過扱いしない。
- unit別 `n_eff = (sum_x g)^2 / sum_x g^2` と正規化版 `/1200`。sum g²=0のunitは担い手なしとして0を割り当て、そのunit数を別列に出す。gは非負ELUのnative微分。float64で集約する。全seedのunit平均を示し、別箱の絶対閾値は流用しない。
- 両層のz/sd、shifted引数、mu2、W行ノルム、bias。SD=0の比はundefinedにし、小さい定数で埋めない。初期出力一致と場の引数/g誤差をseed別に保存する。
- 診断点は分岐直後・更新75/750/7500/15000/30000。これらは測定スケジュールで判定窓ではない。主Gの分岐値は全1200枚評価、経過は同じ評価法。実際の訓練B=16でのgは最初の1 epochを別に記録し、B=1200との差を開示する。
- ρ1=mean(P1)/mean(E(N1)−E(N10))、ρ2=mean(P2)/mean(E(N1r)−E(N10r)) は報告のみ。分母の95%対応差区間が正側に入らなければundefined。負値・1超を丸めない。

## 5. 登録する判定

### 5.1 完全性と適用条件（先に判定する）

1. 登録した **10seed×12腕がすべて揃い有限**、入力/commit/config/hashが整合し、必須検査がPASSでなければ `INCOMPLETE` / `CHECK_FAILED` / `DIVERGED` を理由付きで保存し、主ラベルを出さない。seedを落としてnを縮めない。同一設定からの正しい再開は可。
2. 自然差 D_s=E_s(N1)−E_s(N10) の95%対応差t区間の下端>0、かつ全10seedで自然な分岐のG2(t1)>G2(t10)を要求。満たさなければ **NOT_REPRODUCED**。t_h/t_cや層を取り替えない。これは「差が存在する」条件であり、偶然水準への到達を単独で証明する検定ではない。
3. 表示用の健康/低下は、各自然継続の適合率 `(E−F)/(1−F)` とmemoを併記する。S4の適用条件にQの0.5超過を追加しない。

### 5.2 主比較

`P1_s = E_s(R1_10) − E_s(N10)`、`P2_s = E_s(N1r) − E_s(S1_10r)`。

n=10、差の標本SD s、df=9として、主区間は

`mean(P) ± t_(9, 1−0.05/(2×2)) * s/sqrt(10)`。

2比較のBonferroniによる各97.5%両側区間。95%区間も報告する。名目水準は登録した誤り率で、効果量の固定帯を置かない。s=0ならt統計量を割り算せず、区間を同一点 `[mean,mean]` とし `DEGENERATE_SD` を併記する（正規近似による不確実性を識別できない）。符号は下端>0で+、上端<0で−、他は0。端点=0は0。

| P1符号 | P2符号 | 主ラベル |
|---|---|---|
| いずれか− | 任意 | RESPONSE_REVERSED |
| + | + | RESPONSE_BOTH_WAYS |
| + | 0 | RESTORE_ONLY |
| 0 | + | SINK_ONLY |
| 0 | 0 | RESPONSE_NOT_SHOWN |

0は「差を示せない」であり、同等性や効果ゼロではない。RESTORE_ONLY/SINK_ONLYは検出方向の名前で、非検出側の不可能性を意味しない。

### 5.3 副比較と読みの上限

- R1_10r−N10r、N1−S1_10、N1r−S1_L1_10r、S1_L1_10r−S1_10rを95%対応差区間でREPORT_ONLY。主2比較の代わりにしない。
- 一様階段はN1rをΔ=0の対照として、0/5/10/20のE・G・Q・厳密0・n_effを示す。seed平均Eが非増加なら `MONOTONE_SAMPLE_MEANS`、全て等しいなら `FLAT_SAMPLE_MEANS`、他は `NONMONOTONE_SAMPLE_MEANS`（記述のみ）。微分の大きさ・担い手の形・零化をこの3段だけで分離したとはしない。εの境界ラベルを付けない。
- Eとn_eff等の関係は箱内の記述。別較正seedによる予測モデルの学習・検証は今回は行わず、相関を因果や登録予測精度へ格上げしない。
- 両方向が出れば、**この固定1200枚・ELU/std・1タスク30k更新・固定場で第2層の応答への介入が学習能力を両方向に変えた**と読む。自然なLoPの全因果説明、全媒介、恒久的救済とは書かない。

## 6. 実装前の予測と裁定欄

### Codex（2026-09-20・新規実装／probe／本走前）

適用条件が成立する確率0.90。成立した場合の主ラベル予測は次の分布（合計1）:

| ラベル | 確率 |
|---|---:|
| RESPONSE_BOTH_WAYS | 0.60 |
| SINK_ONLY | 0.20 |
| RESTORE_ONLY | 0.10 |
| RESPONSE_NOT_SHOWN | 0.08 |
| RESPONSE_REVERSED | 0.02 |

追加予測: ρ1≥0.5は0.60（定義可能な場合。自然差の半分という解釈上の予測値で検査ゲートではない）。第2層を奪う平均低下は第1層を奪う平均低下より大きい、確率0.75（`mean[E(S1_L1_10r)−E(S1_10r)]>0`）。階段は `MONOTONE_SAMPLE_MEANS` 0.70。主ラベルは最大確率ラベルの一致と多クラスBrier、追加は真偽とbinary Brier。適用外とundefinedは予測スコアを付けず別報告する。

Claudeの起案予測（既存記録）は両方向0.50、沈降のみ0.20、復元のみ0.15、非検出0.15。仕様差・適用条件差があるため上の分布に同一視しない。

**Issaの予測**: ［未記入］。A6の「全部そうだと思います」はA6の予測だけの採用で、S4へ自動継承しない。確率の記入は必須でなく内容の採用でもよい。

| 設計の裁定 | 推奨案 | 採用記録 |
|---|---|---|
| 健康／崩壊分岐 | t1末／t10末固定、自然差で適用確認 | 未記入 |
| 層対照 | 第1層の沈降1腕を含む12腕 | 未記入 |
| 階段・範囲 | Δ=5/10/20、stdのみ、継続1タスク | 未記入 |
| 実装・本走GO | 上の設計と予測を記録してから | 未記入 |

推奨案をまとめて採用する回答で足りる。結果を見る前に採用commitを作り、そのhashをprovenanceへ入れる。

## 7. 検査（すべて未実施）

自己参照でなく、宿主／独立なfloat64連鎖律／合成fixtureの期待値で照合する。全必須検査・列挙変異を機械可読な一覧から収集し、未実行・空・古いPASSを失敗扱いにする。

| 検査 | 独立の参照・要求 | 検出する変異 |
|---|---|---|
| S-host | 無改変宿主ELU/std・同じR10で2tasks×400epochs、P/m/v/tc・入力・ラベル・順列・onlineがbit一致（検査seed） | lr変更、β時刻ずれ、標本順変更、expm1へ置換 |
| S-prefix | 本走自然接頭部と別再開N1/N10のt2/t11の全状態がbit一致。旧R20との差は報告だけ | m/v/乱数の欠落、tcリセット、R変更 |
| S-branch | d≠0を含む全介入で分岐時の特徴/logit/CEが自然腕とbit一致。全1200枚・最初の全75batchと固定順列を用いる | 凍結補正を落とす、P0を現在Pにalias、対象層を取り違える |
| S-field | §2.3の引数/g誤差上界内。非定数fieldを作りid対応を検査 | d=0、符号反転、他seed・unit・画像id、minibatch位置でlookup |
| S-grad | float64合成網で全6パラメータ勾配が独立連鎖律と丸め上界内。native F.eluのautogradと局所因子を照合 | dをbackwardから落とす、凍結項へ勾配、backwardのみ置換 |
| S-derivative | 負側深部・0近傍・subnormal・厳密零のnative勾配を測り、診断との差を記録 | phi出力+1の床、expm1、解析微分を訓練微分と偽る |
| S-reset | resetのm/v/tc=0、最初の更新がnative Adam演算および−lr*g/(abs(g)+ε)の丸め上界内。g=0も含む | mのみreset、tc持越し、ε省略 |
| S-isolation | arm順を反転して同一短縮走がbit一致。前後のP0/field/checkpoint/RNG hashが不変 | stateの浅いcopy、診断でRNG消費 |
| S-graph | 同じR10でeager/graphが短縮2task・再開でbit一致、capture warmupを巻き戻す | warmupのm/v/acc/stepを戻さない |
| S-verdict | 全ラベル・端点0・SD0・欠測・発散・分母0・符号逆転を独立fixtureで確認 | seed平均の差と対応差混同、片側t、補正忘れ、欠測seed除外 |
| S-resume/STOP | epoch境界の中断→同じP/m/v/tc/RNG/accで継続が無中断と一致。部分結果から判定しない | acc欠落、field忘れ、他commit/hash、完了markerのみ捏造 |
| S-cost/CLI | 検査seedで実際のfrozen-forward・診断・graphの時間/RAM/VRAM、CLIと呼出しの同一性 | arm名取り違え、epochs既定値変更、provenance後書き |

bit一致以外の算術比較は u・γ_n=nu/(1−nu) と和の絶対量・積の伝播で許容を定め、相対1e−6等を一律に置かない。期待差が非零なfixtureを作り、wrong implementationが検出できることを確認する。float64参照にも同じ判定器を使うだけの空虚な検査にしない。失敗時は実測と誤差上界を保存して停止する。

## 8. 実行計画

採用commit→実装→合成・検査seedでの全検査→実装commit/push→自然接頭部→自然再開照合→12腕→集約、の順。未実装の予定入口は `src/resp_cifar_ee_0920.py`、`analysis/resp_cifar_ee_0920/{checks.py,report.py,launch.sh}`。

- 本走の学習量はR10の接頭部11task＋12腕×1task=23task相当、690,000 stack更新。自然腕は照合のため再実行する。起案文の接頭部10taskではt11の照合先がないので1taskを追加した。
- 元の35分/50task目安なら素の更新は約16分だが、凍結forward・診断・検査の費用を含まない。**暫定の本走枠1–2時間**、正式見積りはS-costで更新。実装は半日程度の旧見積り。今回計時はしていない。
- **GPUは1プロセス・腕は直列**。共有 `/tmp/lop_analysis_gpu.lock` を保持し他研究ジョブと重ねない。起動前の空きRAM/VRAMとS-cost peakを記録し、desktopに余裕を残す。履歴を無制限にRAMへ積まず、epoch/task単位に保存する。
- STOPは `results/resp_cifar_ee_0920/STOP`。epoch境界で全状態とacc集計をatomic保存して止める。再開は同一入力・spec・実装・環境・slotでだけ許可。停止や障害でタスク予算を短縮した判定を出さない。
- `provenance_start.json` は起動時にgit_hash・dirty diff/hash・spec/approval hash・入力SHA・環境・argv・UTC/JST・PIDを保存。終了時のgit状態で置き換えない。ログ名は `resp_cifar_ee_0920_<stage>.log` としbranchの `/` を入れない。起動はPIDだけでなくログ初行とprocessを確認する。
- 長時間の進捗確認は5分以上間隔（短時間で終わる検査・異常を除く）。本走中は生存・資源・完全性だけ確認し、途中の効果量を見て腕・窓を変更しない。reportは明示`--src`、スモークは専用`results/_smoke_resp_cifar_ee_0920/`。

## 9. 開示・このspecの範囲

独立監査は未実施。現在は設計の静的確認だけで、表の検査をPASSと扱わない。既知R20軌道から対象・分岐候補を選んだ確認的介入であり、完全盲検ではない。R10の新規自然軌道は旧バトルの数値を再利用したrefではない。

S4の結果でS5のseed・腕・主窓・予測を事後変更しない。別runとして起案するS5が適用外になっても、その理由を保存する。S4の移植方式自体が検査不成立なら、S5を黙って移植の代用にしない。

## 10. 出力・片付け

`results/resp_cifar_ee_0920/` に summary.md、verdict.csv/json、paired.csv、per_seed.csv、per_task.csv、branches.csv、arm_table.csv、secondary.csv、checks.json、provenance_start/end.json、input_manifest.json。unit配列・float32 field・checkpoint・ログはgit外、schemaとhashを残す。

完了時は `CLAUDE.md` §4どおり、git外出力（pycache以外）を `/home/issan/Projects/obsidian-research-data/resp_cifar_ee_0920/` に退避しsource/backup/bytes/SHA256のmanifestをcommit。mainへmerge/push・到達確認後に自分のworktree/branchを削除。共有データ・旧A6/バトル・他セッションの出力は移さない。

**今回のspec納品**: 新規データ・実装はない。文書のcommitをmainへ統合し、専用worktreeを片付ける。実装GO後は同じrun名のworktreeを最新mainから作り直して続ける。
