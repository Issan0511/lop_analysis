# relu_doors_h_ref_0920 — refもR=10で新規実行してH単独を問う

状態: Issaの予測・GOを引き継いだ新規事前登録。追加実装・新規走の前にcommit。
worktree: ~/Projects/claude/wt/relu_doors_h_ref_0920、branch: codex/relu_doors_h_ref_0920。
起点: origin/main b720a7a01618ffbf9457fa5118894f923f4ed6fc。旧登録は変更しない。

## 0. 一行

RL-CIFARのReLU網でH単独が救うか・CH水準にCが必要かを、refとHを同じR=10で新規実行して判定する。

## 1. 出所（既知結果と変更理由）

穴の出所は前登録 `specs/spec_relu_doors_h_0919.md` §1と同じ。
Obsidianの現在地→中心主張v10草案_0916→V10で言えるようになったこと_0917→中心主張v10作業リスト_0917→
Issaの戯言/主張v11を作って見よう.md の第一の柱。「Cは第1層、Hは第2層」の層別対応とCの必要性を埋める。
運用ルール・引用禁止・relu_doors/resp_eeのspec・harness/閾値/メモリ/空虚なS検査のメモを継承する。
本セッションでcontextと共同編集差分を再確認。vault head 0f8a4b04df7d8f0468e9dff70af2d8e8d8d98493、未読peer差分なし。

設計に使った結果は全て既知:
- 親 `results/relu_doors_0919/summary.md`・`verdict.json`：ref/C窓約.113、CH約.987。
  旧ref t1死亡率中央値.9849999547、C0。Cのt50第2層gate=1、CH<1。
  CH seed窓の散らばりから帯を作ったこと、第1層の分類境界を既知ref/Cから作ったことを開示する。
- 前登録 `results/relu_doors_h_0919/summary.md`・`verdict.json`：旧refR20と新実装R20はCSV/状態一致。
  refR10では460項目中193不一致、seed0 t1online .262475 対 .22545625。前走はINAPPLICABLEでH未生成。
  **この事後の検査結果を受けてrefの再利用をやめる新登録**。旧走の停止を取り消さない。
- 前登録の§1に列挙された全既知結果とコード所見も設計背景として引き継ぐ。

C/CHの固定入力: `results/relu_doors_0919/{C,CH}/per_task.csv`。
SHA256: C=7de7284dc8c292a032f750f94cb9f51a8bf0eb1a982529704bd5e2a7dbbfd1dd、
CH=3259e819ec8ad1d06b3bc861f4daa77121dc2acd8d2646422cde9cbaf52467f1。
凍結旧コードと共通host依存のhashを検査し、起動provenanceからdata/subset/seed/slot/optimizerを照合する。
前走のR10の2tasksは新refの本走に継ぎ足さず、初期状態から50tasksを実行する。

## 2. 箱

親spec_relu_doors_0919 §2と前登録§2を継承。CIFAR10、seedごと1200枚、raw=u8/255、
3072–100–100–10、隠れ2層ともclamp(min=0)のReLU、各task独立な10クラス一様乱数ラベル。
Adam lr=.001、β1=.9、β2=.999、eps=1e-8、weight decay=0、状態をtask間で保持。
50tasks×400epochs×75steps、batch16。全腕R10、seed0–9、rawのみ、同一順序・初期化・入力・ラベル・batch系列。
エンジン `src/relu_doors_0919.py` の学習式を維持。H armとrun-id配線は前走の31e4c4aで実装済みだが、H未実行。
Hは**a1とa2の両方**から各seed/unitのEMAを引く。m0=0、beta=.01。
更新前forwardのuncentered clamp(z)のbatch平均をAdam更新後にEMAへ取り込む。
EMAは勾配・optimizerの対象外、train/evalで同じm。raw入力とbiasは自由。

## 3. 腕

| 腕 | C | H | B | 今回 |
|---|---|---|---|---|
| ref | OFF | OFF | OFF | **新規**R10、50tasks |
| H | OFF | ON | OFF | **新規**R10、50tasks |
| C | ON | OFF | OFF | 旧R10をS-reuse通過時だけ再利用 |
| CH | ON | ON | OFF | 旧R10をS-reuse通過時だけ再利用 |

新しいrun idはrelu_doors_h_ref_0920。ref/Hをこの順で直列実行する。新規2腕×10seed。
C/CHの資格不成立なら停止。そこで黙って4腕走へ増やさず、別登録に戻る。

## 4. 登録する読み出し

### 4.1 主量と集約順

\[
W_{a,s}=\frac1{20}\sum_{t=31}^{50}\mathrm{online\_acc}_{a,s,t},\quad
M_a=\operatorname{median}_{s=0}^{9}W_{a,s},\quad
K_a=\sum_{s=0}^{9}\mathbf1[W_{a,s}\ge0.5].
\]

online は **30,000 更新の更新前ミニバッチ argmax 正解率の平均**。1200 枚の新しい乱数ラベルへの学習能力であり、未知画像への汎化精度ではない。比較は同じ seed 内で `d_HR,s = W_H,s − W_ref,s`、`d_HC,s = W_H,s − W_CH,s` を作り、その中央値を主に表示する。**中央値同士の差へ置換しない。** K、全 seed の W、seed の範囲、符号の数と p も出す。

親と同じ副量は `memo_acc` と低下（t1–10 online 平均 − t41–50 online 平均）。主窓を変更する根拠には使わない。

### 4.2 層別量（全タスクを保存、登録した比較窓は固定）

| 量 | 実装上の定義・窓 |
|---|---|
| `dead_frac_l1` | 各ユニットで 1200 枚の診断 gate が全て 0 の割合（ReLU では既存 DEAD_TOL 判定と同じ）。**t1** が層別主量、t50 は補助 |
| `zbar_l1`, `zsd_l1` | 画像平均のユニット中央値、画像 SD のユニット中央値。t1・t50 |
| `zbar_l2`, `zsd_l2`, `sink_ratio_l2` | t1・t50。主に使う深さは既存の **ユニットごとの mean/SD の中央値** `sink_ratio_l2`。`zbar_l2/zsd_l2`（中央値の比）とは別量 |
| `gate_zero_frac_l2` | 1200×100 個の診断 gate で厳密 0 の割合。t1・**t50**。`dead_frac_l2`（全画像で gate 0 のユニット率）と混同しない |
| `bias_over_sd_l2` | **mean_i abs(b₂ᵢ) / median_i sd_x(z₂ᵢ)**。t1・t50。abs(mean bias) ではない |
| `r_a1` | **H を通過した後の a₁** の平均ベクトルのノルム / rms の画像偏差。t1・t50。分母ゼロなら未定義と記録し、0 に埋めない |

依頼文の列例にある `zbar_l1`・`zsd_l1` だけから第2層の深さは得られないが、上記の l2 列は既にエンジンにある。既存列で足りる。EMA 遅れの検査に必要な m₁・m₂・中心化前後の平均は検査用トレース／既存 snapshot から取る。

既存 gate は `z>0` の診断である。訓練は `clamp(min=0)` を使うので **z=0 の autograd の規約との一致を仮定しない**。S-eval でゼロ点を含む配線を確認し、厳密な 0 点の数を報告する。「gate 0」を直ちに訓練勾配ゼロ・永久吸収と言い換えない。

## 5. 登録する判定

以下は前登録§5を維持。機能比較のrefは**本runの新規ref**。
θ1とδCは前登録の既知データ由来の値を固定し、新ref結果で再推定しない。

### 5.1 帯と seed 数の算術

**0.5 は親 spec から指定どおり継承する救済の操作的定義**で、統計から導いた数値ではない。新しい成功閾値を作らず、親との比較可能性を維持する。これ以外の新しい効果の帯・層別分類・検査の許容は以下の算術から決める。

seed の向きについては親と同じ両側厳密二項符号検定、有意水準 0.05。n=10 なら、片側に 9 本以上が並ぶ確率の両側値は

\[
2\{\tbinom{10}{0}+\tbinom{10}{1}\}/2^{10}=0.021484375<0.05,
\quad 2\sum_{j=0}^{2}\tbinom{10}{j}/2^{10}=0.109375>0.05.
\]

従って必要本数は **9/10**。救済本数 K の 9 本規則も親を継承し、この二項分布の境界と対応する。seed 内のタスクや画像を独立標本として n を増やさない。

C の必要性の帯は、既存 CH の seed 別窓 `verdict.json:window.CH` の散らばりから **異なる 2 seed の窓の差の RMS** を導く。h_s=W_CH,s、n=10 として

\[
s_{CH}^2=\frac{\sum_s(h_s-\bar h)^2}{n-1},\qquad
\delta_C^2=\frac{1}{n(n-1)}\sum_{s\ne r}(h_s-h_r)^2=2s_{CH}^2.
\]

正本の全精度値から、`s_CH = 0.000537851600304817`、**`δ_C = 0.0007606370276951454`（0.0761 percentage point）**。√2 は差の二乗和の恒等式から出ており、「SD の適当な N 倍」ではない。本文の丸め値 0.986–0.988 から再計算しない。

これは **既存 CH の seed 差 1 組の RMS を許容損失とする実務上の帯**であり、H の分散や母集団の最小重要差を推定したものではない。その尺度は前登録で Issa が採用済みで、本登録も維持する。H の結果で帯を広げたり狭めたりしない。

対応差 d_HC の推測の不確実性は別に扱う。昇順の対応差を d_(1)…d_(10) として **I_HC=[d_(2),d_(9)]** を使う。独立な seed を標本とする中央値の分布非依存の区間で、連続分布なら被覆率は 1−0.021484375=**97.8515625%**（95%以上を満たす最も内側のこの順位の区間）。同値があれば保守的になる。SD=0 でも同じ順位規則を使う。正規近似・studentized bootstrap・H を見た後の fallback は使わない。

### 5.2 適用条件を先に判定

§7 の必須検査が全て通り、参照 hash が一致し、各腕の raw×seed 0–9×task 1–50 が **重複なく 500 行**あり、online/memo と判定に使う死亡・gate が有限で、設定と登録 commit が一致していること。H の発散、必要列の欠損、seed の欠落、再利用不成立は **`INAPPLICABLE`**。検査不合格を `H_ALONE_FAILS` に数えず、揃った seed だけで主判定しない。有限な診断の比の分母が 0 の場合は未定義を明示し、それだけで機能判定を失効させない。

参照 CH が K_CH≥9、ref と C が K≤1 を満たさなければ **`NOT_REPRODUCED`** とし、本来のラベルは出さない。参照選択の誤りを疑い、別の参照へ黙って差し替えない。失敗もデータを書いた後に停止する。

### 5.3 機能の主ラベル

| 記号 | 判定（適用条件成立後） | ラベル |
|---|---|---|
| **M_H**：H−ref、H 単独で救うか | K_H≥9 | **`H_ALONE_RESCUES`** |
| 〃 | K_H≤8 | **`H_ALONE_FAILS`** |
| **N_C**：H−CH、CH 水準には C が要るか | I_HC の **上端 < −δ_C** | **`C_NEEDED`** |
| 〃 | M_H=`H_ALONE_RESCUES` **かつ下端 > −δ_C** | **`C_REDUNDANT`** |
| 〃 | その他（境界への一致を含む） | **`TIE`** |

M_H には親の詳細分類も併記する: K_H≥9=`RESCUED`、2–8=`SPLIT`、≤1=`COLLAPSED`。従って `H_ALONE_FAILS` は「登録した一貫した救済に届かない」であり、全 seed が床に落ちたことを意味しない。

H−ref の差自体も隠さない。親の `report.py:sign/verdict` と同じ、差が厳密 0 の seed を除いた両側符号検定で、p<0.05 かつ反対符号が1本以下なら `H_HELPS` / `H_HURTS`、それ以外 `TIE` を **副ラベル D_HR** とする。全差 0 の場合は n=0、pは未定義、`TIE`。この副ラベルだけで救済と呼ばない。

`C_REDUNDANT` はこの帯での **非劣性**であり、完全な同等性ではない。H が CH より良い場合も含む。差を検出できなかっただけなら `TIE` のまま。逆に **H が救済していても C_NEEDED はあり得る**。その場合の必要性は「CH 水準を維持するため」であって、「窓0.5を越すため」ではない。主2判定は別々の問いとして報告し、全体の誤り率を制御した一つの検定とは呼ばない。

### 5.4 層別対応の読み（登録）

第1層の「ref 側／C 側」は、既知の t1 の seed 中央値 D_ref・D_C への距離が等しくなる点で分ける:

\[
|D- D_{ref}|=|D-D_C|\quad\Longrightarrow\quad
\theta_1=(D_{ref}+D_C)/2=\mathbf{0.49249997734999995}.
\]

これは死亡率を新しく定義する数ではなく、**既知の死亡側と保護側のどちらに近いか**の境界。D は厳密 gate 0 のユニット率のまま。H の t1 で D_H,s>θ₁ が9本以上なら R1_H=`L1_DIES`、D_H,s<θ₁ が9本以上なら `L1_SAVED`、他（境界同値も含む）は `TIE`。本数は §5.1 の二項算術から出す。

第2層の死亡側は、C の t50 で **`gate_zero_frac_l2 == 1`**（全120,000要素が診断 gate 0）とする。丸めて1にしない。CH は同じ量が1未満かを確かめる。深さ −1.6 や gate 0.9 という親の補助閾値は、新しい主判定に持ち込まない。

**R_LAYER=`LAYER_CORRESPONDENCE`** は次の連言:

1. H の R1_H=`L1_DIES`。
2. C の t1 で D_C,s<θ₁ が9本以上（第1層を保護）。
3. C の t50 で gate₂=1 が9本以上、CH の t50 で gate₂<1 が9本以上。

それ以外は **`LAYER_CORRESPONDENCE_NOT_SHOWN`** とし、どの条件が外れたか出す。2・3 は既知データの確認であり、**新たに試すのは1**。H の第2層が健康なことは要求しない。上流が壊れれば H が正しく働いていても第2層は失敗し得る。

読みは「この箱では C は第1層を保ち、H の追加は第2層の応答保持と対応する」まで。**M_H=`H_ALONE_FAILS`、N_C=`C_NEEDED`、R_LAYER成立**が揃えば「H 単独では足りず、C と H の2枚で救う」という第一の柱を補強する。R_LAYERだけでは機能的救済の必要性を証明しない。H が第1層を保つなら局所対応の予測は外れとして残す。bias/sd・r_a1・深さは機構の記述で、結果を見て追加の成功条件にしない。

## 6. 予測（記名・時刻・追加実装と走の前）

Codex：2026-09-20 00:03:33 JST、追加資料・結果を読む前に/tmp/relu_doors_h_ref_0920_prediction.txtへ記録。
M_H=H_ALONE_FAILS .80（RESCUES .20）、N_C=C_NEEDED .80（REDUNDANT .15、TIE .05）、
R_LAYER=LAYER_CORRESPONDENCE .70（NOT_SHOWN .30）。raw第1層の直接中心化が無いことを理由とし、
下流Hが勾配を変えて間接的に第1層を守る可能性を残す。前走の193/460不一致は既知、H未生成。

Issa：前登録で「①救えない、②必要、③成立する」「GOして実行」。本登録へ同じ3予測を引き継ぐ。
確率は付けない。今回は停止説明と「refも10個で新規実行する案を別登録」に対して「じゃあ新規で」を受領。
記録時刻は上記転記時刻で、発言時刻ではない。追加実装・走の許可を受領済みとして進める。
前回採用済みのC必要性の帯・H両層の範囲は維持。refだけ再利用から新規実行へ変える。
Claude起案予測 .8/.8/.7も既知。独立盲検ではない。INAPPLICABLE/NOT_REPRODUCEDは未採点。

## 7. 検査（変異対照つき）

前登録の検査要求を継承し、S-reuse対象だけC/CHへ変更する。
許容0の検査は0のまま、丸め許容は実際の演算からu=2^-24、γn=nu/(1−nu)で導く。
同じ述語で変異を検出。source mutationはexact-once、状態/入力mutationも件数と変更箇所を保存。
空検査をPASSにしない。チェックを分割実装・実行してよいが、本走は全検査が揃ってから。
修正して再検査する場合は旧失敗を保管、原因・変更・対象commitを開示し、学習中はsourceを編集しない。

| 検査 | 要求 | 検出必須の変異 |
|---|---|---|
| **S-off** | 扉を全部 OFF にした新実装が、別ファイルの親エンジン R と、同一seed・slot配置で共通数値列・最終重み・Adam状態・RNGが bit 一致。正の演算回数と比較要素数を記録 | CをON、seed/ラベル系列をずらす、Adam更新時刻を変えると不一致 |
| **S-reuse** | C/CHは旧起動commit ab983d9802f273af5135e7a3aa3bef55bedc46cfの凍結コードと新実装でraw×seed0–9×R10×2tasks×400epochsを再計算し、保存済みCSVと共通数値列が厳密一致。元コードとP/Adam/EMA/RNGも一致。refは新規R10のため旧R20との一致を要求しない | C→H、seedずれ、cond誤選択、CSV1値改変を拒否 |
| **S-H-route** | Hがraw入力を無加工で使い、両隠れ出力からmを引き、biasを封鎖せず、学習/評価で同じmを使う。初期forwardのz₁はrefと一致。各層を同じ重み・中心化前活性で局所比較し、非零mの引算による次の線形写像の差が−Wmと丸め上界内で一致 | Cを誤ってON、m₁かm₂の引き忘れ、評価時だけm=0、b₁除去 |
| **S-H-EMA** | 下記§7.1の再帰と遅れ上界、非退化fixtureでの0への追随。H単独の学習経路でもm₁/m₂を記録して照合 | β=0、中心化後のaをEMAへ入れる、EMAをforwardより先に更新、片層だけ更新停止 |
| **S-grad** | 実際にoptimizerへ渡すP・勾配グラフを検査し、EMAを定数とする連鎖律と一致。正負zと非一様な上流勾配を含む | mを更新対象Pへ混入、batch meanを勾配つきで引く |
| **S-resume** | Hでタスク1終端ckpt→新プロセス→タスク2が連続2タスクとbit一致。P、Adam m/v/t、**act_stateのm₁/m₂**、RNG、各行、snapshotを比較。m₁・m₂が非零のcheckpointも必須 | m₁・m₂を別々に保存しない／loadを外す／ゼロ化、RNG復帰を外す。設定違いのckptも拒否 |
| **S-graph** | HのCUDA capture前後でEMAを含む状態が復元され、同じslot配置の短いeager経路とbit一致 | capture時のm復元を外す |
| **S-eval** | 保存したfloat32重み・mを**元のR=10配置**でreplayし、記録したmemoと層別列を照合。onlineは短い検査走の各更新前logit・ラベルの独立記録から再集計（終端snapshotからは復元しない）。診断gateと訓練微分はz<0,z=0,z>0で別に確認 | H無しの評価、更新後精度でonlineを記録、l1/l2取り違え、sink_ratioを中央値の比に置換、mを片層だけ無視 |
| **S-diverge** | 1slotへのNaNで他のslotの軌道はbit一致、発散slotを記録して主判定INAPPLICABLE | 発散slotの行を成功扱い、他slotまで停止 |
| **S-verdict** | 合成10seedで§5の全分岐、境界同値、SD=0、0差、K=0/1/2/8/9/10、符号反転、欠測・重複・発散、参照hash違いを検査。期待値は本specの式から手で作る | >と≥の交換、対応差を中央値の差へ変更、TIEをREDUNDANTにする、欠測seedを除外して採点 |
| **S-CLI / provenance / STOP** | CLIとin-processを照合、新run idと登録SHA・起動時git状態を保存。STOPで新規起動と再起動を止める。検査・集計の出力先が新run配下だけであること | 旧EXPERIMENT混入、git状態の終端取得、STOP無視、旧resultsへの出力 |
| **S-collect** | 必須検査名の集合、実行した比較数、全変異の検出数が揃ってからall_pass | 検査1本欠落、空ディレクトリ、古いPASSを混ぜる |


S-offは凍結親Rと新refを**両方R10/raw、seed200–209、2tasks×2epochs**で比較。
通常H検査もseed200–209、同じ短縮箱。S-reuseのみ既知C/CHのseed0–9、2tasks×400epochs。
H0–9はprobeしない。S-reuseのC→H変異はDOORS誤選択の設定照合で拒否し、未知Hを本seedで走らせない。
独立な凍結コードとの比較を自己の二重呼出しで代替しない。
S-costでRSS/VRAM/時間を記録、性能の合否には使わない。

### 7.1 H の平均が 0 の近くにあることの、非空虚な上界

kは更新前の状態、μ_kは固定1200枚の **中心化前** ReLU(a₁の元) の平均ベクトル、μ̂_kはその更新のミニバッチ平均、m_kはforwardで実際に使うEMAとする。q=1−β_H、e_k=μ_k−m_k（Hを通ったa₁の母画像平均）なら、実数算術で

\[
e_{k+1}=q e_k+(\mu_{k+1}-\mu_k)+\beta_H(\mu_k-\hat\mu_k).
\]

従って各ユニットiで、初期化誤差・重み更新による遅れ・標本差を全て含む上界は

\[
|e_{k,i}|\le q^k|e_{0,i}|+
\sum_{j=0}^{k-1}q^{k-1-j}
\bigl(|\mu_{j+1,i}-\mu_{j,i}|+\beta_H|\mu_{j,i}-\hat\mu_{j,i}|\bigr)+F_{k,i}.
\]

Fは浮動小数の演算から積む誤差。float32のunit roundoffをu=2⁻²⁴、γ_n=nu/(1−nu)とし、B=16およびN=1200の平均の誤差を `γ_(B−1) mean(abs(a))`、`γ_(N−1) mean(abs(a))` と最終除算の丸めから評価する。βとqの表現誤差、EMAの乗算・加算、出力の減算の各丸めを絶対値で足し、過去のEMA誤差はqで運ぶ。float64参照側の丸めも同様に足す。演算数と実際の値から算出し、固定atolや安全倍率は入れない。

検査は **2本立て**にする。

1. H単独の短い実際の学習トレースで、raw forwardのaを別に記録し、float64の独立な再帰から上界と残差を計算する。比較対象のm自身から期待値を再構成しない。画像平均とbatch平均を分ける。m更新を各stepから切り離しても同じ検査が落ちることを確認する。
2. **非零で一定の、全画像で同じ中心化前活性**を与えるfixtureでは μ̂=μ、Δμ=0 なので `|e_k|≤q^k|e_0|+F_k` に縮む。K=ceil(log(u)/log(q))=**1656**更新で初期残差はfloat32の相対丸め以下になる。β=0変異には同じ期待β=0.01の上界を適用し、残差μが上界を越えて必ずFAILすることを確認する。正負を含む別のfixtureで勾配・引き算の配線も確認する。

死亡して元のa₁もm₁も0なら平均0は空虚に通るため、1だけでは十分でなく2を必須にする。この上界は **EMAの遅れを許して0の近さを確認する装置検査**で、μ・画像の広がりが急変すれば大きくなる。小さいr_a1を任意閾値で要求しない。実データのt1/t50では、r_a1と分子・分母・mを報告する。平均0そのものを独立した機構の発見と数えない。

## 8. 実行計画

登録commitを固定→追加検査/報告/STOP対応launcherを実装commit→必要検査→ref/H本走→集計。
sourceを変えない分割チェックは段階的に実行してよい。本走の前に全必須名・全変異数・実装SHAを照合する。
run_idと登録SHAをCLIで渡し、結果はresults/relu_doors_h_ref_0920/{ref,H}/のみ。
通常は1process、2腕を直列。合計3,000,000束ね更新（seed別30,000,000）。
親目安なら50–60分、前回refR10短走.81ms/stepを単純外挿すると2腕40.5分＋評価保存。
HはEMA演算を持ち時間が違うので実測で更新する。旧並列親は1腕約105分の実績もあり保証しない。
C/CH資格4本×2tasksと短い変異検査の費用は別計上。

直前にfree/swapon/nvidia-smi/競合を確認。実測RSS/VRAMを使いhost6GiB予備を残す。
共有launch lockで重複起動を防ぐ。nohupログbasenameはlaunch_relu_doors_h_ref_0920.log、ref_raw_seeds0-9.log、H_raw_seeds0-9.log。
branch名の/をbasenameに入れない。PIDだけで成功とせず、実プロセス・start記録・出力先を確認。
STOP=results/relu_doors_h_ref_0920/_launch/STOP。新規起動と再起動の両方を止めるが実行中には効かない。
起動時・初回更新前にgit full status/commit/登録SHA/argv/DOORS/seed/slot/data/subsethash/依存版/GPU/dtype/threads/graphを保存。
終了時gitで上書きしない。resumeは同じcode/configのみ、各起動を追記する。

監視は原則5分間隔。短い検査の想定終了・起動確認・異常時を除き頻繁にpollしない。
本走途中の性能は読まずtask・alive・資源・exitだけ読む。ref完走を見てH開始条件を変えない。
完走後に集計して未再現条件を含めて報告する。失敗seedを捨てずINAPPLICABLE。

## 9. 開示

前走の事後の検査不成立を見て立てた別登録であり、完全盲検でない。Hの結果は未生成・未読。
判定帯とθは既知データ由来。新refのt1/2は前走の短い資格走で既知だが本走は最初からやり直す。
旧結果・前登録は書き換えない。独立監査なし、単一Codexの実装と自己検査。
追加探索・閾値変更・腕追加は本登録に混ぜない。vault本文は今回変更しない。

## 10. 結果の置き場と片付け

results/relu_doors_h_ref_0920/のsummary.mdとverdict.json/verdict.csvを正本とする。
per_seed.csv、layer_table.csv、prediction_score.csv、reuse_manifest.json、checks、provenanceを保存。
ref/Hのper_task.csv、ckpt、snap、histは別ディレクトリ。旧resultsはread-only。
CLAUDE.md §4に従い、追跡外の生データ（__pycache__除く）を
~/Projects/obsidian-research-data/relu_doors_h_ref_0920/へ退避。共有data symlinkは辿らない。
backup_manifest.jsonへsource/backup/bytes/sha256を記録し、移動前後照合してcommit。
worktree内でfetch→merge origin/main→push HEAD:main、remote到達を確認してworktree/自分のbranchを削除。
