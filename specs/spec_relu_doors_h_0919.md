# relu_doors_h_0919 — H 単独腕で「扉 2 枚」の C の必要性を確かめる（S2）

状態: **事前登録 spec・実装前・未実行。Issa の予測と GO 待ち。今回の作業は本ファイルの commit まで。**
作成: 2026-09-19 / 起草: Codex / 依頼: Issa（S2）
run id: `relu_doors_h_0919` / worktree: `~/Projects/claude/wt/relu_doors_h_0919` / branch: `claude/relu_doors_h_0919`
起点: `origin/main` = `c39767f7b977aa26d79a6e3576e93c48b5c78161`。本ファイルの commit が起案の登録証拠となる。Issa の記入・設計裁定は **H の実装前**に追加 commit し、その SHA を最終登録 commit とする。
型: [spec_relu_doors_0919.md](spec_relu_doors_0919.md)、[spec_resp_ee_0917.md](spec_resp_ee_0917.md)。エンジン: [src/relu_doors_0919.py](../src/relu_doors_0919.py)。

## 0. 一行

**raw RL-CIFAR × ReLU MLP に H 単独を 1 腕足し、H が網を救うか、CH の成績を保つには C も要るか、C と H の層別対応が成り立つかを、既存の ref・C・CH と同じ窓・同じ seed で判定する。**

## 1. 出所（設計に使った既知の結果を開示）

### 1.1 埋める穴

vault `~/Projects/obsidian-research/可塑性喪失/` の `現在地.md` → `主張/中心主張v10草案_0916.md` → `主張/V10で言えるようになったこと_0917.md` → `主張/中心主張v10作業リスト_0917.md` → `Issaの戯言/主張v11を作って見よう.md` が出所。V10 §0.1 の平均位置の幾何と層の区別を、V11 の「第一の柱」の中心化対照へつなぐ。これは H5 の他箱への場の移植や H7 の担い手への直接介入とは別の、**V11 第一の柱の必要側が欠けている穴**である。

既存の梯子は ref → C → CH → CHB。C 単独の非救済と CH の救済から「C だけでは足りず、C に H を足すと救う」は言える。しかし H 単独を落としたため、「H だけでは足りず、C も必要」は未判定。親 spec §3.1 で H を落とした見込みを、ここで反証可能な判定にする。

`運用ルール.md` と `引用禁止.md` に従う。中心化の万能性、無限時間の救済、永久吸収は主張しない。過去の condA オラクル中心化の死亡とタスク可識別性交絡も撤回しない。

### 1.2 数値の正本と既知の状態

以下は **H の結果ではなく、設計時に既知の参照結果**。正本は `results/relu_doors_0919/{summary.md,verdict.json}`（起点 commit に固定）。窓は各 seed の t31–50 online 平均を作ってから seed 中央値。

| 腕・量 | 既知の値 | この spec での用途 |
|---|---|---|
| ref の窓 | 0.1132、窓 ≥ 0.5 は 0/10 | H の救済の比較元 |
| C の窓 | 0.1133、0/10、D1 = `TIE` | C 単独では網を救わない既知の側 |
| CH の窓 | 0.9867、範囲 0.9861–0.9878、10/10 | C の追加効果の比較元・§5.1 の帯 |
| ref / C の t1 `dead_frac_l1` の seed 中央値 | 0.9849999547 / 0 | §5.4 の層別分類境界 |
| C / CH の t50 `gate_zero_frac_l2` の seed 中央値 | 1.000 / 0.566 | 第2層の既知の対照 |
| CH の t50 `sink_ratio_l2` | −0.19071839005（seed 中央値） | 深さだけで死亡と呼ばないための対照 |
| C / CH の t50 `bias_over_sd_l2` | 0.0019391851385 / 0.18114691225（seed 中央値） | B を追加せず、残る bias 経路を報告する理由 |

同じ正本の CHB・CHB0 の救済、CS の非救済、LN の非救済も読んでいる。これらの腕は追加しない。親 spec §1・§9 の `rlcifar_mlp_battle_0918` の事後解析（入力平均、層別沈下、bias の寄与、歪み）と CH の seed 100–102 の probe も既知である。**親 spec の ref「死亡率 1.000」という記述は今回の較正に使わず、正本 R1 の 0.985 を使う。**

層別対照の全 seed 確認だけは、正本に seed 別の列が無いため `results/relu_doors_0919/{C,CH}/per_task.csv` の **cond=raw、seed=0–9、task=1 の `dead_frac_l1`、task=50 の `gate_zero_frac_l2`** を読んだ。C の t1 死亡率は全 seed 0、C の t50 gate は全 seed 1、CH の t50 gate は全 seed 1 未満だった。これは既存ログの事後確認であり、新規の独立証拠に数えない。

V10 系資料で読んだ ELU の成長制約・応答移植・float32 の床・担い手の結果、V11 後半の駆動源と初期 Adam 変位の結果は、機構と機能を分ける背景に使った。それらの数値から本走の閾値を選んでいない。外部論文の新規照合はしていない。

### 1.3 参照の固定

`ref` は親の summary/verdict 上の名前で、`results/relu_doors_0919/ref/` は存在しない。行の実体は親 reporter が指定する **`results/rlcifar_mlp_battle_0918/R/per_task.csv` の cond=raw**。C・CH は `results/relu_doors_0919/{C,CH}/per_task.csv`。旧 reporter を本走ディレクトリに向けて実行してはならない。

| ファイル（repo 相対） | SHA-256（本 spec の起点で固定） |
|---|---|
| `src/relu_doors_0919.py` | `8caf2e842852f934c9118e5528b8213d596b2f43ee66f0fd32e751c877376f5f` |
| `src/relu_doors_0919_checks.py` | `a934df13fffaef569bdccc46010d65d350fe649de57ea231cc57a37711401d56` |
| `results/relu_doors_0919/summary.md` | `b0e64924436761075c9bab8e01302ebfef8e5257ac8a26aaeb664c034e624a88` |
| `results/relu_doors_0919/verdict.json` | `3f062ba3398782c20e1747de9e12d13ff8a38ace7032328ef3dc53840c9c2edc` |
| `results/relu_doors_0919/C/per_task.csv` | `7de7284dc8c292a032f750f94cb9f51a8bf0eb1a982529704bd5e2a7dbbfd1dd` |
| `results/relu_doors_0919/CH/per_task.csv` | `3259e819ec8ad1d06b3bc861f4daa77121dc2acd8d2646422cde9cbaf52467f1` |
| `results/rlcifar_mlp_battle_0918/R/per_task.csv` | `f6d1d0a2a5c9d96dad1a3aa58bbba4c41f0711fe38cd14e1dcd2f429ca0d8328` |

C・CH の起動時 code SHA は `ab983d9802f273af5135e7a3aa3bef55bedc46cf`、ref の起動時 code SHA は `e7069acf585ece6914e2d28b3b14def4e94aff11`（各 provenance）。既存結果への書込みは一切しない。新しい出力の `reuse_manifest.json` にこの表、実際に読んだ provenance の SHA-256、入力データ・subset の hash、元 code SHA、検査記録を持たせる。

## 2. 箱

親 spec §2・§2.1 と同じ。Random Label CIFAR-10、seed ごとに訓練画像 1200 枚を固定、入力 raw=u8/255、各タスクで独立一様な 10 クラスラベル。3072–100–100–10、両隠れ層 ReLU、400 epoch × 75 step = **30,000 更新/タスク**、batch=16、50 タスク。Adam lr=1e−3、β=(0.9,0.999)、ε=1e−8、moment 持越し、WD なし、float32。画像集合・初期化・ラベル・標本順の乱数系列を親と一致させる。

**`src/relu_doors_0919.py` の forward・Adam・EMA 更新・CUDA graph をそのまま使う。** 新 runner へ学習式を書き直さない。C の除去は std 化の除去とは違い、元から raw のまま平均画像を引かないこと。

### 2.1 H 単独の定義

`DOORS["H"] = dict(c=False, h=True, b="none")` とする。H は親どおり **a₁ と a₂ の両方**を処理する。

\[
a_{\ell,k}=\operatorname{ReLU}(z_{\ell,k})-m_{\ell,k},\qquad
m_{\ell,k+1}=(1-\beta_H)m_{\ell,k}+\beta_H\operatorname{mean}_{\rm batch}\operatorname{ReLU}(z_{\ell,k}),
\quad\beta_H=0.01.
\]

m₁・m₂ は seed 別・ユニット別、初期値 0、勾配を通さない。前向きでは更新前の m を使い、前向きで得た **中心化前**の ReLU 出力で、Adam の後に EMA を更新する。評価は最後に保存された同じ m を使い、評価では EMA を更新しない。b₁・b₂・b₃ は全て通常の Adam で自由に更新する。平均除去以外の尺度・affine・bias 操作を足さない。

H は第1層の前活性に直接中心化をかけないが、下流の forward と逆伝播を変えるため、学習後の第1層まで ref と同じとは限らない。**「raw だから H でも必ず第1層が死ぬ」は予測であり、検査の合格条件ではない。** また H は a₂（出力層への入力）も変えるので、本走だけで H の効果を a₁ の中心化だけに帰属しない。

### 2.2 GO 後に必要な最小の実装差分（今回は実装しない）

起点の DOORS に H は無く、`EXPERIMENT` は旧 run id 固定である。将来の実装は (i) H の腕登録、(ii) 学習経路を変えず `--run-id` で provenance の run id を指定する配線（旧 CLI の既定値は維持）、(iii) 新 run 専用の検査・集計・launch に限る。`--out` だけでは provenance が旧名のままなので不十分。下記の実行例の `--run-id` は **追加予定の CLI** であり、現在実行可能という意味ではない。

## 3. 腕

| 腕 | C | H | B | 出所・役割 |
|---|---|---|---|---|
| ref | OFF | OFF | OFF | 既存 R/raw の再利用。H−ref |
| C | ON | OFF | OFF | 既存 C の再利用。層別の対照 |
| **H** | OFF | ON | OFF | **新規 1 腕 × seed 0–9**。`results/relu_doors_h_0919/H/` |
| CH | ON | ON | OFF | 既存 CH の再利用。H−CH |

新規本走は **1 プロセス、R=10、cond=raw、seed 順 0–9**。ref・C・CH の本走を追加せず、再利用資格は §7 の短い照合で示す。B・HB・CB・H の片層版、追加 seed、50 タスクより先はこの登録に含めない。

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

## 5. 登録する判定（結果を見て変更しない）

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

これは **既存 CH の seed 差 1 組の RMS を許容損失とする実務上の帯**であり、H の分散や母集団の最小重要差を推定したものではない。その尺度の採否は §6.3 で Issa に返す。H の結果で帯を広げたり狭めたりしない。

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

## 6. 予測（記名・時刻・実装前）

### 6.1 Codex

**2026-09-19 22:10:25 JST**、指定された追加資料・既存 results の読取り前、H の実装・検査・本走前にチャットと `/tmp/relu_doors_h_prediction_0919_codex.txt` に記録。本節へ同じ確率を転記する。記入前から依頼文の既知数値と Claude の予測は見ており、盲検の独立予測ではない。

| 項目 | 予測 | 確率（残りの配分） |
|---|---|---|
| M_H | **`H_ALONE_FAILS`** | **0.80**（RESCUES 0.20） |
| N_C | **`C_NEEDED`** | **0.80**（REDUNDANT 0.15、TIE 0.05） |
| R_LAYER | **`LAYER_CORRESPONDENCE`** | **0.70**（NOT_SHOWN 0.30） |

理由: raw の第1層への直流成分を H は直接取り除かないため、第1層の t1 死亡が本線。ただし下流の中心化で勾配も変わり、H 単独が第1層を間接的に守る余地はある。§5 の精密な帯・ラベルの定義は記録後に既知参照とコードを読んで具体化した。確率は変更していない。採点は適用条件成立時だけ行い、失効時は未採点。

**起案セッションの Claude（依頼文からの転記）**: `H_ALONE_FAILS` 0.80、`C_NEEDED` 0.80、層別対応 0.70。日付は今回依頼の記録、元の記入時刻は未提示。Codex の上記時刻を Claude に帰属させない。

### 6.2 Issa（空欄のまま停止）

記名:

記入時刻（JST）:

実装前・H の結果未読の確認:

| 項目 | Issa の予測（確率は任意） |
|---|---|
| M_H | |
| N_C | |
| R_LAYER | |

GO（記名・時刻）:

**この欄の記入と明示の GO、その追加 commit が済むまで H の実装・検査の学習走・本走に進まない。** 今回は spec だけ commit し、作業用 branch/worktree を残す。

### 6.3 Issa の判断が要る設計上の分岐点

| 分岐点 | 本案に登録した選択 | 別の選択をするなら |
|---|---|---|
| C の必要性の水準 | CH の seed 差 RMS を帯とし、非劣性＋H の救済で REDUNDANT。「窓0.5の救済」と「CH水準」を分ける | 0.5だけを必要性の対象にする、別の許容損失を採る場合は H の実装前に式ごと改訂 commit |
| H の範囲 | 親の H（a₁・a₂ の両方）を保持。層別の読みは操作的対応 | a₁だけの中心化を問うなら別腕・別登録。本S2へ途中追加しない |
| 既存結果の再利用 | §7で R=20→10 の ref 配置差も確認。不一致ならSTOP | 同じ配置で参照を新規本走する案は費用・run id・登録を別に決め、無断で差し替えない |

H の DOORS 登録と run id の配線はこの選択を実行可能にする最小の作業として予定し、学習式は維持する。追加の腕や独立監査をこの起案作業で自動起動しない。

## 7. 検査（GO 後。変異が同じ検査で落ちることを要求）

元の `src/relu_doors_0919_checks.py` を参照して新 run 専用の `analysis/relu_doors_h_0919/checks.py` に検査を用意する（未実装）。既存 checks の結果をそのまま PASS とみなさない。特に旧 S-H は「β=0ならm=0」と確認するだけ、S-grad の parameter membership は空タプルに対する検査、S-resume の変異は別腕拒否だけであり、**H の壊れ方を全て捕まえる保証がない**。

新しい各検査は正しいコードで PASS、列挙した変異に **同一の合格述語**を適用して FAIL しなければならない。変異はチェック用コピーに exact-once で施し、置換数を検査する。本実装を汚さない。未実行・空配列・空ファイルでの `all([])` は PASS にしない。

| 検査 | 要求 | 検出必須の変異 |
|---|---|---|
| **S-off** | 扉を全部 OFF にした新実装が、別ファイルの親エンジン R と、同一seed・slot配置で共通数値列・最終重み・Adam状態・RNGが bit 一致。正の演算回数と比較要素数を記録 | CをON、seed/ラベル系列をずらす、Adam更新時刻を変えると不一致 |
| **S-reuse** | §1.3 hash・設定・入力を照合。旧起動時commitのコードと新コードで ref/C/CH の最初の2タスクを **400 epoch** で照合し、保存済み行の共通数値列ともbit一致。refの元配置は **R=20（raw,std）**、C/CHはR=10。refはさらに新実装R=10/rawと元R=20のraw行を比較。メタデータのarm名・slot番号・時間だけ比較から除外 | CをHに誤写、seed対応ずれ、別condの選択、参照CSVの1値改変を各々拒否 |
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

全扉OFFを比較する相手は**独立した凍結コード**で、同じ関数の2回呼出しを親対照にしない。bit一致を求める箇所の許容は0。演算順序・slot形状・CUDA scalar除算の逆数乗算を保存する。不一致を相対1e−6等で通し直さない。S-reuseで未保存の状態は元コードとの比較で補い、保存済み行の照合と区別して記録する。

チェック用の新規学習は seed **200–209**（本走0–9および親probe100–102と別）、通常2タスク×2epoch。S-reuseだけは既知のref/C/CHの0–9と元の配置で2タスク×400epochを使う。**Hの0–9を先にprobeしない。** チェック用Hの性能は判定帯や予測の更新に使わない。費用計測はS-costとしてRSS・VRAM・step時間を記録するだけで、科学的なPASSを主張しない。

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

## 8. 実行計画（本セッションでは実行しない）

1. Issaが§6.2を記入し、§6.3を裁定してGO。最終登録commitを固定してから、§2.2の最小変更と新run専用のchecks/report/launchを実装・commitする。
2. 既存resultsの読取りは§1.3に限定した明示パスを使う。報告器には入力・出力先を明示する引数と本走の完全性ガードを持たせ、動作確認は `results/relu_doors_h_0919/_smoke/` の合成データだけで行う。
3. §7の検査を `results/relu_doors_h_0919/_checks/` に出し、必須検査と変異が全部揃ってから本走。再利用に失敗したら§6.3へ戻り、Hを開始しない。
4. **本走1プロセス×H×10 seed**。GPUとhostメモリを競合ジョブと共有するため、起動直前に `free -h`、`swapon --show`、GPU使用量・他ジョブを確認。S-costの実測peak RSS/VRAMと残りの記録領域から1本が収まることを確かめる。hostには既存運用の6GiB予備を残し、共有起動lockで同時launchを避ける。足りなければ待ち、seedを分割してRを変更しない。
5. 起動予定のコマンド（**GO後の実装済みCLI用**）:

   ```bash
   python3 src/relu_doors_0919.py run --arm H --seeds 0-9 --conds raw --tasks 50 --epochs 400 --beta 0.01 --run-id relu_doors_h_0919 --out results/relu_doors_h_0919/H
   ```

   実際はSTOP対応の新launcherから起動する。launcherとrunnerのログbasenameは `launch_relu_doors_h_0919.log` / `H_raw_seeds0-9.log` とし、**branch名の `/` をbasenameへ入れない**。ログディレクトリを先に作る。`nohup` / `setsid` で起動後はPIDだけで成功とせず、実プロセスと最初のstart記録・出力先を即時確認する。
6. **STOPファイルは `results/relu_doors_h_0919/_launch/STOP`**。新規起動・再起動の両方を止める。実行中の学習には効かないことを明記し、設定やcodeを変更したいときはSTOPでlauncherを止め、実行中のプロセスがなくなったことを確認してから編集する。再開は同じ設定・code・登録SHAのckptだけを許す。学習中のcode差し替えはしない。
7. `provenance_start.json` を**プロセス起動時、初回更新前**に永続化する。最終provenanceはそのコピーを引き継ぐ。記録: run id、最終登録SHA、実装SHA、起動時刻、全git status（untrackedを含むsrc/analysis/specs）、argv、DOORS、seed/slot順、data/subset hash、依存版・GPU・dtype・実際のthread数・CUDA graph設定。`git hash` を終了時に取り直して置換しない。再開ごとの起動時状態も追記する。
8. 進捗監視は **5分より短い間隔でpollしない**（起動直後の生存確認、想定終了が早い短い検査、異常時を除く）。本走途中のonline・層別値は読み進めず、task番号・生存・資源・exit statusだけ確認する。完走後に集計・予測採点を行う。途中結果を見たら時刻・読んだ範囲を開示し、判定を変えない。

**費用**: Hだけで50×30,000=**1,500,000束ね更新**（10seed、seed別には15,000,000更新）。依頼文の既知目安は1腕25–30分で、本走の暫定見積りに採る。ただし既存C・CHのprovenanceのwall_clock_sは各約104.8分で、単独所要時間とは区別する（親の計画は複数腕並列）。S-costでstep当たりmsを測り、`ms×1,500,000/60,000`分＋保存・評価時間と既存記録を併記する。競合による延長を隠さず、25–30分を保証しない。親と同じ資源条件なら約105分までかかった実績がある。

再利用検査はref/C/CHの2/50ずつ（旧コード対照も含む）、それ以外は短縮走であり、本走費用と別計上。S-costで得た秒数を並列数で割って短く見積もらない。Hの層別故障や性能を理由とする早期打切り・成功するまでの再試行は行わない。発散・検査不良・運用停止は理由を保存する。

## 9. 開示

- §6.1は追加資料・結果の読取り前に記録したが、依頼文には既知のref/C/CHの数値とClaudeの予測があった。完全盲検でも独立予測でもない。以後に読んだ既存結果は§1に列挙した。**Hの結果は生成も読取りもしていない。**
- Obsidianのcontext APIは複数の正本を同時に返すため、最初のcontext取得で背景ノートも返った。その後に指定資料を順に確認した。共同編集の確認では未読のpeer差分は無かった。vaultの本文・正本・結果ノートは変更していない。
- ref/C/CHを既知の対照として再利用する追補で、未使用seedによる全4腕の独立再現ではない。帯と第1層の境界は**既知データを用いた事後の設計**だが、未知のHに対する判定は実装・本走前に固定する。
- Hは親と同じ二つの隠れ出力への中心化。逆伝播と出力層への入力も変わる。Hが第1層を直接中心化しないことから、その学習軌道まで不変と推論しない。
- 既存refのR=20/raw+stdと本走R=10/rawの差、H未登録、provenanceの旧run id固定、旧checksの弱い変異検査をコード読取りで確認した。これらは§2.2・§7で処理する予定で、今回修正・実行していない。
- **親走は独立監査なし。本specも単一のCodexによる起案・自己点検で、独立監査なし。** 将来の実装検査PASSと独立監査を同一視しない。後に独立監査を行えば担当・対象commit・範囲・結論を別記する。
- Hが救う・Cが冗長になる・層別対応が外れる結果もそのまま登録ラベルで残す。結果を見た追補や感度分析は別登録または事後と明記し、本登録を差し替えない。

## 10. 結果の置き場と片付け（CLAUDE.md §4）

将来の出力はすべて **`results/relu_doors_h_0919/`** の下:

- `H/{per_task.csv,provenance_start.json,provenance.json,ckpt.pt,snap/,hist/}`。
- `_checks/{checks.json,S-*.json,mutation_manifest.json,...}`、`_smoke/`、`_launch/{plan.json,events.jsonl,logs/,STOP}`。
- `reuse_manifest.json`、`summary.md`、`verdict.json`、`verdict.csv`、`per_seed.csv`、`layer_table.csv`、`prediction_score.csv`、図。**新しい数値と判定の正本はこのrunのsummaryとverdict**。そこへ参照の出所・窓・集約順・独立監査の有無も記す。

既存 `results/relu_doors_0919/` と `results/rlcifar_mlp_battle_0918/` は読取り専用。必要な旧snapshotは親 `backup_manifest.json` のsource→backup対応とhashを検証して参照し、旧resultsを作り直さない。

**実験完了時（結果commit後）**は `~/Projects/claude/CLAUDE.md` §4に従う:

1. `git status --porcelain --ignored --untracked-files=all` でgit外ファイルを洗い出す。ckpt、snap、hist、生ログ、検査・スモーク生成物等（`__pycache__`を除く）を `~/Projects/obsidian-research-data/relu_doors_h_0919/` に相対配置を保って退避する。**data等の共有symlinkを辿らず、git追跡ファイルを移さない。**
2. 移動前後でbytesとSHA-256を照合し、`results/relu_doors_h_0919/backup_manifest.json` に **source・backup・bytes・sha256** を全ファイル分記録・commitする。旧runの退避先へ混ぜない。検証前に元ログを消さない。
3. worktree内で `fetch origin` → `merge origin/main` → `push origin HEAD:main`。競合はこのworktreeで解決する。結果commitのremote到達を確認してから共有する。push済みcommitのrebase・squash・amendは行わない。
4. branchがorigin/mainの祖先であること、退避manifestの整合を確認してからworktreeと自分のbranchを削除する。remote branchを作っていた場合だけその枝も削除する。他セッションの枝は消さない。

**今回の終了点はspecのcommit**。実験は未完了なので、mainへの統合・push・退避・worktree削除はまだ行わない。Issaへspecのパス、commit SHA、本案の要約と§6.3の分岐点を返し、予測欄を空けて止まる。
