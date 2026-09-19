# ch_chb_200_0919 — S3: 中心化 ReLU の CH・CHB を 200 タスクへ延ばす（遅延か治癒か）

状態: **事前登録案を commit。Issa の予測・設計判断・GO 待ち。実装・検査走・本走は未着手。**
作成: 2026-09-19 / 起草: Codex / 依頼・設計骨子: Issa、起案セッションの Claude。
run id: `ch_chb_200_0919` / worktree: `~/Projects/claude/wt/ch_chb_200_0919` / branch: `claude/ch_chb_200_0919`。
分岐元: `origin/main = c39767f7b977aa26d79a6e3576e93c48b5c78161`。日付 0919 は worktree 作成日の JST。
親: `specs/spec_relu_doors_0919.md`、vault `可塑性喪失/Issaの戯言/主張v11を作って見よう.md`（第一の柱・中心化の長期限定）。
手続きの型: `specs/spec_relu_doors_0919.md` と `specs/spec_resp_ee_0917.md`。本稿の commit は実行許可ではない。§6 の記入と GO、および別途の実装指示の後に進める。

## 0. 一行

**同じ RL-CIFAR × ReLU MLP の CH・CHB の t50 状態をそのまま t200 まで延ばし、保つ／幅に伴う緩やかな目減り／崩壊を区別する。CH だけの崩壊なら bias を含む B 扉が遅延を分け、両方の崩壊なら B 扉だけでは防げない経路が残る。** `CURED_200` はこの更新予算までの登録ラベルであり、無限時間の治癒ではない。

## 1. 出所（設計に使った既知結果を開示）

指定順に `CLAUDE.md`、vault の `現在地.md` → `主張/中心主張v10草案_0916.md` → `主張/V10で言えるようになったこと_0917.md` → `主張/中心主張v10作業リスト_0917.md` → `Issaの戯言/主張v11を作って見よう.md`、`運用ルール.md`・`引用禁止.md`、手本の 2 spec、指定の memory 4 本を読んだ。追加で以下の正本・実装を読んだ。数値の正本は各 run の `summary.md` と `verdict`、例外の較正計算はファイル・行を以下に明記する。

| 既知の内容 | 出所・格・今回への使い方 |
|---|---|
| condA では両層中心化も 5M で機能が悪化した。オラクル中心化でも死亡し、タスク可識別性の交絡がある | vault `ハブ/論点マップ.md` §22b–23、`引用禁止.md` B。長い地平線を置く動機だけに使う。「centering を完全にすれば防げる」「µ は保護的」とは書かない。condA の数値閾値を CIFAR へ移さない |
| V10 の EE 箱では成長制約・応答移植の接続があるが、cap12 の下でも bias は沈み、観測期間より先は未確認 | 上記 V10 3 ノートと作業リスト A3。有限期間の救命と経路の消滅を区別する動機。MNIST・ELU の応答閾値は移さない |
| ReLU の CH・CHB は 50 タスクで高い成績を保った | `results/relu_doors_0919/{summary.md,verdict.json}`。t31–50 online の seed 中央値は CH 0.9867、CHB 0.9854。登録 N=`NEED_CH`、D3=`TIE`。既知の同じ seed の継続であり、独立 seed の再現ではない |
| H を足すと B 経路が相対的に目立つ | 同正本の B_ROUTE=`BIAS_TAKES_OVER`。**0.0019 → 0.1811 は t50 の C → CH の腕間差であり、CH の時間変化ではない。** CH の t50 `sink_ratio_l2` は seed 中央値 −0.1907。将来の bias 崩壊をこのラベルだけでは証明しない |
| C・CS は崩壊、LN も性能は低いが第2層ゲートは開いている | 同正本。低性能をすべて負側の死と呼ばず、性能と応答を別々に出す動機。新しい対照腕は足さない |
| 親 spec の「−1.6」は既知の崩壊付近の経験値 | `spec_relu_doors_0919.md` §0–1、`results/rlcifar_mlp_battle_0918/R/per_task.csv` の `cond=std, seed=0..9, task=2,3`。本稿 §5.2 で同じ CSV から算術的に再導出した。Gaussian の消灯定理ではない |
| ω が半分になると約 1 pt の目減り | vault `測定/回る速さが幅の正体か_結果_0912.md`、`results/turn_rate_0912/summary.md` K1–K4。5 腕×3 seed の当てはめは `L[pt] = −9.11 − 1.05 log₂ω`、K1=`K1_PARTIAL`。625 更新の leaky の別箱で、長期・ReLU・30,000 更新への定量移送は未検証。§5.3 の桁の目安だけに使う |
| 再開は完了 checkpoint からも可能 | `src/relu_doors_0919.py:run` の docstring と復元処理、`src/relu_doors_0919_checks.py:s_resume`。既存 `results/_checks_relu_doors_0919/S-resume.json` は bit 一致・別腕拒否を確認済み。ただし 3 seed・2 epoch・短い走の検査で、今回の変更の合格証にはしない |
| CHB の λ、実ファイル、費用 | 元 `CHB/provenance.json` の `door_lam=0.1713`。checkpoint の bytes・sha256 は §2.2。旧 `S-cost` と provenance の時間は §8 に開示 |

手本 spec の事後追補・予測の外れ・λ probe も既読である。relu_doors の設計根拠だった raw/std ReLU の層別崩壊、H の EMA、B の b₁ 除去＋b₂ WD を継承する。親の `0.2 sd`・「8 倍の余裕」や固定 `1e−4` は今回の新しい判定・検査の帯には流用しない。λ は介入を同一にするため元の実値を継承する。

## 2. 箱

### 2.1 同じ学習過程を継続する

- Random Label CIFAR-10。seed ごとに訓練画像 1200 枚を固定、画素は raw `u8/255`。タスクごとに各画像へ独立一様な 10 クラスラベル。自然な CIFAR 分類のテスト精度ではない。
- 3072–100–100–10、両隠れ層 ReLU、float32。400 epoch × 75 step、batch 16、**30,000 更新/タスク**。Adam lr=1e−3、β=(0.9, 0.999)、ε=1e−8。W の WD なし。
- CH・CHB とも C: その seed の 1200 枚の平均画像だけを引き、尺度では割らない。H: 両隠れ層の ReLU 出力からユニット別 EMA を引く。β=0.01、EMA に勾配を通さず訓練と評価で同じ値を使う。
- seed 0–9、slot 順も元 provenance と同じ。**1 プロセス=1 腕×10 seed、腕は逐次**。元と同じマシン・GPU・数値環境・CUDA graph・演算順を使用する。精度・flush・batch・epoch・学習率・EMA を途中で替えない。
- t1–50 は既知の接頭部、追加は **t51–200 の 150 タスク**。総 6,000,000 更新/seed、追加 4,500,000 更新/seed、2 腕×10 seed で追加 90,000,000 更新。t51 の task index を 1 に戻さない。

### 2.2 継続元とハッシュ

退避 root は `/home/issan/Projects/obsidian-research-data/relu_doors_0919/results/relu_doors_0919/`。元 manifest は repo の `results/relu_doors_0919/backup_manifest.json`。

| 腕 | 元 checkpoint（上記 root から） | bytes | sha256 |
|---|---|---:|---|
| CH | `CH/ckpt.pt` | 38922344 | `5135c3b634c292177f8d26fb6243ba8a6540b2aa22aaef9f2c9972e7ade9ec97` |
| CHB | `CHB/ckpt.pt` | 38923880 | `8e43503921212d4239818271a9153c3fc4dc64ff0f8e88ccdf4677d52f8f2866` |

2026-09-19 の起草時に両ファイルの bytes・SHA-256 と manifest の一致だけを確認した。checkpoint のロード・学習・新しい評価はしていない。実装時にも再照合する。元 manifest の SHA-256 は `27df5aa8da591e98ff9abd1c211eeaba9a034c8ad0ac24d04136e07d139cf790`。

GO 後、**新しい `results/ch_chb_200_0919/{CH,CHB}/` にコピー**する。元・退避先は変更しない。コピー前後の hash を `source_manifest.json` に保存し、元 checkpoint の不変コピーを `inputs/<arm>/t50_ckpt.pt` に残す。runner が上書きしてよいのは新しい `<arm>/ckpt.pt` のみ。

**checkpoint だけのコピーでは足りない。** 元 engine は過去 histogram を `<arm>/hist/` から読み、毎タスク全履歴を書き直す。CH・CHB の `hist/` 全 10 seed、`snap/` の t00–50、元 `per_task.csv`・`provenance.json` も manifest／git の対応ファイルからコピーし、接頭部の不変コピーを `inputs/` に保存する。hist が欠けると 51 番目の値を 1 番目として保存する危険があるため、hist・acc・snapshot の task 軸の長さと対応を S-history で必ず検査する。データセット symlink はコピー・退避で辿らない。

復元対象は `P`、Adam `m/v/tc`、`act_state.m`（両層 EMA）、各 seed の `g_lab/g_batch`、`alive`、`rows`、task `t=50`。`tc=1,500,000`、10 slot 生存、元の 500 行、meta の全設定を検査する。元 `door_lam` は **CH=0.0、CHB=0.1713**。CHB に provenance と同じ浮動小数点値を渡し、再較正しない。

`src/relu_doors_0919.py` の `run(... n_tasks=200, checkpoint=True, resume=True)` を利用する（現 CLI は再開が既定で `--no-resume` が無効化指定）。checkpoint が無い場合、現 engine は初期化から走れるため、将来の今回用 launcher は**欠落時に必ず拒否**する。引数に `--resume` が存在するとは仮定しない。

`continuity.json` に元ファイル hash、コピー hash、meta、slot 順、data/subset hash、t50 の canonical state hash と **t51 の最初のラベル draw 直前**の canonical state hash を記録し一致を要求する。state hash はキー順・dtype・shape・CPU contiguous bytes を固定し、P・Adam・EMA・乱数・alive・完了 task 数（どちらも 50）・tc を含める。ファイル hash と、シリアライズ方法に依らない state hash を混ぜない。CUDA warm-up/capture 後も同じ状態であること、最初のラベル・標本順が元 RNG の次の draw と一致することを別に確かめる。

## 3. 腕

| 腕 | t1–50 から続ける操作 | 問い |
|---|---|---|
| CH | C＋H、b₁・b₂・b₃ は従来どおり自由 | H で弱めた輸送の代わりに bias 等が長期に効くか |
| CHB | C＋H、b₁ は厳密 0、Adam 更新後 b₂←(1−lr·0.1713)b₂、b₃ は自由 | 同じ B 扉を保つと CH の遅延崩壊を防ぐか |

新規の seed、CHB0、ref、λ 探索、t50 から新たに B を加える介入は無い。**二腕は t50 の重みまで同じではない**（t1 から別の介入を受けた軌道）。`B_ROUTE_DELAY` はこの予防介入履歴の差のラベルであり、t50 以後の b₂ の因果だけを単独同定しない。B は b₁ 除去と b₂ WD の組なので、層の帰属も診断で限定する。

## 4. 登録する読み出し

### 4.1 性能・窓・集約順

主量は元と同じ `online_acc`（30,000 更新の**更新前**ミニバッチ argmax 正解率の平均）。`memo_acc` はタスク終端で同じ 1200 枚を評価した副量。

seed s、腕 a について、`B_as = mean_{t=31..50} online_acc_ast`、`W_as = mean_{t=181..200} online_acc_ast`、`D_as = B_as−W_as`。**先に seed 内で窓平均・差を作り、その後 seed を集約**する。全 20 個の W と D、seed 平均・中央値・範囲・対応差の 95% t 区間（df=9）を出す。SD=0 なら一点区間。新しい集約の偶数標本の中央値は中央 2 値の平均とする。旧 evaluator のユニット中央値（torch の下側中央値）は変えない。タスクを独立な 200 標本として扱わない。窓の途中打ち切り・best window・失敗 seed の除外は禁止。

t51–70、61–80、…、181–200（20 タスク幅・10 タスク刻み）の窓も時系列図に出すが、主判定には最終窓だけを用いる。`RECOVERED_AFTER_CROSSING` は、t51–180 のいずれかの単一タスクで online<1/2 または depth<d_c を満たした後、最終窓は非 COLLAPSE となった seed に併記する。単一タスクの一時越えと最終窓の主判定を混ぜない。

### 4.2 診断の定義（既存列を変えない）

| 列・量 | 定義と扱い |
|---|---|
| `dead_frac_l2` | 元 evaluator の `max_x abs(dphi) < DEAD_TOL` のユニット割合。ReLU の `dphi=(z>0)` では 100 ユニット中、1200 枚すべてで z≤0 の個数/100 |
| `gate_zero_frac_l2` | 元 evaluator の `dphi==0` の割合、分母 1200×100。dead と区別する |
| `zbar_l2`, `zsd_l2` | ユニット別の画像平均の中央値、画像 SD（元の correction=1）の中央値。別々に集約された列 |
| **追加 `depth_ratio_l2`** | **`zbar_l2 / zsd_l2`**。§5.2 の崩壊線を当てる量。元の **`sink_ratio_l2 = median_i(mean_x z_i / sd_x z_i)` とは別**で、後者も残して報告する |
| `bias_over_sd_l2` | 元実装どおり `mean_i abs(b₂i) / median_i sd_x(z₂i)`。`abs(mean b₂)` ではない。b₁ についても記録 |
| `w_norm_l1/l2` | 元実装のユニット別 **非中心化** W ノルムの中央値を保存。これを W̃ と呼ばない |
| **追加 `wtilde_gmean_l1/l2`, `omega_l1/l2`** | 入力方向に平均を引いた各ユニット重み W̃ᵢについて、`N_l=exp(mean_i log ‖W̃ᵢ‖₂)`、`omega_l=lr/N_l`。保存 W は入力×出力の配列なので中心化は入力軸に行う。W は保存 float32、集計は float64。この ω は**角速度の proxy**で、実 Adam 変位の角度を測った値ではない |
| `r_a1` | **H を引いた後**に第2層へ実際に入る a₁ の `‖mean_x a₁‖ / sqrt(mean_x ‖a₁−mean_x a₁‖²)`。ReLU 出力そのものの平均と混ぜない |
| `abs_zbar_l1` | CHB の全ユニットの `abs(mean_x z₁)` の最大。S-pin の単位別検査結果・上界も別保存 |

両層の dead・gate・深さ・bias、b₂ の符号つき平均と絶対平均、z の SD、生の W̃ ノルムを保存し、比の悪化を分母の変化から分けて読む。`p_pos/onesided_frac/skew/eff_rank` 等の既存列も削らない。z==0 の要素数も追加で記録する。**現実装の `clamp(min=0)` の z=0 での autograd と診断 `dphi=(z>0)` は区別する**。旧列を訓練微分そのものへ無断で定義変更せず、厳密な訓練停止を述べるなら S-gate の照合結果を添える。

追加列は新 run の `per_task.csv` の末尾に加える。**t1–50 の旧列は元 CSV の文字列・解析後の浮動小数点ビットとも保持**し、更新経路も不変とする。t1–50 の W̃ は元 snapshot の W から、比は旧列から読む。depth は t51 以後も同じ旧列の保存表現から計算する。過去の gate 個数等の照合には、snapshot の W/b/EMA と元の R=10・slot 順・同じ GPU による `replay_stack` の float32 前活性を使い、保存済み float16 の z を厳密な符号・個数の正本にしない。既存 snapshot で足りない量を未来の値で埋めない。再学習による接頭部の作り直しはしない。診断フックは RNG・P・Adam・EMA を一切進めない。

追加指標の分母ゼロは clip で隠さない。`zsd=0, zbar<0` の depth は −∞、`zbar>0` は +∞、0/0 は `DEPTH_UNDEFINED`。W̃ ノルム 0 の ω は `OMEGA_UNDEFINED` とし、GENTLE/OMEGA 判定は保留できるが、有限の性能が崩壊した判定は残す。数値発散の NaN と数学的な未定義を区別する。旧列内の clamp は再定義せずそのまま残す。

## 5. 登録する判定（算術・優先順・保留を固定）

### 5.1 HOLDS の帯

CH の既知の 10 個の `B_CH,s` を参照とする。`verdict.json` の `window.CH` から、平均 b=0.986881041666、標本 SD s_B=0.000537851600304817。

「下がらない」の片側判定なので参照帯を **[L_H, 1]** と登録する。上端 1 は正解率の定義域。下端は同じ長さの新しい seed 窓の散らばりを含む片側 95% t 予測限界の算術:

$$h=t_{9,0.95}s_B\sqrt{1+1/10}=0.0010340654531,\qquad
L_H=b-h=0.9858469762129.$$

`sqrt(1+1/10)` は新しい 1 窓の分散 s_B² と参照平均の分散 s_B²/10 の和。`t_{9,0.95}=1.83311293265`。**20 task を独立と置いて SD を √20 で割らない。** 同じ seed を継続するため厳密な独立標本の予測区間ではなく、既知の CH のばらつきから固定した操作的な帯である。改善を HOLDS から外さない片側帯を採ることは Issa の判断点（§6.3）。

### 5.2 COLLAPSE の線（−1.6 の再導出）

性能の線 0.5 は、更新前の正解予測と不正解予測の数が等しい `n_correct = n_incorrect` から `n_correct/(n_correct+n_incorrect)=1/2`。偶然水準 1/10 と健康時成績の中点ではなく、依頼で指定された**過半数を正解できるか**の操作的な境界である。

深さの線は未来の CH/CHB から推定しない。既知の親 `R,std` 各 seed で初めて `online_acc<1/2` となり直前が ≥1/2 のタスクを k_s とする。全 10 seed で k_s=3。CSV の定義と一致する `d_st=zbar_l2/zsd_l2` を使い、

$$d_- = \operatorname{median}_s d_{s,k_s-1}=-1.431058520239941,\quad
d_+ = \operatorname{median}_s d_{s,k_s}=-1.7737228999267733,$$
$$d_c=(d_-+d_+)/2=\mathbf{-1.6023907100833572}.$$

タスク境界の二つの観測値のどちらにも寄せない中点、すなわち両端からの最大距離を最小にする線である。旧 spec の丸め値 **−1.6** をこの箱の既知データから再構成した。判定には丸めない d_c を使う。**中点の選択は本稿で初めて登録する経験的較正であり、旧 spec がこの導出を使っていたとの主張ではない。** 正規分布なら Φ(−1.6)>0 なので、この線からゲートの厳密 0 や不可逆吸収は導けない。

seed ごとに `Z_as = mean_{t=181..200} depth_ratio_l2` とし、**`W_as<1/2` または `Z_as<d_c`** を COLLAPSE とする（等号は非崩壊側）。理由を `PERFORMANCE_ONLY / DEPTH_ONLY / BOTH` に分けて出す。`DEPTH_ONLY` を機能的 LoP が証明された、と読まない。初回の一時越えは副報告で、最終窓の線を後から変えない。

### 5.3 GENTLE と OMEGA_ONLY の帯

ω から各層の半減回数を出す。t31–50 と t181–200 の **log ω の平均**を使い、

$$k_{as,l}=\max\{0,\operatorname{mean}_{31:50}\log_2\omega_{aslt}
-\operatorname{mean}_{181:200}\log_2\omega_{aslt}\}.$$

同じ lr なのでこれは W̃ の幾何平均ノルムの成長の log₂。既知の当てはめの 1.05 pt/半減を正解率単位に直し `β_ω=1.05/100=0.0105`。一つの層が律速の場合と二層の寄与を足す場合を事前に両端とした**桁の目安**を

$$g^-_{as}=\beta_\omega\max(k_{as,1},k_{as,2}),\qquad
g^+_{as}=\beta_\omega(k_{as,1}+k_{as,2})$$

とする。`[g^-−h,g^++h]` が OMEGA 型の帯。固定「2 pt まで」や、許容の固定倍は置かない。例: 各層のノルムが √(200/50)=2 倍なら k₁=k₂=1 なので 1.05–2.10 pt、これに §5.1 の標本由来の h=0.1034 pt を加減する。**√t 成長は説明例だけで判定には使わず、実際のノルムで計算する。** 別箱の傾き・proxy・二層の最大/和を置く仮説を今回試すのであり、普遍的な精度則や因果媒介率ではない。係数を今回の精度に回帰し直さない。

「dead・gate が t50 から増えず」は **数え上げとして厳密**に登録する。`n_dead` は 100 個中、`n_gate0` は 120,000 要素中の数。両方について

$$\sum_{t=181}^{200}n_{ast}\ \le\ 20\,n_{as,50}$$

を要求する。これで float32 の割合比較の固定許容を不要にする。旧割合から個数を復元する場合は最寄り整数に戻し、元 float32 丸め幅×分母が 1/2 未満で復元が一意と確かめる。snapshot との照合も行う。**小さな真の増加にも合格を与えない**。揺らぎを許す同等性帯を採るなら、Issa が GO 前に §6.3 で変更を選び、導出を追補 commit する。結果を見て緩めない。

GENTLE の条件はすべて満たすこと:

1. COLLAPSE でなく `1/2≤W_as<L_H`（CH の維持帯より下だが崩壊線以上）。
2. 上記の dead・gate の両個数が増えず、Z_as が定義され `Z_as≥d_c`。
3. `D_as≤g^+_as+h`。目減りは**自分の**既知窓から測る。CHB が t50 時点ですでに CH より少し低い水準だったことを、新しい時間劣化に加算しない。D≤0 なら `BASELINE_OFFSET` を併記し、ω 型とは数えない。

GENTLE のうち **`D_as>h`、`max_l k_as,l>0`、`g^-_as−h≤D_as≤g^+_as+h`** を満たす seed に `OMEGA_COMPATIBLE` を付ける。減ったとは言えないものを、ω の説明成功へ数えない。

### 5.4 seed の三型と腕の判定

優先順は **完全性 → COLLAPSE → HOLDS → GENTLE → 保留**。HOLDS は W が §5.1 の帯内で、Z が定義され深さの崩壊条件が偽であること。HOLDS における dead・gate の増加は別に `RESPONSE_ERODING` と記録し、隠さない。

三型は数学的に全事象を覆わない。帯の外で gate が増える、ω からの帯を超えて下がる、0/0 で必要な深さが不明、等は **`UNCLASSIFIED`** とし、`RESPONSE_ERODING / EXCESS_DROP / DEPTH_UNDEFINED / OMEGA_UNDEFINED` の理由を出す。上から COLLAPSE に押し込まず、下から GENTLE に吸収しない。

各腕で 10 seed 中 9 本以上が同じ型なら腕ラベル HOLDS/GENTLE/COLLAPSE、型が混じるが 9 本以上が HOLDS または GENTLE なら `HOLDS_OR_GENTLE`、その他は `SPLIT`。9 の根拠は二項の算術: `P_{p=1/2}(K≥9)=(10+1)/2^10=0.0107421875`、K≥8 なら 0.0546875。2 腕に片側 α=0.05/2 を配ると最小の合格数は 9。p=1/2 は記述上の多数支持基準であり、seed の交換可能性まで保証する検定とは言わない。

### 5.5 二腕の組合せ（主ラベル）

seed を対応させ、各 seed の **二腕の組**を下表へ分類する。主ラベルは同じ組が 9/10 以上に成立したときだけ付ける（異なる seed を都合よく合わせない）。残りの seed と腕別型を必ず同じ表に出す。

| CH | CHB | 組のラベル・読む範囲 |
|---|---|---|
| HOLDS または GENTLE | HOLDS または GENTLE | **CURED_200**。この箱・200 タスクまで大きな崩壊を避けた |
| COLLAPSE | HOLDS または GENTLE | **B_ROUTE_DELAY**。B 扉の有無と延長時の崩壊が分かれた。bias 比と符号・gate を併記 |
| COLLAPSE | COLLAPSE | **OTHER_ROUTE**。登録した B 扉の閉じ方では防げない。b₂ WD は完全除去ではないため、残存 bias・歪み・別経路をこのラベルだけで区別しない |
| HOLDS または GENTLE | COLLAPSE | **REVERSED_B_EFFECT**。CHB だけが崩れる想定外の向き |
| 上記以外 | | **INCONCLUSIVE**（型の隙間・seed 間混在を含む） |

`CURED_200` の下で、少なくとも 1 seed が GENTLE になった腕の集合を J とする。**J が空でなく、すべての a∈J について OMEGA_COMPATIBLE が 9/10 以上**なら主ラベルに **OMEGA_ONLY** を併記する。両腕とも HOLDS のみ、または少数 seed の GENTLE だけなら付けない。`OMEGA_ONLY` は「登録した幅型の帯と応答条件に整合」の略号で、未測定経路の排除や ω への因果介入の代用ではない。

完全性: 2 腕×10 seed の全 t51–200、元の t1–50、checkpoint・設定・検査証跡が揃わなければ主判定 `INAPPLICABLE`。STOP・計算障害は再開して補える。NaN/Inf の数値発散は `NUMERICAL_FAILURE` を記録し、死亡 seed や精度 0 として補完しない。データを保存してから停止し、有効 seed だけへ黙って分母を減らさない。

## 6. 予測（記名・時刻・実装前）

### 6.1 起案セッションの Claude（依頼文から転記）

依頼文で与えられた予測。**原記入時刻は依頼文に無いため不明**。2026-09-19 に Codex が受領・転記した。50 タスクの既知結果を踏まえた延長の予測であり、50 タスク走以前の予測ではない。

| 主張 | 確率 |
|---|---:|
| CURED_200 かつ OMEGA_ONLY | 0.55 |
| B_ROUTE_DELAY | 0.30 |
| OTHER_ROUTE | 0.15 |

数値予測: CH の t181–200 は t31–50 から 1–2 pt 以内の低下。元予測は他の分岐に確率を配っていないため、その分岐が出たら外れとし、後から再配分しない。

### 6.2 Codex（2026-09-19T22:10:31+09:00）

**CLAUDE.md と git 配置を確認した後、vault・既存 spec・summary/verdict・CSV・checkpoint を読む前、実装前**に `/tmp/ch_chb_200_0919_codex_prediction.md` へ保存した予測を変更せず転記する。依頼文の既知結果と Claude の予測は既読。資料を読む前でも、依頼文の結果まで未読とは言わない。

| 排他的な帰結 | 確率 |
|---|---:|
| CURED_200 かつ OMEGA_ONLY | 0.45 |
| CURED_200、OMEGA_ONLY なし | 0.10 |
| B_ROUTE_DELAY | 0.27 |
| OTHER_ROUTE | 0.13 |
| それ以外・判定保留（逆向き非対称、三型の隙間、検査不成立を含む） | 0.05 |

数値予測: CH の t181–200 の平均 train accuracy は CH の t31–50 より **0–2 percentage points 低い区間に入る確率 0.60**。CHB の低下は CH 以下と予測する確率 **0.65**。ここで train accuracy は主量 online_acc、比較する平均は先に seed 内窓を作った 10 seed の平均。資料を読んでから確率を動かしていない。

採点: 主ラベルは登録分岐への一致、確率は上の 5 分岐の Brier score、数値区間は `0≤mean_s D_CH,s≤0.02`、二腕の低下順は `mean_s D_CHB,s≤mean_s D_CH,s`。実装不能等で endpoint が無い場合、数値予測は未採点とし当たりへ数えない。

### 6.3 Issa（**空欄で停止**）

- 記名・時刻（JST）:
- CH の型 / CHB の型:
- 主ラベル / OMEGA_ONLY の有無:
- CH の t181–200 の低下予測:
- 設計分岐の判断（下表）:
- GO（対象 spec commit）:

**この欄を Codex が代筆・推測して埋めない。今回の仕事は spec の commit まで。** Issa の予測、GO、別途の実装指示の後に実装・検査・本走へ進む。GO 前の修正は追加 commit とし旧案も残す。延長結果を一行でも見た後は本稿の閾値・ラベルを変更しない。

| Issa の判断が要る分岐 | 本稿の具体案 |
|---|---|
| 「CH の帯の内側」を片側にするか | §5.1 の [L_H,1]。改善も HOLDS。厳密な両側帯が意図なら、GO 前にその算術と上振れ分岐を追補する |
| 「dead・gate が増えず」をどこまで厳密に読むか | §5.3 の個数で増加 0。微小な増加も UNCLASSIFIED にする。許容する場合は接頭部由来の帯を GO 前に導出・凍結する |
| 別箱の ω 則の移送と二層の扱い | 0.0105×半減回数、最大～和の帯。OMEGA_ONLY は機構確定でなく整合ラベルとする |
| 深さだけの COLLAPSE を主ラベルに含めるか | 依頼どおり含めるが DEPTH_ONLY を必ず併記。−1.6 は §5.2 の中点 −1.602390710…を使用 |
| 三型の隙間・逆向き・seed 混在 | UNCLASSIFIED、REVERSED_B_EFFECT、INCONCLUSIVE を先に用意し、9/10 の同じ seed 対で主ラベルを付ける |
| GPU 占有時間と独立監査 | 2.5–3 h は期待値、旧並列走の時間をそのまま使うと約 10.5 h。§8 の占有枠を確保し、独立監査を付けるか選ぶ |

## 7. 検査（将来の実装後に行う。今回の合格宣言ではない）

検査・smoke は `results/ch_chb_200_0919/checks/` に限定し、本走ディレクトリに集計の空打ちや `--partial` を向けない。元の検査を import すると旧 `results/_checks_relu_doors_0919` に書くものがあるため、今回用の明示 `--out/--src` と出力先ガードを実装する。短縮学習には未使用 seed **200–209** を使う。元 probe の 100–102、元 cost 検査の 100–109 と混ぜない。

本物の経路が PASS し、表の**各変異が対応する検査で FAIL** したことを記録する。検査関数自身を別の定数と比べるだけ、入力の無い all([])、呼ばれない assertion、非ゼロでない摂動は合格ではない。変異が検出できないなら検査を修正し、主 endpoint を緩めない。比較前の生値・誤差上界・不一致・変異の差分を保存してから assert する。

| 検査 | 実経路での要件 | 落とす変異 |
|---|---|---|
| S-source | §2.2 の全入力ファイルの source/backup/bytes/hash と設定。checkpoint t=50・tc=1.5M・500 行・全 slot 生存 | checkpoint 1 byte 改変、別腕・別 seed、λ を隣の表現可能値へ変更、checkpoint 欠落で初期化から進む |
| S-continuity | ファイル hash 一致に加え、load 前の期待 state と warm-up 後/t51 draw 前 state の canonical hash 一致。次のラベルと最初の標本順が独立 RNG clone と一致 | Adam m/v/tc を初期化、EMA を 0 にする、RNG を一回余計に進める、t51 を t1 に戻す、capture の状態復元を省く |
| S-resume | **CH と CHB の両方**で同じ 2 epoch×3 タスクを uninterrupted と「完了した t1 checkpoint → t3 へ延長」で比較。R=10・同じ graph。全数値旧列、終端 P/Adam/EMA/RNG の bit 一致 | moment・EMA・RNG の各復元を一つずつ落とす。別腕拒否だけを唯一の変異にしない |
| S-history | 接頭部の旧列セル・snapshot は bit 一致。hist 50 点と acc 50 点が t1–50 に対応し、t51 追加後は両方 51 点。新規データで t1 を上書きしない | hist をコピーしない、1 点欠損、task 軸を 1 ずらす、旧 CSV を別 seed へ差し替える |
| S-observer | 無変更の frozen engine と追加診断 ON/OFF で、短縮継続の旧列と全学習状態が bit 一致。**t1–50 の旧 CSV 500 行×2 腕全列**は元セルと一致 | logger から EMA を更新、乱数を消費、旧列の値を再丸め、層を入れ替える |
| S-columns | 保存 W と z を使う独立 float64 再計算で W̃・ω・depth・bias・r を照合。非同値な数値を持つ二層/二 seed の fixture も通す | W の中心化を省く、入力軸を取り違える、lr を掛けず割る、layer1 を layer2 に流す、mean(abs b) を abs(mean b) に替える、sink_ratio を depth_ratio に流す |
| S-gate | dead/gate 個数を独立の bool 集計で再現。z<0・z=0・z>0 の点と、一部画像だけ正のユニットを明示して旧診断と autograd を照合・区別 | dead と gate を同一列にする、z=0 の比較符号を替える、分母を 100 と 120000 で取り違える |
| S-H | 元の β と未中心化 ReLU 出力を用いる EMA 漸化式。train/eval が同じ m を読む。m を外すと独立に計算した Wm だけ前活性が変わる | β=0、中心化済み出力を EMA に入れる、eval だけ m を更新、m 経由で勾配を通す |
| S-B / S-pin | b₁ は厳密 0、b₂ は Adam 後の指定 WD、b₃ 無傷。CHB の全ユニット・全 t51–200 の平均位置が下記丸め上界内に釘付け | b₁ に非ゼロを注入、λ=0、WD 対象を b₃ に変更、C を外す。期待値側を同時に変えない |
| S-verdict | 合成 10 seed の全三型・境界等号・各保留理由、両腕の全組合せ、8/9/10 支持、NaN/欠損・片腕のみ完了を期待ラベルと照合。d_c、h、g を独立算術で再計算 | < を ≤ に変更、窓を t180–199 にずらす、median と mean を交換、seed の対応をずらす、NaN を 0 として数える、8/10 を合格 |
| S-output / S-launch | 新 run id・出力先限定・起動時 provenance、STOP、異常終了、ログ作成・実 PID・heartbeat の初動確認 | 旧 results へ向ける、完走時 HEAD を記録、親 git_hash を新起動 hash とする、STOP を無視、basename に `/` を入れる |
| S-cost | 単独 R=10 の速度・CPU peak RSS・GPU peak memory・snapshot 1 task bytes を測り、後半の診断/保存を含めて外挿する（合否の科学判定には使わない） | GPU 利用者がいる fixture/状態では起動拒否。PID が出ただけを起動成功とする変異を S-launch で検出 |

**丸めの算術**: bit 比較できる学習状態に許容を置かない。独立演算との比較は精度 u と演算数 n に対する `γ_n=nu/(1−nu)`、積和の絶対値の和、保存値の丸め幅から列別に伝播する。比は分母の区間が 0 を跨ぐと比較未定義にし、相対固定 1e−6 で押し通さない。標本分布の幅と数値丸めを混ぜない。

**S-pin の上界**: 元データと C の仕様から独立に作った期待入力 X_c の残差平均 μ_res を使う。b₁=0 なら厳密算術では `zbar_i=w_i·μ_res`。float32 の積和誤差は画像ごと `γ_3072 Σ_j|w_ij x_c,xj|`、画像平均の誤差は `γ_1200 mean_x|z_xi|` で抑える。これらと残差 `|w_i·μ_res|`、上界計算自身の float64 丸めを足した B_i に対し `|zbar_i|≤B_i` を要求する。**μ_res は変異側の実入力から作り直さない**（C を外しても通る恒真チェックを避ける）。変異では独立な期待中心化入力と実入力の差も検出する。`|zbar|<1e−4` の固定値や `||w||` の増大そのものによる不合格を使わない。

`checks.json` は必要な検査名と変異名の明示集合に対し件数・PASS・各 mutation の FAIL を照合して `all_pass` を作る。既存検査の成功を転記しただけでは今回の合格にならない。**独立監査は本起草時点では未実施**。同じ担当の再計算を独立監査と呼ばない。

## 8. 実行計画（GO と別途指示の後）

1. Issa が §6.3 を記入し、設計判断・GO を commit に結びつける。必要な追補を先に commit。その後に実装・検査を行い、全検査を記録したコードを commit する。起動時点の spec/実装 commit を固定する。
2. 入力 manifest を照合し、新 run へコピー。追加列と閾値入力を既知接頭部だけから構成して `registration.json` に保存。h・d_c・係数・窓・集約順・予測の hash が本稿と一致しなければ起動しない。
3. `src/relu_doors_0919.py` の resume を使う今回専用 launcher で **CH→CHB の順、並列数 1**。seed 束ね数 R=10 は変えない。同じマシンの他の GPU 走がある間は検査 GPU 走も本走も起動しない。他セッションのプロセスを停止しない。起動前に `nvidia-smi`・プロセス一覧・`free -h`・`swapon --show`、入力データの存在を確認する。共有 lock を取り、非協調の他ジョブも検出する。
4. 依頼の速度見積もり **50 task/腕=25–30 分**なら、追加は `(150/50)×2×(25–30)=150–180 分`、**2.5–3 時間**。ただし元 provenance は CH/CHB とも wall_clock 約 6,288 秒/50 task、旧 S-cost は 64.2/103.5 分相当で、この値では 150 task×2 が約 **10.5 時間**。元の並列・GPU 競合を含み得る時間であり単独性能とは同一視しない。単独 S-cost で見積もりを更新し、占有枠に収まらなければ待つ。遅さを理由に task/seed を減らさない。
5. 元 manifest では CH・CHB 各約 0.935 GB、うち t00–50 snapshot 約 0.891 GB。線形外挿なら 200 task の二腕で約 7.2 GB、これに接頭部の不変コピー約 1.9 GB、checks と一時書出しを加える。実際の空き容量と S-cost の 1 task bytes から必要量を計算する。旧 peak RSS 1.94 GB/GPU 0.42 GB は参考で、履歴長による増分も見積もる。メモリ不足で R を割らない。
6. `results/ch_chb_200_0919/STOP` を launcher が毎起動前、runner がタスク境界で確認する。STOP 時はそのタスクのデータ・atomic checkpoint を保存して停止し、次腕を起動しない。source を変更するときは STOP→全今回プロセス停止確認→編集・検査・追加 commit→明示再開。変更の理由と再開時 state hash を保存する。
7. ログは **basename に `/` を含めない** `logs/ch_chb_200_0919_CH.log`、`logs/ch_chb_200_0919_CHB.log`。log directory を先に作り、launcher の構文確認後 `nohup` で起動。直後に実 PID・コマンド・ログの開始時刻・provenance・heartbeat を照合する（PID 表示だけでは成功としない）。以後の監視は原則 5 分以上あけ、task/状態・使用メモリのみを読む。live online 値・途中 verdict を人が読む運用をしない。
8. **provenance の git 状態は各起動の開始時に採る**。元 engine は再開時 `git_states` を連結し先頭を top-level `git_hash` に使うので、そのままだと親走の hash が見える。今回の `provenance.json` は `run_id=ch_chb_200_0919`、起動時 git_hash/dirty、登録 commit、engine source hash、親 provenance/ckpt hash、resume history、環境・threads・GPU・データ・slot・λ を明示する。親の記録は `parent_provenance` として保存し、完走後 HEAD で置換しない。各 restart の起動記録を別ファイルにも残す。
9. 両腕が揃った後に明示 `--src results/ch_chb_200_0919` で集計し、`summary.md`・`verdict.json`・`verdict.csv` を正本として固定する。主判定、全 seed 型、理由、予測採点、検査・監査の有無を先に書き、事後解析は別節へ置く。結果を見て追加腕や延長を自動で起案・実行しない。

## 9. 開示

- 今回は **spec だけ**を作成した。新しい学習、checkpoint ロード、GPU 検査、t51 以後の観測はゼロ。Codex の予測は資料閲覧前に固定したが、依頼文の既知結果に条件付いている。起案 Claude の時刻は不明、Issa は未記入。
- §1 の既知結果に加え、起草時に旧ファイルだけで行った算術は §5.1 の CH 窓の平均/SD/帯、§5.2 の R,std t2/t3 の較正、§2.2 のファイル hash/bytes と §8 の manifest 容量集計。これらは **S3 の事前較正、親走については事後・登録外**。親の判定を更新しない。
- 較正の固定出所: `results/relu_doors_0919/verdict.json` SHA-256=`3f062ba3398782c20e1747de9e12d13ff8a38ace7032328ef3dc53840c9c2edc`（`window.CH`）、`results/rlcifar_mlp_battle_0918/R/per_task.csv` SHA-256=`f6d1d0a2a5c9d96dad1a3aa58bbba4c41f0711fe38cd14e1dcd2f429ca0d8328`（R,std、seed 0–9、t2/t3、online/zbar/zsd）、`results/turn_rate_0912/summary.md` SHA-256=`9105efa9b7a0b8025baa02d5e16514a78404c4df3184c5be95477a838afc83c0`（K4）。読んだ engine の SHA-256=`8caf2e842852f934c9118e5528b8213d596b2f43ee66f0fd32e751c877376f5f`。
- 既知の t1–50 を使う同一 seed 継続なので未使用 seed の検証ではない。200 task までを登録対象とし、より長い時間・他の箱・一般的な中心化手法へ広げない。condA の反例は撤回しない。
- CH と CHB の差には両 bias の操作と t1–50 の履歴差がある。深さは応答の十分統計量ではなく、OMEGA_ONLY も独立した ω 介入の結果ではない。`CURED_200` と呼んでも、GENTLE の性能低下や少数 seed の崩壊・応答侵食は明記する。
- relu_doors 本走は **独立監査なし**。本稿も独立監査なし。S3 の将来の走を独立監査なしで報告する場合は、summary 冒頭にもそのまま書く。チェックの自己実行だけで格を上げない。
- vault の read-only connector で現行文書と共同作業差分を確認し、未取込の peer 差分は無かった。vault の本文・既存登録済み results は変更していない。今回の実験以外の worktree・branch は触らない。

## 10. 結果の置き場と片付け（CLAUDE.md §4）

正本は新しい `results/ch_chb_200_0919/` のみ。予定構成:

```text
results/ch_chb_200_0919/
  inputs/{CH,CHB}/             # 不変の t50 checkpoint・親 CSV/provenance・接頭部
  source_manifest.json         # 元 manifest と全コピーの照合
  registration.json            # 凍結式・入力 hash・既知較正値
  checks/                      # smoke・変異の証跡・checks.json・費用
  {CH,CHB}/                    # per_task.csv・ckpt.pt・snap/・hist/
    provenance.json
    continuity.json
  logs/
  summary.md
  verdict.json
  verdict.csv                  # 全 seed の型・理由・組合せ・主ラベル
  backup_manifest.json
```

判定と compact な証跡・provenance を commit し、大きい checkpoint/snapshot/hist・検査 raw・生ログは `~/Projects/obsidian-research-data/ch_chb_200_0919/` へ元の相対パスを保って退避する。`git status --porcelain --ignored --untracked-files=all` で漏れを確認し、`__pycache__` と symlink を除外して **共有 data/ を辿らない**。`backup_manifest.json` は各実ファイルの **source・backup・bytes・sha256**、run id、退避時刻を記録し、コピー先で検算して commit。元 relu_doors の manifest/退避を差し替えない。

**今回は spec commit で停止するため、実験は未完了。worktree と branch を保持し、push・main 統合・削除は今回行わない。** 結果 commit が済んだ将来の実行セッションで、その日のうちに CLAUDE.md §4 の順序どおり、退避→manifest commit→fetch/merge origin/main→push HEAD:main→取り込み確認→今回の worktree/branch の削除を行う。push 済み commit を rebase・squash・amend しない。provenance の git_hash を失わせない。
