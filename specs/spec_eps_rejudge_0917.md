# ε の再判定 — float32 の床を入れて「ε が止めた」を系統ごとに読み直す（保存データの再解析＋分岐点から 1 エポックの ε 対照）

状態: **事前登録**（本ファイルの commit が登録。§1.4 に登録前に見たものを列挙した。それ以外の判定量はまだ見ていない）
作成: 2026-09-17 / 起草・実装・判定: Claude / 依頼: Issa（設計案の題名「εの再判定_float32の床を入れて読み直す_設計案_0917」を指示）
親: obsidian-research `可塑性喪失/spec/εの再判定_float32の床を入れて読み直す_設計案_0917.md`（同日・Claude 起草・走ゼロ）
読み直す対象: `層別キメラでELU崩壊の所在を決める_結果_0914` §5（「止めたのは Adam の ε」）・`第2層のµ₂壁にεとLayerNormを当てる_結果_0916` Part B（ε 介入 `MIXED`）・`第1層の成分上限をELU→ELUに当てる_結果_0917` 範囲（t100 の第1層で $\sqrt{\hat v}<\varepsilon$ がほぼ全座標）
実装: `analysis/eps_rejudge_0917/{kernels.py, l1_branch.py, tables.py, checks.py, verdict.py}`（`src/` は触らない）
branch: `claude/eps_rejudge_0917`

## 0. 問い

**崩壊した ELU の層で更新が止まったのは、勾配が Adam の ε（1e−8）を下回ったからか（EPS）、optimizer に渡った微分が厳密に 0 だったからか（ZERO）。**

## 1. 設計案からの変更（登録前に見つけた前提の穴）

### 1.1 「この箱の ELU」は 3 系統あり、訓練の微分の実装が違う

設計案は、過去の走がすべて autograd の expm1 実装（$z<-16.64$ で訓練の微分が厳密に 0）だという前提で書かれていた。コードを読むと、次の 3 系統に分かれる。

| 系統 | 走 | 計算機・装置 | 第2層 ELU の訓練の微分 | 厳密に 0 になる $z$ |
|---|---|---|---|---|
| **L1** | `mucap_el_0916`（第1層のみ ELU）・`mucap_ee_0917`・`resp_ee_0917`・`l2cap_ee_0917` | white-san CPU・1 スレッド・flush_denormal（EE 系） | autograd: $\mathrm{expm1}(z)+1$（`src/mucap_el_run_0916.py` forward2 + `torch.autograd.grad`） | $z\le-16.635534$（実測・K1） |
| **L2** | `elu_environment_0913`・`layer_chimera_rl_0914`・`relu_gelu_silu_rl_0914`（ext150 含む）・`l2_wall_0916`（全 Part） | white-san RTX 5060 Ti・CUDA graph | **手書きの逆伝播で $e^{z}$**（`src/layer_chimera_rl_0914.py:44-57` の `gate`、`src/l2_wall_0916.py` の `vgradients` は `C.gate`） | $z<-103.97$（float32 の exp の下溢れ・K3） |
| **L3** | `act_chimera_0913` の rlmnist（400 epoch 箱・0914 §5 の表の出所） | GCP c2d-standard-32（AMD EPYC 7B13）CPU・flush なし（VM は削除済み） | autograd: ELU1 は $\mathrm{expm1}(z)+1$、SMINH の深部は $e^{z}$（`src/act_chimera_0913.py:92-94`） | ELU1: カーネル依存で未測定（§1.2）／SMINH: $z<-103.97$ |

- L2 には −16.64 の床が無い。設計案 §2 の「`tinygate` は厳密 0 の下界」は L2 では成り立たず、`tinygate` は「訓練の微分が ε 未満の (ユニット, 画像) の割合」そのものになる。設計案 §1 の「l2_wall の `MIXED` は ZERO と整合」も、L2 の ε 介入は床の無い実装での介入なので読めない。
- 0914 §5 の「80 epoch 箱は t50 でもまだ走行中」は L2、「t24 で凍結・6.09e−34」は L3 の値。別の実装の別の走をつないだ読みだった。

### 1.2 境界の実測（white-san・torch 2.13.0+cu130・`kernels.py` が `kernels.json` に書く）

- **K1** CPU の autograd の係数: 0 になる最大の float32 は −16.635534、その次の float32 で $2^{-24}$。flush の有無で同じ。
- **K2** CUDA（RTX 5060 Ti）の同じ係数: 0 になる最大は −16.982107（CPU と別の丸め）。L2 はこの経路を使っていない。
- **K3** float32 の exp が 0 になるのは $z<-103.972$（CPU flush なし・CUDA。$[-103.97,-87.34]$ は非正規化数）。CPU で flush ありなら $z<-87.3365$。
- **K4** 勾配が 0 のときの Adam の moment（float32）: flush なしでは $m\leftarrow0.9m$ は $k\cdot2^{-149}$（$k\le4$）で止まって 0 にならない。$v\leftarrow0.999v$ も $k\le500$ で止まる。flush ありなら両方とも厳密に 0 に落ちる（`kernels.py` で反復して段数を記録）。
- **L3 の ELU1 の境界**: VM のカーネルは測れない。ただし float32 で $-1+x$ を最近接丸めか切り捨てで返す限り、$x<2^{-25}$ では −1 になるので、**境界は $\ln 2^{-25}=-17.3287$ 以上**（K1 の white-san は切り捨てで −16.6355）。

### 1.3 L3 は白さんで bit 再現できない

`_smoke_act_chimera_0913`（white-san CPU）の ELU1 s0 の task 1 の online は 0.9550625、VM の記録は 0.9543625。LR の決定性検査の t2 の sha256 も両機で違う。L3 は VM が保存した readout（`readout_s*.npz`: ユニットごとの `gbar_l2`・`depth_vel`・`step_num`・`adam_ratio`・`adam_epsfrac`）だけで読む。

### 1.4 登録前に見たもの（開示）

1. K1–K3 の値（§1.2）。
2. 系統表（§1.1）のコード。
3. **L3 の ELU1 s0–s2 の readout を t1–t31 まで**: 上界 $\ln(1200\,\bar g_i)$ が $\ln2^{-25}$ 以上のユニット数、両隠れ層の `depth_vel`・`step_num` の最大、`adam_ratio`・`adam_epsfrac` の中央値。凍結（両隠れ層の移動が厳密に 0 のまま t50 まで）は s0 t24・s1 t18・s2 t30 に始まり、その直前のタスク終端で上界が $\ln2^{-25}$ を超えるユニットは s0・s1 で 0、s2 で 1（上界 −16.65）。凍結中の第2層の `adam_ratio` の中央値は 5.61e−37。
4. L3 の ELU1 s0 の `per_task.csv` 全列（0914 §5 の表の元）。
5. `specs/spec_l2cap_ee_0917.md` §1.1 の表（L1 の ref の $\bar z_2$ は t5/t10/t100 で −8.3/−50.9/−152.9）と、記憶にあった「L1 の t10 の第2層は (ユニット,画像) の 91–96% が訓練の微分 0」「崩壊の時刻は $\bar z_2$ のユニット平均が −16.64 を越える時刻と ±1 でそろう」「t100 の第1層で √v̂<ε がほぼ全座標」。

したがって **B-ELU1（§2.2）は記述であって予測ではない**。L1 の分岐点の 1 エポック解析（C2）、L1 の時系列の凍結と全滅（C1・C3）、L3 の SMINH（B-SMINH）、L2 の ε 腕の診断（D1）はまだ見ていない。

## 2. 解析

### 2.1 A — 系統表（§1.1 を `lineage.csv` に、根拠の file:line 付きで）

### 2.2 B — L3（VM の readout・ELU1 と SMINH・seed 0–2・t1–50）

- 凍結: `frozen(t)` = 両隠れ層の全ユニットで `depth_vel[t]` と `step_num[t]` が厳密に 0。$T_{\rm freeze}$ = t から t50 まで `frozen` が続く最小の t（無ければ None）。
- **ELU1（記述）**: 上界 $u_i=\ln(1200\,\bar g_{2,i})\ge\max_x z_{2,i}(x)$（$\bar g$ は float64 の解析的な $e^{z}$ の平均。$e^{\max}\le\sum e^{z}$）。
  - `ZERO_R`: $T_{\rm freeze}$ があり、$T_{\rm freeze}-1$ の終端で全ユニットが $u_i<\ln2^{-25}$（どの丸めでも訓練の微分は 0。凍結中はパラメータが動かないので境界は凍結期間を通して成り立つ）
  - `ZERO_T`: $\ln2^{-25}$ では言えず、$u_i\le-16.635534$（white-san と同じ切り捨てのカーネルなら 0）
  - `OPEN`: それ以外
  - 凍結中の `adam_ratio` の中央値を $k\cdot2^{-149}/\varepsilon$（K4 の止まった残り）と並べる。
- **SMINH（予測あり）**: 深部の訓練の微分は float32 の $e^{z}$ で、$z>-103.97$ なら 0 でない。下界 $\ell_i=\ln\bar g_{2,i}\le\max_x z_{2,i}(x)$。
  - `TRAVEL`: $T_{\rm freeze}$ が無い（t50 でも隠れ層が動く）
  - `EPS_REAL`: $T_{\rm freeze}$ があり、$T_{\rm freeze}-1$ 以降の全タスク終端で $\ell_i>-70$ のユニットが 1 つ以上ある（その画像の訓練の微分は $e^{-70}\approx4\times10^{-31}$ 以上で、誤差と入力の積が 1e−6 以上なら勾配は float32 で 0 にならない。それでも 1 bit も動かない＝ε と重みの分解能で一歩が消えている）
  - `ZERO_X`: $T_{\rm freeze}$ があり、$T_{\rm freeze}-1$ の終端で全ユニットが $u_i<-103.97$
  - `OPEN`: それ以外
  - 腕の判定: 3 seed が同じならその値、割れたら `SPLIT`。

### 2.3 C — L1（white-san CPU・flush あり）

**C1（時系列・`mucap_ee_0917` の 4 腕 × seed 0–9 × t1–100 のタスク終端・記録の `units.npz`）**
- 全滅: $\mathrm{dead}_i(t)\iff U_{2,i}(t)\le-16.635534$（1200 枚の float32 の $z$ の最大。全画像で訓練の微分 0）。$n_{\rm dead}(t)=\sum_i\mathrm{dead}_i$。$T_{\rm dead}$ = t から t100 まで $n_{\rm dead}=100$ が続く最小の t。$|U-(-16.6355)|<10^{-4}$ のユニット数を「境界ぎわ」として報告（学習のミニバッチの行列積は 1200 枚と bit が違いうる）。
- 凍結: `frozen(t)` = $b_1,b_2$、行ノルム（両層）、$\bar z$（両層）がすべて $t-1$ の終端と bit 一致。$T_{\rm freeze}$ は §2.2 と同じ定義。
- seed ごと: `FREEZE_AT_DEATH`（$T_{\rm freeze}-T_{\rm dead}\in\{1,2\}$。flush ありでは最後の非零勾配の後 m が約 700 更新で 0 に落ちるので、動きは全滅したタスクか次のタスクで終わる）／`MOVES_WHILE_DEAD`（$t\ge T_{\rm dead}+3$ で `frozen` でないタスクがある）／`FROZEN_WHILE_LIVE`（$n_{\rm dead}(t-1)<100$ なのに `frozen(t)`）／`NO_FULL_DEATH`（$T_{\rm dead}$ なし・凍結なし）。複数に当たれば全部書く。
- 腕ごと: `ZERO_NOT_EPS`（8/10 seed 以上が `FREEZE_AT_DEATH` のみ）／`EPS_REAL`（3 seed 以上が `FROZEN_WHILE_LIVE`）／`NOT_REACHED`（6 seed 以上が `NO_FULL_DEATH`）／`MIXED_C1`（それ以外）。

**C2（分岐点・`resp_ee_0917` の `branch_states.pt`・t ∈ {2,5,7,10,15,20} × seed 0–9 = 60 状態）**
- 各状態から task t+1 を自然に 6000 更新走らせ（`resp_ee_0917.train_task` と同じ演算。記録は読むだけ）、座標ごとに 1 エポック目（75 更新）と全タスクの「勾配が厳密に 0 でない更新の数」、勾配の二乗和、1 エポック後に値が変わったか（`moved8`）を取る。
- 同じ状態から、同じラベル・同じ並びで **ε = 1e−30 に替えた 75 更新**を走らせ、1 エポック後に値が変わったか（`moved30`）を取る。
- 座標（$W_1,b_1,W_2,b_2$。$W_3,b_3$ は参考）の分類:
  - **Z**: `moved8` でなく、1 エポック目の勾配がすべて厳密に 0 で、分岐点の $m$ が厳密に 0
  - **E**: `moved8` でなく、Z でなく、`moved30`（ε を外すと動く）
  - **Q**: `moved8` でなく、Z でなく、`moved30` でもない（ε に依らない分解能以下の一歩・momentum の残り）
  - **M**: `moved8`
  - Z のうち `moved30` のもの（ε=1e−30 の腕で上流が動いて勾配が戻った）は別に数える。
- 判定（第2層 = $W_2$ と $b_2$ の座標を 10 seed でプール、止まった座標 Z+E+Q の中の E の割合 $s_E$）: `ZERO_NOT_EPS`（$s_E<0.1$）／`EPS_REAL`（$s_E\ge0.5$）／`BOTH`（その間）。**主判定は t15 と t20**、t10 は報告。第1層（$W_1,b_1$）も同じ表で報告。
- 報告のみ: t10・t15・t20 の状態から ε=1e−30 で task t+1 を 6000 更新まるごと走らせた online 精度と終端の $n_{\rm dead}$（自然継続と並べる）。

**C3（第1層の $\sqrt{\hat v}<\varepsilon$・C1 と同じデータ）**
- seed ごと: `HISTORY_DECAY` = $T_{\rm dead}$ があり、$t\ge T_{\rm dead}+2$ の全タスク終端で第1層の `mhat_rms_l1` が全ユニット厳密に 0（第2層が全滅すると第1層への勾配は厳密に 0。flush ありなら m も 0 に落ちる）／`NOT_DECAY` = 同じ範囲で `mhat_rms_l1` > 0 のユニットがある／`NOT_REACHED` = $T_{\rm dead}$ が無いか 99 以上。
- 腕ごと: 全 seed が `HISTORY_DECAY` なら `HISTORY_DECAY`、`NOT_DECAY` が 1 つでもあれば `NOT_DECAY`、それ以外は `PARTIAL`（`NOT_REACHED` の seed の t100 の第1層は C2 の t15/t20 の第1層の分類で補足する）。

### 2.4 D — L2（`l2_wall_0916` の変種の診断・`diag_variant.npz`）

- **D1**: RL の ELU→ELU と leaky→ELU × ε ∈ {1e−8, 1e−6, 1e−30} × seed 0–2（18 モデル）の t10・t20・t50 の終端で、$U_{2,i}<-103.972$（全画像で exp ゲートが float32 で 0）のユニットの割合。`L2_NO_FLOOR` = t50 で 18 モデルとも 0.1 未満／`L2_FLOOR_REACHED` = それ以外。あわせて `u_tiny_l2`（解析的なゲートが 1e−8 未満の割合）と第2層の `u_adam_epsfrac_l2` を報告。
- **報告のみ**: L1（mucap_ee ref）と L2（l2_wall の none の EE）の $\bar z_2$ の時系列（t10/20/30/40/50、L1 は t100 も）と t41–50 の 1 タスクあたりの変位。§1.4 の 5 で L1 の t100 の値を見ているので判定にしない。

## 3. 予測（Claude・§1.4 に挙げたもの以外は未見）

| 番号 | 対象 | 予測 | 確信 |
|---|---|---|---|
| P1 | C2 第2層 t15・t20 | どちらも `ZERO_NOT_EPS` | 75% |
| P1b | C2 第2層 t10 | `ZERO_NOT_EPS`（外れるなら `BOTH`） | 55% |
| P2 | C1 ref | `ZERO_NOT_EPS` | 35%（`NOT_REACHED` 40%・壁の上端の画像が生き残る） |
| P3 | C3 ref | `HISTORY_DECAY` | 35% |
| P4 | B-SMINH | `EPS_REAL`（凍結はするが、凍結中も訓練の微分が 0 でないユニットがある） | 45%（`TRAVEL` 35%） |
| P5 | D1 | `L2_NO_FLOOR` | 80% |

Issa の予測欄: 空（指示は設計案の題名のみ）。

## 4. 検査（`checks.py` → `checks.json`・判定の前に全部通す）

- **S0 カーネル**: K1 の境界が −16.635534286499023 と一致し、次の float32 の係数が $2^{-24}$。K4 の止まる値（flush なし）と 0 に落ちる段数（flush あり）を記録。変異: 係数を `exp(z)`（手書き）にすると −16.64 に境界が出ないこと。
- **S1 アンカー**: C2 の自然な 6000 更新の終端の `state_sha256` が、`results/resp_ee_0917/runs/s*/prefix.csv` の task t+1 と 60/60 一致。外れた状態は C2 の集計から外し、その t の判定は `NOT_TESTABLE`。
- **S2 分類の空振り対照**: (a) ε=1e−8 を「対照」として同じ 75 更新を 2 回走らせると E が 0 で `moved30`=`moved8` が全座標で一致。(b) s0 の t15・t20 で、第2層の微分だけを解析的な $e^{z}$（手書きの autograd.Function）に替えて 75 更新走らせると、第2層の Z が自然の場合より少なくなる（Z が expm1 の床から来ていて、分類器がその有無を見分けることの確認）。
- **S3 上界と下界の有効性**: L1（mucap_ee 4 腕 × 10 seed × 全タスク終端）で、`gate_mean_l2`（float32 の $e^z$ の平均。flush ありなので $z<-87.34$ の画像は 0）> 0 の全エントリで $\ln(1200\,\bar g)\ge U_2-10^{-5}$ と $\ln\bar g\le U_2+10^{-5}$（許容: float32 の exp の丸めの相対誤差は 1 ulp ≈ $6\times10^{-8}$、平均の float64 の丸めはそれより小さいので、100 倍しても $10^{-5}$ に届かない）。変異: 上界の 1200 を 1 に替えた $\ln\bar g\ge U_2-10^{-5}$ が落ちるエントリが 1 つ以上ある（検査が空振りでない）。
- **S4 全滅の判定の一致**: `l2cap_ee_0917` の ref（mucap_ee の ref と 100 タスク bit 一致）の `dtrain_zero_l2` で、$\mathrm{dead}_i\iff$ `dtrain_zero_l2`=1 が全タスク終端で成り立つ（不一致は数を報告し、境界ぎわのユニットだけであることを確かめる）。

走を殺す assert は出力を書いた後に置く。

## 5. 費用と並列

- C2: 1 状態あたり 6000 + 75（+ t10/15/20 は 6000）更新。1 seed で約 5.4 万更新。EE・flush ありの CPU 1 スレッドで 1 seed 数分。seed を 3 本のプロセスに分けて並列 3（他セッションの CNN 5 本と bench が走っているので上限 3・起動前に `free` と `ps` を見る）。
- B・C1・C3・D は保存ファイルの読み出しだけ。`diag_variant.npz`（480 MB）は必要なキーだけを 1 回ずつ読む。
- 新しい学習走は C2 の 1 エポックの ε 対照と、報告のみの ε=1e−30 の 1 タスクだけ。

## 6. 書き換え方（結果が出たら対象ノートへ 1 段ずつ）

- 0914 §5 の 2（「止めたのは Adam の ε」）: B-ELU1 が `ZERO_R` の seed では「凍結は第2層の全ペアが float32 の expm1 の床より下に入った直後に始まり、optimizer に渡った微分は厳密に 0。6.09e−34 は解析的な $e^z$ の中央値で訓練には使われていない。m に残るのは非正規化数の丸め残りで、ε はその残りを分解能以下にしているだけ」に。B-SMINH が `EPS_REAL` なら「ε が止める」は exp で微分を作る実装（SMINH・L2）の性質として残す。
- 0914 §5 の含意（「到達深さは optimizer の定数で決まる」）: 系統ごとに書き分ける。
- l2_wall Part B: L2 は手書きの exp ゲートで、−16.64 の床は無い。D1 の値を添えて、ε 介入の読みは L2 の実装の中でのものと明記。
- mucap_ee 範囲: C3 の結果。
- V10 作業リスト H4 の ε の段落と仮説 §3.4: 「このプロジェクトの ELU」を「autograd 系統（L1・L3）の ELU」に直し、L2（手書きの exp）には床が無いことを足す。
- 設計案: 状態を「実行済み」にして結果ノートへリンク。

## 7. 範囲

- L3 の ELU1 の境界は測れない。`ZERO_T` の seed は「white-san と同じ切り捨てなら」の条件付き。
- L3・L2 は保存された量だけで読む。L2 の分岐点からの ε 対照は本 spec に含めない。
- C2 の ε=1e−30 は 1 エポックだけの介入で、崩壊を救うかは報告のみ。
- 「ε が止めたか」は ELU（expm1 か exp か）の実装と、Adam の実装（flush の有無で moment の残りが違う）の両方に依る。一般の Adam の性質とは書かない。
