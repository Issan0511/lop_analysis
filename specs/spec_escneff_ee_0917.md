# 残り 0.11 は移植先の「逃げ」か（再登録）— escape_ee_0917 の 12 腕を未使用の seed 10–19 で・操作の確認は $\bar n_{\rm eff}$

状態: **事前登録**（本ファイルの commit が登録。seed 10–19 では、どの腕もまだ走っていない）
作成: 2026-09-17 / 起草: Claude / 依頼: Issa「前提条件を平均の登りではなく n̄_eff に替えて、今回使っていない seed（10–19）で同じ腕を回し直す」
前の走: `specs/spec_escape_ee_0917.md`（seed 0–9・主 `ESCAPE_NOT_MANIPULATED`）
実装: `src/escneff_ee_0917.py`（`src/escape_ee_0917.py` の薄い包み）・`analysis/escneff_ee_0917/{checks.py,verdict.py,launch.sh}`
branch: `claude/escneff_ee_0917`

## 0. 一行と問い

escape_ee_0917 では、respdyn の主腕の移植先（健康な t2 の網）の成長を t2 の値で止めると、応答の床の上の残りが 0.112 → 0.066 に下がった。しかし、登録した操作の確認（網自身の第2層の平均前活性の登り）が成り立たず、「残り＝移植先の逃げ」には答えていない。事後に見ると、塞ぎ方を変えた 5 腕の学習能力は、平均の登りではなく、**課題中の $\bar n_{\rm eff}$**（第2層の訓練の微分を運ぶ実効的な画像数・ユニット平均）の順に並んだ。

**同じ 12 腕を、同じコード（escape の runner をそのまま import）で、未使用の seed 10–19 に当てる。変えるのは判定の 2 か所だけ。**

1. **操作の確認 (C)** を $\Delta n=\bar n_{\rm eff}(\text{S2dyn\_10r})-\bar n_{\rm eff}(\text{S2dyn\_c12})$ に替える。
2. **並び**を「5 腕の $\bar n_{\rm eff}$ と E の順位相関」に替える（平均の登りの順位相関は報告のみ）。

さらに副判定を 1 つ足す: **seed 間で $\Delta n$ が大きいほど $\Delta R$ が大きいか**（傾き）。

- **Q1（主）**: 両層の上限（c12）で移植先の微分の担い手が減ったとき（C）、床の上の残りは下がるか。0 まで下がるか。
- **Q2（並び）**: 塞ぎ方を変えた 5 腕で、$\bar n_{\rm eff}$ の順に学べるか。
- **Q3（傾き）**: seed ごとの担い手の減り $\Delta n$ と残りの減り $\Delta R$ は同じ向きに並ぶか。
- **Q4（固定した場）** と **対照（保持が学習を削らないか）**: escape と同じ。

## 1. なぜ・何を知っているか（REPORT_ONLY）

### 1.1 escape_ee_0917（seed 0–9）の値（登録済みの走・すべて既知）

| 腕 | E（継続 1） | $\bar n_{\rm eff}$ | 平均の登り |
|---|---|---|---|
| S2dyn_10r（保持なし） | 0.324 | 0.0095 | +3.50 |
| S2dyn_c1 | 0.300 | 0.0074 | +2.73 |
| S2dyn_c2 | 0.307 | 0.0094 | +3.77 |
| S2dyn_c12 | 0.278 | 0.0064 | +2.96 |
| S2dyn_c12b | 0.275 | 0.0060 | +2.48 |
| S2u30r（床） | 0.212 | 0 | 0 |

$\Delta R$ = +0.046 [+0.031, +0.061]、$R_{c12}$ = +0.066、$\Delta n$ = +0.0031（10/10 正・95% [+0.0020, +0.0043]）、$\Delta{\rm climb}$ = +0.54（5/10）、$\bar n_{\rm eff}$ と E の順位相関は平均 +0.88（10/10 正）、seed 間の $\Delta n$ と $\Delta R$ の相関 r = +0.947。

**本 spec の判定規則を事後に seed 0–9 へ当てると**、主 REMAINDER_REDUCED_BY_HOLD・並び NEFF_TRACKS・傾き SLOPE_POSITIVE（p = 3e−5）・固定した場 FIX_REMAINDER_REDUCED・フラグなし になる。規則はこの値を見てから決めたので、seed 0–9 は証拠に数えない。**本走の seed 10–19 だけで判定する。**

### 1.2 seed 10–19 について見たもの

- neff_pred_0917 が seed 10–12 の自然軌道（同じ CPU 箱・51 タスク）を走らせ、崩壊の時刻と自然な $n_{\rm eff}$ を見ている（seed 10 の online は t5 0.64・t8 0.19・t10 0.14）。本走の prefix（自然軌道 t1–12）はこの記録と状態 hash で照合する。
- 検査 S1b で seed 10 と 13 の prefix（自然軌道のみ）を計算した。腕は回していない。
- 検査の S-cost と S9 は seed 20–23（登録外）で回す。**seed 10–19 のどの腕の値も、本走前には存在しない。**

### 1.3 $\bar n_{\rm eff}$ の定義（respdyn の probe・変更なし）

継続タスク中の 750 更新ごと（と終わり）に、1200 枚で網自身の第2層前活性 $z_2$ に場 $d$ を足した $z_2+d$ の float32 の訓練の微分 $g$ を計算し、ユニットごとに $(\sum_x g)^2/(N\sum_x g^2)$（全画像で $g=0$ のユニットは 0）、ユニット平均。その 9 点の平均が $\bar n_{\rm eff}$。**学習の結果も含む量**（網が学ぶと $z_2$ が動く）なので、操作の確認としては「保持が担い手を減らした」までしか言わない。

## 2. 設計（escape_ee_0917 §2 と同一）

- 箱: Random Label MNIST（1200 枚固定）、784–100–100–10、両隠れ層 ELU(α=1)、Adam lr 1e−3、batch 16、6000 更新/タスク、float32、CPU 1 スレッド、`flush_denormal`。
- **seed 10–19**。1 seed = ref の自然軌道 task 1–12 → t2 の網から 12 腕 × 2 タスク（Adam 初期化）。腕・保持・場・影・探針は escape の spec §2.2 のまま（`src/escape_ee_0917.py` を変更せずに import）。

## 3. 測る量

escape の spec §3 と同じ。加えて prefix の各タスクで neff_pred_0917 の記録との照合（`hash_match_neff_pred`・seed 10–12 のみ記録あり）。resp_ee / respdyn の記録は seed 10–19 に無いので、その照合は空欄になる。

## 4. endpoint

escape の §4 に加えて:

- **$\Delta n=\bar n_{\rm eff}(\text{S2dyn\_10r})-\bar n_{\rm eff}(\text{S2dyn\_c12})$**（継続 1 タスク目）。
- 並び: seed ごとに 5 腕（none・c1・c2・c12・c12b）の $\bar n_{\rm eff}$ と E の Spearman 順位相関。平均の登りとの順位相関は報告のみ。
- 傾き: 有効 seed をまたいだ $\Delta n$ と $\Delta R$ の Pearson 相関 r。t = r√(n−2)/√(1−r²)、自由度 n−2 の両側 p。

## 5. 判定

### 5.1 帯

escape と同じ（主の 2 量 $\Delta R$・$R_{c12}$ は 97.5%、他は 95%）。

### 5.2 適用条件

- **(A)** seed ごとに: prefix が 12 タスクで、記録のあるタスクは記録と一致（記録が無いタスクは可・矛盾は不可）。resp_ee / respdyn の記録と矛盾しない（記録が無ければ可）。影 = N10、分岐点の logits、動く腕の $d(0)$、保持の外れ ≤ 1e−4、S2dyn_c12 が各タスクで上限を書く、12 腕 × 2 タスクの E・climb・$\bar n_{\rm eff}$ が有限。有効 seed が 8 未満なら INAPPLICABLE。
- **(B)** $R_{\rm none}$ の 95% 区間の下端 > 0（respdyn の残りが新しい seed でも出る）。満たさなければ NOT_REPRODUCED。
- **(C)** **$\Delta n$ の 95% 区間の下端 > 0**。満たさなければ ESCAPE_NOT_MANIPULATED。

### 5.3 ラベル

- **主**（97.5%）: escape と同じ表（$s(\Delta R)=+$ かつ $s(R_{c12})\ne+$ → REMAINDER_REMOVED_BY_HOLD、両方 + → REMAINDER_REDUCED_BY_HOLD、$\Delta R$ が 0 → HOLD_LEAVES_REMAINDER、− → HOLD_RAISES_REMAINDER）。
- **並び**: 10 seed の順位相関の符号で正確な両側符号検定、p < 0.05 で NEFF_TRACKS / NEFF_OPPOSES、それ以外 NEFF_UNRESOLVED。
- **傾き**: p < 0.05 で r > 0 なら SLOPE_POSITIVE、r < 0 なら SLOPE_NEGATIVE、それ以外 SLOPE_UNRESOLVED。
- **固定した場**・**読みのフラグ**: escape と同じ。

### 5.4 読み方（登録）

- **REDUCED / REMOVED かつ SLOPE_POSITIVE** → 移植先の成長を止めると、微分を運ぶ画像が減り、その減った分だけ残りが減る。「respdyn の残りは、移植先が成長で担い手を保てた分」が新しい seed で介入として支持される。ただし $\bar n_{\rm eff}$ は学習の結果も含むので、「担い手が残りを作る」の因果の向きまでは言わない。
- **REDUCED で SLOPE_UNRESOLVED** → 残りは減るが、減り方が担い手の減りと seed ごとに並ぶとは言えない。
- **HOLD_LEAVES_REMAINDER** → escape の $\Delta R$ は seed 0–9 に固有だった。
- **ESCAPE_NOT_MANIPULATED** → 保持が新しい seed では担い手を減らさなかった。問いには答えない。
- **HOLD_IMPAIRS** → 主のラベルを「逃げ」の証拠にしない。
- 範囲は escape の §5.4 と同じ（ELU→ELU・乱数ラベル・6000 更新・t2 の網・t10 の動く場・Adam 初期化・継続 2 タスク・t2 の値での保持）。

## 6. 検査（登録前に開発実行。許容は導出から）

`python3 analysis/escneff_ee_0917/checks.py` → `results/escneff_ee_0917/checks.json`。**登録前の開発実行（`--only`）で S1・S1b・S8・S9 は pass、変異 29/29 を検出**（S9 の開発実行は seed 13 の 2 epoch で行い、本 spec の時点で seed 20 に移した）。全検査（S-cost を含む）は本 spec の commit の後、runner に登録 commit を書き込んでから 1 回通しで実行する。

| 検査 | 要求 | 変異 |
|---|---|---|
| S1 腕（seed 0・この runner を通して） | escape の S1 と同じ 7 項目（分岐点の logits、S2dyn_10r = respdyn・S2u30r = resp_ee と respdyn、影と $d(0)$、保持と書き込み、凍結 = 床、上限つき床 = 床）に加えて、provenance がこの実験と engine（escape）を名乗り、seed 0 の prefix が resp_ee と 12/12 一致・neff_pred の記録なし。E は出力しない | 4（escape の runner に入れて、この runner に engine として渡す） |
| S1b prefix の照合 | seed 10 の prefix 12 行がすべて neff_pred の記録と一致し resp_ee の記録なし、seed 13 は記録なし（全照合が空欄） | 3（記録の場所・タスクのずれ・seed のずれ） |
| S8 判定 | 合成 seed 10–19 の 21 場面でラベルと推定値（1e−9）、t 分位点、SEEDS | 18 |
| S9 CLI | seed 20・2 epoch で provenance（実験名・engine・spec・flush・両 runner の hash・記録なし）、定数（seed 10–19・12 腕・6 保持） | 4 |
| S-cost | seed 20–23（登録外）で 4 プロセス同時・腕 3 本、exit 0・4 スロット以上 | — |

## 7. 事前予測（Claude・seed 10–19 のどの腕も回す前）

seed 0–9 の値（§1.1）を知ったうえでの予測。新しい seed の群のばらつきは escape の SD（$\Delta R$ 0.018・$\Delta n$ 0.0016）程度と見る。

| 項目 | 予測 |
|---|---|
| (A) | 満たす 95% |
| (B) $R_{\rm none}>0$ | 95% |
| (C) $\Delta n>0$ | 90% |
| **主** | **REMAINDER_REDUCED_BY_HOLD 70%** / REMOVED 5% / HOLD_LEAVES 10% / ESCAPE_NOT_MANIPULATED 10% / その他 5% |
| $\Delta R$・$R_{c12}$ | +0.045（+0.02〜+0.07）・+0.06 |
| 並び | NEFF_TRACKS 80% |
| 傾き | SLOPE_POSITIVE 70%（seed 0–9 の r 0.95 は外れ値に引かれている可能性がある） |
| 固定した場 | FIX_REMAINDER_REDUCED 85% |
| 読みのフラグ | 立たない 85% |
| 平均の登りの順位相関（報告） | 10 seed 平均 +0.5 未満 70% |

**Issa の予測**: （登録後・本走前にチャットで記入）

## 8. 費用と実行

- 1 seed 7–9 分（escape の実測）。最大 4 並列・flock で直列起動（escape の launch.sh と同じ）。約 25 分。
- 起動は本 spec の登録 commit を runner に書き込み、全検査を 1 回通して all_pass になってから。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- `results/escneff_ee_0917/runs/s<seed>/`（seed 10–19。`prefix.csv`・`arms.csv`・`traj.csv`・`provenance.json` は git、`units.npz` は退避）。判定: `analysis/escneff_ee_0917/verdict.py` → `results/escneff_ee_0917/{summary.md,verdict.csv,paired.csv,provenance.json}`。
- `units.npz` と `results/_checks_escneff_ee_0917/` は `obsidian-research-data/escneff_ee_0917/` へ退避し `backup_manifest.json` を commit。結果 commit の後、main へ統合し worktree と branch を消す。
