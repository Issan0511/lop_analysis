# spec_mlp2_centering_delay_posthoc_0830: 5M 層別 centering 遅延仮説

proj_004 / 作成 2026-08-30 / 対象リポジトリ: `lop_analysis` / 対象データ: `results/mlp2_phase1_0829/layer_stats.csv`

> **状態: 解析手順登録・未実装・未実行。** この commit では本 spec だけを追加する。
> 集計コード、結果ディレクトリ、図、数値表は作らない。本 spec の push を確認した後、
> Issan が明示的に実装を指示した場合だけ次段へ進む。

---

## 0. 格の宣言（重要）

これは盲検の事前登録ではない。5M 走はすでに完了し、次の情報は spec 起草前に既知である。

- `results/mlp2_phase1_0829/summary.md` には末尾50 taskの腕別・層別
  `strict_dead` / `alive` / `dose` と、腕別 `unfit` が掲載済み
- 同 summary の既存判定 `NO_LOP_PREVENTION_SIGNATURE` も閲覧済み
- 起草依頼には、1層1Mログの傾き・到達時刻と、
  「`L2_A1` の第2層死が `L2_none` に追いつき、`L2_Aall` は低く残る」
  という予測が数値つきで提示されている

したがって本 spec が固定するのは、**既存ログに対する新しい問い、集計単位、同値幅、
判定分岐、出力形式**である。結果を独立確認・事前登録効果とは呼ばない。
格は一貫して **post-hoc reanalysis registration** とする。

---

## 1. 問い

第1層入力だけを center した `L2_A1` は、第2層入力に dose を残す。
そのため centering は第2層の LoP を消すのではなく、5M の時間軸上で遅らせるだけなのか。

ただし「LoP」を一つの量として扱わず、次の二つを分離する。

1. **形態 endpoint $D$**: 第2層の `strict_dead_frac`（現在タスクの32点全支持で
   $\hat p_i=0$ のユニット割合）
2. **機能 endpoint $U$**: ネットワーク全体の `unfit = residual_var / signal_var`

$D$ が baseline に追いついても $U$ の改善が残る可能性がある。その場合は
「死の遅延」は支持されるが、**「centering は LoP を治しておらず観測窓が隠しただけ」
という強い主張は支持しない**。

---

## 2. データ源とスコープ

### 2.1 使用ファイル

- `results/mlp2_phase1_0829/layer_stats.csv` — 主データ
- `results/mlp2_phase1_0829/provenance.json` — S-pair / S-pair-final の確認
- `results/mlp2_phase1_0829/config_used.yaml` — 腕・seed・時刻の照合
- `results/mlp2_phase1_0829/summary.md` — 既知結果と新集計の整合確認のみ

### 2.2 対象

- 条件: condA、$T=10^4$、batch=1、lr=0.01、隠れ2層×100、5M step
- seed: 0–9
- task: 1–500、task末尾のみ（step = task × 10,000）
- 腕: `L2_none` / `L2_A1` / `L2_Aall`
- 層: 1 / 2

`L2_A2` は数値発散したため除外する。`L1w100_A1` は深さが違い、
本 spec の層別対比には使わない。新規学習、checkpoint再計算、乱数再生は行わない。

### 2.3 対応づけ

3腕は同じ seed 内で init・教師・入力列・flip軌道が対応する。
`provenance.json` の S-pair と S-pair-final が両方 PASS であることを必須条件とし、
腕間はすべて paired 集計とする。どちらかが FAIL なら解析を停止し、
`PAIRING_INVALID` だけを出す。unpaired への事後変更は禁止。

---

## 3. 記号と窓

腕 $a$、seed $s$、層 $l$、50-task block $b$ について、

$$
D_{a,s,l,b}=\operatorname{mean}_{t\in b}\;\texttt{strict\_dead\_frac}_{a,s,l,t}
$$

とする。機能量はlayerで重複保存されているのでlayer 1行だけを読み、

$$
U_{a,s,b}=\operatorname{mean}_{t\in b}\log_{10}(\max(\texttt{unfit}_{a,s,t},10^{-23}))
$$

とする。$10^{-23}$ は元走の較正済みfloorを継承し、変更しない。

### 3.1 非重複 block

`B01 = task 1–50`, `B02 = 51–100`, …, `B10 = 451–500` の10本。
主要な「1M窓」は B02、「5M窓」は B10 とする。

- B02 は最初の1Mの末尾50 task
- B10 は元走の登録済み末尾窓と一致
- exact task 100 / 500 の一点値は sensitivity として出すが判定に使わない
- rolling window、窓幅変更、結果を見た後の区間除外は禁止

### 3.2 Paired gap

形態 gap（percentage pointではなく割合）:

$$
g^D_{A1,s,l,b}=D_{L2\_A1,s,l,b}-D_{L2\_none,s,l,b}
$$

$$
g^D_{Aall,s,l,b}=D_{L2\_Aall,s,l,b}-D_{L2\_none,s,l,b}
$$

機能 gap:

$$
g^U_{A1,s,b}=U_{L2\_A1,s,b}-U_{L2\_none,s,b}
$$

負値は centering 腕が baseline より良いことを表す。

---

## 4. 推定と区間

- 点推定: seedごとの paired gap 10本の中央値
- 不確実性: seed-cluster paired percentile bootstrap、B=20,000
- RNG: `np.random.default_rng(20260829)`（元の多層系列から継承）
- 95% percentile CI
- 補助: paired gap の符号数（negative / zero / positive）
- taskを独立標本としてbootstrapしない。選ばれたseedの50-task軌道を丸ごと保持する

同値幅は形態量のみ事前固定する。

$$
\epsilon_D = 0.05
$$

すなわち第2層 `strict_dead_frac` の±5 percentage point。
100ユニットなので5本に相当する。CI全体が $[-0.05,+0.05]$ に入った場合だけ
`EQUIVALENT` とする。点推定だけが入る場合は `POINT_NEAR` であり同値とは呼ばない。

機能量 $U$ の同値幅は

$$
\epsilon_U = 0.1\;\log_{10}\text{ unit}
$$

（約1.26倍）と固定する。5M窓のpaired CI全体が $[-0.1,+0.1]$ に入る場合だけ
`FUNCTION_EQUIVALENT`。点推定だけが入る場合は `FUNCTION_POINT_NEAR` で、
正式な同値とは呼ばない。

---

## 5. 主判定

### P1. `L2_A1` 第2層の形態的遅延とcatch-up

層2の $g^D_{A1,2,B02}$ と $g^D_{A1,2,B10}$ を使う。

1. **早期保護**: B02 の95% CI上端 < 0
2. **gap閉鎖**: `median(g_B10 - g_B02)` の95% CI下端 > 0
3. **5M同値**: B10 のgap CI全体が $[-0.05,+0.05]$

| 1 | 2 | 3 | P1 verdict |
|---|---|---|---|
| yes | yes | yes | `MORPHOLOGICAL_DELAY_AND_CATCHUP` |
| yes | yes | no、B10 CI上端 < -0.05 | `DURABLE_MORPHOLOGICAL_PROTECTION` |
| yes | yes | no、B10 CI下端 > +0.05 | `LATE_OVERSHOOT` |
| yes | no | — | `EARLY_GAP_WITHOUT_CLOSURE` |
| no | — | — | `NO_EARLY_MORPHOLOGICAL_ADVANTAGE` |
| その他 | | | `INCONCLUSIVE_MORPHOLOGY` |

「catch-up time」は、B02以降でgapが `EQUIVALENT` になり、次blockでも同値が続く
最初のblockの終端とする。一度だけの交差をcatch-upとしない。B10で初めて同値なら
次blockを観測できないので時刻は確定せず、`CATCHUP_BY_5M_SINGLE_BLOCK` と記録する。

### P2. 強い「LoPは遅延だけ」仮説

P1が `MORPHOLOGICAL_DELAY_AND_CATCHUP` であることに加え、機能gap
$g^U_{A1,B10}$ が `FUNCTION_EQUIVALENT` である場合だけ、

`DELAY_ONLY_ACROSS_MORPHOLOGY_AND_FUNCTION`

とする。

P1はcatch-upだがB10で機能gapの95% CI上端 < 0なら、

`MORPHOLOGICAL_CATCHUP_WITH_DURABLE_FUNCTIONAL_BENEFIT`

とする。この場合、**「centeringは病気を治しておらず、evalに届く前に窓が終わっただけ」
は棄却**する。死ラベルと機能の乖離（冗長性 / 機能盲目性）が残る。

P1がcatch-upでなく、機能gapも負なら `DURABLE_PROTECTION`。
その他は `INCONCLUSIVE_FUNCTIONAL_KINETICS`。

### P3. `L2_Aall` は低く残るか

二つの強さを分ける。

1. **相対的な持続保護**: 層2 B10で、`Aall-none` と `Aall-A1` の
   paired gap CI上端がどちらも $-0.10$ 未満
2. **絶対的に低い**: 層2 B10のseed中央値が0.25以下

両方なら `AALL_ABSOLUTE_AND_RELATIVE_LOW`、1のみなら
`AALL_RELATIVE_ONLY`、2のみなら `AALL_ABSOLUTE_ONLY`、どちらもなければ
`AALL_NOT_LOW`。

「低いまま」という文言は両方を満たした場合だけ使う。

### P4. 層局在の負対照

`L2_A1` の層1 B10について `A1-none` gap CI上端が $-0.10$ 未満なら、
第1層の保護は残るとする。

- 層1は保護、第2層だけcatch-up: `LAYER2_LOCALIZED_CATCHUP`
- 両層catch-up: `GLOBAL_CATCHUP`
- 層1も悪化: `A1_LAYER1_PARADOX`

P4はP1/P2を上書きしない。

---

## 6. 副次・REPORT_ONLY

- block別の `alive`, `strict_dead`, `eff_rank`, `eff_rank_per_alive`, `dose`
- block別 `unfit`, `eval_loss_exact`。腕間の機能比較は無次元 `unfit`だけで判定する
- exact task 100 / 500 の腕差
- 各腕・層の block trajectory（seed中央値＋seed軌道）
- B02→B10 の各seed内変化

`dose` は機構整合の記述にだけ使い、P1–P4の判定に入れない。
`strict_dead` は現在タスク支持上の $\hat p=0$ であり、ユニットIDの恒久死・
吸収時刻・生存解析として扱わない。

---

## 7. 必須サニティ

1. `layer_stats.csv` は対象3腕×10seed×500task×2層 = 30,000行
2. `(arm, seed, task, layer)` は一意
3. seedは0–9、taskは1–500、layerは1–2が欠損なく揃う
4. 全行で `task_end == 1`、`step == task * 10000`
5. `strict_dead + alive == 100`
6. `strict_dead_frac == strict_dead / 100`（絶対誤差 < 1e-12）
7. centered flag:
   - `L2_none`: layer 1/2とも0
   - `L2_A1`: layer 1のみ1、layer 2は0
   - `L2_Aall`: layer 1/2とも1
8. P1–P4に使う列はfinite、`0 <= strict_dead_frac <= 1`、`unfit > 0`
9. layer 1行とlayer 2行の `unfit` / `eval_loss_exact` はbit一致
10. `provenance.json` の S-pair / S-pair-final がPASS

どれか一つでも落ちたら数値判定を出さず `SANITY_FAIL`。
欠損seed除外、腕除外、補間、窓変更による救済は禁止。

---

## 8. 将来の実装時の出力契約

実装が別途承認された場合のみ、`results/mlp2_centering_delay_posthoc_0830/` に出す。

- `block_levels.csv` — arm×seed×layer×block
- `paired_gaps.csv` — P1–P4のseed別gap
- `verdict.csv` — P1–P4、点推定、CI、符号数、判定
- `summary.md` — 格、主判定、限定、引用禁止
- `centering_delay_layers.png` / `.pdf`
- `provenance.json` — 入力SHA256、spec commit、実装commit、実行環境

解析コードの想定先は `analysis/mlp2_centering_delay_posthoc/analyze.py` だが、
**本 spec commitでは作成しない**。

---

## 9. 引用上の禁止

- 本解析を事前登録・独立再現と呼ばない
- 第2層 `strict_dead` のcatch-upだけで「eval LoPもcatch-upした」と書かない
- `strict_dead` をユニットID付きの恒久死や吸収ハザードと呼ばない
- `L2_Aall` がbaselineより低いだけで「低いまま」と書かない（P3の両条件が必要）
- B10一点の接近だけで「追いついた」と書かない（P1の早期保護・gap閉鎖・同値が必要）
- 1層 `center_selfcov_0814` と2層 `mlp2_phase1_0829` を同一個体・同一走として結ばない
- condA・2層×100・T=10,000・5M外へ外挿しない

---

## 10. 停止点

このspecをcommit・pushした時点で停止する。
`layer_stats.csv` の再集計、閾値判定、bootstrap、図作成、vaultへの結果転記は行わない。
