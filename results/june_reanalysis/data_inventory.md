# Phase 0 — データ発見結果

生成: 2026-08-11 / `analysis/june_reanalysis/phase0_inventory.py`
対象仕様: `june_reanalysis_spec.md` §1

## 0. 最重要の前提訂正（先に読むこと）

仕様書が前提にしている構図と、実在するデータの構図が **3 点で食い違う**。
解析は止めず、両方の読みを並走させる形で実行した。

| # | 仕様書の前提 | 実データ | 対応 |
|---|---|---|---|
| 1 | リポジトリ `lop_analysis` | 存在しない。作業は `proj_004_drift`（実験ディレクトリは `results/drift_0809` 1 本のみ） | パス読み替えで続行 |
| 2 | 6月データ `exp2` / `exp4c` | **存在しない**。該当する出力・命名のディレクトリもファイルも 0 件 | **MISSING**（§4 参照）。B1 は drift 実験の重みで代替実行 |
| 3 | A1 は「ニューロン**重み** w_i の対ごと整列」 | A1 は **凍結測定の期待勾配 E[g_{W_i}]** の対ごと整列。重み W では整列は**まったく起きていない**（下記 §3） | B1・B4 は `obj='W'` と `obj='Eg'` の**両方**で実行 |

**3 が本再解析の結論を最も強く規定する。** 既報 A1 = 0.61–0.68 は再現したが、
それは重みの整列ではない（§3 の表）。

## 1. リポジトリと出力

- ルート: `/home/issan/Projects/claude/proj_004_drift`（git リポジトリ、venv は `.venv`、torch 2.13.0 / numpy / pandas あり）
- 実験ディレクトリ: `results/drift_0809` のみ
- 実行系: `src/train.py`（学習）→ `src/freeze.py`（凍結測定）→ `src/followup.py`（拡張凍結測定）→ `src/followup_analysis.py`（A1–A6）
- config: `configs/drift_0809.yaml` / 実行時コピー `results/drift_0809/config_used.yaml`

### run 一覧

`results/drift_0809/runs.csv`: **130 run = 26 条件 × 5 seed**（seed 0–4）。列は
`exp, width, period, enc, c, lr, seed, run_id`。

- 条件A（Slowly-Changing Regression, `m=20`, `f=15`, LTU 教師 100 unit, β=0.7）
  - width ∈ {5, 100}、T ∈ {100, 1000, 10⁴, 10⁵}、enc ∈ {std, centered}
  - lr グリッド {0.01, 0.003, 0.001} は `width=100, T=10⁴, std` にのみ適用
- 条件B（ガウス入力・教師切替, `d=21`）
  - width ∈ {5, 100}、K ∈ {100, 10⁴}、c ∈ {0.0, 2.0}、lr=0.01

### 重みスナップショット

- 形式: **PyTorch `.pt`**（`results/drift_0809/ckpts/{exp}_w{width}_step{step}.pt`、24 ファイル）
- チェックポイント時刻: `[0, 10000, 50000, 100000, 300000, 1000000]` × 4 グループ（A_w5, A_w100, B_w5, B_w100）
- キー: `step`, `net{W,b,v,c}`, `env{...}`, `teacher{W,b,v,c,t}`, `running_mean`, `runs`
  - `net.W`: `[R, h, d]`（R = そのグループの run 数、バッチ次元で並列化）
- 形式: **npz**（`followup_Eg_{exp}_w{width}_step{step}.npz`、24 ファイル）。キー:

| キー | 形状 | 内容 |
|---|---|---|
| `W`, `v` | `[R,h,d]`, `[R,h]` | 学習器の第1層重み・出力重み（その ckpt 時点） |
| `Eg_W` | `[R,h,d]` | 凍結測定の期待勾配 Ê[g_{W_i}]（**A1 の対象**） |
| `Eg_W_odd`, `Eg_W_even` | `[R,h,d]` | 奇数/偶数サンプル分割（A6 用） |
| `mu_inter`, `mu_odd`, `mu_even` | `[R,d]` | 測定窓の経験平均入力 µ̂ |
| `mu_true` | `[R,d]` | µ の解析値（A: flip_state+0.5、B: 真の µ） |
| `dead` | `[R,h]` bool | per-neuron dead 判定 |
| `finite` | `[R]` bool | 発散していない系列か |
| `gbar_norm`, `gbar_norm_i`, `gbar_count` | | 周期別平均勾配（A5 用） |
| `M`, `period` | `[R]` | 測定サンプル数・周期 |

## 2. 入力分布の定義（解析的に書ける）

`src/envs.py` を参照。**両条件とも µ と Σ が解析的に書ける。**

### 条件A（`SCREnv`, `envs.py:39-82`）

x = [flipping bits (15), U{0,1} (5)]、d = 20（論文の21番目の定数1ビットは層 bias で実現）。
flipping bits は周期 T ごとに 1 ビットだけ反転し、**周期内では定数**。

- 周期内: µ = [flip_state, 0.5·1₅]、**Σ = diag(0×15, 0.25×5)**
- 測定窓（50 周期）全体では flip 側にもわずかな分散が乗る

### 条件B（`GaussEnv`, `envs.py:133-156`）

x = µ + z, z ~ N(0, I_d), **µ = (c/√d)·1**, d = 21。

- **Σ = I₂₁（完全等方）**。→ 仕様書 §5 の「Σ の最小固有ベクトル」検定は
  条件B では **構造的に空虚**（B3 でこの通り判定した）

入力サンプル自体は保存されていないが、測定用 generator が
`make_gens(exp, width, device, offset=999000+step)` で完全再現できる（`freeze.py:31`）ので、
**任意精度で再サンプルできる**。

## 3. A1 の実装定義（コードからの読み取り）

`src/followup_analysis.py:83-155` `a1_pairwise()`。

- **対象は重みではなく期待勾配** `Eg_W`（`followup_analysis.py:97`: `Eg = z["Eg_W"]`）
- **signed**（|cos| ではない）。列 `vpcos_alive_mean` = `float(np.nanmean(vpca))`
- v 符号補正あり: `stats(U[i][alive], sv[i][alive])` で `sv = np.sign(z["v"])` を掛ける
  （`followup_analysis.py:97,121`）。生の符号付き `pcos_*` は構造的に ≈0 になる
- **dead 除外あり**（`alive = ~dead[i]`, `followup_analysis.py:118`）
- 層: 第1層（入力→隠れ）の重みに対する勾配

RESULTS.md がなぜ v 符号補正を使うかの根拠（RESULTS.md:157-160、`nets.py:44-52`）:

> 1 サンプルの勾配は **g_{W_i} = 2δ·v_i·1[pre_i>0]·x** なので、全ニューロンの瞬時勾配は ±x に厳密に平行。

これは本再解析の中心的な事実である（B2・B4 の判定に直結）。

### 既報 A1 の再現（seed 平均 ± std、signed、v 符号補正、alive のみ）

| 条件 | ckpt | obj=`Eg`（既報の定義） | obj=`W`（仕様書の文面） |
|---|---|---|---|
| B_w100_K10⁴_c0.0 | 1e6 | **+0.610 ± 0.236** | +0.016 ± 0.007 |
| B_w100_K10⁴_c2.0 | 1e6 | **+0.681 ± 0.046** | +0.009 ± 0.006 |
| B_w100_K10⁴_c0.0 | 1e5 | **+0.683 ± 0.294** | +0.015 ± 0.007 |
| B_w100_K10⁴_c2.0 | 1e5 | **+0.637 ± 0.060** | +0.006 ± 0.006 |
| A_w100_T10⁴_std | 1e5 | +0.620 ± 0.294 | +0.002 ± 0.013 |
| A_w100_T10⁴_centered | 1e5 | +0.431 ± 0.297 | +0.001 ± 0.003 |

→ **既報 A1 = 0.61–0.68 は条件B の 4 セルとして完全に再現**（RESULTS.md:140 と一致）。
→ **同じ前処理を重み W に適用すると全条件で ≈0**（ランダム床 0.174 未満、
signed なので 0 が期待値）。整列は勾配にあり、重みには無い。

## 4. dead 判定基準（コードからの読み取り）

`src/followup.py:36-59` `per_neuron_dead()` / config `configs/drift_0809.yaml:26-27`:

- 固定 eval バッチ **N = 2000**（学習時と同じ generator, offset=0）で活性 a を計算
- ニューロン i が dead ⇔ **mean_n 1[|a_{n,i}| < 1e-7] > 0.95**（`dead_tol=1e-7`, `dead_tau=0.95`）
- 判定は測定で env を進める**前**に行う（`followup.py:74`）

### alive 数の実態（重要な制約）

| ckpt | グループ | alive/ニューロン min / 中央値 / max | finite |
|---|---|---|---|
| 1e5 | A_w100 | 7 / 65 / 100 | 50/50 |
| 1e5 | B_w100 | 0 / 65 / 100 | 19/20 |
| 1e5 | A_w5 | 1 / 3 / 5 | 40/40 |
| 1e6 | A_w100 | 1 / 38 / 96 | 50/50 |
| 1e6 | B_w100 | 0 / 45 / 100 | 19/20 |
| 1e6 | A_w5 | 0 / 1 / 5 | 40/40 |

→ **width=5 は最終 ckpt でペアがほとんど作れない**（中央値 1–3 alive）。
B1–B4 の主判定は **width=100** で行い、width=5 は参考に留めた。
→ 条件B の 1 系列（`B_w100_K10000_c2.0_lr0.01_s?`）は lr=0.01 で発散し `finite=False`（既知、
メモリ記載どおり）。全解析で除外。

## 5. teacher の復元可否（B2 の前提）

**復元可能。** 各 ckpt の `teacher` キーにパラメータが完全保存されている
（条件A: LTU の `W,b,v,cout,tau` / 条件B: MLP の `W,b,v,c` と `t`）。
`src/freeze.py:31-56` `_restore()` がそのまま使え、教師の forward も
`envs.py` の `__call__` で計算できる。→ **B2 実行可**。

ただし「各 period t の teacher」については注意:

- 条件B: 教師は K ステップごとに**再サンプル**される。ckpt に入っているのはその時点の 1 つ。
  過去の period の教師は保存されていないが、測定用 generator から**同じ系列を再生成できる**
  （`MLPTeacher.maybe_resample()` を回す）。B2 ではこの方法で period 列を作った。
- 条件A: 教師は固定（非定常性は入力側の flipping bits）。period ごとに変わるのは
  **入力分布**であって教師関数ではない。したがって c_t = Cov(f(x), x) の period 依存は
  条件A では「x の分布が変わることによる依存」になる。

## 6. MISSING（依存する解析をスキップ／代替した項目）

| 項目 | 状態 | 影響 |
|---|---|---|
| `exp2` / `exp4c` の生データ | **存在しない** | B1 の一次データ源が無い。→ drift 実験（`drift_0809`）の重み・勾配で代替実行 |
| 怪文書§5 の inter-unit mean\|cos\| = 0.27 | **再現不能** | 本リポジトリのどの条件の重みにも該当する値は無い（実測 ≈0.17 = ランダム床）。B1 §「既報との差分」に記録 |
| 怪文書§5 の µ 方向 mean\|cos\| = 0.38 | **再現不能** | 同上（実測 ≈0.18 = ランダム床）。B1 に記録 |
| 独立残差モデルの予測 0.145 / 第二共通方向強度 0.36 | **検証対象が消滅** | 上 2 つが再現しない以上、この差分 0.125 自体が本データには存在しない。B1 は「実測値で再計算した予測」と比較した |
| 「c=0 でも centering でも A1 が落ちない」 | **部分的に成立** | 条件B（c=0 vs c=2）では確かに落ちない。**条件A では centering で落ちる**（0.620→0.431 @1e5、0.882→0.222 @1e6）。仕様書の前提「µ が整列を作るは反証済み」は条件B に限った話 |
| `sign_rand` の定義 | 該当コードなし | 本リポジトリに `sign_rand` という識別子は存在しない。B4 の置換ヌルは仕様書 §6-2 の定義をそのまま実装した |

## 7. 実行環境

- Python: `.venv/bin/python`（3.12）。`numpy`, `pandas`, `torch`, `matplotlib` 利用可
- **システム python には pandas が無い**。必ず `.venv/bin/python -m analysis.june_reanalysis.*` で実行する
- 乱数: `numpy.random.default_rng(20260811)`（`common.py:SEED`）
- 新規学習は一切実行していない。B2/B3 の入力サンプリングと teacher/learner の forward のみ
