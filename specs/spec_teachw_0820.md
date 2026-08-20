# spec_teachw_0820: 教師複雑度スイープ(生存者数 = 必要ユニット数か)

proj_004 / 作成 2026-08-20 / 対象リポジトリ: lop_analysis

位置づけ: 盲点4の介入ダイヤル。surv_hist_0820(観察)と対をなす。役立ち説の複雑度予言「生存者数はタスクを表現するのに必要なユニット数で決まる」を、**入力ストリームを固定したまま教師の複雑さだけを振って**検証する。**実行前に本仕様(特に §6)を commit すること(事前登録)。**

---

## 0. 一行

condA の教師 LTU 幅 H_T ∈ {1, 2, 4, 8, 32, 100} を振り(入力ビット列は seed 固定で全アーム同一)、最終生存者数 alive_final が H_T に単調増加するかを paired 設計で判定する。

## 1. 背景と仮説の構造・ダイヤル選定の理由

- H_earn(役立ち説)の予言: alive_final は教師複雑度に**単調増加**(サポート次元 2^(m−f)=32 付近で飽和は許容)。
- 対立(罠幾何説 / H_marg 系): 生存者数は罠の幾何(μ̂ と消灯点)で決まり、入力統計が固定なら教師複雑度に**非感応**。
- **ダイヤル選定**: 当初案の f(flipping bits 数)は棄却。f を動かすとサポートサイズ 2^(m−f)・μ̂ の構成・境界ジャンプ幾何が同時に動き交絡する(src/envs.py SCREnv: 入力 = [flip_state(f), U{0,1}^(m−f)])。教師幅 `target_hidden` なら **SCREnv の乱数ストリームに一切触れず**、seed を固定すれば全アームで入力・flip 系列がビット同一になる。教師(LTUTarget, src/envs.py L12)だけが変わる、最も清潔な介入。f スイープは Phase 2 の収束証拠に降格(§9)。
- キルライン: P1 の傾き CI が 0 を跨ぐ or 負 → 「生存者数 = 必要ユニット数」予言は**このダイヤル・このスコープで棄却**。
- 限定: 単調性は役立ち説の**必要条件であって十分条件ではない**(複雑度と相関する第三因子の可能性)。PASS でも「支持」に留める。

## 2. 実行前に読むファイル

- `src/envs.py` — `LTUTarget`(W,b,v,cout ∈ ±1、τ = β(m+1) − S、**出力の正規化なし** → §3 のスケーリングが必要)、`SCREnv`
- `src/freeze.py` L40 — condA 教師の構築規約(`gens["teacher"]` が env と別ストリームであること。別でなければ §7 S2 は成立しないので実装を確認)
- `src/train.py` — `train_group` / `save_snapshot`
- `configs/methods_sde_0813.yaml` — condA A_w100 の土台
- `results/ratchet_log_0819/` — H_T=100 のアンカー(per_seed_metrics.csv の dead_frac_final)
- `src/lop_metrics.py`、ratchet probe の厳密 p̂ 計算(最終時点の alive_final に流用)

## 3. 設計

- レジーム: condA A_w100(m=20, f=15, T=1e4, std, batch=1, lr=0.01)、seed 0–9、1M step。**アーム = H_T ∈ {1, 2, 4, 8, 32, 100} の 6 本**。介入・分岐なし。
- **教師出力スケーリング**: y_scaled = y_raw · √(100 / H_T)(cout 込み全体に乗算)。根拠: LTU 出力は正規化されておらず Var[y] ≈ O(H_T) で複雑度と損失スケールが交絡するため。H_T=100 は係数 1 で**既存実装とビット一致**(アンカー)。
- 計測: 既存 lop_metrics(lop_every=1000)+ **t=1M で厳密 p̂ を 1 回計算**(ratchet probe の p̂ 実装を流用、フル probe 常駐は不要)→ alive_final = #{i: p̂_i ≥ 0.05}。最終スナップショットを保存(事後の台帳解析用)。
- 計算量: ratchet_log 実測 363 s / 10 seed 群 → 6 アーム ≈ 40 分(CPU)。

## 4. Phase 0(本走前に完了)

1. **恒等性**: H_T=100・seed 0 の 50k スモークが ratchet_log_0819 と state hash 一致(スケーリング実装が係数 1 で恒等であることの検証)。
2. **Var[y] 測定**: 各 H_T、t=0 の全サポート上で Var[y_scaled] を測り記録。中央値が H_T=100 比 [0.5, 2.0] 内であること(外れたらスケーリング則を見直してから本走)。
3. **H_T=1 の学習可能性**: 50k スモークで eval_loss_exact が初期値から低下すること(タスクが自明壊れしていないか)。

## 5. 走行

6 アーム × seed 0–9(R=10 ベクトル化群 × 6)。`OMP_NUM_THREADS=1`。config は本 spec と同時に `configs/teachw_0820.yaml` として commit。

## 6. 判定基準

| ID | 予測 | 統計量 | 基準 |
|---|---|---|---|
| P0 | 前提ゲート: 各アームで堆積が起きる | dead_frac_final = 1 − alive_final/100 | 各レベルで ≥ 0.5。FAIL レベルは void。**有効レベル < 4 なら全体を判定保留**(posreset G0 規約) |
| P1 | 主判定: alive_final は複雑度に単調増加 | (i) seed ペア回帰 alive_final ~ log2(H_T)(有効レベルのみ)の傾き、(ii) per-seed Spearman ρ の中央値 | (i) 傾き > 0 かつ seed bootstrap CI ゼロ非含有、**かつ** (ii) ρ 中央値 ≥ 0.6。両方満たして PASS |
| P2 | 効果量(報告のみ) | alive(H=100) − alive(H=1) の paired 差と CI | 判定なし |
| P3 | 探索的 | レベル別の alive median p̂(surv_hist T1 の複雑度依存)、eval_loss_exact の plateau、dead 進行の t50 | 判定なし |

- H_T=1 等の低複雑度で P0 FAIL(死ななくなる)が出た場合、それ自体を「堆積には未フィット残差の持続が必要(T4)」の傍証として**別枠で**報告する(P1 の判定には混ぜない)。

## 7. サニティ

- S1: `OMP_NUM_THREADS=1`(meta に記録)。
- S2: **flip_state 軌跡の hash が seed ごとに全 6 アームで一致**(入力ストリーム同一の保証。教師 gen が env gen と別ストリームであることの実効検証)。不一致なら設計前提が崩れているので中止して原因究明。
- S3: H_T=100 アームの dead_frac_final が ratchet_log_0819 per_seed_metrics と **seed 別に厳密一致**(アンカー再現)。
- S4: Phase 0-2 の Var[y] 帯を本走ログでも確認。

## 8. 統計・出力規約

- bootstrap B=10000、`np.random.default_rng(20260820)`、run_id ソートで seed ペアリング。
- 出力: `results/teachw_0820/`: `verdict.csv`、`summary.md`(P0 ゲート表・レベル×seed の alive 行列・逸脱節)、`per_seed_metrics.csv`、`runs.csv`、`snapshots/`、`figures/`(alive vs log2 H_T の seed 線 + 中央値、dead_frac 時系列のレベル別)。

## 9. スコープ・Phase 2・逸脱

- スコープ: **condA・w100・T=1e4・batch=1・LTU 教師族・このスケーリング則**。condB・他教師族へ外挿しない。
- Phase 2(本走の判定確定後に起案): f ∈ {17, 13} の f スイープを**交絡明記の収束証拠**として。P1 と同方向なら頑健性、逆なら交絡の切り分けを設計。
- 逸脱は summary §逸脱 に列挙(spec_ratchet_log_0819 §9 と同形式)。null 結果も同じ形式で報告する。
