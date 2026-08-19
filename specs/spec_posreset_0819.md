# spec_posreset_0819: E1 同方向・小ノルムリセット判別（2×2要因）

proj_004 / 作成 2026-08-19 / 対象リポジトリ: lop_analysis

位置づけ: 夏休み検証計画_0819 §3 E1 の実装仕様。**Phase 1 実行前に本仕様（特に §6 判定基準）を commit すること（事前登録）。**

---

## 0. 一行

新しいランダム特徴を一切供給しない「座標復元」リセット（同方向・初期ノルム・b初期化）が full 再初期化の便益をどこまで再現するかを 2×2 要因で測り、位置仮説 B1（reinit が回復するのは操舵権限 1/‖w‖ とゲートマージン β）を判別する。

## 1. 背景と仮説の構造

再初期化の作用機序の候補は2つ：

- **B1 位置仮説**: 効くのは座標が戻るから（‖w‖ 縮小＝操舵権限回復、b 初期化＝β 復帰）
- **H_feat 新特徴供給説**: 効くのは新しいランダム方向 u が供給されるから

**キルラインの所在（重要）**: レジームにより β の依存構造が違う。

- **レジーム B（µ=0, K=100, b経路）**: β = b/‖w‖。posonly（b←0）は β を直接 0 に戻す＝座標が完全復元される。**ここで posonly≈none なら B1 は棄却**（座標を全部戻しても効かない＝座標は回復変数ではない）。
- **レジーム A（condA, µ経路）**: b=0 のとき β = ŵᵀµ/‖ŵ‖_Σ でノルム非依存。posonly は操舵権限のみ回復し β は戻らない。回復は「残存ゲート流 p=Φ(β)>0 ＋ 回復した操舵で u が +µ̂ 半空間へ回る」の二段。**A で posonly>0 は「操舵単独で仕事をする」証拠であり、H_feat の直接反証**（新特徴ゼロで効いた）。A で posonly≈none は B1 の棄却ではなく改訂（操舵単独では不十分、マージン必須）。

dironly（新方向・ノルムと b は保持）は逆向き対照。レジーム B では b が深い負のままなのでゲートが開かず、**どの仮説でも効かないはず**——効いたら β/ゲート機構自体への警報。

## 2. 実行前に読むファイル

- `src/train.py`（resume 機構: spec_rank_int_0814 §3 で実装済みの完全再開スナップショット。save/load と決定論性）
- `src/nets.py`（VecMLP: W, b, a の初期化分布と state_dict。fresh draw g_i は t=0 と同一分布から引く）
- `src/lop_metrics.py`（eval_loss、ゲート発火率系の既存定義）
- CBP のリセット実装（`src/` 内を grep。読み出し重み a←0 の規約を確認し同じ関数を再利用）
- `configs/methods_sde_0813.yaml`（レジーム A の土台: condA, batch=1, lr_main=0.01, eval_batch=2000, 1M step）
- `configs/cbp_harm_0815.yaml`（レジーム B の土台: routeK, K=100, c=0（µ=0 厳密）, kappa=1, freeze_bias=false）
- `results/methods_sde_0813/`（A_w100 none アームの lop_metrics 時系列 → Phase 0）
- `results/cbp_harm_0815/`（routeK K=100, rho=0 の時系列 → Phase 0）

## 3. 設計

### 3.1 レジーム

| ID | 継承元 config | 設定 | 病理 |
|---|---|---|---|
| A | methods_sde_0813 の condA A_w100 none | batch=1, lr=0.01, µ≠0 | µ経路 dead（方向性、fullbatch/methods で dead≈0.97 @1M） |
| B | cbp_harm_0815 の routeK K=100 | µ=0 厳密, K=100, b 学習可 | b経路 dead（β=b/‖w‖ 沈降、dead≈0.99 @1M） |

### 3.2 トランクとブランチ

- トランク: 各レジーム × seed 0–9 を 0→t_int まで走らせ、完全再開スナップショットを保存。**主 t_int = 500,000**（適格性は Phase 0 で確認、満たさない seed も除外せず記録・層別報告）
- ブランチ: 各トランクから 4 アームへ分岐し t_int+500,000 まで（合計 1M step 相当/系）。none アームはトランクの無介入継続

### 3.3 treated 集合の定義

t_int 時点で、固定 eval バッチ（eval_batch=2000、seed 固定、同一 seed 内で全アーム共通）上の経験発火率 p̂_i = mean(1{w_iᵀx + b_i > 0}) を計算し、

    treated = { i : p̂_i < 0.05 }

（methods_sde の neg_gate 基準に一致）。treated_frac を記録。**適格性（事前登録）**: treated_frac ≥ 0.3 が 10 seed 中 8 未満のレジームは主判定を参考格に降格。

### 3.4 介入アームの数値定義

treated の各ユニット i について、t=0 と同一分布から fresh draw g_i を1本引き（アーム間で同一 seed 系列を共有し、g_i はユニットごとに1回だけ生成して3アームで再利用する）:

| アーム | w_i | b_i | a_i（読み出し） |
|---|---|---|---|
| none | 変更なし | 変更なし | 変更なし |
| posonly | ‖g_i‖ · (w_i/‖w_i‖) | b_init（=0） | 0 |
| dironly | ‖w_i‖ · (g_i/‖g_i‖) | **保持** | 0 |
| full | g_i | b_init（=0） | 0 |

- a←0 は CBP と同一規約（出力破壊を避ける）。**3つの reset アームで a の扱いを揃える**ことでランダム特徴回帰の便益を整合させ、アーム間差を w 座標の差だけに帰着させる
- ガード: ‖w_i‖ < 1e-8 のユニットは posonly を full にフォールバックし件数を記録
- treated 外のユニット・v2 層・その他 state は全アームで不変

### 3.5 実行規模

トランク 2×10 本 ×500k ＋ ブランチ 2×10×4 本 ×500k ＝ 計 50M step（methods_sde の約 1/5）。seed 10 本（境界判定の較正、計画 §5 準拠）。本走前に seed 0–1 で 4 アームのスモークを行い、判定コードまで通してから残りを流す。

## 4. Phase 0（再学習なし）

既存時系列から t_int=500k の適格性を確認する：

1. レジーム A: methods_sde_0813 A_w100 none の dead 系指標が 500k で ≥0.6 かつ概ねプラトーであること（seed 別に記録）
2. レジーム B: cbp_harm_0815 routeK K=100 rho=0 の同様の確認
3. 不適格なら t_int を {300k, 700k} から選び直し、**変更理由を phase0_summary.md に記録してから** Phase 1 に進む

出力: `results/posreset_0819/phase0_summary.md`

## 5. 指標

- **主指標 M**: ブランチ窓 [t_int, t_int+500k] の clean eval_loss の平均（既存 eval グリッド上）。**判定は clean eval_loss のみ。dead_frac は判定に使用禁止**（cbp_harm 規約）
- M_late: 窓末尾 100k の平均（A の二段回復の遅さ対策）
- 便益: Δ_arm = M(none) − M(arm)、seed でペア
- 副次（treated ユニットのみ、10k ごと）: p̂_i(t)、‖w_i‖(t)、レジーム B は解析 β_i = b_i/‖w_i‖、レジーム A は cos(u_i, µ̂)（符号つき。ゲート開放は wᵀµ>0 半空間なので符号つきが正しい）
- 保存: `unit_traj_{regime}_{seed}_{arm}.npz`

## 6. 判定基準（事前登録 — 実行前に固定）

統計手続き: paired seed bootstrap、run_id ソートでペア、`rng = np.random.default_rng(20260819)`、B=10,000、percentile 95%CI。**OMP_NUM_THREADS=1**。

**G0（前提ゲート、各レジーム）**: Δ_full > 0（CI が 0 を除外）。不成立のレジームは以降の判定を void（記録して保留）。

| ID | レジーム | 予測 | 判定 | FAIL の帰結 |
|---|---|---|---|---|
| P1 | B | Δ_posonly > 0 | CI 0 除外 | P2 と併せ **B1 棄却**の主成分 |
| P2 | B（主判定） | Δ_posonly ≥ 0.5·Δ_full | bootstrap(Δ_posonly − 0.5·Δ_full) の CI が 0 除外で PASS、点推定のみ正で弱 PASS。0.75 超えを「強」併記 | B1 弱体化（座標必要だが特徴鮮度も寄与＝混合説へ） |
| P3 | B | Δ_dironly ≤ 0.25·Δ_full | bootstrap(0.25·Δ_full − Δ_dironly) CI 0 除外 | **β/ゲート機構への警報**（b が深いままゲートが開かない所で新特徴が効いた） |
| P4 | A | Δ_posonly > 0 | CI 0 除外 | B1 の棄却ではなく**改訂**（操舵単独不十分、マージン必須）。ただし B で P1 PASS が条件 |
| P5 | A | Δ_full − Δ_posonly の符号・大きさを記録 | report only（PASS/FAIL なし） | —（マージン寄与の分解量） |
| P6 | B（副次） | ゲート再開率（p̂>0.05 到達 @+100k）: posonly ≫ dironly | 点推定順序のみ | 機構署名の不発（判定に波及しない） |
| P7 | A（副次） | posonly-treated の median Δcos(u, µ̂) > 0（+500k 窓） | 点推定のみ | 二段回復署名の不発（同上） |

**帰結マッピング**:

- B で P1 FAIL（かつ G0 PASS）→ **B1 棄却**。座標完全復元で効かない＝回復変数は座標でない
- B で P1 PASS・P2 FAIL → 混合説（座標＋特徴鮮度）へ改訂
- A で P4 FAIL・B で P1 PASS → B1 改訂: マージン必須・操舵は補助
- P3 FAIL → ゲート理論の見直しを最優先課題に昇格
- P1–P4 PASS → §11 の主張へ（ただし「確定」とは書かない）

## 7. サニティ

- **S1**: OMP_NUM_THREADS=1 を環境で固定（全 run・全解析）
- **S2**: resume bit 一致 — ブランチ開始時 state hash がトランク ckpt と一致（rank_int S1 踏襲）
- **S3**: 介入の数値保証 — posonly: |cos(u_pre, u_post) − 1| < 1e-12 かつ ‖w_post‖ = ‖g‖（相対誤差 < 1e-12）; dironly: ‖w_post‖ = ‖w_pre‖ 厳密; full: w_post ≡ g; 全 reset アームで treated の a_i == 0、b は表の規約通り; treated 外パラメータの hash 不変
- **S4**: treated 集合の hash が同一 seed 内で 4 アーム一致。treated_frac を runs.csv に記録

## 8. 交絡と限界

- a←0 は none と非対称（意図: none は無治療対照）。ランダム特徴回帰の便益は 3 reset アーム間で整合済みだが、none との比較には reset 共通コスト（cbp_harm PC-4）が乗る——アーム間コントラスト（posonly vs full vs dironly）が主戦場である理由
- A の posonly は二段回復で遅い可能性 → M_late 併記。窓不足が疑われる場合は +250k 延長の resume オプションを感度分析として事前許可（実施したら逸脱として記録）
- B（µ=0, Σ=I）では旧方向の鮮度劣化が小さく、P2 の等価性は H_feat の弱い反証にしかならない。**H_feat の主戦場は P4**（旧方向が真に病的な condA で新特徴ゼロが効くか）
- one-shot リセット ≠ CBP 継続型。継続型への外挿は E3 で扱う
- スコープ: 2層 ReLU・MSE・SGD・トイ・K=100（B）

## 9. 主張してはいけないこと

- dead_frac に基づくいかなる判定・主張（clean eval_loss のみ）
- P1–P4 が全て PASS しても「B1 確定」（E2 の燃料溶接・E3 の組合せ代数・E4 の基準対決が残る）
- 「新特徴は無用」への飛躍（B の等価性は弱い証拠。強い証拠は A の P4 のみ）
- CBP（継続適用）・K=10⁴・実スケール・Transformer への外挿
- 忘却側（OP10）への言及

## 10. 出力

`results/posreset_0819/` に:

- `phase0_summary.md`（t_int 適格性、変更があれば理由）
- `runs.csv`（regime, seed, arm, treated_frac, M, M_late, ガード発動数）
- `lop_metrics_*.csv`、`unit_traj_*.npz`（treated のみ）
- `verdict.csv`（G0, P1–P7 の PASS/FAIL/void/report、根拠数値と CI）
- `summary.md`: G0 結果、Δ 表（4 アーム×2 レジーム、CI 付き）、P 表、treated_frac 表、S1–S4 結果、逸脱記録、§9 の再掲
- `figures/`: (i) レジーム別 eval_loss 時系列（4 アーム重ね、seed 帯）、(ii) treated ユニットの p̂ 再開曲線、(iii) A の cos(u, µ̂) 軌跡、(iv) Δ の forest plot

## 11. 通ったときの主張（先取りメモ、確定後に使う）

「一切の新特徴を供給しない座標復元（同方向・初期ノルム・b 初期化）だけで、full 再初期化の便益の 50% 以上（b 経路）および有意な便益（µ 経路）が再現された。再初期化の作用機序の主成分は座標復元——操舵権限 1/‖w‖ とゲートマージン β の回復——である」※スコープ条件（§8 末尾）を必ず併記。

---

## 12. 実装ノート（事前登録コミット時に追記、§6 は不変）

本文 §6 の判定基準・§3 の設計は上記のまま凍結する。以下は実装上の事実確認のみ。

- **リポジトリ**: 本文ヘッダは「lop_analysis」だが、§2 が参照する `src/train.py` の resume 機構・`configs/methods_sde_0813.yaml`・`configs/cbp_harm_0815.yaml`・`results/{methods_sde_0813,cbp_harm_0815}/` が実在するのは **`proj_004_drift`**（`lop_analysis` は 2026-08-12 で更新が止まった旧リポジトリで、これらを含まない）。実装・実行は proj_004_drift で行う。
- **§3.4 の記号 a_i**: 本文の「a_i（読み出し）」は実装上の `VecMLP.v`（隠れ→出力の読み出し重み）を指す。実装の `a` は活性ベクトルであり別物。CBP の reset 規約（`train.py:apply_method`）は `net.W[sel]=kaiming再サンプル / net.b[sel]=0 / net.v[sel]=0` なので、full アームはこの規約と厳密に一致する。
- **レジーム B の幅**: §3.1 が継承元とする `cbp_harm_0815` の condB は `widths: [20]`、`target_hidden: 100`、`spike_dir: alt`。レジーム B は w=20 で実施する。
- **b_init = 0**: `envs.kaiming_mlp_params` が b を zeros で返すことを確認（§3.4 の「b_init（=0）」と一致）。
- **fresh draw g_i の分布**: 同関数の入力層 W と同一、すなわち成分独立の一様分布 U(−√(6/d), +√(6/d))。
- **S3 の数値許容**: 介入計算は rank_int_0814 と同様 float64 で行い、S3 の 1e-12 判定は float64 の介入結果に対して行う。学習再開用に float32 へ丸めた後の誤差（float32 の eps ≈ 1.2e-7 に律速）は別列で併記する。
