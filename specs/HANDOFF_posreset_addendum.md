# 引き継ぎ資料: spec_posreset_0819_addendum の実装・実行

宛先: 外部エージェント（OpenAI Codex）/ 作成 2026-08-19 / 作成者: Claude Code (Opus 5)

これまでの会話コンテキストを持たない実行者向けに、**必要な背景を全部**書き出したもの。
まずこれを通読し、次に `specs/spec_posreset_0819_addendum.md`（事前登録済み・凍結）と
`specs/spec_posreset_0819.md`（本体・凍結）を読むこと。

---

## 0. あなたのタスク（一行）

`specs/spec_posreset_0819_addendum.md` を**そのまま**実装・実行し、§7 の成果物を
`results/posreset_0819_add/` に出す。判定基準（§4 の Q1–Q8）は事前登録済みで**変更禁止**。

---

## 1. 研究の背景（何を調べているのか）

**proj_004 = ニューラルネットの可塑性喪失 (Loss of Plasticity, LoP) の機構解明。**
継続学習で NN が新タスクを学べなくなる現象を、2層 ReLU・MSE・SGD のトイ設定で
SDE（確率微分方程式）として分解し、「どの項・どの座標が病理の本体か」を判別している。

中心命題: **治療ハンドルは SDE の項と引数（源＝データ統計、項＝ドリフト/拡散/ゲート/
ジャンプ、座標＝β・‖w‖・u）にしかない。力学に陽に現れない集計量（ランク・dead 数・
基底配置）への介入は効かない——それらは症状である。**

### 今回関係する2つの対立仮説

再初期化（reinit）はなぜ効くのか？

- **B1「位置仮説」**: 効くのは**座標が戻るから**。すなわち
  - 操舵権限 `1/‖w‖`（‖w‖ が大きいとタスクからの操舵信号が 1/‖w‖ で減衰し、方向が
    タスクに操舵されなくなる＝「ハンドルが切れる」）
  - ゲートマージン `β`（ユニット i の発火確率は p_i = Φ(β_i)。β が深い負に沈むと
    ゲートが閉じっぱなしになり dead 化する）
  この2つが初期値に戻ることが便益の本体、という説。
- **H_feat「新特徴供給説」**: 効くのは**新しいランダム方向 u が供給されるから**。

### β の定義（レジームで構造が違う。ここが実験設計の肝）

ユニット i の事前活性は `pre_i = w_iᵀx + b_i`。入力を x ~ (µ, Σ) とすると
`β_i = (w_iᵀµ + b_i) / sqrt(w_iᵀΣw_i)`。

- **レジーム A（condA、µ≠0）**: b は 0 付近に留まり、β ≈ ŵᵀµ/‖ŵ‖_Σ。
  **ノルム非依存**（ŵ は単位ベクトル）。→ ‖w‖ を縮めても β は戻らない。
- **レジーム B（µ=0 厳密、K=100）**: µ=0 なので `β_i = b_i/‖w_i‖` ちょうど。
  → b を 0 に戻せば β は厳密に 0 に戻る（座標の完全復元）。

---

## 2. E1 本走（既に完了・コミット済み `cec7faf`）で何が起きたか

### 設計

t_int=500,000 でトランクから4アームに分岐、seed 10 本、+500,000 step。
treated = 固定 eval バッチ（N=2000）上の経験発火率 `p̂_i < 0.05` のユニット。

| アーム | w | b | v（読み出し） |
|---|---|---|---|
| none | 変更なし | 変更なし | 変更なし |
| posonly | ‖g‖·(w/‖w‖) = 方向保持・ノルムのみ初期化 | 0 | 0 |
| dironly | ‖w‖·(g/‖g‖) = 新方向・ノルムと b は保持 | 保持 | 0 |
| full | g（標準 reinit） | 0 | 0 |

g は t=0 と同一分布からの fresh draw（`envs.kaiming_mlp_params` の W と同じ、
成分独立 U(−√(6/d), +√(6/d))）。**3アームで同一の g を共有**。

主指標 M = 窓 [t_int, t_int+500k] の clean eval_loss の平均（小さいほど良い）。
Δ_arm = M(none) − M(arm)（正 = そのアームが良い）。**判定は clean eval_loss のみ。
dead_frac は判定に使用禁止**（過去実験 cbp_harm_0815 の教訓：CBP は定義上 dead を
下げるので dead で判定すると循環する）。

### 結果（数値は `results/posreset_0819/` に全部ある）

**レジーム A（condA A_w100、µ経路）** — M の seed 平均:

| none | posonly | dironly | full |
|---|---|---|---|
| 0.4267 | 0.2194 | 0.1509 | 0.1321 |

- G0（前提ゲート Δ_full > 0）: **PASS** 0.2946 CI [0.2230, 0.3606]
- P4（Δ_posonly > 0、§8 が「H_feat の主戦場」と名指し）: **PASS** 0.2073 CI [0.1613, 0.2517]
  → **新方向をゼロ供給しても大きな便益 → H_feat の強い形は棄却**
- P5（report）: Δ_full − Δ_posonly = 0.0873 CI [0.0502, 0.1288] → posonly は full に有意に劣る
- P7（report）: posonly-treated の median Δcos(u, µ̂) = **−0.0327 CI [−0.0536, −0.0071]**
  → 仕様が予測した「操舵権限が戻って u が +µ̂ 半空間へ回る」二段回復の署名は**出ず、逆符号**
- 事後追加のアーム間比較（`results/posreset_0819/posthoc_arm_contrasts.md`）:
  - Δ_dironly − Δ_posonly = **+0.0685 CI [0.0382, 0.1039]** ← **今回の追補が狙う対象**
  - Δ_posonly/Δ_full = 0.704 CI [0.616, 0.797]
  - Δ_dironly/Δ_full = 0.936 CI [0.895, 0.979]
  - 順序 full ≳ dironly > posonly > none で隣接差が全て有意。70%+94% で明確に**劣加法**

**レジーム B（µ=0, K=100, w20、b経路）** — 全アーム M ≈ 0.68 で動かない:

- G0: **FAIL** Δ_full = 0.0133 CI [−0.0089, 0.0337]
  → 規約どおり P1/P2/P3 は全て **void**。treated_frac=0.945、つまり
  **95% のユニットを丸ごと初期化しても clean eval が動かない**
- ただし機構署名は完璧: ゲート再開率 posonly **0.498** vs dironly **0.011**
  （b を戻せばゲートは開き、b が深い負のままなら新方向を入れても開かない）
  → **β/ゲート理論は正しく動いているのに eval は 1 ミリも改善しない**

サニティ: S1–S4 全 PASS（S2 = resume が連続runと bit 一致、S3 = 介入の数値保証
float64 で最大 4.4e-16、S4 = treated hash が4アームで一致）。発散 0 本、全 run 501 点。

---

## 3. 今回の追補が解こうとしている2つの交絡（← 最重要）

**交絡1**: レジーム A の β は `(wᵀµ + b)/‖w‖_Σ` なので、dironly が u を引き直すと
`wᵀµ` も引き直される＝**マージンがランダムに再抽選される**。よって A の dironly は
「新特徴」と「新マージン」を同時投入しており、両者が分離されていない。
→ **posflip**（新特徴を一切引かず、元の方向の符号だけ反転。b_init=0 なので β の符号が
厳密に反転する）を入れれば、posonly との差＝マージン反転のみ、になる。

**交絡2**: 3つの reset アームは全て `v[treated] ← 0` を共有している。v=0 のユニットは
出力に寄与せず、勾配も（δ が v を経由するので）ほぼ止まる。よって
- Δ_arm は全て共通床 V（v←0 だけの便益）を含む → 比 0.704/0.936 は 1 へ膨らむ上限
- P7 の負値は「操舵が戻っても信号が届かない（窒息）」の副作用かもしれない
→ **vzero**（w も b も保持し v だけ 0）で V と Q7 を直接測る。

---

## 4. 実装環境（ここを間違えると全部壊れる）

- **リポジトリ**: `~/Projects/claude/proj_004_drift`（origin は github.com/Issan0511/lop_analysis）。
  **注意**: `~/Projects/claude/lop_analysis` という別クローンもあるが 2026-08-12 で
  止まった古いもの。**必ず proj_004_drift で作業すること**。
- **Python**: `~/Projects/claude/proj_004_drift/.venv/bin/python`（torch 2.13.0）。
  実行は必ずリポジトリルートから `python -m src.<module>` 形式。
- **`OMP_NUM_THREADS=1` を全 run・全解析で必須**（本体 §7 の S1）。
  理由: LAPACK の SVD はスレッド数で集約順序が変わり `eff_rank` 等が %.6g の最下位桁で
  ずれる。トランクとアームが違うスレッド数で走ると bit 一致検査が誤 FAIL する。
- **CPU 並列の落とし穴**: このマシンは28コアだが torch は既定でプロセスごとに
  全コアを掴みに行く。複数プロセスを同時に走らせるなら必ず `OMP_NUM_THREADS` を絞ること
  （絞らないと load average が40超・swap 発生で 100倍遅くなった実績あり）。
  今回は規模が小さいので**単一プロセス逐次で十分**。
- **速度実測（OMP_NUM_THREADS=1）**: A_w100 R=10 で約 2,650 steps/s。
  1アーム 500k step ≒ 3分。追補3アーム（A のみ）なら**10分弱**で終わる。
- **既存実験の bit 互換を絶対に壊さないこと**。新機能は必ず opt-in（既定で no-op）にする。
  検証方法: 既存 config（`configs/methods_sde_0813.yaml` 等）を短く走らせて
  変更前後で出力 CSV が byte 一致することを確認する。

---

## 5. 既存コードの地図（読むべき順）

- **`src/posreset.py`** ← 本体のランナー。今回の主な改修先。
  - `fresh_draws(regime, R, h, d, seed_base, device)`: g を引く。
    `REGIME_SEED_OFFSET = {"A": 0, "B": 1000}` で決定論的。**追補も同じ g を使うこと**
    （同じ関数を同じ引数で呼べば同じ g が出る）。
  - `build_arm_params(net, G32, treated, norm_guard)` → `(arms64, guard)`。
    float64 で計算し、呼び出し側で `.float()` して snapshot に載せる。
    **ここに posflip / vzero / dirkeep を足す**。
  - `RESET_ARMS`: S3 検査の対象アーム集合。
  - `s3_row(i, net, G32, arms64, arms32, treated, guard, tol, c_ref)`: S3 検査。
    **S3a/S3b を足す**。
  - `run_regime(...)`: cont トランク → snapshot → treated 判定 → 介入 → 各アーム resume。
    `--reuse-snapshot` で既存 snapshot を再利用しトランクを再実行しない。
  - `make_probe(treated, acc)` / `new_acc(R)`: unit_traj 用のプローブ。
  - `arm_runs(base_runs, arm)`: run_id に `_<arm>` を付ける。
- **`src/train.py`**: `train_group(gkey, runs, cfg, device, outdir, total_steps, ckpts,
  start_step, resume_state, gname, snapshot_steps, probe, probe_steps)`。
  `probe`/`probe_steps` は本体で追加したフック（既定 None で厳密 no-op）。
  `save_snapshot` / `load_resume` が完全再開（RNG 状態込み）を担う。
- **`src/nets.py`**: `VecMLP`。**`self.v` が読み出し重み**（spec の「a_i」）。
  `self.W`/`self.b`/`self.v`/`self.c`。形状は W [R,h,d]、b [R,h]、v [R,h]、c [R]。
  R = seed 数（10本を並列に走らせるベクトル化）。
- **`src/envs.py`**: `kaiming_mlp_params(R,h,d,gen,device)` が t=0 初期化。b は zeros。
- **`src/lop_metrics.py`**: `compute_lop_metrics`。`eval_loss` が主指標の素。
  `open_frac = (pre>0).float().mean(0)` が p̂。`neg_gate_frac` = (p̂ < 1−dead_tau=0.05) の割合
  → **treated の定義と厳密一致**。
- **`src/figures_posreset.py`**: 本体の判定・作図。paired seed bootstrap の実装
  （`rng = np.random.default_rng(20260819)`, B=10000, percentile 95%）と
  M/M_late の窓定義がここにある。**追補の判定はこれを再利用/踏襲すること**。
  `--selftest` で既知真値フィクスチャによる自己検証が走る（83 チェック）。
- **`src/posreset_posthoc.py`**: 事後アーム間比較。比の CI ガード（分母が 0 を跨ぐ
  bootstrap 標本が 5% 超なら CI 非報告）の家内規約の実装例。

### M / M_late の定義（本体と厳密に揃えること）

- `M` = 窓 `t_int <= step <= t_int + post` の eval_loss 平均（**閉区間、501 点**）
- `M_late` = `t_int + post - 100000 < step <= t_int + post`（**左端排他、100 点**）
- eval グリッドは `lop_every = 1000`（本体と同じ。理由は
  `results/posreset_0819/phase0_summary.md` §5.1 — condA の 10k 格子は課題周期
  T=10000 と同期していて M を 4.7〜29.1% 過小評価するため 1k に統一した）

---

## 6. 設計上の注意（実装前に必ず考えること）

1. **出力先の分離**: 本体ランナーは `resolve_outdir(config)` で config 名から出力先を
   決める。追補は `results/posreset_0819_add/` に出す一方、**スナップショットは
   `results/posreset_0819/snapshots/` から読む**必要がある。この非対称をどう実現するかを
   設計すること（新 config + snapshot ソースディレクトリの指定オプション追加が素直）。
   **本体ディレクトリへの書き込みは一切禁止**（read-only で開く）。
2. **同一 g の再現**: 追補アームは本体と同じ `fresh_draws("A", ...)` の結果を使うこと。
   posflip は g のノルム `‖g_i‖` だけ使い方向は使わない。dirkeep は g の方向を使う。
3. **treated 集合の同一性**: 同じスナップショット・同じ eval バッチ・同じ閾値なので
   自動的に一致するはずだが、**S4a として hash を本体の `intervention_log.csv` の
   `treated_hash` 列と突き合わせて確認すること**。
4. **posflip のガード**: 本体 posonly と同じく `‖w_i‖ < 1e-8` のユニットは方向が
   定義できない。本体は posonly を full にフォールバックしている。posflip も同様の
   扱いにし、件数を記録すること（本走では `n_guard_fallback = 0` だったので実際には
   発生しない見込み）。
5. **vzero と none の関係**: vzero は w も b も触らないので、**S3b は「treated の w,b が
   bit 不変」**を要求する。実装上は snapshot の W,b をそのまま使い v だけ差し替える。
6. **unit_traj の形式**: 本体と同一（`steps`[T] int64, `unit_idx`[U] int64,
   `p_hat`/`w_norm`/`beta`/`cos_u_mu` [T,U] float32, 0次元で regime/seed/arm/t_int/
   treated_hash）。レジーム A では `beta` は NaN、`cos_u_mu` は符号つき cos(w_i, µ) で
   µ = concat(env.flip_state, 0.5*ones(m−f))。
7. **Q1 の統計量**: `Δ_posflip − 0.9·Δ_dironly`。Δ_dironly は**本体の結果**から取る
   （同一 seed・同一スナップショット由来なので厳密ペア）。本体の
   `results/posreset_0819/runs.csv` を読んで join すること。
8. **事後登録の明記**: Q1–Q3 は本体 §6 になかった判定。summary_addendum.md に
   「本追補の commit が実行より前」であることを根拠として明記すること。

---

## 7. 品質基準（この研究室の作法）

- **仕様は凍結。判定基準を実行後に動かさない。** 想定外の結果が出たら、基準を変えるのでは
  なく「どのセルにも落ちなかった」ことを記録する（本走が実際そうなった）。
- **数値は必ず根拠つきで報告**。「PASS」だけでなく点推定と CI を出す。
- **交絡・限界を自分から書く。** 本走の summary は自分で「比は上限としてしか読めない」と
  書いている。同じ水準を保つこと。
- **サニティは検出力を確かめる。** 検査を書いたら、わざと壊した入力（mutant）を作って
  その検査が実際に FAIL することを確認する。本走ではこれで S3 の盲点を3つ潰した。
- **コメント・docstring は日本語**。既存ファイルの密度・トーンに合わせ、
  `[posreset_0819_add §2]` のように仕様の節を角括弧で引用する。
- 迷ったら既存の `src/rank_int.py`（同型の介入実験）と `src/posreset.py` の書き方に倣う。

---

## 8. 完了後の報告

`results/posreset_0819_add/summary_addendum.md` に加えて、**最終メッセージで以下を報告**:

1. Q1–Q8 の結果（点推定・CI・PASS/FAIL/report）
2. §4 帰結マッピングのどのセルに落ちたか（どれにも落ちなければそう書く）
3. 統合 Δ 表（none/posonly/posflip/dironly/vzero/full の M と Δ）
4. S1/S2a/S3a/S3b/S4a の結果
5. 実装で仕様を解釈した箇所（曖昧だった点と、どう決めたか）
6. **`specs/spec_posreset_0819_addendum.md` §8 の「完了後にすること」向けの下書き**を
   `results/posreset_0819_add/followup_drafts.md` に書く。
   §8 の 1〜4 は Obsidian と外部資料の更新で、あなたはそれらにアクセスできない。
   **下書きテキストだけ作れば、後で人間/Claude が貼り付ける。**

## 9. git

- 作業ブランチは切らず `main` で良い（この研究室の慣習）。
- **コミットは実装完了後にまとめて1つ**。メッセージは日本語、本文に判定結果の要約を含める。
- push はしてよい（origin/main、fast-forward のはず）。push 前に `git status` で
  意図しないファイルが混じっていないか確認すること。
- コミットメッセージ末尾に `Co-Authored-By: Codex <noreply@openai.com>` を付ける。
