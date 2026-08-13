# spec_rank_int_0814: 実験(4) 低ランク崩壊への介入（SVD ランク回復 vs 基底保持シャッフル）

proj_004 / 作成 2026-08-14 / 対象リポジトリ: lop_analysis
位置づけ: 先生提案実験 (4) の実装仕様。**Phase 1 実行前に本仕様（特に §6 判定基準）を commit すること（事前登録）。**

---

## 0. 背景とレジーム選定の根拠（要約）

srank 低下と可塑性喪失は相関しか示せていない。ランクだけを人為的に回復させ、可塑性が戻るかを介入で判別する。摂動一般の効果（Shrink & Perturb 型）と区別するため、「同量の手術をするがランクは戻さない」対照アームを置く。

レジームは `coupling_fbw_0813` の P2′ に基づき **condA / batch=full / width {10, 20}** とする:

- srank_alive 崩壊 t50 ≈ 30–37k、dead t50 ≈ 410–490k → 「ランクは崩れたが dead は未蓄積」の窓が数百 k step 開いている
- 最終 eval_loss 0.42 (w10) / 0.23 (w20) で LoP が存在
- b=1 広幅は dead が先行/同時（w100 で反転）のため **不適格**（介入ノイズによる dead 復活交絡）
- w5 はランク上限 5 で回復余地が小さいため副次のみ

full-batch（exp A は全サポート厳密列挙）は決定論なので、同一 seed のアーム間差は介入操作のみに帰着できる。

---

## 1. 実行前に読むファイル

- `src/train.py`（train_group、save_ckpt、postswitch 計測、full_support 分岐）
- `src/nets.py`（VecMLP.state_dict / load_state / grads_batch）
- `src/lop_metrics.py`（stable_rank_W_alive、dead_frac の定義）
- `configs/coupling_fbw_0813.yaml`（本実験の土台。common: lr_main=0.01, dead_tau=0.95, eval_batch=2000, lop_every=1000, loss_bin=1000, seeds 0–4）
- `results/coupling_fbw_0813/`（Phase 0 の入力データ: lop_metrics_A_w10_bfull.csv 等、t50_runs.csv）

---

## 2. Phase 0: 既存データ解析（再学習なし）

`results/coupling_fbw_0813/` の A_w10_bfull / A_w20_bfull（seed 0–4）に対して:

1. **t_int の妥当性確認**: 主 t_int = 150,000、副 t_int = 300,000。各 seed で
   (a) stable_rank_W_alive が t_int 時点で t50 を通過済みであること、
   (b) dead_frac(t_int) ≤ 0.15 であることを確認。満たさない seed があっても除外せず、記録して層別報告。
2. **回復 / 予防のラベル付け**: eval_loss の離陸時刻（定義: 1M 時点値と初期値の半値を最初に上抜く step。1k 移動平均で平滑化）を seed 別に算出。t_int より前に離陸 → その seed は「回復」実験、未離陸 → 「予防」実験。混在してよいが summary で必ず区別する。
3. **B アームの目標ランク**: step=0 の stable_rank_W_alive を seed 別に記録し、介入の目標値 `srank_target` とする。

出力: `results/rank_int_0814/phase0_summary.md` と `phase0_targets.csv`（seed, width, srank_t50, dead_at_tint, evalloss_takeoff, label, srank_target）。

---

## 3. Phase 1 実装: resume 機構

現状 save_ckpt は checkpoints で指定した step に呼ばれる。以下を満たす完全再開スナップショットに拡張する:

- 保存対象: W, b, v, c、env の内部状態（現在の flip 状態・step カウンタ）、teacher 状態、running_mean、RNG 状態
- `train_group` に warm-start オプション（スナップショット path + 開始 step）を追加

**サニティ S1（必須）**: 連続 run（0→350k）と、150k で保存→resume した run（150k→350k）の lop_metrics が**全行一致**すること（full-batch 決定論なので厳密一致が要求できる。float 誤差を許すなら相対 1e-10 以内）。S1 が通るまで Phase 1 本体に進まない。

---

## 4. Phase 1 実装: 介入アーム

t_int はタスク境界（period=10,000 の倍数）に一致させ、処理順を「**介入 → タスク切替 → 学習継続**」に固定する（介入直後の新タスク適応を測るため）。介入対象は **W のみ**（b, v, c は不変。srank の主張は W についてのものだから）。

W = U S Vᵀ（thin SVD、特異値 s_1 ≥ … ≥ s_h）として:

### arm = none（対照 A）
介入なし。連続 run そのもの（S1 が通っていれば pre 区間の run を延長するだけでよい）。

### arm = svdrec（条件 B: ランク回復、ノルム・基底固定）
1. s'_i = max(s_i, ε·s_1)。ε は「介入直後の stable_rank_W_alive ≈ srank_target（Phase 0 の step0 値）」となるよう bisect（stable rank は ε に単調増加）
2. Frobenius ノルムを介入前と一致させる一様再スケール
3. W' = U S' Vᵀ（U, V 不変）

### arm = shuffle（条件 C: 基底保持・ランク不変の等量摂動）
1. エネルギー 99% を張る最小の k を取る（Σ_{i≤k} s_i² / Σ s_i² ≥ 0.99）
2. 乱数 skew-symmetric G ∈ R^{k×k}（seed 固定、‖G‖_F=1 に正規化）、Q(θ) = expm(θG)
3. W'(θ) = U_k Q(θ) S_k V_kᵀ + U_⊥ S_⊥ V_⊥ᵀ
   → 列空間 span(U_k)・行空間 span(V_k)・特異値・Frobenius ノルム・ランクすべて不変、重みだけが変わる
4. θ を bisect して **‖W'−W‖_F を svdrec の ‖ΔW‖_F に一致**させる（相対誤差 < 1e-6）。θ ∈ (0, π] で到達不能なら abort して報告（top-k のエネルギーが大きいため通常は到達可能なはず）

注記（先生への確認事項として summary に転記）: 先生の記述は「空間内要素のみシャッフル」だが、ΔF マッチングを可能にするため連続化（部分空間内ランダム回転）を採用した。

### 介入ログ `intervention_log.csv`
seed, width, arm, t_int, ‖ΔW‖_F, ε, θ, k, 介入前後の stable_rank_W_alive / dead_frac / ゲート開閉数（eval_batch での発火ユニット数変化）。

**サニティ S2**: (i) shuffle の特異値変化が相対 1e-6 以内, (ii) svdrec と shuffle の ‖ΔW‖_F 一致が相対 1e-6 以内, (iii) 両アームの ‖W'‖_F 保存。

---

## 5. Phase 1 実行計画

- セル: condA × batch=full × width {10, 20} × seed {0–4} × arm {none, svdrec, shuffle} × t_int {150k}（副次: t_int=300k は主判定の後、必要なら）
- pre 区間（0 → t_int）は seed × width ごとに 1 本（決定論なので全アーム共有）。t_int でスナップショット保存
- 介入後 **200k step（20 タスク）** 継続。ログは既存踏襲（lop_every=1000、loss_bin=1000、coupling.postswitch_n=10 も併録）
- 同一 seed のアーム間で env / teacher の系列が完全同一であることを要件とする（介入以外の差をゼロに）
- 出力規約: `configs/rank_int_0814.yaml`、`results/rank_int_0814/`（runs.csv、lop_metrics_*.csv、online_loss_*.csv、postswitch_err_*.csv、intervention_log.csv、summary.md）

---

## 6. 判定基準（事前登録 — 実行前に固定）

主指標 M: 介入後 20 タスクそれぞれの online loss タスク内平均（loss_bin から算出）を、タスクについて平均した seed 別スカラー。
副指標: 各タスク末尾 2k step の平均 loss（到達精度）、eval_loss 時系列、dead_frac、stable_rank_W_alive。
検定はすべて paired seed bootstrap（n=5、95%CI）。width 別に判定し、両 width で同符号なら「頑健」と記す。

- **P-int-1**: M(svdrec) < M(none)、差の CI がゼロ非含有 → ランク回復に効果あり
- **P-int-2**: M(shuffle) − M(none) の CI がゼロ含有、または M(svdrec) < M(shuffle) がゼロ非含有 → 摂動一般では説明できない
- **判定表**:
  - P-int-1 ∧ P-int-2 → **ランク因果を支持**（実験 (4) の本来の目標）
  - svdrec ≈ shuffle < none → 摂動効果。ランク因果は不支持
  - 全アーム同等 → この regime では回復不能。Phase 0 ラベルが「予防」の seed が多い場合は P-int-3 に主軸を移す
- **G1（ガード）**: 介入直後の Δdead_frac の svdrec − shuffle 差の CI が ±0.05 内。破れたら dead 復活交絡を疑い、感度分析（alive 行のみに SVD 介入を適用する版）を追試
- **P-int-3（予防読み出し = 連鎖仮説 (5) の介入テスト）**: t_int → t_int+200k の dead_frac 増分が svdrec < none（CI ゼロ非含有）、かつ shuffle ≈ none → 「srank 低下 → dead 蓄積」経路の介入的証拠
- **反証条件**: svdrec が介入直後に srank_target の 80% 以上までランクを回復させたにもかかわらず、M も dead 増分も none と区別できない → **この regime において低ランクは LoP の原因ではなく随伴症状**、と結論して報告する

---

## 7. 図（figures_rank_int.py）

1. アーム別 online loss vs step（t_int 起点、タスク境界線入り、seed 平均±SE）
2. stable_rank_W_alive / dead_frac 時系列の 3 アーム重ね描き（介入時点マーカー）
3. M の paired 差（svdrec−none、shuffle−none）の CI 森プロット（width × t_int）

---

## 8. summary.md に必ず含めること

- Phase 0 の回復/予防ラベル分布（結論の主張文言がこれで変わる）
- 判定表のどのセルに落ちたか + G1 の結果
- 先生への確認事項 2 点: (i) shuffle の連続化（部分空間内回転）の妥当性、(ii) ε の決定規則（step0 srank 復元）
- 全サニティ（S1, S2）の合否
