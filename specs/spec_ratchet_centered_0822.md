# spec_ratchet_centered_0822: プローブ付き centered アーム（消灯点の入力統計依存性）

proj_004 / 作成 2026-08-22 / 対象リポジトリ: lop_analysis

位置づけ: [[フレーム前の穴]] §6「Q3 の詳細（前-1）」(c) 消灯点のスケーリング（task_104）の実装仕様。Q3（片側消灯）は機構節三本柱のうち「なぜ止まるか」を担う最弱の柱で、(a) 誤差棒と (b) ‖w‖ 層別は既存データで済む一方、**(c) 消灯点が入力統計から予言できるのか・この設定固有かだけが新規走を要する**。**Phase 1 実行前に、§5.4 の事前予測欄を記入した上で本仕様（特に §5 集計・§6 判定基準）を commit すること（事前登録）。** 走の要否自体は [[フレーム前の穴]] §9-5 のとおりフレーム決定後に判断する（本仕様はその Go に備えた準備）。

---

## 0. 一行

ratchet_log_0819（condA A_w100・std）と**乱数実現まで同一**の走を enc=centered（running_mean 減算、center_alpha=0.01）に切り替えて 10 seed × 1M step 取り直し、p̂ / cos のプローブログ付き centered アームを作って、ゲート曲線の消灯点（p̂=0 になる cos の上端、std では ‖w‖ 四分位で −0.15 / −0.20）が入力平均に追随して動くか・動かないかを判別する。

## 1. 背景と問いの構造

- Q3 の主張: 移動度は片側消灯（p̂ < 0.05 で滑落 ≈ 0）。std のゲート曲線で cos < −0.15 → p̂ = 0.000、−0.20 より下は 10 seed 全部で厳密ゼロ。ガウス β 近似の −0.46 より遥かに急峻で、condA は有界入力なので文字通りの吸収が実在【(a)(b) は既存データの事後計算で確認済み・未事前登録】
- **開いている問い（本実験）**: この消灯点の**値**は入力統計（µ̂ の大きさ・向き）から予言できる量なのか、それとも m=20/f=15/std という設定に固有の定数なのか。condA 側で入力平均を動かすレバーは enc（std / centered）のみ
- centering の効果自体は既測（dead 0.964 → 0.294 [drift_0809]）だが、**ratchet_log_0819 は std のみで、centered アームには p̂ / cos のプローブログが無い**。よって新規走
- `c` は condB 専用の入力平均パラメータ（µ = c/√d·1、`src/envs.py:176`。condA の run は `c=None`、`src/common.py:73`）であり**使わない**。Q3 の主張は condA スコープで、condB への外挿は禁止（[[論点マップ]] 未カバーの穴 ②）

**判別の構造（結論は先取りしない）**: centered では学習器入力が x_in = x − running_mean になり、µ̂ = E[x_in] = E[x] − running_mean。この µ̂ の大きさ・時間挙動は std と異なりうるため、「消灯点が動く／動かない／そもそも cos 軸が可比でない」の三つ組を §6 の手続きで判別する。どれが出るかの予測は §5.4 の記入欄に**実行前に**書く。

## 2. 実行前に読むファイル

- `specs/spec_ratchet_log_0819.md` ＋ `configs/ratchet_log_0819.yaml` — 土台。設計・グリッド・記録量・S1–S4 は全て継承
- `src/train.py` L349–361 — centered の配線。x_in = x_raw − cmask·running_mean は**更新前**の running_mean で計算され、EMA 更新（alpha=0.01）は同一反復内その直後。probe はループ本体先頭で呼ばれるので、その step の forward が使うのと同じ running_mean を読む（整合）。教師は生入力を見る（中心化は学習器入力の前処理）
- `src/ratchet_log.py` — `exact_record`（µ̂ 以下を x_in から数値計算。0819 §3.1 の enc=std 専用ガードは本実験で解禁）、`full_support_ro`（読み取り専用列挙）、S2/S3/S4 の実装と運用判定基準
- `src/common.py:57–100` — build_runs（condA は c=None、enc 軸）、group_runs（enc はグループキーに含まれず単一グループ A_w100 のまま）
- `results/ratchet_log_0819/logs/seed{0..9}.npz` — std 側の比較データ（commit 済み）。§5 の集計は両アームともこの npz 形式だけを読む
- [[フレーム前の穴]] §6 — (a)(b) の事後計算値（参照値であって判定基準ではない、§5.3）

## 3. 設計

### 3.1 レジーム（1つのみ）

condA A_w100: m=20, f=15, T=1e4, **encoding=centered（唯一の変更）**, center_alpha=0.01, width=100, batch=1, lr_main=0.01, seed 0–9。config は `configs/ratchet_centered_0822.yaml`（ratchet_log_0819 と encodings / spec キー以外同一）。**condB へは外挿しない**（スコープ）。

### 3.2 走行

トランク 0 → 1,000,000 step、介入なし、seed 10 本、probe フックで読み取り専用ロギングのみ（S2 で無擾乱を保証）。分岐アームなし。

**ペア設計（本実験の設計上の利点）**: `make_gens` は exp/width からしか generator を seed しないため、R=10・seeds [0..9] を保つ限り、init・教師・入力列・flip 軌道は **ratchet_log_0819 と bit 一致**する。差は encoder だけ。準備時に R=10 短走（30k step）で flip_state の全共通記録点 bit 一致・step 0 の w_norm/b/v/cos_u_mu/p̂/mu_norm 一致を確認済み。したがって std↔centered は seed 単位で**対応のある比較**になる（§5.3 の paired bootstrap の根拠）。**R や seeds リストを変えるとこのペアリングは壊れる**（スモークの R=1 は非ペア）。

### 3.3 ロギンググリッド

ratchet_log_0819 §3.3 と同一: 境界 [−100, +100] を毎 step ＋ バルク 1000 step ごと、計 20,901 記録点/run。

### 3.4 記録量

ratchet_log_0819 §3.4 と同一キー（`cos_u_mu / p_hat / w_norm / b / v / F_self / F_rest / F_gate`、run レベル `G / flip_state / E_delta / mu_norm / ratio_mu_cov / cos_G_mu / G_dot_mu / eval_loss_exact`）。float64 計算・float32 保存。保存先 `results/ratchet_centered_0822/logs/seed{k}.npz`。

**サイズ見込み**: 非圧縮 ≈ 70 MB/seed。std 実績 83 MB（10 seed 計）は dead 96.4% の凍結値が圧縮で潰れた結果であり、centered は dead が少ない既測（0.294）のため**数百 MB 級（上限 ~700 MB）**を見込む。ディスクを 1 GB 確保しておく。

### 3.5 µ̂ の定義（本実験で唯一意味が変わる量）

µ̂_t = E[x_in] を **32 パターンからの数値計算**で取る（実装は enc に依らず共通）。std では従来どおり flip_state ‖ 0.5·1、centered では E[x] − running_mean_t。cos_u_mu / cos_G_mu / F の µ̂ 射影はすべて x_in 座標系・単位ベクトル µ̂/‖µ̂‖ に対する量で、‖µ̂‖ は mu_norm として保存されるので生の内積は事後復元できる。**centered の cos 軸が std の cos 軸と直接比較可能かどうか自体が判別対象の一部**（§6 C1）。

## 4. Phase 0（本走前に完了。準備時 2026-08-22 に実測済みの項目は値を記す）

1. **恒等式サニティ（centered 版）【済】**: enc=centered で 2k step（seed 0–1）学習した状態の exact_record を、本番勾配コード `nets.VecMLP.grads_batch` を x_in 上で float64 実行した独立経路と突き合わせ。実測: G 相対誤差 0.0、F_gate 4.45e-16、分解閉包 3.71e-16、いずれも判定 1e-10 を PASS（ratchet_log_0819 phase0 §4.1 と同形の検査）
2. **グリッド健全性スモーク【済】**: seed 0 × 30k step ＋ `--s2-steps 10000`。正常終了、S2 PASS（probe あり/なし bit 一致。state_hash は running_mean を含むので、probe が running_mean を読んで書かないことも検査対象に入っている）、S3 PASS（max|z|=2.39）、S4 PASS（境界 3・flip 遷移 2）。npz は 22 キー全て期待形状・NaN/Inf ゼロ、p̂ は [0,1] 内の厳密な 1/32 倍数、run_id = `A_w100_T10000_centered_lr0.01_s0`。成果物は repo 外（`~/q3_out/c/smoke/`）に置き、results/ には残していない
3. **ペア設計の確認【済】**: §3.2 のとおり（R=10 短走 vs `results/ratchet_log_0819/logs/seed0.npz`）
4. **ゲート曲線・消灯点は読まない**: スモークおよび準備段階では上記の構造チェックのみ行い、centered の p̂ × cos の結合集計は一切しない（本走・事前登録後まで答えを見ない）

## 5. 集計（事前登録）

集計対象サンプル = 各アームの (記録点 t × unit i × seed k) の三つ組 (cos_u_mu, p̂, ‖w‖)。全 20,901 × 100 × 10 ≈ 2.09e7 点/アーム、時間による除外はしない。std 側は `results/ratchet_log_0819/logs/` を**本節の推定量で引き直す**。

### 5.1 ゲート曲線

- cos ビン: 幅 0.05、区間 [−0.60, +0.60) の 24 ビン（事前固定）。範囲外サンプルは曲線に入れず件数のみ報告
- 有効ビン: プール後サンプル数 ≥ 1000 のビン
- 曲線値: ビン内 p̂ の**中央値**
- CI: **seed 束ねブートストラップ** B=10,000、`rng = np.random.default_rng(20260822)`（config の bootstrap_seed）。seed 10 本を復元抽出 → 選ばれた seed の全サンプルをプール → ビン中央値を再計算 → percentile 95% CI
- **‖w‖ 四分位層別**: 各アーム内でプールした ‖w‖ の四分位境界を取り（境界値は報告）、各層で同じ曲線・同じ CI。層はアーム内の相対位置（絶対値は enc で動くため）

### 5.2 消灯点の推定量（事前固定）

有効ビンを cos 昇順に並べ、

- **θ̂_med（主）** = 「その上端以下の**すべての**有効ビンで中央値 p̂ = 0」が成り立つ最大のビン上端。最下位の有効ビンから既に中央値 > 0 なら θ̂_med = NA（消灯領域なし）
- **θ̂_all（副・厳格版）** = 同じ構成で「全サンプル p̂ が厳密に 0」（p̂ は 1/32 の倍数なので = 0 は厳密判定）。θ̂_med 以下のビンでの非ゼロサンプル数も報告
- CI: 5.1 と同じ bootstrap resample ごとに θ̂ を再計算した分布の percentile 95% CI（0.05 格子上の離散分布になる）
- 全体＋‖w‖ 四分位の各層で算出。副次として時間半割（t < 5e5 / t ≥ 5e5）でも θ̂_med を出し、定常性の確認に用いる（報告のみ）
- 頑健性（報告のみ）: プール中央値の代わりに「seed 別中央値の seed 間中央値」を使う層化版 θ̂_med^strat も併記する（[[フレーム前の穴]] §6(b) の事後計算を再実装した `analysis/q3_gate_curve_ci.py` と同じ作法。主判定はあくまで θ̂_med）

### 5.3 std との比較

- **Δθ̂ = θ̂_centered − θ̂_std**（全体および四分位層別）。§3.2 のペア設計に基づき **paired bootstrap**: 同一の seed 復元抽出を両アームに適用して Δθ̂ を再計算、B=10,000、rng は 5.1 と共通、95% CI
- 参照値の規律: [[フレーム前の穴]] §6(a)(b) の −0.15 / −0.20 は**事後計算・未事前登録**。比較の基準は本節の推定量で引き直した θ̂_std のみとし、引き直し値と (a)(b) の値がずれた場合はずれ自体を summary に記録する

### 5.4 事前予測の記入欄（**実行前に記入して commit**。どれが出るかを本仕様は予断しない）

選択肢の並置:

- **予測A（入力統計説）**: 消灯点は µ̂（enc）に追随して動く — Δθ̂ ≠ 0。向き・大きさの見込み: ＿＿＿＿
- **予測B（設定固有説）**: 消灯点は std の値（−0.15〜−0.20）のまま — Δθ̂ ≈ 0
- **予測C（軸退化）**: centered では cos(u, µ̂) 軸の構造自体が失われ、ゲート曲線が std と可比な形にならない（§6 C1 で不可比）

> **Issa の事前予測**: **B（設定固有説）**。Δθ̂ ≈ 0、centered でも消灯点は std と同じ −0.15〜−0.20 付近に残る。
> **根拠**: centered は初期の `running_mean = 0` なので step 0 の実効入力 `x_in` は std と同一であり、std では cos < −0.15 に生存ユニットが一つもいなかった。centering は消灯点へ到達するユニットの数を減らしても、消灯点そのものは変えないと予測する。
> **記入日時**: 2026-08-22 18:54 JST（本走前。commit 時刻を最終根拠とする）

## 6. 判定基準（事前登録・commit 前に固定）

| ID | 問い | 判定手続き | 帰結の書き分け |
|---|---|---|---|
| C1 | centered のゲート曲線は std と可比か | (i) θ̂_med（全体）が NA でない、かつ (ii) cos < 0 側の有効ビンが 4 個以上 | 満たさなければ**不可比**: C2 の数値比較は行わず、曲線の形と mu_norm の記述（E1）だけを報告して「消灯点は cos 軸ごと入力統計依存」とは**言わない**（予測C の実現として記録するのみ） |
| C2 | 消灯点は動いたか（主判定） | Δθ̂_med（全体）の paired bootstrap 95% CI | **追随**: CI が 0 を含まない ／ **固有**: CI ⊆ [−0.05, +0.05] かつ 0 を含む ／ **保留**: それ以外（1 ビン幅 0.05 = 分解能。保留は保留と書く） |
| C3 | 層別で一貫するか | 四分位 4 層の Δθ̂_med の符号と C2 の方向の一致数を報告 | 報告のみ。C2 を覆さない（層別は交絡の記述であって主判定ではない） |
| E1 | 探索的（PASS/FAIL なし・報告のみ） | 推定対象のみ事前固定: (i) 両アームのゲート曲線の重ね描き、(ii) mu_norm の記述統計（中央値・IQR・境界直後 [0,+100] の挙動）、(iii) 最終 dead_frac（p̂<0.05 基準。centering 既測 0.294 との整合確認、判定には使わない） | — |

判定は `results/ratchet_centered_0822/verdict.csv` に C1–C3 と θ̂ 値（点推定・CI）を記録。E1 は summary.md。**dead_frac は判定に使用しない**（0819 と同じ規律）。

## 7. サニティ（事前登録）

ratchet_log_0819 §7 を全て継承（運用判定基準は `src/ratchet_log.py` 実装のもの）:

- **S1**: `OMP_NUM_THREADS=1`
- **S2（無擾乱）**: `--s2-steps 100000` で probe あり/なしの最終 state（net・env・running_mean）が bit 一致。centered では probe が running_mean を読むため、この検査が「読むだけで書かない」ことの保証になる
- **S3（厳密 vs 経験）**: 記録点 3 箇所で厳密 p̂ と eval_batch=2000 経験値を突き合わせ（経験側も同じ centering を通る実装なので判定はそのまま有効）。median|z| ≤ 1.0 かつ |z|>3 個数の二項上側 p ≥ 0.001 かつ退化ユニット厳密一致
- **S4（境界検出）**: flip が t ≡ 0 (mod 10⁴) 直後のみ・1 ビット

## 8. 実行

- 実装: `src/ratchet_log.py`（本実験のための変更は §「Claude Code 実行手順」の diff 一覧のとおり）＋ `configs/ratchet_centered_0822.yaml`
- 規模: 10 run × 1M step、CPU。probe コストは std と同一（cmask 演算は enc に依らず常時実行）
- 順序: Phase 0（§4、済）→ **§5.4 記入 → spec・config・コード diff を commit（事前登録）** → 本走 → S2–S4 確認 → 集計モジュール実装・commit → verdict.csv / summary.md
- 出力: `results/ratchet_centered_0822/{logs/, meta.json, config_used.yaml, verdict.csv, summary.md, figures/}`

## 9. 禁止事項・留保（本文に転記すること）

- **condB への外挿禁止**。主張スコープは「condA・w100・T=1e4・batch=1・center_alpha=0.01」。`c` スイープは対象違い（condB）なので行わない
- alpha（EMA 時定数）依存性は本実験では測らない。消灯点が動いた場合に「入力統計から予言できる」の定量式（何の関数か）を立てるのは次段の仕事であり、本実験の主張は Δθ̂ の有無と向きまで
- centered の cos 軸は µ̂_t = E[x] − running_mean に対する量で、‖µ̂‖ のスケール・時間挙動が std と異なりうる。**消灯点の比較は無次元の cos どうしで行うが、その比較が適切かどうか自体を C1 で先に判定する**。mu_norm を常に併記する
- std 側参照値（−0.15 / −0.20）は事後計算・未事前登録。本仕様 §5 の引き直し値のみを比較に用いる
- ペア設計（§3.2）は R=10・seeds [0..9] の維持が前提。部分再走・seed 追加をした場合は対応が壊れるので paired bootstrap を非ペア版に落とすこと
- 本走前に centered の p̂ × cos 結合集計を行わない（§4-4）。スモーク・準備段階の検証は構造チェックに限る

---

## Claude Code 実行手順

Issa のマシンの Claude Code に渡す実行台本。**上から順に**。

### 前提（このコミットに含まれるもの）

- 本ファイル `specs/spec_ratchet_centered_0822.md`（§5.4 記入済みであること）
- `configs/ratchet_centered_0822.yaml`（ratchet_log_0819 と encodings / spec キー以外同一）
- コード diff（2 ファイル・実験条件に影響するのは ratchet_log.py の 2 点のみ）:
  - `src/ratchet_log.py` — (1) `exact_record` の enc=std 専用ガード（NotImplementedError）を撤去し、µ̂ = E[x_in] の数値計算に一本化（std 側の出力は bit 不変。docstring 更新込み）。(2) meta.json の `spec` 参照を config の `spec:` キーから取るように（既定値は従来どおり 0819）
  - `src/ratchet_log_phase0.py` — コメントの事実更新のみ（挙動不変）
- 依存: torch / numpy / pyyaml（README の venv 手順どおり）。GPU 不要（device: cpu）

### 1. 事前登録コミット（走らせる前に必ず）

```bash
cd <repo>
git add specs/spec_ratchet_centered_0822.md configs/ratchet_centered_0822.yaml \
        src/ratchet_log.py src/ratchet_log_phase0.py
git commit -m "<チャット名>: spec_ratchet_centered_0822 事前登録 (config+コード)"
```

repo の規律: **spec（判定基準・事前予測記入済み）を commit してから走らせる**。commit 時刻が「予測が実行前に固定されていた」ことの唯一の根拠になる。§5.4 が空欄のまま commit しない。

### 2. 本走

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.ratchet_log \
    --config configs/ratchet_centered_0822.yaml --s2-steps 100000 \
    2>&1 | tee run_ratchet_centered.log
```

- 出力先は config 名から自動で `results/ratchet_centered_0822/`（--outdir 不要）
- **想定所要時間: 約 7–9 分**（std 実績: train+probe 363 s ＋ S2 で全体 441 s [results/ratchet_log_0819/meta.json]。centered の step 当たりコストは std と同一。参考: 準備コンテナ（遅め）では R=10 × 30k が 39 s → 単純外挿で本体 ~22 分 ＋ S2 ~5 分が上限の目安）
- ディスク: logs は数百 MB 級になりうる（§3.4）。1 GB 確保

### 3. 実行直後の確認（集計より前）

- 標準出力 / `results/ratchet_centered_0822/meta.json` で **S2 / S3 / S4 が全て PASS** であること、`n_record_steps=20901`、`n_realized_flips=99`、`spec` が本ファイルを指すこと
- `logs/seed{0..9}.npz` が 10 本、キー欠落・NaN が無いこと（Phase 0-2 と同じ構造チェック。p̂ × cos の結合集計はまだしない）

### 4. 集計と転記

- §5 を唯一の正として集計モジュールを実装（`analysis/` 配下、surv_hist の流儀）し、**commit してから**走らせる
- `verdict.csv` / `summary.md` へ転記する際の規約: **転記元は commit 済みコードの出力であることを確認する**（[[命題リスト]] 運用メモ 8/22。git status がクリーンな状態で集計を再実行できること）。§5.3 のとおり std 引き直し値と [[フレーム前の穴]] §6(a)(b) の事後計算値のずれも記録
- 成果物 commit 後、[[命題リスト]] Q3 行と [[フレーム前の穴]] §6(c) の状態を更新（チャット名入りコミット）

### してはいけないこと

- spec commit 前に走らせる / §5.4 空欄のまま走らせる
- スモークや途中確認で centered の p̂ × cos を集計する（消灯点の答えを先に見ない）
- `results/ratchet_log_0819/` 配下の std 成果物を上書き・再走する（比較基準は commit 済みの npz）
- seeds / R / total_steps を変える（ペア設計が壊れる。変えるなら spec 改版）

