# proj_004: ドリフト項の非無視性・数値検証 (v2 仕様書実装)

非定常学習におけるミニバッチ勾配の期待値 E[g](ドリフト項)が (1) ゼロに沈まないか、
(2) 入力平均 µ(潮流)の方向を向くか、(3) 入力中心化で消えるか、を重み凍結法で測定する。

- 条件A: Slowly-Changing Regression (Dohare et al. Nature 2024 [D] / Joudaki et al. ICLR 2026 Appx.B [J] 準拠)
- 条件B: ガウス入力・教師切り替え(理論ノート対応、本プロジェクト独自 [NEW])

## 1コマンド再現手順

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy matplotlib pyyaml pandas \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

C=configs/drift_0809.yaml; R=results/drift_0809

.venv/bin/python -m src.run_all --config $C --device cpu  # フル実行 (~35分, CPU の方が速い: 行列が小さい)
.venv/bin/python -m src.make_figures $R                   # $R/figures/ に図 1-7 を生成

# フォローアップ (Part A / task_074) — 再学習不要、既存 $R/ckpts/*.pt から測定
.venv/bin/python -m src.followup $R                       # 拡張凍結測定 -> $R/followup_Eg_*.npz (~20分)
.venv/bin/python -m src.followup_analysis $R              # A1/A2/A4/A5/A6 の CSV と図
```

スモークテスト(数秒): `.venv/bin/python -m src.run_all --config $C --smoke`
(出力は常に `results/_smoke/`。捨てる前提なので `.gitignore` 済み)

## 実験の追加方法

**コードを触らず config を1枚足すだけ**で新しい実験を定義する。

1. `configs/<短いタイトル>_<月日>.yaml` を作る(例: `configs/resid_cos_0820.yaml`)。
   タイトルは英小文字スネークケース2語以内、月日は実行日。同日に取り直したら末尾に `_b` `_c`。
2. `python -m src.run_all --config configs/resid_cos_0820.yaml` を走らせる。
   出力先 `results/resid_cos_0820/` は **config のファイル名から自動決定**される。
3. 図も CSV もその中に出る。図がどの実行のものかは path で確定するので上書き事故が起きない。

この名前(`drift_0809` 等)は怪文書・レポート・Canvas で実験を参照するときにもそのまま使う。

## 原典照合の結果 (configs/drift_0809.yaml にも記載)

公式リポジトリ `shibhansh/loss-of-plasticity`(main, 2026-08-09 取得)で照合:

| 項目 | 値 | 出典 |
|---|---|---|
| β (LTU 閾値) | **0.7** | `cfg/sgd/bp/relu.json` |
| T 標準値 | **10000** | 同上 `flip_after` |
| lr グリッド | **{0.01, 0.003, 0.001}** | 同上 `step_size`(仕様書 v2 の 1e-4 は公式と不一致 → 公式値を採用) |
| 学習器初期化 | 入力層 kaiming_uniform(relu gain)・bias 0、出力層 kaiming_uniform(linear gain)・bias 0 | `lop/nets/ffnn.py` |
| 入力次元 | 実装上 **20**(論文の「定数 1 ビット」は層 bias として実現) | `lop/nets/fix_ltu_net.py` |
| LTU 閾値式 | τ = β(m+1) − S, S = (m − ΣW + b)/2(バイアス込み) | 同上 |
| flip 方式 | 1 ビット一様選択で反転(`flip_one=True` 経路)、初期値 randint{0,1} | `slowly_changing_regression.py` |
| 損失 | `F.mse_loss`(δ² — 勾配に係数 2 が乗る)| `lop/algos/bp.py` |

## 実装上の選択([J] の定義が仕様書に無い/曖昧な箇所)

- **オンライン中心化** (enc=centered): 学習器入力から生入力の EMA(α=0.01, 時定数 100 step)を
  減算。教師は生入力を見る(中心化は学習器側の介入)。減算に使う平均は前ステップまでの値。
  - **A3 確認結果 (2026-08-09)**: 中心化条件でも教師 (LTU / MLP) に渡すのは生の {0,1} 入力で、
    µ̂ の減算は学習器入力にしか掛かっていない (`train.py:111-115` の `y = teacher(x_raw)` /
    `x_in = x_raw - cmask * running_mean`、凍結測定も `freeze.py:99-107` で同じ、eval バッチも
    `train.py:128-130` で同じ)。**タスクは不変で、eval_loss の改善は介入効果と解釈してよい。
    再走は不要。**
- **凍結測定** (`src/freeze.py`): 周期境界(全 T, K は 100 の倍数)でセグメントを切り、
  セグメント内は時間方向に完全ベクトル化した閉形式勾配で積算。中心化系列の EMA は
  下三角 Toeplitz 行列で逐次値を厳密再現(逐次実装と一致することをテスト済み、誤差 ~1e-6)。
  測定用乱数は学習系列から分離。
- **µ̂ の 2 定義** (M2): `cos_inter` = 測定窓全体の生入力経験平均、`cos_intra` = A: 測定終了時の
  flip_state + 0.5(周期内平均の解析値)/ B: 真の µ。cos の基準は生入力ベース
  (中心化系列でも「潮流」との整列を測るため)。
- **Ĉov(e·1_i, x)** (M4): 学習器が実際に見る入力 x_in で計算(標準系列では生入力と同一)。
  `cov_proj` は µ̂_inter 方向への射影。
- **saturated** の「勾配」: ニューロン i のプリ活性勾配 |2δ v_i 1[pre_i>0]| を使用。
- **M3**: `E_adelta` = Ê[a_i δ](係数 2 を除いた出力重み勾配の半分)。
- **発散系列**: lr=0.01 の条件Bなどで学習が発散した系列は NaN として記録される(除外しない)。
- **ベクトル化と再現性**: 条件×シードを R 次元に平坦化して並列学習(各系列は batch=1 の
  plain SGD を厳密に維持)。乱数はグループ単位の generator(入力/教師/初期化/eval で分離)
  なので、再現はグループ単位(config + コード + シードで完全決定)。

## ディレクトリ

```
configs/
  drift_0809.yaml            全パラメータ+出典コメント (実験1つにつき1枚)
src/                         コード (実験に依存しない)
results/
  drift_0809/                実験ディレクトリ = 出力の単位
    config_used.yaml         実行時の config の実体コピー
    meta.json                title / date / git_hash / device / smoke / elapsed_sec
    runs.csv                 130 系列の一覧
    online_loss_*.csv  lop_metrics_*.csv  freeze_global_*.csv  freeze_neurons_*.csv
    followup_A*.csv          Part A の集計
    followup_Eg_*.npz        Ê[g] の生テンソル
    ckpts/                   .pt (gitignore)
    figures/                 図 1–7 と Part A の fig_a1〜fig_a6
  _smoke/                    --smoke の出力先 (gitignore, 上書き前提)
```

- `src/` の内訳 — `envs.py`(SCR / ガウス環境・教師)、`nets.py`(閉形式勾配の学習器)、
  `train.py`(学習ループ)、`freeze.py`(凍結測定 M1–M4)、`lop_metrics.py`(LoP 症状)、
  `run_all.py`(CLI)、`make_figures.py`(図 1–7)、
  `followup.py`(Part A の拡張凍結測定)、`followup_analysis.py`(A1/A2/A4/A5/A6)
- CSV も図も git に commit して残す。どちらも小さく、共有と再現性の証拠になる
- `meta.json` の `git_hash` で「この図を出したコードの状態」が一意に復元できる
- 実行環境(CPU/GPU、smoke か否か)はディレクトリ名ではなく `meta.json` のフィールドで表す

## Part A 追加解析の CSV スキーマ

- `followup_A1_pairwise.csv`: run_id, ckpt, `pcos_*`(生の符号付き pairwise cos),
  `vpcos_*`(sign(v_i)sign(v_j) 補正版), `vpcos_perp_alive_mean`(µ̂ 成分を抜いた残差),
  `eig1_frac*`(単位化 Ê[g_i] の Gram 最大固有値/h = 共線性), `n_alive`, `finite`
- `followup_A1_hist.csv`: 条件×ckpt×kind(raw/vsigned) の 40 ビンヒストグラム(alive のみ)
- `followup_A2_deadcross.csv`: cos 符号(step 1e4)× dead(step 1e6) の分割表
- `followup_A4_ckptcos.csv`: 連続 ckpt 間の cos。`cos_null` は「別 run と組ませた」対照
- `followup_A5_period.csv`: `ratio` = ‖Ê[g]‖/mean_τ‖ḡ_τ‖ と非整合予測 `ratio_pred` = 1/√n
- `followup_A6_splithalf.csv`: 同一サンプル版と split-half 版の |cos|、`cos_chance` = √(2/πd)

## CSV スキーマ(抜粋)

- `freeze_global_*.csv`: run_id, ckpt, M, n_periods(50 未満なら周期を跨げていない),
  noise_floor(1/√M), snr_all, snr_W/b/v/c(レイヤー別 SNR)
- `freeze_neurons_*.csv`: run_id, ckpt, neuron, w_norm, snr_i, cos_intra, cos_inter,
  E_adelta, sign_v, cov_norm, cov_proj
- `lop_metrics_*.csv`: dead_frac, dup_frac, sat_frac, eff_rank, stable_rank,
  sign_match_mean, sign_clone_frac, eval_loss(いずれも [J] Appx.B 定義、固定バッチ計測)
