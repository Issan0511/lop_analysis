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
.venv/bin/python -m src.run_all --device cpu   # フル実行 (~1時間, CPU の方が速い: 行列が小さい)
.venv/bin/python -m src.make_figures           # figures/ に図 1-7 を生成
```

スモークテスト(数十秒): `.venv/bin/python -m src.run_all --smoke`

## 原典照合の結果 (config.yaml にも記載)

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

- `config.yaml` — 全パラメータ+出典コメント
- `src/` — `envs.py`(SCR / ガウス環境・教師)、`nets.py`(閉形式勾配の学習器)、
  `train.py`(学習ループ)、`freeze.py`(凍結測定 M1–M4)、`lop_metrics.py`(LoP 症状)、
  `run_all.py`(CLI)、`make_figures.py`(図 1–7)
- `results/` — CSV(`runs.csv`, `online_loss_*`, `lop_metrics_*`, `freeze_global_*`,
  `freeze_neurons_*`)、`config_used.yaml`、`meta.json`(git hash, デバイス)、`ckpts/`
- `figures/` — 仕様書 §6 の図 1–7

## CSV スキーマ(抜粋)

- `freeze_global_*.csv`: run_id, ckpt, M, n_periods(50 未満なら周期を跨げていない),
  noise_floor(1/√M), snr_all, snr_W/b/v/c(レイヤー別 SNR)
- `freeze_neurons_*.csv`: run_id, ckpt, neuron, w_norm, snr_i, cos_intra, cos_inter,
  E_adelta, sign_v, cov_norm, cov_proj
- `lop_metrics_*.csv`: dead_frac, dup_frac, sat_frac, eff_rank, stable_rank,
  sign_match_mean, sign_clone_frac, eval_loss(いずれも [J] Appx.B 定義、固定バッチ計測)
