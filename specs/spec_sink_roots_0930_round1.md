# sink_roots_0930 追補 1 — 本体の導出の一巡目が出した走（登録・走る前）

書いた: Claude（fork2）、2026-09-30。依頼: 本体セッション「ELUが沈降しない事件」（0930 のメッセージ）。依頼の全文と予言は、原文のまま `specs/sink_roots_0930_round1_requests.md` に写した（出所 `obsidian-research-data/drive_recon_0930/run_requests_round1.md`。各組の導出の全文は同じフォルダの `derivations_round1/<組>/final/result.md`）。**予言はこの写しが正本**で、走の前に commit する。加えて adam の組の `PREDICTIONS.md`（`src/sink_roots_round1/adam/`）、return の組の `PREDICTIONS_wcap.txt`（`src/sink_roots_round1/return/`）も写した。

## 1. 今回回すもの（本体の優先順位 1〜4）

| 順 | 依頼 | 実装 | 腕 |
|---|---|---|---|
| 1 | valley R_eps_wall | `src/sink_roots_round1/valley/alpha_ladder_1layer.py`（elu_alpha_ladder_1layer_0925 の v4 の写し、`--eps1`） | GELU・SiLU × ε₁ ∈ {1e−6, 1e−12} × seed 0・1 × 200 課題（ε 1e−8 は既存の v4 の runs） |
| 2 | return Q1_replacement_branch | `src/sink_roots_round1/return/rho_test.py`（写し、変更なし） | ELU seed 0（既定: 課題 6・8・10・12・14、+400 → 2,000）、leaky seed 0 `--w0 500` |
| 3 | return R6_ink_normalize | `src/sink_roots_mnist_0930.py --ink-normalize`（R3 と同じ記録・切替時の状態を課題 5・10・20・30 で保存） | ELU・leaky・SiLU × seed 0〜2 × 30 課題 |
| 3 | return R1_W1_rownorm_cap | `src/sink_roots_mnist_0930.py --wcap-from 5` | ELU・SiLU × seed 0〜2 × 30 課題 |
| 4 | sign replay_adam_gap_2layer | `src/sink_roots_round1/sign/replay.py`（写し、出力先だけ環境変数 REPLAY_OUT で生データの場所へ） | 状態 L2_{ELU,LR,SILU}_K10_s{0,1}_t00{2,3,4,6}（24）と L1_{ELU,LR}_K10_s0_t00{2,3,5}（6）。状態は `obsidian-research-data/sink_roots_0930/round1/sign_states/` に写した |
| 4 | adam F1_twolayer_a6 | `src/sink_roots_round1/adam/twolayer_a6_probe.py`（写し、変更なし） | 空回し `--steps 50 --probe 2 --R 2`、本走 seed 0 `--probe 2,3 --R 16`・seed 1 `--probe 2,3 --R 16`・seed 0 `--probe 10 --R 12` |

優先順位 5 の依頼（tau_eff_vs_fit_speed ほか）は、この 4 つの後に別の追補で登録する。

## 2. 依頼から変えたこと

- **R6・R1 はこのエンジンで回す。** 依頼は R1 に `derive/retprobe2.py` か `refute/quick_arm.py` を挙げているが、判定に R3 と同じ記録（35 点の m・上端・s・k と S・A の場）が要るので、R3 と同じエンジンに入力の墨をそろえる操作と W1 の行ノルムの上限を足した。上限は quick_arm.py と同じ規則（課題 5 の頭の行ノルムを保存し、課題 5 以降の毎更新の後に超えた行を縮める）。既定のままの計算は R3 の基準とビット一致する（ELU seed 0 の課題 1・2 で確認）。墨の量は各 seed の 1,200 枚の平均（seed 0 で 100.89）。
- **本体の script の「CPU の門」は外した。** R7 が 10 本動いているので、門（CPU > 50% の python が 8 本以下）は開かない。同時の本数はこちらの起動 script（2 本まで、R7 と合わせて 12 本）で決める。
- **判定の量（a_P・κΔs_task/|P|・f_top など）は本体が導出の定義で出す。** こちらはデータ（arrays.npz と状態の npz）と、R3 と同じ比（ρ_med）を出す。
