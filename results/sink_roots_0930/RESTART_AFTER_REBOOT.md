# 再起動（2026-09-30 13:0x、GPU のドライバ不一致を直すため）の後の再開手順

再起動の理由: 9/25 21:50 の NVIDIA パッケージの自動更新（580.173.02 → 580.178.04）の後、再起動していなかった。読み込まれたカーネルモジュール 580.173 とライブラリ 580.178 が食い違い、CUDA が使えなかった（`nvidia-smi`: Driver/library version mismatch）。Issa の判断で再起動。

## 再起動の直前の状態

- 全 launcher の新しい起動を止めた（`results/sink_roots_0930/STOP`・`STOP_round2`〜`STOP_round6`）
- R7（`analysis/sink_roots_0930/launch.py`）: A は 10/10 済み。F は s0・s1・s4 済み、s2・s3・s5・s6・s7 は走行中、s8・s9 は SIGSTOP で一時停止中。F99 は未着手。どれも課題ごとに `r7/<arm>/s<seed>/ckpt.pt` を書く（tmp → os.replace で原子的）。再起動で止まっても、最後に終わった課題から `--no-resume` なしで起動すれば再開する（エンジンの resume はビット一致）
- 追補 4: 49 走のうち 45 走済み（R_eps_zero_v2 の 4 本は終わりかけで、終わるのを待ってから再起動）
- 追補 5: 31 走のうち 7 走済み、追補 6: 45 走のうち 8 走済み（途中だったものは `.done` が無いので、launcher を起動し直せば最初からやり直す）
- 追補 2 の lr の梯子（8 本）と追補 3 の残り（R2p 7 本・R3p 54 本）は未着手

## 再起動の後にすること（順に）

1. `nvidia-smi` で GPU が見えること、`python3 -c "import torch; print(torch.cuda.is_available())"` が True であることを確かめる
2. R7 をどこで続けるかを決める
   - (a) CPU のまま再開: STOP を消して `python3 analysis/sink_roots_0930/launch.py` を起動（走っていた F は ckpt.pt から再開、`caps.json` の枠のまま）
   - (b) GPU: 登録（spec_postfit_elu_cifar_0924）の本来の形（R = 10 の同時走・CUDA graph）で A・F・F99 を GPU で回し直し、CPU の A・F は別機材の再現として残す。GPU を他の走と共有するかを先に見る
3. MNIST の queue を起動し直す: `STOP_round4`〜`STOP_round6`・`STOP_round2`・`STOP_round3` を消して、`launch_round5.py`・`launch_round6.py`（優先）、続いて `launch_round2.py`・`launch_round3.py`。合計の枠は `caps_round2.json`（本体の依頼で 10）。R7 を CPU で回すなら MNIST は 5 本
4. 本体セッション（ELUが沈降しない事件）へ、追補 4 の結果を送る（集計は `analysis/sink_roots_0930/r4_readouts.py`、R_eps_zero_v2 は round 1 の R_eps_wall と同じ集計）
