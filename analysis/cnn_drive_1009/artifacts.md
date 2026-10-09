# 1009 追加成果の位置づけ

CE の arbitrary-state 条件: ce_trained_state.md。
Adam の arbitrary-state 条件と一歩反例: adam_state.md。
旧ラベル依存と小 overlap の障害: overlap_obstruction.md。
定常二点分布の解析反例: adam_stationary_twopoint.md。
定常17点・batch16分布の厳密証明書: adam_stationary_b16.md。
その独立監査: adam_stationary_b16_review.md。

全て実現可能な有限の状態・構成を含む。実 RL-CIFAR の CE/Adam 全期間で条件が維持されることは未証明。

チェック用 Python はいずれも単独実行可能。torch を使うものはプロジェクトの .venv、stationary certificate は Python 標準ライブラリだけで動く。stationary B16 の --mc は補助計算であり通常の証明検算には不要。

2026-10-09 の再確認: src/rlcifar_cnn_0908.py は task 間で Adam moments を保持する。write_hist が保存するのはヒストグラム・平均・前活性サンプルで、全パラメータと Adam moments の checkpoint ではない。obsidian-research-data で見つかった rlcifar の .pt は MLP 比較走であり、この CNN の新しい state certificate の実測には使わなかった。

全 Conv/head が実際に動く CE-SGD の長期定理と元の容量自己項への接続: moving_ce_longtime.md。
小振幅における定常 Adam の有限残差・対称雑音の正例: adam_smallscale.md。
実際の moving Adam に対する累積恒等式・残差の十分条件・balanced-label 正例: adam_cumulative_positive.md。
全 Conv/head の CE-SGD 長期証明の独立監査: moving_ce_longtime_review.md。
実 moving Adam の累積恒等式と限定正例の独立監査: adam_cumulative_positive_review.md。

課題内 label reuse を含む長期共同SGD: task_reuse_longtime.md / task_reuse_longtime_review.md。
一般CNN有限taskとnegative-mean/mixed-gate正例: task_reuse_finite.md / task_reuse_finite_review.md。
過去momentsを保った通常Adamのfinite reused-label task: adam_task_reuse.md / adam_task_reuse_review.md。
本来の空間Conv・hiddenFC・全biasでの元の容量自己項のopen符号保証: full_ntk_positive.md。
本来の空間Conv/FCの全NTK符号と32×32全構造の非空性の独立監査: full_ntk_positive_review.md。

通常iid binary labelsを課題内再利用し、momentsを保持した全raw Adamの高確率長期沈降: adam_iid_longtime.md。
actual moving historyから定常task-block平均への定量的橋: adam_task_averaging.md。
36rawweights不変線とfull/literal-self全NTKの独立導出: adam_invariant_line_derivation.md。
平均化・永久保持・実学習率総和・有理数確率証明書の独立監査: adam_iid_longtime_review.md。
有限raw AdamとNTKの検算、および非常に保守的な存在証明用の学習率証明書: verify_adam_iid_longtime.py / ../../results/cnn_drive_1009/adam_iid_longtime.json。
