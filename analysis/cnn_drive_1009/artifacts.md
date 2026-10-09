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
