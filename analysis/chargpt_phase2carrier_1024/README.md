# C3の全経路の担い手

Spec: `specs/spec_chargpt_phase2carrier_1024.md`（本測定前のcommit b71fa304）。

`measure.py --gate`で3状態の独立照合と有限差分を通した後、`measure.py`で90状態を測る。`report.py`は全512unitの平均・中央値・方向割合とseed別の方向照合を作る。`plot.py`はCSVだけから図を作る。学習・optimizer作成は無い。

`P2`と`P2ctx`は既存の自由head LS模型。`switch`は独立一様新ラベルに対する期待CE勾配、`fit`は現課題の真の遷移分布への期待CE勾配。実網の共有headの直接勾配も保持する。

`own`は対象FFNのup行だけが位置を動かす経路。誤差信号の自己項という意味ではない。`upstream`は他の非ゼロparameter群のJVPの和。全parameter同時JVPと群ごとの和は各unitで照合する。

位置は固定一様probeのr=mu/s。列`mean_term`はdmu/s、`width_term`は−mu*ds/s²。`mean`はS_rの全unit平均で、`mean_term`とは違う。各列はgradient方向gに沿う微分であり、更新−ηgはその逆向きなので、正のS_rが沈める側。Adam_fixedは保存二次モーメントの分母だけを固定し、momentumを含む実更新ではない。

90状態が完了した後に`archive.py`で生配列を研究データ置き場へコピー・SHA256照合し、worktree内の複製を削除する。元checkpointのtarには書かない。
