# 実装上の約束（結果を見る前、2026-10-23）

登録 spec 8c4ecf05 の J1〜J3・閾値・予測・窓は変更しない。保存状態を読むだけで、optimizer・学習・再生・ワーカーは使わない。

- `kappa_cnn5p1_1022.Bundle`・データローダ・訓練の `E.forward` を再利用する。順伝播は float32、縮約と softmax は float64。softmax に渡す logit は訓練の順伝播の出力で、再計算しない。
- φ′ は訓練で使った `act.phi` 自体を、保存状態の float32 の z で autograd 微分する。再現検査 (a) の φ′ は旧解析との比較のため旧 `act.dphi` の画像平均→unit 中央値を使う。両者の役割を混ぜない。
- FC1 は FC2 を通る連鎖律を含む: residual を q とすると `((q @ W_head) * φ′_FC2) @ W_FC2` を FC1 の局所微分に掛ける。a は conv2 の活性化を pool して flatten した 1024 次元。自分の重みと bias の勾配だけを位置の変化に入れる。
- 生存は spec どおり `sum_n K_n abs(φ′_in) > 0`。K の負値の数と、生存条件を満たさない unit の非ゼロの S も記録する。比の分母のゼロは exact zero で除き、eps を足さない。中央値は偶数個なら中央2値の平均。
- 相対誤差は `max_abs(x-y) / max(max_abs(x), max_abs(y), 1e-300)`。恒等式の各辺を独立に計算する。中心化の検査は logits を固定し、縮約に使う head だけを中心化する。
- 置換は hard の全状態、5 回。CPU の torch generator を `1023 + seed*10000 + boundary*100 + task` で固定し、腕・予算間では同じ置換を使う。
- **spec に数値化されていない (d) の「0 付近」**: 5 回平均した L の生存 unit 中央値の絶対値が、釣り合ったラベルの置換の厳密な帰無標準偏差 `/sqrt(5)` の unit 中央値の3倍以内、として PASS/FAIL を付す。この帯は検査の実装上の約束で、登録済みの統計的有意水準ではない。raw の L の中央値・絶対値中央値・5 回の個別値を併記する。agree_rand は5回の平均が本物以上かを各状態で比較する。FC2 の全非除外状態で両条件を満たした場合だけ (d) PASS。FC1 は別記。失敗した状態も隠さない。J1〜J3 の閾値へは使わない。
- 崩壊の判定は旧解析の train_acc < 0.3 を再利用し、その状態が ReLU・自分の課題の f2 が全死と一致することを確認する。行は保存し、集計からだけ除く。初回にはさらに train_acc = 0 を確認して停止したが、旧解析の記録に train_acc = 0.2 の全死状態が含まれていた。下記の経緯を参照。
- 境界別の表は seed 中央値・最小・最大。判定用は各 seed 内で境界を等重み平均し、その seed 値の中央値・最小・最大。ReLU の欠けた境界は除外し、seed ごとの使用境界数を保存する。独立な追試とは数えない。
- この実行環境から CUDA が見えない場合は、明示的な `--device cpu` で検査から行う。比較許容は変えない。Python 環境と CIFAR archive は旧解析と同じ場所から読み取り専用で使い、他の worktree・clone・vault のファイルは変更しない。

実行:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python src/cnn5p1_labelterm_1023_checks.py preflight --device cpu
PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python src/cnn5p1_labelterm_1023.py measure --device cpu
PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/cnn5p1_labelterm_1023/summarize.py
```

preflight が FAIL なら全状態の計算は行わない。

## 初回停止の診断と修正（2026-10-23、判定の集約前）

初回の (a)・(c) は PASS。全状態の計測で 1,440 行まで保存したところで、予算7800・境界19・ReLU seed2009 の `train_acc = 0.2` が、追加した `train_acc == 0` の確認に掛かった。既存 `kappa_cnn5p1_1022/rows.csv` では、この状態は `collapsed_state=True, n_alive_f2=0, F_norm=0` で、旧実装の判定 `train_acc < 0.3` と一致する。その時点の順伝播検査は440/440 PASS。

spec の「同じ規則で除く」に従い、旧実装の `train_acc < 0.3` を維持し、余分な `==0` の assertion だけを削除した。spec の括弧「train 精度0」は、元のデータでは全例について正確ではない。この不一致は summary の冒頭と最終報告にも明記する。FC2全死の確認は維持する。J1〜J3・量・閾値・窓は変更しない。失敗時の provenance と preflight は `results/cnn5p1_labelterm_1023/attempts/attempt1_*.json` に保存し、修正後のソースで preflight から再実行する。同じ状態の再計算であり、独立な追試ではない。
