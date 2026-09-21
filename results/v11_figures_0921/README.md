# V11 原稿の図 — 図 1–9（0921）

作図スクリプトは `analysis/v11_figures_0921/`。committed CSV だけを読み、走は起こさない。

```
python3 analysis/v11_figures_0921/make_all.py     # 全部作り直す
python3 analysis/v11_figures_0921/fig02_doors.py  # 1 枚だけ
```

体裁は vault `可塑性喪失/主張/V11原稿_図表一覧と9月30日までの計画_0921` §2.5 の作図規約に従う。
規約の実装は `style.py` に集約してあり、色・ラベル・格の表示・登録窓の帯・精度軸の固定は
すべてそこを直せば全図に効く。ラベルは日本語で、英語化は `style.py` の LABEL / AXIS /
GRADE を差し替える（§2.5-9）。

各図の `*.txt` はその図のキャプションの下書き（集約・帯の意味・窓・格・元データ）。

## 体裁の選択肢が残っている図

| 図 | 出力 | 残っている選択 |
|---|---|---|
| 2 | `fig02_doors_1x4` / `_2x2` | 4 パネルを 1 行に並べるか 2×2 にするか |
| 3 | `fig03_bias_route_seeds` / `_median` | seed 個別線（規約の既定）か中央値＋帯か |
| 4 | `fig04_real_doors_dotted` / `_hard_only` | easy 課題を点線で添えるか hard だけにするか |
| 9 | `fig09_5p1_points` / `_bars` | 16 腕を点＋範囲で出すか棒で出すか |
