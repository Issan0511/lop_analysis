# l2split_rlmnist_0914 — 判定（Random Label MNIST・L2 を中心化 W̃ の減衰と残りの減衰に割る 2×2）

> 自動生成: `analysis/l2split_rlmnist_0914/verdict.py`。spec: `specs/spec_l2split_rlmnist_0914.md`。数値はこのファイルと `verdict.csv` から転記する。

## 0. 実行したもの

- 新規 run の commit: `e0a3db519683dd0450f11a69127ce07ab7251a93`（種類 1・未 commit 変更: ['False']）
- 集計時の HEAD: `e0a3db519683dd0450f11a69127ce07ab7251a93`
- 再利用: **REUSE_OK**（照合 4/4 一致）
- shard: 16/80 を読んだ（新規 8・再利用 8）・欠損 64・有効 seed 0（無効: {'0': 'diverged or incomplete', '1': 'diverged or incomplete', '2': 'missing shard', '3': 'missing shard', '4': 'missing shard', '5': 'missing shard', '6': 'missing shard', '7': 'missing shard', '8': 'missing shard', '9': 'missing shard'}）
- 新規 run の壁時計 中央値 0.0 分（最大 0.0 分）
- 検査: `checks.json` all_pass = **True**

## 1. 結論（spec §5.2 を上から適用）

| act | arm | ラベル（95%） | 理由 | Bonferroni 版（97.5%） |
|---|---|---|---|---|
| R | l2wt | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |
| R | l2rest | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |
| LR | l2wt | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |
| LR | l2rest | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |

| act | 2×2 の型（95%） | Bonferroni 版 |
|---|---|---|
| **R** | **MIXED** | MIXED |
| **LR** | **MIXED** | MIXED |

主の問いは R の型。限定: Random Label MNIST・784–100–100–10・Adam lr=1e−3・λ=1e−3・400 epoch・50 タスク・seed 0–9・white-san CPU。G は画像固定による正の転移を含む正味。

