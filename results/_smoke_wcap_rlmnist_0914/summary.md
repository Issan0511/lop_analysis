# wcap_rlmnist_0914 — 判定（Random Label MNIST・隠れ層のユニット別中心化ノルムの上限）

> 自動生成: `analysis/wcap_rlmnist_0914/verdict.py`。spec: `specs/spec_wcap_rlmnist_0914.md`。数値はこのファイルと `verdict.csv` から転記する。

## 0. 実行したもの

- run の commit: `0b8ef5384ca8f223db53c0f954913381d4495c37`（run 時の code の未 commit 変更: False）
- 集計時の HEAD: `0b8ef5384ca8f223db53c0f954913381d4495c37`
- 環境: host white-san・torch 2.13.0+cu130・python 3.12.3・スレッド [1]
- shard: 16/80・欠損 64・有効 seed 0（無効: {'0': 'diverged or incomplete', '1': 'diverged or incomplete', '2': 'missing shard', '3': 'missing shard', '4': 'missing shard', '5': 'missing shard', '6': 'missing shard', '7': 'missing shard', '8': 'missing shard', '9': 'missing shard'}）
- 1 run の壁時計 中央値 0.0 分（最大 0.0 分）・peak RSS 中央値 1072 MiB
- 検査: `checks.json` all_pass = **True**

## 1. 結論（spec §5.2 の表を上から適用）

| 問い | ラベル（95%） | 理由 | Bonferroni 版（97.5%） |
|---|---|---|---|
| Q1（主）LR capT1 | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |
| Q2 LR cap2 対 l2 | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |
| Q2 付記 LR cap2 の ρ | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |
| Q3 R cap2 | **INCOMPLETE** | 0 valid seeds < 8 | INCOMPLETE |

限定: Random Label MNIST・784–100–100–10・Adam lr=1e−3・400 epoch・50 タスク・seed 0–9・white-san CPU。G は画像固定による正の転移を含む正味。

