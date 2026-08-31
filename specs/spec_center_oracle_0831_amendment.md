# center_oracle_0831 実行前監査 amendment

状態: **事前登録追補（実行前）** / 作成: 2026-08-31 / 親: `spec_center_oracle_0831.md`

本追補は結果を見る前の実装監査で検出した、元 spec の S0′ と S-taut の論理的不整合を修正する。元 spec は履歴として変更しない。

## A1. S0′ の訂正

元 spec は同時に次を要求する。

1. Aexact の step 0 の全記録量が A1 と bit 一致する（S0′）
2. Aexact は支持平均を引くため step 0 から `M ≡ 0`（S-taut）

既存 A1 の step 0 は EMA がゼロ初期化されており `M ≠ 0` なので、両者は同時成立しない。介入が forward に入れば、介入後の `M`・`B`・`p_hat` 等が異なるのは設計どおりである。

したがって operative な S0′ を以下へ置換する。

- **S0-state**: Aexact の介入適用前の step 0 で、net・teacher・env・raw running_mean が既存 `L1w100_A1_step0.pt` と bit 一致
- **S0-stream-preflight**: A1 と Aexact を同じ10-seed vectorで30,000 step進め、raw `x`・`y` の stream digest、初期 state、最終 env state が一致
- **S0-final-env**: Aexact 5M の env `flip_state`・`t` が既存 A1 5M log の state hash と一致
- Aexact と A1 の**介入後の記録量は一致対象にしない**。step 0 で異なる列を `verdict.csv` に列挙する

この訂正はペアリングの本来の意味（init・teacher・raw input realization・flip trajectory の共有）を検査し、介入効果そのものを一致条件に入れない。

## A2. 判定の実装順

- P1 は Aexact の `Δβ_bnd` CI が0を含む場合、比 `R` より先に `BOUNDARY_DESCENT_ELIMINATED` を採る
- P2 の判定ラベルは5M `strict_dead_frac` の paired差を主とする。直近100タスク連続dead率は独立にREPORT_ONLY
- 任意の alpha sweep（P3）は採用しない。主判定の Aexact 1腕だけを実行する

## A3. 新規学習前の停止条件

S0-state または S0-stream-preflight がFAILなら5M走を開始しない。S-taut・S1/S2・S3・S7は元 spec のまま。
