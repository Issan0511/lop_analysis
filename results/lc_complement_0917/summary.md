# lc_complement_0917 — 実装検査

C1–C4の実行コード・集計器・登録仕様・実行手順を実装。**本走の判定結果ではない。**

- unittest 15件が合格（checks.json）。
- 保存済みseed0,t5から2task・12,000更新の自然継続で、元のstate hashとonline精度に完全一致。
- C1 CPU: 26腕×2層の再解析CLI smokeが完了。
- C1 GPU: B/L全54モデルの最初の1task・6000更新を再生。既存online精度と両層のz平均に完全一致。150task全体は未実行。
- C2: t2/t5×6腕×2継続taskを1epoch/taskでCLI smoke。全6000更新の衝撃境界と2task目の解除は別にunit test。
- C3: 5腕すべて2task×1epochのCLI smoke完了。標準nn.Linear/nn.ELU/Adamとの3更新後のparameter bit一致。
- C4: Cauchy–Schwarz上界を検査。一般の勾配履歴で3.16ηを超える反例と、約7.27ηの上界への一致を確認。回復時間について得られるのは下限だけ。

C1–C3の本走は未開始。C4の算術は登録後に実行済み（c4/verdict.csv）。C3は彼らの手順を採用したため、先頭1200枚・固定順・native ELU/CELU。C2のexpm1系とは区別する。

生データはbackup_manifest.jsonの場所に保存。実行手順はanalysis/lc_complement_0917/README.md。

## 本走完了

C1 CPU10seed、C1 GPU150task、C2の10seed、C3の5腕×3seedが完了。各部のsummary.mdとverdict.csvを参照。C4は既存の算術結果を使用。
