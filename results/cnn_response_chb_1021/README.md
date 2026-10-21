# CNN CH′/CH′＋B 全4層応答交換（1021）

**BIDIRECTIONAL**：初期online accuracyはCHへのCHB場移植で+43.583pt（10/10）、逆移植で−11.468pt（9/10）。失敗0、全4腕×10seed×30000更新を完了し、独立再集計と一致。

ただしCHへの移植は平均CEを悪化させ、逆方向の精度低下は終盤に残らない。初期accuracy、損失、時間的な応答変化を分けて読む。

- [監査済み結果・全seed・全固定窓のaccuracy/CE](report/report.md)
- [解釈・例外・応答操作・検証](report/interpretation.md)
- [全数値と監査](report/report.json)
- [図表の読み方](report/supplement/READING_GUIDE.md)
- [凍結した登録spec](../../specs/spec_cnn_response_chb_1021.md)
- [実行記録](execution_record.json)
- 生データの退避先は完了時の backup_manifest.json にSHA256/bytes付きで記録する。

実装の再現はcommit338a9443を参照する。後から追加した図表・退避コードとCEの表示追加は学習を変更していない。退避後の独立監査は、リポジトリルートから次で再生成できる（NumPyとGitが必要）：

```sh
python analysis/cnn_response_chb_1021/result_report.py --out /tmp/cnn_response_chb_audit_replay
```

Vaultの結果正本は「CNNのCHとCHBが保った全層応答の交換_結果_1021」。V13の部品1/5と段落地図へ接続する。
