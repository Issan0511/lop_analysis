# 主走後の集計出力の型修正（1021）

全4腕の30,000更新完了後、summarize末尾の標準出力JSONでNumPy int64の直列化エラーが発生した。保存JSONには数字・boolが文字列化されていた。sign_testの正負数をPython boolの和へ変更し、同じ保存結果から再集計した。統計式・閾値・比較・学習状態・更新・登録specは変更していない。

本走実装commit: 92a2214d。登録spec: aa991f9e。修正後の型はint/bool、値は修正前と同じ。独立監査では全腕400epoch×10seed、summaryのhash、preflight/spec/実行identity、failed=0、epochからの再集計一致（最大1.1e-16）を確認。
