# CNN cap監視・完了報告（2026-10-21 14:17 JST）

担当会話: 01a15cb1-652a-7ee1-aa0c-f10194255fc8。宛先: 01a14da9-6ed1-7d12-9eb3-ed79510a9c72。

natural/capとも課題30までCOMPLETED、各580行、bad_step=0、欠落・非有限学習なし。14:12:51 JSTの確認でdriver/学習プロセスは消失。保存済みprimary.jsonと独立再計算が一致し、report.jsonはCOMPLETE_AUDITED。CPUレポートと図を生成した。元のunified exec 68136はこの会話から取得できず、driver自体のexit codeは未確認。

## 登録済み主評価（late hardのfirst780）

- ReLU natural22.184%、cap33.823%、差+11.639pt、正/負/同点5/4/1、Holm p=1、INCONCLUSIVE。
- ELU α3.6 natural83.657%、cap78.547%、差−5.111pt、正/負/同点0/10/0、Holm p=.00390625、WORSENS。
- 新naturalの課題1→30幅増大はReLU c1/c2/f1各10/10、f2 9/10、ELU全4層10/10（EMAおよび固定probe）。capは幅を抑えた。EMA上限逸脱unit-step/境界unitとも0。
- ELUの局所応答は保たれる/増えるが初期性能は悪化。救済とは呼ばず、幅が原因でないとも断定しない。未定義rの40記録はゼロ幅による診断上の未定義で、学習失敗ではない。

## 出力

worktree: /home/issan/Projects/claude/wt/cnn_width_cap_1021

- results/cnn_width_cap_1021/report/report.md
- results/cnn_width_cap_1021/report/report.json
- results/cnn_width_cap_1021/report/figure.png
- results/cnn_width_cap_1021/report/figure.pdf
- results/cnn_width_cap_1021/report/width_growth.csv 他、全seedの詳細表

## CPUレポートの型不整合

元analysis/cnn_width_cap_1021/result_report.pyはdata_sha256を文字列として比較したが、実provenanceではファイル名→SHAのsingleton dictのためexit1。学習・元コードは変更せず、/tmp/cnn_cap_result_report_monitor_1021.pyへコピーし、ROOTを元worktreeへ固定し、データ項目だけactual archive basenameと完全一致するsingleton dictの値を使って実ファイルSHAを照合した。これでexit0、独立primary一致。元result_figure.pyもexit0。

変更内容と元/一時ファイルSHA: /tmp/cnn_cap_report_hash_format_fix_1021.json

監視担当はcapの生データ退避、main統合、Vault更新、worktree整理をしていない。担当元会話でこの形式対応を恒久修正し、結果保存・整理をお願いします。旧結果を混ぜたり学習を再実行したりしていません。

CH/CHB応答交換は事前登録338a9443をpush済み、CPU17検査PASS。cap完了を受けGPU検査と4腕の実行に進みます。こちらの実験は最後まで当会話で所有します。
