# 完了後集計の監査準備（1021）

独立集計スクリプトは学習コードとは別ファイル。主走中のsrc/specは変更していない。

- 両条件COMPLETED、580件ずつのtask/seed集合、非有限なし、condition以外の全metadata一致を先に要求する。
- 主判定は保存CSVから再計算し、登録集計の全seed差・early/late・符号検定・Holm・判定と照合する。
- 保存unit配列のキーとshapeを `cap_arrays` / `unit_stats2` に照合。固定probe幅とbatch分散EMA、unit中央値とseed平均、hard/easy、未定義rを分ける。
- ソースは実行時commitのGit内容とhash照合。退避後のrawはmanifestのsize/SHA256を確認する。
- 未完走状態での実行を拒否し、科学的レポートを作らないことを確認した。完全データでの実行は完走後に行う。

別workerの読み取り監査はモデルcapacityで完了せず、親がコードと保存形式を読んで照合した。追加のFable APIコード監査呼び出しは自動承認レビューに拒否され、実行されていない（一般の相談許可とは別に、コードの外部送信への個別許可が不足との判定）。このコード監査をFable実施済みとは扱わない。設計相談と応答交換の結果解釈のFable往復は、別記録で実施済み。
