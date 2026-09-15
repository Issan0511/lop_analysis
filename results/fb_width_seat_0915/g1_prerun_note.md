# G1 事前監査メモ（本走結果を見る前）

2026-09-15、本走投入後かつ結果確認前に、親腕10 seedの10,000-step予備走を退避済み同機材ログと比較した。既存の学習・測定配列は全10 seedでbit一致した。`lr_used` metadataだけが本runnerで `NaN`、参照で `0.01` だった。原因はbranch runnerが学習に使う `st["lr"]` は0.01のまま、edge log writer用の付加metadata初期化だけを呼んでいないためである。追加moment列は短縮走と5M走でdense-last-20-taskの記録scheduleが異なるため予備比較から除外した。本走同士では同scheduleで比較する。

G1は二つを保存する。(1) 原G1は`lr_used`を含め不一致、(2) G1測定配列は既存全測定列・同schedule moment列・state hashを厳密比較する。元rawを修正しない。原G1 metadata不一致だけを理由に主判定を後から変更せず、学習数値bit一致、G2、M1、M2とともに明記する。
