# ViT 最終 FFN の迂回を調べる保存状態介入（1025）

既存 `vit_long_1024` に対する事後解析。Issa の「最終層がスキップされてる可能性」を検証する。以下を介入結果の閲覧前に commit する。元の走・事前登録の判定は変更しない。

## 対象と介入

- 6 ブロック・ReLU・LayerNorm あり、R0L（head reset）と SL（head shared）、全 5 seed。
- task 1, 10, 30, 60 の update 100 と update 500 の保存状態（80 checkpoints）。
- 各 task の元の 250 画像・iid ラベル・loss head rows を復元する。これは課題内の記憶精度であり、未見画像の一般化精度ではない。
- 重みは固定し、eval mode、augmentation なし。最終 FFN 出力を alpha = 0, .25, .5, .75, .9, 1 倍する。残差入力・最終 LN・head はそのまま。alpha = 0 が FFN 除去、1 が元のモデル。
- 補助対照として最終ブロック全体（attention と FFN）を恒等写像にする。FFN 単独の除去と混同しない。
- 再学習・optimizer step は行わない。

## 指標と解釈

各 checkpoint で精度、cross entropy、元のモデルからの予測変更率、クラス内中心化 logit 差の RMS と相対 RMS、KL(full || intervention) を計算する。CLS で FFN 出力ノルム / 残差入力ノルムも記録する。結果は seed ごとの値と中央値・範囲で表示し、画像数を独立 seed 数に数えない。

精度が変わらなくても損失・logit が変われば「未使用」としない。alpha の段階的変化で除去の大きさへの依存を見る。除去が有害なら、その保存状態の予測は最終 FFN に依存している。変化が小さければ、その時点・その画像集合で依存が小さいという証拠。恒久的な不要性や全入力での不要性は主張しない。勾配・実際の Adam 更新・沈降の原因はこの介入単独からは決めない。結果に応じた二値判定の閾値は追加しない。

## 検算と保存

- 元の tar 内の manifest の SHA256 と各 checkpoint を照合する。
- 元の provenance の source hashes と、元の git commit から凍結したソースを照合する。
- 復元した課題・loss rows を元のログと照合する。
- alpha = 1 の手動 forward と通常 forward、alpha = 0 と forward hook による除去が一致することを各 checkpoint で確認する（float32 の許容差 2e-5）。
- 元の intask 記録の同じ update の記憶精度が一致し、平均 CE の絶対差が 2e-4 以下であることを確認する。
- 各介入でモデル parameter の in-place version が変わらないことを確認する。
- 生の example logits、ラベル、復元した画像のハッシュ、source snapshot は `~/Projects/obsidian-research-data/vit_ffn_ablation_1025/`。コードと集計結果は同名の analysis/ と results/ に保存する。
