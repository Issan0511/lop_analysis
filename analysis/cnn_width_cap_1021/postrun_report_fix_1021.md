# 完走後のレポート修正と再照合（1021）

学習・介入・事前仕様・登録主統計は変更していない。

## データhashの表現

レポートが`data_sha256`を文字列と仮定していたが、元データloaderのprovenanceは`{archive_basename: sha256}`である。監視担当は一時コピーでこのsingleton dictに対応し、両腕COMPLETED・580行ずつ・独立主統計の一致を確認した。

親は`registered_file_digest`を恒久追加した。データだけは実ファイル名をキーとするsingleton dictも許容し、別名・複数ファイル・不正digest・source/calibrationへのdict混入を拒否する。実際のファイルのSHA256との照合は維持した。

検査: 正常な文字列とsingleton dictが通り、別名・複数キー・非データdict・短いdigest・Noneの5変異を拒否した。修正後の元レポートをCPUで実行してexit0。生成JSONは監視担当の一時版出力と比較し、`report_script_sha256`以外の全フィールドが一致した。新しいforward・更新・GPU計算はしていない。

退避後の再生成も検査した。provenance内の旧worktree絶対パスを保ったまま、追跡済みの同じrun配下metadataは現在checkoutへ解決し、生データはmanifestのsize/SHA256を照合した退避先へ解決する。これにより追跡済みreference.jsonが旧絶対パスだけを見て失敗する問題を防ぐ。実行時のprovenanceを書き換えていない。

## 図

Aの検定注釈をプロット内の専用余白へ移し、図の高さを増やした。Cの見出しとの近接を解消した。PNGを目視して全4パネルの文字・凡例・注釈が読めることを確認した。データや統計量は変更していない。

## 実行終了の記録

監視担当は元driverのexit codeを取得できなかった。親は元のunified exec session 68136から、以下の子工程終了記録を回収した（UTC）。

```
EXIT natural 0 2026-10-21T04:29:38Z
START cap 2026-10-21T04:29:38Z
EXIT cap 0 2026-10-21T05:12:30Z
START summary 2026-10-21T05:12:30Z
EXIT summary 0 2026-10-21T05:12:31Z
```

両腕の保存provenanceはCOMPLETED、各580行で全行bad_step=-1（非有限更新なし）。登録集計と独立再計算が一致した。上記はdriverが記録した各子工程の終了値であり、未返却のdriver自身のexit codeを推定したものではない。

ReLUは+11.639ポイントでも正/負/同点5/4/1、Holm p=1でINCONCLUSIVE。ELUは-5.111ポイント、0/10/0、p=.00390625でWORSENS。ReLUを改善成立と記さない。ELUの高応答と低い主精度は、幅・局所応答の保持だけで学習を説明し切れないことを示す。ただし課題末診断を最初780更新と同一時刻の媒介測定とは扱わない。
