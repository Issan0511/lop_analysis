# 実行前の実装修正

- 登録 dffbbbf8 後、CPU synthetic 7組は通過したが、CUDA syntheticのcaptureでPyTorch 2.13のstale AccumulateGrad streamエラーが出た。実データ・本走は未実行。
- 共有hostを変えず、CapEngineのwarmupとcaptureに同じCUDA streamを明示。チェックはcaptureの状態巻き戻しに加え、2更新をeagerと直接照合する。数値判定・介入・seed・予算は変更しない。
