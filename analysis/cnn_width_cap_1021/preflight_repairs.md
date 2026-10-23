# 実行前の実装修正

- 登録 dffbbbf8 後、CPU synthetic 7組は通過したが、CUDA syntheticのcaptureでPyTorch 2.13のstale AccumulateGrad streamエラーが出た。実データ・本走は未実行。
- 共有hostを変えず、CapEngineのwarmupとcaptureに同じCUDA streamを明示。チェックはcaptureの状態巻き戻しに加え、2更新をeagerと直接照合する。数値判定・介入・seed・予算は変更しない。
- b33b0c8a の同一stream指定だけでは解決しなかった。合成データのeager検査で作られたAccumulateGradにもdefault streamの参照が残るため、PyTorch 2.13が提供する `torch.autograd.graph.set_override_stale_capture_stream(True)` をこの実験プロセス内で明示する。設定値をruntime/provenanceに記録し、graphとeagerの全状態の完全一致、保存b01の完全再現を引き続き必須とする。共有hostのコードは変更しない。
