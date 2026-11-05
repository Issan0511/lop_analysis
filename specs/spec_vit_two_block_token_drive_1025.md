# CLSの沈降とpatch由来の更新を分ける診断

2026-10-25、通常学習の符号を追う追加診断。ゲート対照で浅層の沈降が残り、深層の浮上がrate_gateで消えた結果を見て起案。token別の結果は未見。既存のμ/σはCLS上の指標で、FFNの重みは全tokenで共有される。このため「浅いFFNのCLSが沈む」と「全tokenが沈む」を分ける。

同じseed0–4、課題10/60、起点100/500、block1/2の40状態。入力・fc1以外固定、元CE・元ReLU。fc1のgradientをzのtoken軸でCLS分とpatch分に厳密に分ける。all=cls_only+patch_onlyをautogradで照合。

生SGDおよび保存Adam分母固定でCLSの一次Δμ/Δr。通常・CLSだけ・patchだけを勾配normを揃えて固定計量と実Adamの一歩で比べる。対照のzero gradientは線形0を記録し、norm合わせ不能なので有限腕を実行しない（深いblockはCLS-only headの構造でpatch gradient=0になるはず）。過去Adam履歴が一歩の符号を保つ可能性を分け、固定計量・momentumなしを必ず併記。

一歩前後のCLSと全patch（画像×64位置をpool）のμ・σ・μ/σを別々に測る。また既に保存済みの局所full-batch50更新（fixed/Adam）について、始点と終点のpatch統計を追加集計する。再訓練しない。CLS統計は既存測定と照合し、同じ画像・cached inputで計算する。全patchのpool指標は空間位置内のばらつきも含み、CLS指標と同じ母集団ではない。

一歩でCLS-onlyが浮上しpatch-onlyが沈むなら、当該状態のCLS沈降をpatchから共有重みへの更新で説明する。そうならなくても報告する。画像・tokenを独立seedに数えない。これは駆動源T5の検証ではない。
