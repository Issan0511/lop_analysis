# swish_battle_0917 の CNN 箱は打ち切り（2026-09-18 03:20）

Issa の指示（「もうまじでいらんからなー」「終了させて」）で、CNN 箱の残りを止めた。
走行中だった 5 本（SWA1・SW1・SWA3・SWA1u・SWA3u の seed 6、各 1 時間半経過）は kill した。この runner に途中再開は無いので、その 5 本は残っていない。

**残っているもの（46 本）**
- `SNAc3`: seed 0–9（10 本、完走）
- `SW1`・`SW3`・`SWA1`・`SWA1u`・`SWA3`・`SWA3u`: seed 0–5（各 6 本、完走）

**登録した判定（spec §5 の CNN の呼び出し）は未解決**。10 seed で登録したので、6 seed で読むと検出力が足りない
（符号検定は n=6 なら 6/6 で p=0.031、5/6 では p=0.219）。読むときは「seed 0–5、検出力不足」と明記する。
MLP 箱は 110 本が完走していて、`results/swish_battle_0917/summary_mlp.md`・`verdict_mlp.json` のとおり（判定は有効）。
