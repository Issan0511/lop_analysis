# ch_chb_200_0919 — 登録結果

**独立監査なし。自己検査・変異対照のみ。**

主判定: **INCONCLUSIVE**

同一 seed の t50 checkpoint から t200（6M 更新）への継続。有限予算のラベルであり永続的治癒ではない。深さだけの COLLAPSE は機能的 LoP の証明ではない。

|腕|型|最終 online（seed平均）|低下（pt）|
|---|---|---:|---:|
|CH|SPLIT|0.52818144|45.86996|
|CHB|HOLDS|0.99042762|-0.46859|

全 seed の型・理由・境界値は verdict.csv、窓別診断・対応差の t 区間は verdict.json。少数 seed も除外していない。

|seed|CH|CHB|組|
|---|---|---|---|
|0|COLLAPSE (RESPONSE_ERODING,BOTH)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|
|1|UNCLASSIFIED (RESPONSE_ERODING,EXCESS_DROP)|HOLDS (RESPONSE_ERODING)|INCONCLUSIVE|
|2|COLLAPSE (RESPONSE_ERODING,DEPTH_ONLY)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|
|3|COLLAPSE (RESPONSE_ERODING,BOTH)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|
|4|COLLAPSE (RESPONSE_ERODING,BOTH)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|
|5|UNCLASSIFIED (RESPONSE_ERODING,EXCESS_DROP)|HOLDS (RESPONSE_ERODING)|INCONCLUSIVE|
|6|UNCLASSIFIED (RESPONSE_ERODING,EXCESS_DROP)|HOLDS (RESPONSE_ERODING)|INCONCLUSIVE|
|7|COLLAPSE (RESPONSE_ERODING,BOTH)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|
|8|COLLAPSE (RESPONSE_ERODING,BOTH)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|
|9|COLLAPSE (RESPONSE_ERODING,BOTH)|HOLDS (RESPONSE_ERODING)|B_ROUTE_DELAY|

予測採点: `{"codex_brier": 1.2048, "codex_interval": false, "codex_order": true, "issa_main": false, "issa_types": false, "issa_interval": false, "issa_order": true}`

登録閾値・窓・応答個数条件は実行後に変更していない。bias WD は完全除去でなく、OMEGA_ONLY も機構同定ではない。

## 完走後の確認と読み方

CH の COLLAPSE は 7/10（性能と深さの両方が 6、深さのみが 1）。残る 3 本も 17.9–30.6 pt 低下し、応答侵食と ω 帯を超える低下により UNCLASSIFIED。B_ROUTE_DELAY の組は 7/10 で、登録した 9/10 に届かないため主判定は INCONCLUSIVE のまま。CHB は 10/10 HOLDS、平均 online は 99.04%（既知窓より約 0.47 pt 改善）だが、全 seed に RESPONSE_ERODING が付く。HOLDS は応答不変の証明ではない。

CURED_200＋OMEGA_ONLY、CH の低下 2 pt 以内という予測は不一致。CHB の低下が CH 以下という順序予測のみ一致。Claude の提示した三分岐には今回の INCONCLUSIVE が含まれず、主ラベル予測は不一致。判定後に型や支持数を変更していない。

実行 commit: `a9563cb99ac8d3a67e5e447c2b39f025f81af922`。CH 約63.2分、CHB 約63.6分、逐次実行。起動時 src/analysis は clean。事前検査69件、変異対照75件を検出。完走後、旧列全セル・接頭部snapshot・t51直前状態/RNG・CHBの全150タスク全ユニットの釘付け上界を自己照合した（completion_verification.json）。独立監査はしていない。
