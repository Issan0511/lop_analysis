# NaP比較の順位予測追補

対象: `nap_cifar5p1_1018`。予測者: Issa。記録者: Codex。

Issaの2026-10-18 02:07:26 JSTの「NaPとSNA/KKT1の順位はKKT1にして」と、02:08:13 JSTの「NaP対SNAもSNAで」を受け、次の二つの方向予測を追加する。

| 比較 | 追加する順位予測 | 対応差の予測方向 |
|---|---|---|
| NaP対KKT1 | **KKT1 > NaP** | 後期窓のseed対応差NaP−KKT1の中央値<0 |
| NaP対SNA | **SNA > NaP** | 後期窓のseed対応差NaP−SNAの中央値<0 |

KKT1とSNAの間の順位、数値確率、精度、効果量、有意差が出る確率は指定されていない。NaP対Rの既存方向予測はそのままとする。この二つの順位は**Issaの追加予測**であり、論文が既にNaPとの順位を示していたという主張にはしない。

## 時点と旧登録

原登録は両比較ともOPEN_COMPARISONだった。原specと共通spec、ソース、検査のbytesは変更しない。本追補とcommit/hashを新しい予測記録として保存し、元のOPEN履歴を残す。

rootと別担当 `/root/run5_nap` が、精度値を開かず、パス・更新時刻・起動状態を照合した。独立確認時刻は02:08:57 JST。確認時のHEADは54effd4585992892e996729a0e1f8440a2e3164a、clean。NaPの較正preflightは01:26:39 JSTのBLOCKED_CUDAで、per_task・selected.json・評価ディレクトリ・checkpointはなかった。確認したworktreeでは、NaPの較正・評価結果が生成される前の追加である。

正確な記録時刻、確認ファイルのmetadata、原ソース/specのhashは同名JSONと独立監査JSONに保存する。既存SNA/KKT1の成績が既知であることは変わらず、研究全体で未観測だったとは称さない。

## 検定と観測後の扱い

主指標はhard t21,23,25,27,29のonline平均、seed0–9。原spec §5.4の両側正確符号検定、2比較のHolm補正、中央値・平均・全10対応差、同等性幅0.005を維持する。片側検定へ変えず、検定数も減らさない。

両比較で予測方向が統計的に支持されたと記すには、従来の `COMPARATOR_WINS`（Holm補正p<.05、対応中央値<0）を要求する。方向予測と有意差判定は別に報告する。未決・非有意を勝利へ変換しない。数値や判定を見て本追補を修正しない。

固定済みrunner/verdictのprediction欄がOPEN_COMPARISONを表示する場合、それは原登録の記録である。最終報告では本追補のcommit/hashを併記して追加予測を示す。性能値や科学判定を予測に合わせて変更しない。
