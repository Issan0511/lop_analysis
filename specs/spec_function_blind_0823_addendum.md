# spec_function_blind_0823_addendum: 初回再現の定義不一致を受けた追補

proj_004 / 作成 2026-08-23 / 対象リポジトリ: lop_analysis / **再学習なし**

> **位置づけ**: `spec_function_blind_0823.md` と実装 commit `b6b6410` の初回実行後に作成した追補。したがって完全に事後であり、盲検事前登録ではない。本 commit 後に追補実装・再実行する。

## 1. 初回実行で判明したこと

1. H の `event = 300k 内の保存記録点で一度でも p_hat<0.05` は low/mid/high の seed 等重み率が `1.000 / 0.997 / 0.995` となり、旧チャット値 `0.837 / 0.839 / 0.850` を再現しない。**any-hit 転帰には天井効果がある**。初回結果は削除せず、H-any として正本に残す。
2. O の current / primary repair は、10 seed の平均 `nmse = MSE/Var(y)` で **0.1433666 → 0.0091483**。旧チャット値 `0.143 → 0.009` と一致する。一方、初回 spec が主表示にした seed 中央値 `unfit_var` は `0.109 → 約0` で、旧見出しの集約法ではなかった。
3. O の3破壊対照は初回 spec で固定した操作では旧チャット値を再現しなかった。元の使い捨てコードは保存されておらず、作業リストにシャッフルの軸・乱数反復数・集約法がない。初回 spec の対照結果は**新しく固定した対照**として有効だが、旧対照値の再現とは呼ばない。

## 2. H-end: 300k 後の凍結占有率

H-any の天井効果と旧値の定義不一致を切り分けるため、転帰を次のように固定する。

- 起点: bulk グリッド `t0 = 200000, 201000, ..., 600000`（401点）
- 主リスク集合: t0 で `p_hat >= 0.05`
- 共変量 `x`, `r` と各 t0 内三分位は初回 spec §3 と同一
- 主転帰 `end_strict_dead`: **t0+300000 の保存記録点で `p_hat == 0`**
- 副転帰 `end_dead_0.05`: 同時点で `p_hat < 0.05`
- 中間時点の消灯・再点灯履歴は見ない。これは「300k内に一度でも死んだ率」ではなく、ランドマーク300k後の**状態占有率**である
- 同一 seed×unit は最大401回入る。点推定・CI・EQUIV基準は初回 spec §3.3 と同じ seedブロック bootstrap（B=10000, RNGは主 `20260825`、副 `20260826`）
- p_hat×x の3×3統制と `RD_adj` も初回 spec §3.4 と同じ
- 出力は `endpoint_exposures.csv`, `endpoint_rates.csv`, `endpoint_verdict.csv`, `endpoint_cells_3x3.csv`

H-end は初回結果を見て追加した切り分けであり、H-any の主判定を置き換えない。論文で使う場合は **事後追補**と明記する。

## 3. O-legacy: 旧見出しの集約を再現

- seedごとの指標は初回 spec §4.1 の `nmse`
- 点推定 `legacy_nmse = mean_seed(nmse)`
- `legacy_recovery = 1 - mean_seed(nmse_repair) / mean_seed(nmse_current)`
- primary repair は `repair_dead_0.05_k0.5`
- `legacy_recovery >= 0.90` を O1-legacy PASS とする（初回 O1 と同じ90%基準）
- seedブロック bootstrap（B=10000, RNG `20260827`）で recovery の95% CIを付ける。分母0のリサンプルは生じた場合のみ除外し件数を記録する
- summary は current / repair の `nmse mean` を先に、`unfit_var median` を併記する

4対照については初回 spec の結果を保持するが、旧チャット値との一致を主張しない。

## 4. 追加サニティ・再現性

- H-end S1: 401起点と全 `t0+300k` が10 seedすべてに存在
- H-end S2: 曝露数、seed×t0のリスク集合数、個体ごとの反復回数を出力
- O-legacy S1: current / repair の平均 nmse をCSVから再集計し、verdict と最大絶対誤差 `<1e-12`
- 同一 commit・同一コマンドで2回実行し、`*.csv` の sha256 が全一致することを外部照合する。照合結果を `determinism_check.md` に保存する

## 5. 禁止事項

- H-end を H-any の事前登録結果にすり替えない
- H-end を連続時間ハザード、不可逆な恒久死、unit独立の標本と呼ばない
- O の対照値が旧チャットと不一致である事実を隠さない
- 旧値へ近づく変換を結果を見ながら追加しない。対照操作の原コードが回収できない限り、旧対照値は未再現のまま残す
