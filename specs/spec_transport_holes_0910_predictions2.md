# transport_holes_0910 — 追補 2 で足したラベルの記名予測（走行中・**判定量を読む前**）

親: `specs/spec_transport_holes_0910.md`（単独 commit `9e5bc62` / 追補 1 `d35e565` / 追補 2 `6f2e542`）

**別ファイルにした理由**: 本走が実行中で、各 provenance は書き出しの瞬間に `spec_sha256` を計算する。走っている最中に spec 本文を編集すると、**古いコードで走った腕の provenance に新しい spec のハッシュが入る**（追補 2 R1 と同じ「後から見ると検証できない記録」を作る）。新規ファイルは既存のどのハッシュも動かさない。

## A1″ 反転したユニットは凍るか（追補 2 で主判定に昇格）

Δ = `frozen_units`(t400) − `frozen_units`(t20)。`frozen_units` = 谷の向こう（z < z_c）に居て、ゲートが float32 で厳密に 0 に潰れたプローブ標本が 50% を超えるユニット数。**0 から始まる**（スモークの t8 で 0）。

3 seed とも Δ ≥ 15 → `ESCAPE_FREEZES`／3 seed とも Δ ≤ 3 → `ESCAPE_DOES_NOT_FREEZE`／害の帯 Δ ≤ −3 → `PARTIAL` 側／他 `PARTIAL`。

- **Issa: `ESCAPE_FREEZES`**（2026-09-10・本人から直接取得・走行中・判定量を読む前）
- **Claude: `ESCAPE_FREEZES`**

**ここでも両者は一致している。**V9 §2 段階 2 の「谷を越えたユニットは φ′→0 で凍るまで逃走する」が名指す終状態そのものなので、外れれば**導出の終状態が外れる**。

## A1 の扱い（開示への裁定）

モジュールの `FINISHED` 行が判定量を印字していたため、**SiLU seed 1・2 の `inv_units`(t400) = 80 / 84 を見てしまった**（追補 2 の開示 2）。Claude は A1 を併記へ落とすことを提案したが、**Issa の裁定は「A1 も普通に採点する」**（2026-09-10）。

理由（Issa の選択の読み）: 記名予測は commit `9e5bc62`（Claude `INVERSION_ACCUMULATES`）と `d35e565`（Issa `INVERSION_SATURATES` → A1′ で `ACCUMULATES2` に更新）で**見る前に固定されている**ので、見たことは予測を動かせない。したがって A1 のラベルは有効な記録として数える。

**ただし結果ノートには次を必ず書く**: (1) A1 の 3 seed のうち 2 つは著者が走行中に値を見ている、(2) A1 の閾値は ELU の `dead_hard` からの借り物で、追補 1 が「水準が高く走る前からほぼ決まっている」と診断したものである、(3) **重みは A1″ に置く**。

## 記名予測の現在の総覧（4 副走・8 ラベル）

| ラベル | Issa | Claude | 取得 |
|---|---|---|---|
| A1 反転の水準 | `INVERSION_SATURATES` | `INVERSION_ACCUMULATES` | `9e5bc62` / `d35e565`（走る前） |
| A1′ 反転の蓄積 | `INVERSION_ACCUMULATES2` | `INVERSION_ACCUMULATES2` | `d35e565`（走る前） |
| A1″ 凍結の蓄積 | `ESCAPE_FREEZES` | `ESCAPE_FREEZES` | 本ファイル（走行中・判定量を読む前） |
| B（R）A 深さ | `DEPTH_NO_EFFECT` | `DEPTH_NO_EFFECT` | `9e5bc62`（走る前） |
| B（R）B 幅 | `WIDTH_REMOVES_LOSS` | `WIDTH_REMOVES_LOSS` | `9e5bc62`（走る前） |
| C（谷）A 深さ | `DEPTH_REMOVES_LOSS` | `DEPTH_REMOVES_LOSS` | `9e5bc62`（走る前） |
| D 駆動源の符号 | （未取得） | `VALLEY_SIGN_FLIPS` | `9e5bc62`（走る前） |
| A2・A3 | （未取得） | `VALLEY_FALLS_FASTER` / `WIDTH_ORDER_HOLDS` | `9e5bc62`（走る前） |

**A1 を除く 7 ラベルで両者は一致しているか、Issa が予測を置いていない。**つまりこの走は「どちらが正しいか」を分ける走ではない。分けるのは spec §5 追補 1 に列挙した**両者外れの 5 形**だけである。
