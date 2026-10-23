# S-J6 の続き: 符号の食い違いは近零の unit に限られるか（2026-10-24・compute_J）

script: `analysis/chargpt_detlabel_1023/j6/check_sj6_unit.py`。数は `sj6_unit_level.json` に記録した。

## 近零の定義（固定）
- v は、符号の割合の列が数える unit ごとの値である。例えば `s_original_other` なら、第二段階の配列 original_other[i] = mean_n K_n φ′(z_ni)·other_ni を指す。`s_*_self_plus_*` は 2 つの配列の和、`s_nonzero` は original_exact_K である。
- 母集団は、その列の割合の分母になる unit（生きた unit）である。`s_nonzero` だけは、そのうち v ≠ 0 の unit に限る。
- max は、同じ状態・同じ列・同じ側の母集団の上で取った max |v| である。
- 近零 ⇔ |v_i| ≤ 1e−12 × max。v = 0 も含む。
- 符号が機械で変わりうる unit（flippable）は、近零から次の unit を除いたものである: 勾配の開かない unit（gradient_active でない）で v = 0 のもの。この unit は全項に門 0 が掛かるので、どの機械でも 0 になる。
- S > 0 の数え方（母集団の中で v > 0 の割合）は変えていない。

## 1e−12 でよいか
- 全 37 列 × 350 状態の、0 でない |v|/max の分布を見た。≤1e−16 が 314、(1e−16, 1e−15] が 221、(1e−15, 1e−14] が 11、(1e−14, 1e−13] が 0、(1e−13, 1e−12] が 2、(1e−12, 1e−11] が 23、(1e−11, 1e−10] が 265 である。ちょうど 0 は 33,892 個あった。
- 全体としては 1e−12 に谷は無い。1e−13〜1e−10 の値は self・residual・第三段階の self の列にあり、そこでは食い違いが 0 件だった。
- 食い違いの起きた 2 列（`s_original_other`・`s_loaderror_other`）では、0 でない近零 542 個（270 と 272）はすべて ≤1e−14 にある。(1e−14, 1e−12] は 0 個で、その上は 1e−10 まで空く。例外は、近零の外の (1e−12, 1e−10] にある `s_loaderror_other` の 1 個だけである（`sj6_unit_level_tau.json` の bycol）。
- 食い違いの説明に要る幅は最大でも 1e−15·max だった（下の表）。1e−12 は、この 2 列の雑音と本物の値の間に入っている。

## 件数（A2 の C3 seed 0〜9 × 35 状態、3 file の符号の列 37 = states 11・components 14・loading 12）
| 量 | 値 |
| --- | --- |
| 状態 × 列 | 12,950 |
| 正の数が食い違った状態 × 列 | 86（`s_original_other` 40・`s_loaderror_other` 46。ほかの 35 列は 0） |
| 食い違った unit の数の下限（Σ\|正の数の差\|） | 112 |
| loadsign の割合 × n が整数でない | 0 |
| 必要条件（loadsign の正の数 − こちらの近零でない正の数 ∈ [0, flippable の数]）が破れた状態 × 列 | 0 / 12,950 |
| 食い違いを説明するのに要る最小の幅 τ | 0（勾配の開いた unit のちょうど 0）: 6、1e−16: 70、1e−15: 10、それより大きい τ: 0 |
| 後期（課題末 89〜100）の食い違い | 0 |

各状態の近零の unit の数（`s_original_other`、350 状態）:
- 0 でない近零: 0 個が 259 状態、1 個が 34、2 個が 15、3 個が 18、4 個が 6、5 個が 3、6 個が 7、7 個が 4、9 個が 3、16 個が 1。
- 勾配の開いた unit のちょうど 0: 0 個が 341 状態、1 個が 9。
- 勾配の開かない 0: 0〜69 個（分布は json の `near_units_per_state_by_column`）。

## unit を 1 つずつ特定する手順 (2) は、この機械ではできない
- loadsign の unit の配列（`raw/`・`raw_components/`・`raw_loading/` の `C3_seed<k>.npz`、計 30 file）は、`results/chargpt_loadsign_1023/backup_manifest.json` によると white-san の `/home/issan/Projects/obsidian-research-data/chargpt_loadsign_1023/…` にだけある。byakko の SSD の作業場所（/home/kubo/project/Nakatsuka 以下）には無い。
- 上の結果は、loadsign の csv の割合から数え上げで出せる必要条件である。どの unit が食い違ったかは特定していない。
- 配列が届けば、`check_sj6_unit.py --ref-dir <dir>` で unit ごとの照合が回る。照合では manifest の sha256 を確かめ、食い違った unit の両側の値・両側の max・両側で近零かを並べる。自分の配列を ref にした試しでは食い違い 0 だった（`sj6_unit_level_selftest.json`）。

## Issa の条件
1. 判定に使う量: 後期の `s_original_other` と `s_original_exact_K` の seed 内中央値は、10/10 seed で完全に一致した。記録の窓は次のとおり: `s_original_exact_K` は早期・中期とも 10/10 で一致した。`s_original_other` は早期が 7/10、中期が 6/10 で一致した（記録だけで、判定には使わない）。
2. 恒等式・autograd・連続量の照合は、今の条件で通っている（`sj6_check.json`・`A2_full_vs_loadsign.json`）。
