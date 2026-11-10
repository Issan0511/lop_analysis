# 追補 E の結果（1111）: RMSNorm の腕と 5 seed

**結論。** head を共有する腕は、LayerNorm でも RMSNorm でも 5 seed すべてで、後期の学び直しが新品より遅くなった。head を課題ごとに 0 に戻す腕は、5 seed すべてで新品より速いままだった。追補 D の 1 seed の結果は seed によらず、平均を引かず β も無い RMSNorm でも同じだった。

箱は追補 D と同じ（2 ブロック、FFN 3072、ReLU、seed ごとに固定の 2,500 枚、課題ごとに乱数ラベル 10 値、U = 5,000、100 課題、Adam 1e−4）。seed 10252010 の LN の組は追補 D の走（借りた機械）で、ほかの 18 本と新品 36 本はすべて white-san で回した。byakko（先生の PC）の分も 1108 から white-san で回した（E.2 の 1108 の記録）。最後の列は 1111 03:42 に終わり、30 本とも rc = 0。判定表の全体は `judge_e.md`、課題ごとの T90 は `e_t90.png`、道具は `posthoc_2blk/fable_e_tables.py`。

## 1. 判定（spec E.3。規則は走る前のまま）

| 正規化 | 共有の腕 | 後期の T90 | Q | 戻す腕 | 後期の T90 | 組 | 盾 |
|---|---|---|---|---|---|---|---|
| LayerNorm | SL `LOSS` 5/5 | 675〜925 | 1.32〜1.76 | R0L `NO_LOSS` 0/5 | 75 | `SHARED_WORSE` 5/5 | `PHASE1_BREAKS_SHIELD` |
| RMSNorm | SRMS `LOSS` 5/5 | 775〜1,138 | 1.44〜2.22 | R0RMS `NO_LOSS` 0/5 | 75 | `SHARED_WORSE` 5/5 | `PHASE1_BREAKS_SHIELD` |

- 後期の T90 は課題 81〜100 の中央値。Q はそれを同じ seed の新品 4 本の中央値（512〜538）で割った値。
- 止める規則（E.6）に当たった走は無い。

予測の採点（E.5）: PE1〜PE5 は当たり。PE6（SRMS の早期の s_load の中央値が 0.75 以上の seed が 3 以上、確率 0.30）は外れで、0/5 だった。

## 2. 分かったこと

- **共有 head の遅れは seed によらず、RMSNorm でも出る。** 課題 2〜11 の T90 は SL 475〜525・SRMS 425〜525 で新品並み。課題を重ねるにつれて遅くなり、後期は新品の 1.3〜2.2 倍になる。
- **遅れの大きさは正規化では決まらない。** seed ごとに SRMS と SL の Q を比べると、差は −0.32〜+0.90 で、向きが seed によって入れ替わる。どちらが大きいとは言わない。
- **戻す腕は両方の正規化で新品の 7 分の 1 の更新で覚える。** R0L と R0RMS の T90 は、課題 2〜11 の 125 から後期の 75 へ速くなり続ける（5 seed とも同じ）。
- **装填の押しの向きは RMSNorm でも LN と同じに削られる（記録）。** 早期（課題 1・2・5 の末）に装填の成分を通した押しが沈める側の unit は、SRMS で 54〜57%、SL で 54〜57%。直通の項だけなら SRMS 86〜93%・SL 87〜91% が沈める側で、分散微分の項がそのそろいを打ち消す。最後の正規化が平均を引かなくても同じだった。この量の道具（`j5_rlfix.py` に RMSNorm の逆伝播を足し、`fable_eg_j5.py` で 5 seed）は、seed 10252010 の SL で追補 D の値と 19 状態すべて一致した。
- **盾の判定は両方の正規化で `PHASE1_BREAKS_SHIELD` だった。** 戻す手順では喪失せず、head を共有して切替の押しを通すと喪失する。spec §11.1 がこの場合に用意した文（正規化は、押しが無いときだけ盾だった）と同じ形で、この箱では 5 seed と RMSNorm で成り立つ。

## 3. 言わないこと

- RMSNorm が LN より悪い・良いとは言わない（上の Q の差は seed で向きが変わる）。
- この箱（2 ブロックの ViT、固定 2,500 枚、乱数ラベル、共有 head）の外へ一般化しない。
- seed 10252010 の LN の組は借りた機械の走で、ほかの組と丸めが違う。正規化の間の比較は seed 10252011〜14 の同じ機械の組で読む。

## 4. 生データ

`~/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs/{SL,R0L,SRMS,R0RMS,freshL,freshRMS}/seed*/`、列の記録 `run_queue_jobs_rlfix_e_byakko_whitesan.log`（white-san の前半は `run_queue_jobs_rlfix_e_whitesan.log`）、装填の記録 `posthoc_j5/j5_<腕>_seed<seed>.json` と `posthoc_j5/eg_j5.log`。
