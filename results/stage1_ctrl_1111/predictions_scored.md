# stage1_ctrl_1111 の予測の当たり外れ（spec §6。予測は本走の前に 5773944b で登録）

数は `judge.md`（登録の規則の判定）と `describe_T.md`（登録外の記述）から。

| 予測 | 結果 | 当たり外れ |
|---|---|---|
| PR1 J1（D200）は ReLU・ELU・SiLU・GELU の課題 4 以降で ALIGN_NOT_NEEDED か MIXED（点推定 < 0.3） | ReLU・ELU・GELU は ALIGN_NOT_NEEDED。SiLU の 3 帯は INCONCLUSIVE（帰無 P0m・P0 の区間が 0 をわずかに外した。ρ_align の点推定は −0.05〜0.01） | 一部（SiLU は判定不能） |
| PR2 leaky ReLU の課題 9 以降の ρ_align は 0.2〜0.6 | 0.09・0.06 | 外れ |
| PR3 課題 2〜3 の ρ_align は 5 活性化とも 0.1〜0.5 | 0.09・0.10・0.04・0.12・0.09 | 外れ（3 つが 0.1 未満） |
| PR4 J2 は課題 4 以降の全ての帯で NORM_NEEDED | leaky ReLU・SiLU・GELU は全て NORM_NEEDED。ELU は NORM_NOT_NEEDED（ρ_norm 0.37〜0.44）、ReLU の 9〜19 は MIXED | 外れ |
| PR5 D200_R と D200_H0 の差は全ての帯で 10 ポイント以内 | R のほうが 13〜58 ポイント多く沈む | 外れ |
| PR6 J3(a) は課題 4 以降の 3 帯 × 5 活性化のうち 15 以上で SINK_SAME か SINK_MORE | 14 帯の全てで SINK_LESS（A0 は 1〜9 ポイント少なく沈む） | 外れ |
| PR7 J3(b): leaky ReLU は LATE_LOSS_MOSTLY_ADAM、ReLU は LATE_LOSS_NOT_ADAM | leaky ReLU は LATE_LOSS_NOT_ADAM（rem_A0 −0.05）。ReLU はデータ不足 | 外れ |
| PR8 課題 3〜10 の rem_H0 > 0.5 が 5 活性化とも | 0.56・0.55・1.33・0.83・0.64 | 当たり |
| PR9 課題 3〜10 の rem_P は 4 以上で rem_H0 より小さい | 5 活性化とも小さい（rem_P 0.04〜0.21） | 当たり |
| PR10 SGD の S は 5 活性化とも課題 2〜3・4〜8 で SINK_PRESENT | SiLU・GELU の課題 2〜3 は SINK_ABSENT（読み出しを 0 にした腕も同じだけ沈む）。他は SINK_PRESENT | 外れ |
| PR11 SGD の課題 21〜30 の不足は 4 以上で Adam より小さい | 比べられる 4 活性化（leaky ReLU・ELU・SiLU・GELU）全てで小さい（SGD 0.00・0.28・−0.17・−0.00 対 Adam 0.38・0.54・0.58・0.76）。ReLU は Adam 側がデータ不足 | 当たり |
| PR12 Lyle の Adam の S は 5 活性化とも課題 2〜8 で SINK_PRESENT | 全て SINK_PRESENT | 当たり |
| PR13 Lyle の Adam の ReLU の課題 21〜30 の不足は、既定の Adam の 0.5 倍未満 | 0.66 対 0.88（制限なしの記述）で 0.75 倍 | 外れ |
| PR14 H0all の不足の比は 5 活性化とも上端 < 1、ReLU は LOSS_REMOVED にならない | 比は 0.02〜0.18 で 5 活性化とも LOSS_REMOVED（ReLU も 0.13） | 外れ（前半は当たり） |
| PR15 Pall は 4 以上で LOSS_KEPT | 5 活性化とも LOSS_KEPT（0.74〜1.04） | 当たり |
| PR16 A0all は leaky ReLU と ELU で LOSS_REMOVED | 1.04・0.70（LOSS_KEPT） | 外れ |
| PR17 Σpush_P/Σpush_S は精度 20%・50% で 0.5 未満、課題 1 の末で 0.7 以上 | 0.07〜0.19（ELU −0.23）・0.18〜0.42・0.83〜1.49 | 当たり |
| PR18 D200_P − D200_S は精度 20%・50% で −15 ポイント以下、課題 1 の末で ±10 ポイント | 精度 20% で −6〜−11、50% で −9〜−18、末で −4〜−6 | 外れ（末だけ当たり） |

当たり 6（PR8・PR9・PR11・PR12・PR15・PR17）、一部 1（PR1）、外れ 11。

## 外れたときの読み（spec §6 に登録した物）との照合

- PR1（整列が要る）は外れなかった。事後の一歩の期待の計算（和の比 0.77〜1.56）と同じく、学習の動きでも整列は沈みに要らない。
- PR4 は ELU で外れたが、読み出しの大きさを縮めると沈みの 37〜44% は消える（NORM_NOT_NEEDED は「半分に届かない」の意味）。毎回縮める軌道（Qall）では ELU の喪失が消えた（比 0.03）。「大きさも整列も要らない」には当たらない。
- PR6・PR7 は外れたが、A0 は沈みも不足も消さなかった。登録した読み（A0 が沈みを消すなら Adam の状態が担い手）は当てはまらない。隠れ層の Adam の状態だけでは、沈みも喪失も説明できない。
- PR10 は、SGD の SiLU・GELU の課題 2〜3 だけで外れた。そこでは読み出しを 0 にした腕も 94〜95% のユニットが沈み、読み出しは沈みを足していない。他の帯では SGD でも沈みが起きる。
- PR14 は「喪失が段 ① を要さない」の向きには外れなかった。全ての切替で出力層を 0 にすると、5 活性化とも喪失が消えた（この箱では Dohare らの作法で喪失が出ない）。
