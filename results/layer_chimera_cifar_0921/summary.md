# layer_chimera_cifar_0921 -- 層別キメラ (S-A) の登録判定

窓 = t31-50 の online 平均。raw のみ・R=10・seed 0-9。

| cell | act1 | act2 | 窓(中央値) | t1 online | 低下 | 段 | 型 | T_A 中央値 |
|---|---|---|---:|---:|---:|---|---|---:|
| `LL` | LR | LR | 0.7277 | 0.8621 | +0.0433 | ALIVE | SLOW_SINK | - |
| `EL` | ELU | LR | 0.1397 | 0.8286 | +0.1900 | PARTIAL | LEARNS_THEN_PARTIAL | 3 |
| `GL` | GELU | LR | 0.1151 | 0.1425 | +0.0034 | AT_FLOOR | DEAD_FROM_T1 | 1 |
| `LE` | LR | ELU | 0.1002 | 0.8717 | +0.1350 | AT_FLOOR | LEARNS_THEN_DIES | 3 |
| `EE` | ELU | ELU | 0.1011 | 0.8324 | +0.1724 | AT_FLOOR | LEARNS_THEN_DIES | 3 |
| `GE` | GELU | ELU | 0.1159 | 0.1167 | +0.0005 | AT_FLOOR | DEAD_FROM_T1 | 1 |
| `GG` | GELU | GELU | 0.1155 | 0.1316 | +0.0011 | AT_FLOOR | DEAD_FROM_T1 | 1 |

**Q1 主判定: `MIXED`**

## C1
- EL - EE: median +0.0388, 10/10 seeds, p=0.0020 -> A_WINS
- GL - GE: median -0.0004, 4/10 seeds, p=0.7539 -> TIE
- LL - LE: median +0.6270, 10/10 seeds, p=0.0020 -> A_WINS

## C2
- GL - LL: median -0.6115, 0/10 seeds, p=0.0020 -> B_WINS
- GE - LE: median +0.0160, 10/10 seeds, p=0.0020 -> A_WINS
- EL - LL: median -0.5890, 0/10 seeds, p=0.0020 -> B_WINS
- EE - LE: median +0.0000, 6/10 seeds, p=0.7539 -> TIE

## C3
- LL - EL: median +0.5890, 10/10 seeds, p=0.0020 -> A_WINS
- EL - GL: median +0.0242, 10/10 seeds, p=0.0020 -> A_WINS

## Q2 機能喪失時点で深い層
- `LL`: NO_LOSS
- `EL`: DEEPER_L1_AT_LOSS
- `GL`: DEEPER_L1_AT_LOSS
- `LE`: DEEPER_L2_AT_LOSS
- `EE`: DEEPER_L1_AT_LOSS
- `GE`: DEEPER_L1_AT_LOSS
- `GG`: DEEPER_L1_AT_LOSS

## Q3 担い手 ||mu2||
- EE - EL @t10: median +449.894, 10/10, A_WINS, non_overlapping=False
- `GL`: MU2_BELOW_INIT (10/10, t00 1.529 -> t10 0.078)
- `GE`: MU2_BELOW_INIT (10/10, t00 1.529 -> t10 0.137)
- `GG`: MU2_BELOW_INIT (10/10, t00 1.529 -> t10 0.085)

## Q4 第1層は下流に依らないか
- GELU 列: **L1_INDEPENDENT_OF_L2**（帯は `GG` の seed 範囲）
- ELU 列: **L1_INDEPENDENT_OF_L2**（帯は `EE` の seed 範囲）

## 予測の採点（§6 の登録。Issa は P1 を含め Claude の確率にすべて同意）

的中 11 / 外れ 6。

| 外れ | 登録した述語 | 実際 |
|---|---|---|
| P1 | 主ラベル `BOTH_LAYERS_THREE_TYPES`（0.35） | `MIXED`（次点に置いた 0.40 の枝） |
| P2 | `EL` の型 = `SLOW_SINK`（0.55） | `LEARNS_THEN_PARTIAL`（窓 0.1397・次点 0.30 の枝） |
| P6 | C1 の 3 対すべてで `L2_FLOOR_HELPS`（0.55） | `GL − GE` が `TIE`（両方 t1 で死んでいるので差が出ない）。`EL − EE`（0.85 と置いた）は A_WINS |
| P10' | `EE` が `DEEPER_L2_AT_LOSS`（0.65） | `DEEPER_L1_AT_LOSS` |
| P14 | ELU 列が `L1_DEPENDS_ON_L2`（0.60） | `L1_INDEPENDENT_OF_L2`。ただし帯が広い（下記） |
| P15 | `EL` が `ALIVE` のまま Q₁(t10) > 0.9（0.50） | Q₁(t10) = 0.995 だが `ALIVE` ではない |

的中: P3・P4・P5・P7・P8・P9（2 セル）・P10・P11（順位のみ）・P12・P13・P17。

## 読み（すべて事後。登録判定は上の表が正本）

1. **3 型は 1 つの箱に出たが、並びが予測と違う。** 生き残ったのは `LL` だけで、**どちらか一方の層を ELU にするだけで落ちる**（`EL` 0.140・`LE` 0.100 対 `LL` 0.728）。GELU はどちらの層でも t1 から死ぬ。
2. **沈む深さではなく床の有無が分けている。** `LL` の第1層も沈む（t50 で p⁺₁ = 0.021・d₁ = −2.13）のに 0.73 を保つ。`EL` の第1層は p⁺₁ = 0.0011・d₁ = −3.44 で、**Q₁ が 0.998**（leaky なら構成上 0）。幾何の沈下は近いのに、ゲートに床があるかどうかで生死が分かれた。
3. **`LE` は raw で「第2層の死」を作る。** 第1層は全セル中いちばん健康なまま（p⁺₁ = 0.226 で `LL` の 0.021 より高い）、第2層だけが Q₂ = 1.000・d₂ = −3.68 へ落ちる。A6 が「raw は `L1_FIRST` 10/10」と読んだ箱でも、第1層に leaky を敷けば第2層の死が単独で作れる。
4. **担い手 ‖µ₂‖ は第2層の活性化が決める。** t50 で `LE` 847・`EE` 734 に対し `EL` は 9.5（初期 2.5 の 4 倍で頭打ち）。第2層が leaky だと µ₂ は育たない。GELU 列は初期値以下（0.24–0.39 対 1.53）で 10/10。
5. **低応答の先行と機能喪失は別**（A6 の注意の再確認）: `EE`・`EL` は t1 で Q₁ ≈ 0.80 を越えているが online が 0.5 を割るのは t3。
6. **Q4 の帯の広さ**: GELU 列は帯が狭く（p⁺@t10 で 6.7e−5）差はその 1/8 なので独立は鋭い。ELU 列は帯が 0.05–0.08 と広く、`L1_INDEPENDENT_OF_L2` は**この帯では何も除外していない**（spec §5.5 の「両列とも独立なら帯が広すぎたと読む」に該当）。ELU 列の独立性はこの走では未決とする。
