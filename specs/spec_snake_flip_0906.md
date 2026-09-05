# Snake の「反転」だけで足りるか — 周期を外した腕と零点を外した腕で Issa の仮説を切り分ける（幅 100 ＋ 幅 5）

親: [[命題1-5_上端則_spec_0905]]（結果: 押し下げは一次の力・特徴は初期支持の中にないと見つからない）／[[前活性の力学_事後_0904]] §8（Snake は零点に着座・sinc 則） / 状態: **事前登録（2026-09-06・実装前 commit）** / 出典チャット: `活性化プロット_0904`（続き・09-06 朝の Issa の仮説）

> **run id: `snake_flip_0906`**。幅 100 側は `edge_law_0905` の runner に `--config configs/snake_flip_0906.yaml` を足して回す（記録列・フック・窓はそのまま）。幅 5 側は `src/w5_snake_0905.py`（`width5_gate_b_0901` 宿主・20 seed・U で適合を測る）。**config と本 spec を実装より先に commit する。**
> lr は登録値ではない（幅 100: Snake α=1 が完走する 0.005・幅 5: ‖J‖² 則）。**登録済み腕との水準比較には使わない**。比べるのは本 spec の腕どうし＋同 lr の対照だけ。

## 0. 一行

**Issa の仮説「Snake が効くのは周期ではなく、原点付近の下に凸のすぐ下に上に凸があること（最初の 1 回の曲率反転）」**を、(H1) 周期を外した `snake1` と (H2) 零点を外した `snake_amp` で切り分ける。

## 1. 仮説

命題 2（曲率の反対へ動く）を境界の両側に当てると、**上側が下に凸・下側が上に凸**の境界 $z_1=-\pi/4\alpha$ には両側から押しが集まり罠になる（逆の並びの $+\pi/4\alpha$ は分水嶺）。幅 100・α=1 の実測は 977/1000 が $z_1$ に着座、18 が上の零点、5 が下の零点 — **周期の 2 本目以降はほぼ使われていない**。

- **H1（周期は要らない）**: 第 1 罠までの 1 葉だけ残して外を線形にした `snake1` は、Snake と同じ分布・同じゲート・同じ適合になる。
- **H2（零点は要らない・反転が本体）**: $\varphi'=1+A\sin 2\alpha z$（$A<1$、零点なし・反転位置は同じ）でも罠は残り、沈まず、ゲートは開いたまま。

対立仮説: 罠の強さは $\varphi'$ の落ち込みの深さ（零点に達すること）が要る → `snake_amp` は罠が弱くて拡散する／沈む。

## 2. 腕

### 2.1 幅 100（`edge_law_0905` runner・10 seed・5M・lr フック 0.005・記録列は edge_law と同じ）

| 腕 | 活性化 | 役割 |
|---|---|---|
| `SN_a1_1216` | `snake` α=1 | 参照（新列つきで取り直す） |
| `SN1_a1_1216` | `snake1` α=1 | H1 |
| `SNA05_a1_1216` | `snake_amp0p5` α=1（$\varphi'\in[0.5,1.5]$） | H2 |
| `SNA025_a1_1216` | `snake_amp0p25` α=1（$\varphi'\in[0.75,1.25]$） | H2（浅い反転） |

### 2.2 幅 5（`w5_snake_0905`・20 seed・5M・lr は ‖J‖² 則・教師幅 100 で残差が残る）

| tag | 活性化 | lr | 対照 |
|---|---|---|---|
| `SN1_a1` | `snake1` α=1 | 則 | `asweep_a10`（Snake α=1・lr 0.0037・U 0.068） |
| `SNA05_a1` | `snake_amp0p5` α=1 | 則 | 同上、`LR5x_lr00037`（leaky 0.1・同 lr・U 0.38）、`LIN5` |
| `SNA025_a1` | `snake_amp0p25` α=1 | 則 | 同上 |

‖J‖² 則: $\mathrm{lr}=0.01\cdot\lVert J_{LR5}\rVert^2/\lVert J_{arm}\rVert^2$、$J=\partial\hat y/\partial\theta$ を init・32 パターン・20 seed で測る。**S-lr-rule: snake α=1 で 0.0037 を再現すること。**

## 3. 定義

- `snake1`: $[-3\pi/4\alpha,\ +\pi/4\alpha]$ で Snake、外は $z+1/(2\alpha)$（連続・傾き 1・継ぎ目は $\varphi'$ の極大なので $\varphi'$ が 2→1 に跳ぶ $C^0$）。残るのは原点周りの下に凸と、そのすぐ下の上に凸だけ。
- `snake_amp{A}`: $\varphi=z+A\sin^2(\alpha z)/\alpha$、$\varphi'=1+A\sin 2\alpha z$、$\varphi''=2\alpha A\cos 2\alpha z$。`snake_amp1` は S-limit 用（`snake` と bit 一致）。
- 第 1 罠 $z_1=-\pi/4\alpha$、罠占有 $=|\bar z-z_1|<\pi/4\alpha$。窓: 幅 100 は タスク 451–500（settle 3 窓 301–350/376–425/451–500）、幅 5 は 491–500。ALIVE = `layer1_denom`>0.25。CI = seed bootstrap 2000 回・`rng(20260906)`。

## 4. 判定（事前登録）

### H1（`SN1` 対 `SN`）
- (a) 幅 5: 対応 seed の $\Delta\log_{10}U$ 中央値と CI。
- (b) 幅 100: 罠占有率の差（seed 対応）と CI。
- (c) 幅 100: `mob` 中央値の差と CI。
- **`ONE_FLIP_EQUIV`**: (a) CI ⊂ [−0.1, +0.1] dex かつ (b) CI ⊂ ±0.05 かつ (c) CI ⊂ ±0.05。**`LOBE_MATTERS`**: いずれかで CI が 0 を外し |中央値| が帯の外。それ以外 **`INCONCLUSIVE`**。

### H2（`SNA05` 対 `SN`・`SNA025` は併記）
- (i) **ゲート**: `mob` 中央値が `SN` の −0.05 以上 → `GATE_OK`。
- (ii) **沈まない**: 末尾 $\bar z$ 中央値 $>-1.5$ かつ settle 3 窓の単調ドリフトが CI 幅以下 → `NO_SINK`；中央値 $<-1.5$ または下降が続く → `SINKS`。
- (iii) **罠の位置**: 末尾 $\bar z$ 中央値が $z_1\pm0.3$ → `TRAP_AT_INFLECTION`；それ以外 `TRAP_ELSEWHERE`（REPORT: 罠占有率・$\bar z$ の IQR）。
- (iv) **適合（幅 5）**: $\Delta\log_{10}U$ 対 `SN`（`FIT_EQUAL` ±0.1 dex / `FIT_WORSE` / `FIT_BETTER`）、対 `LR5x_lr00037`（`BEATS_LEAKY` CI<0 / `NOT`）。
- **`FLIP_SUFFICES`** = GATE_OK ∧ NO_SINK ∧ TRAP_AT_INFLECTION（＋ (iv) の修飾子）。**`ZERO_NEEDED`** = SINKS または TRAP_ELSEWHERE。**`NOT_DETERMINED`** = 発散 3/10 超・未定着。

### REPORT
罠まわりのドリフト場 $E[\Delta\bar z\,|\,\bar z-z_1]$ の符号反転（±0.3 の帯・幅 100）、$W$ と $1-\mathrm{sinc}(\alpha W)$、‖w‖・|v| の推移。

## 5. 検査（本走前 PASS）

| 検査 | 内容 |
|---|---|
| S-fd | 3 族が自分の forward の真の導関数（float64 中心差分・`snake1` の継ぎ目 ±1e−3 除外）・`act_curv` は `act_grad` の導関数 |
| S-limit | `snake_amp1` ≡ `snake`（forward/backward・バイト）・`snake1` ≡ `snake`（葉の内側の格子点・バイト） |
| S-fallthrough | 3 族 × `act_fn`/`act_grad`/`act_curv` が ELU 分岐と一致しない |
| S-guard | 3 族が `ACTIVATIONS` と `WEIRD_FREQ_ACTIVATIONS` に入る・`snake` を `act_curv` に登録 |
| S-lr-rule | ‖J‖² 則の実装が snake α=1 で lr 0.0037 を再現 |
| S-cfg | `--config` で腕表・出力先・活性化マップが差し替わり、`--config` 無しの `edge_law_0905` の挙動は不変（`build_cfg()` の腕表が commit 済みと一致） |
| 短縮走行 | 4 腕 × 30k step: 有限・新列あり・`lr_used`=0.005 |

## 6. 事前予測

- **Issa**: H1 → `ONE_FLIP_EQUIV`／H2 → `FLIP_SUFFICES`（零点は不要・反転が本体）。
- **Claude**: H1 `ONE_FLIP_EQUIV` 90%（幅 5 の |Δlog₁₀U| < 0.05 dex）。H2 `FLIP_SUFFICES` 70%（対立: 罠が弱く拡散して `TRAP_ELSEWHERE` 20%）。適合は `FIT_WORSE` 60%（非線形振幅が半分）・`BEATS_LEAKY` 65%。`SNA025` は罠がさらに弱く $\bar z$ の IQR が 2 倍・U 0.15〜0.25。
- 外れたときに疑うもの: (i) 幅 100 の lr 0.005 で 5M 内に定着していない、(ii) `snake1` の継ぎ目（$\varphi'$ 2→1 の跳び）が新しい折れ目として効く、(iii) `snake_amp` で E[φ′²] が下がり ‖J‖² 則の lr が上がって発散。

## 7. コスト

幅 100: 4 腕 × ≈15 分（並列）。幅 5: 3 腕 × ≈27 分（並列）。壁時計 ≈ 30 分・RSS 7 本 × ≤0.9 GiB。

## 8. 引用上の注意

1. lr が登録値でないので、登録済み腕（`SN_a1_1216`@0.01 は発散・`LR_1216` など）との**水準比較はしない**。
2. H2 の `GATE_OK` は $\varphi'\ge 1-A$ で下駄が入っている。ゲートの主張は「Snake と**同程度**」までで、「leaky より良い」は幅 5 の U でしか言えない。
3. 幅 5 の `U` は容量不足箱の適合で、可塑性の直接指標ではない。

## Log
- 2026-09-06: Issa の仮説（09-06 朝）から起票。config `configs/snake_flip_0906.yaml` と同時に実装前 commit。
