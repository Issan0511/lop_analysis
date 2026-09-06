# 押し「下げる」非対称の出どころ — φ に定数を足して「|φ| の偏り」と「φ′ の偏り」を切り分ける

親: [[命題1-5_上端則_spec_0905]]（結果 §10.3: 活性化方向へ引き戻す復元場は実在・固定点は場の零点）／[[前活性の力学_事後_0904]] §8（ビー玉 R = E[φ²] + κE[φ′²]） / 状態: **事前登録（2026-09-06・実装前 commit）** / 出典チャット: `活性化プロット_0904`（09-06 夕・Issa の 2 候補）

> **run id: `act_offset_0906`**。`edge_law_0905` の runner に `--config configs/act_offset_0906.yaml` を渡して回す（記録列・窓・フックはそのまま）。**config と本 spec を実装より先に commit する。** 実装と本走は**別マシン（lab・i9-13900KF・62 GiB）で行ってよい**（§9）。比較は本走の腕どうしだけで、元マシンの登録済み腕とは**水準比較しない**。

## 0. 一行

引き戻す力（φ² の井戸）は確認できた。残る問いは**押し「下げる」非対称がどこから来るか**。Issa の 2 候補は R の 2 つの項そのもの——(1) **|φ| の正負偏り**（E_支持[φ²] の最小点が「φ の小さい側」にずれる）、(2) **曲率がだいたい正**（φ′ が左で小さい → E_支持[φ′] / E[φ′²] を下ると左へ）。leaky・ELU・softplus では 2 候補が同じ向きを予測するので今までの実験では区別がつかない。**φ に定数 c を足す**と φ′ は変わらず |φ| の偏りだけが変わるので、これで切れる。

## 1. 仮説

- **H-slope（候補 2）**: 落ち着き先は c に依らない。押し下げは φ′ の非対称（可動度 E_支持[φ′] のラチェット）だけで決まる。
- **H-mag（候補 1）**: |φ+c| の小さい側へ動く。leaky a=0.1 では |φ+c| の零点は c>0 で z=−c/a（負側・**深く沈む**）、c<0 で z=+|c|（**正側に浮く**）→ 着座点が **c の符号で反対に動く**。
- **H-v（第 3 の可能性）**: 定数は勾配では ∂L/∂v にしか入らない（∂L/∂b, ∂L/∂w は φ′ しか見ない）。だから c が効くなら **v の学習経由**で、符号パターンは H-mag と違い得る（例: c の符号に依らず |c| で深くなる）。v 凍結で沈下が 2.9 浅くなった結合（edge_law §5）と同じ経路。

背景の事実: このハーネスは層の**入力**を中心化していて φ の出力は中心化しない（`forward_gate_batch`: `cur − layer_means`）。定数 c は関数としては出力バイアス c_out に吸収されるが、E_支持[φ²] は変わる。

## 2. 腕（10 seed・5M・lr 0.01・a=0.1・用量 12.16・幅 100・記録列は edge_law と同じ）

| 腕 | 活性化 | c | 役割 |
|---|---|---|---|
| `LRoff0_1216` | `leaky_off_0` | 0 | 参照（**同じマシンで** `leaky_relu` と bit 一致・S-limit） |
| `LRoffm2_1216` | `leaky_off_m2` | −2 | H-mag: 浮く |
| `LRoffm0p5_1216` | `leaky_off_m0p5` | −0.5 | H-mag: 少し浮く |
| `LRoffp0p5_1216` | `leaky_off_p0p5` | +0.5 | H-mag: 少し沈む |
| `LRoffp2_1216` | `leaky_off_p2` | +2 | H-mag: 深く沈む |
| `Eoffm1_1216` | `elu_off_m1` | −1 | 併記: ELU は左が −1 で有界。c=−1 で |φ+c| の零点が z=+∞ 側（正側）へ |
| `Eoffp1_1216` | `elu_off_p1` | +1 | 併記: φ+1 ≥ 0、零点は z→−∞ |

c の目安: leaky a=0.1 の末尾は z̄ −3.7・zmax −0.08・半幅 3.8・|v| 0.61・‖w‖ 4.0（`LRnull_1216`）。c=±0.5 は半幅の 1/8、±2 は半幅の 1/2。

## 3. 定義

- `leaky_off_{c}`: φ(z) = leaky_relu(z; a) + c、φ′ = leaky と同一（1 / a）、φ″ = 0（`ZERO_CURVATURE_ACTIVATIONS` に加える）。**c=0 は加算せず `leaky_relu` の式を逐語で返す**（`x + 0.0` は −0.0 の符号だけ変え得るので、bit 一致の担保はこの書き方で）。`act_alpha` は傾き a（[0,1] のガードを leaky と共有）。
- `elu_off_{c}`: φ = elu(z; α) + c、φ′・φ″ は elu と同一。
- 名前と定数は `LEAKY_OFFSET = {"leaky_off_m2": -2.0, "leaky_off_m0p5": -0.5, "leaky_off_0": 0.0, "leaky_off_p0p5": 0.5, "leaky_off_p2": 2.0}`、`ELU_OFFSET = {"elu_off_m1": -1.0, "elu_off_p1": 1.0}` の**明示 dict**（snake_amp と同じ流儀）。
- 窓: 末尾 タスク 451–500、settle 3 窓 301–350 / 376–425 / 451–500。ALIVE = `layer1_denom` 末尾平均 > 0.25。CI = seed 対応 bootstrap 2000 回・`rng(20260906)`。
- 主量は **zmax の末尾中央値**（上端則の定義量）、副量は z̄ の末尾中央値。差 Δ(c) = 腕 c − 腕 c=0（seed 対応）。

## 4. 判定（事前登録・leaky 5 腕で判定・ELU 2 腕は併記）

- **`OFFSET_IRRELEVANT`**（H-slope）: 4 つの c すべてで Δzmax の CI ⊂ [−0.3, +0.3] かつ Δz̄ の CI ⊂ [−0.5, +0.5]。
- **`OFFSET_SIGNED`**（H-mag）: Δz̄(c=+2) の CI < 0 かつ Δz̄(c=−2) の CI > 0（符号が反対）で、|Δz̄| が c=±0.5 より ±2 で大きい（単調）。
- **`OFFSET_OTHER`**（H-v など）: いずれかの c で CI が 0 を外すが `OFFSET_SIGNED` の型でない（例: 両符号で深くなる、または浅くなる）。
- **`NOT_DETERMINED`**: 発散 3/10 seed 超、または settle 3 窓の単調ドリフトが CI 幅を超えて未定着。
- それ以外 **`INCONCLUSIVE`**。

REPORT（登録外・併記）: |v|・‖w‖・線形化率（|v|<0.05）・出力バイアス c_out の末尾値（c·Σv を吸収しているか）の c 依存、ELU 2 腕の Δzmax/Δz̄、復元場（条件 z̄(t−2task)・変位 z̄(t+1task)−z̄(t)）の零点の c 依存。

## 5. 検査（本走前 PASS・**空虚防止の変異対照つき**）

| 検査 | 内容 |
|---|---|
| S-fd | 7 活性化の `act_grad` が forward の中心差分（float64・折れ目 ±1e−3 除外）と一致。`act_curv` は `act_grad` の導関数 |
| S-limit | `leaky_off_0` ≡ `leaky_relu`（forward/backward/curv・格子でバイト一致）。**30k step の短縮走行で `LRoff0_1216` の `state_hash_1m` 相当（30k のハッシュ）が同じ config で `activation: leaky_relu` にした腕と一致**（同じマシン）。変異対照: `leaky_off_p0p5` で同じ比較が**落ちる**こと |
| S-shift | 格子で `leaky_off_c(z) − leaky_relu(z) == c`（バイト厳密）・`act_grad` はバイト一致 |
| S-fallthrough | 7 活性化 × `act_fn`/`act_grad`/`act_curv` が ELU 分岐（`elu_off_*` は `leaky` 分岐）と一致しない |
| S-guard | 7 名が `ACTIVATIONS` に入る・`leaky_off_*` は `ZERO_CURVATURE_ACTIVATIONS` に入る・`act_alpha` の範囲ガードを継承 |
| S-cfg | `--config` 無しの `edge_law_0905` の腕表（30 腕）が不変・`--config configs/act_offset_0906.yaml` で 7 腕の表になる |
| 短縮走行 | 7 腕 × 30k step: 有限・`lr_used`=0.01・新列あり |

## 6. 事前予測

- **Claude**: `OFFSET_IRRELEVANT` 70%。`OFFSET_OTHER` 25%（v 経由で |c| が大きいほど深い・符号は非対称）。`OFFSET_SIGNED` 5%。ELU は c=+1 で沈下群が増え、c=−1 で 0 張り付き群が増えると予想（|φ+c| の零点の側）——ただしこれは H-mag 寄りの予想なので自分の主予測と整合しない。半々。
- **Issa**: （記入待ち。走らせる前にここに書く）
- 外れたときに疑うもの: (i) c_out が c·Σv を一瞬で吸収して E[φ²] の差が消える（そのときは c を出力で相殺できない設定——c_out 凍結——が次の腕）、(ii) c=±2 が lr 0.01 で発散、(iii) 5M で未定着。

## 7. コスト

7 腕 × ≈14 分（1 腕 = 10 seed・1 プロセス・peak RSS ≤0.9 GiB）。全並列で壁時計 ≈ 15 分。生ログ ≈ 7 × 200 MB。

## 8. 引用上の注意

1. 元マシンの登録済み腕（`LRnull_1216`・`Enull_1216`）とは bit 一致しない別マシンの走なので、**水準比較はしない**。参照は本走の `LRoff0_1216`。
2. 定数 c は c_out と冗長。「c が効いた」と言えるのは E[φ²] の差ではなく **v の学習を経由した効果**である可能性を常に併記する（§1 H-v）。
3. ELU 2 腕は判定に入れない（φ′ が非対称かつ |φ| も非対称で、2 候補が同時に動く）。

## 9. 別マシンでの実装手順（lab・Claude Code 宛て）

1. `git pull origin main`。**作業ツリーの未 commit 変更（`snake_phase_0904` など）は触らず、`git worktree add ../act_offset_0906 origin/main` で別ツリーに出す**。
2. `src/nets.py` に**加法のみ**で 7 活性化を登録（§3 の dict・`ACTIVATIONS` 末尾に追加・`act_fn`/`act_grad`/`act_curv` に分岐）。既存分岐の式は一文字も変えない。
3. `src/test_act_offset_0906.py` に §5 の検査を書く（様式は `src/test_snake_flip_0906.py`）。`OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_act_offset_0906 -v` が全 PASS してから走らせる。既存の `src/test_edge_law_*_0905.py`・`src/test_snake_flip_0906.py` も PASS のまま。
4. 本走: `OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.edge_law_0905 --config configs/act_offset_0906.yaml --launch-plan` の出力どおり、**1 腕 = 1 プロセスで 10 seed**（`--seeds` で割らない・S-seed-split）。並列数 = min(コア数, floor(空き GiB / 1.5))。
5. 終わったら `--tail-extract` で `results/act_offset_0906/logs_tail/` を作り、`results/act_offset_0906/{arm_status,logs_tail}` と `src/nets.py`・検査・config の commit を push。生ログ（`logs/*.npz`・約 1.4 GB）は gigafile で送る。
6. 解析（§4 の判定）は `src/act_offset_analyze_0906.py` として `snake_flip_analyze_0906.py` に倣って書く。元マシン側でも同じ script を `logs_tail` に当てて再計算する。

## Log
- 2026-09-06: Issa の 2 候補（|φ| の偏り／曲率が正）から起票。config `configs/act_offset_0906.yaml` と同時に実装前 commit。元マシンは pmnist_rlmnist_0906 が走行中のため、実装・本走は lab に委託。
