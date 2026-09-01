# オラクル中心化 spec 追補2 — 打ち切り補正（生存中 1 境界あたりの Δβ・事後解析の登録）

proj_004 / 作成 2026-09-02 / 対象リポジトリ: `lop_analysis` / run id: `center_oracle_0831_followup2`
出所: Obsidian `可塑性喪失/spec/オラクル中心化_spec_0831_追補2.md`（vault 作成 commit `4c68d85`・同名衝突解消 `ecd05ee`・読取時 sha256 `be2ee6dc15ef42e3825d55d2a83538f12c75bfb486136bd5226e083c2a85e339`）
親: `specs/spec_center_oracle_0831.md`・`specs/spec_center_oracle_0831_amendment.md`・`specs/spec_center_oracle_0831_followup1.md`

> **状態: 事後解析の登録。** vault 側 §2 の数値は `center_oracle_0831` の**結果を見た後**にチャットで算出済みであり、事前登録ではない。本 spec が固定するのは、その計算を repo の committed 出力として再現し、格を明示して保存する手順だけである。格は **post-hoc reanalysis registration**（`analysis_grade = registered_posthoc_not_preregistered`）。出力する全行に `registered = 0` を立てる。

---

## 0. 格の宣言（**ここを間違えない**）

- **P1（総和版）の登録判定 `BOTH_CONTRIBUTE` は撤回しない。** 本追補は撤回ではなく、読み方の限定と補正量の追加である。
- **独立確認・事前登録効果とは呼ばない。**
- 走り直し・再計装はしない。**入力は committed の生ログだけ**（§2）。

## 1. 直す欠陥 — 総和 endpoint が生存時間で打ち切られる

P1 の登録量 $\Delta\beta^{\rm bnd}$ は 5M 全体の**総和**である。`L1w100_Aexact` は $\mu\equiv0$ で吸収域が一度も引き直されず、中央値 36 境界で全ユニットが凍結し、残り 463 境界では $\Delta\beta=0$ を積む（`L1w100_A1` は中央値 387 境界を生存）。**総和どうしの比 $R_{499}=0.5167$ は生存境界数の比を測っている。**

親 spec 追補1（499 点の真の切替マスク）はそのまま継承する。**変えるのは集計方法だけで、マスクも閾値も動かさない。**

## 2. 入力（**これ以外を読まない**）

`results/center_oracle_0831/logs/L1w100_Aexact_seed{0..9}.npz`（10 本）と
`results/mlp2_phase1_0829/logs/L1w100_A1_seed{0..9}.npz`（10 本）、および照合用に
`results/center_oracle_0831/verdict.csv`。**いずれも committed**（生ログは `git add -f` 済み・tracked）なので fresh clone で完全に再計算できる。

sha256 は実装側 `INPUT_SHA256` に列挙して固定する。**一致しなければ何も書かずに中止する**（`OracleSanityError`）。

読む配列は `step` / `flip_state` / `layer1_M` / `layer1_B` / `layer1_p_hat` / `unfit` のみ。

## 3. 定義（**ここを曖昧にしない**）

- $\beta$ = `layer1_M + layer1_B`（親と同一）。$\Delta\beta$ は隣接記録点の差。
- **境界遷移**: 追補1 の `boundary_499`（`step % 10000 == 1000` かつ `flip_state` が実際に変化）。
- **生存 (alive)**: 当該遷移の**直前の記録点**で `layer1_p_hat > 0`。`p_hat == 0` が strict dead（親と同一）。
- **生存中 1 境界あたりの $\Delta\beta$（rate）**: ユニットごとに、**直前に alive だった境界遷移のみ**を平均する。
- **除外**: 当該窓での生存境界が **10 本未満**のユニットを、その窓・その腕・その seed の集計から外す。除外は窓ごとに判定する。
- **seed 代表値**: 保持ユニットの rate の中央値（親の "seed median unit decomposition" を継承）。
- **CI**: seed クラスタ bootstrap・$B=10{,}000$・percentile・seed `20260829`（親の `shared_draws` / `estimate` をそのまま呼ぶ）。
- **窓**: 全 499 境界 ／ 最初の 150 境界 ／ 最初の $K$ 境界。

### $K$ の決定規則（**vault §2 では「Aexact の生存境界数の中央値」としか書かれていないので、ここで確定する**）

$K = \mathrm{round}\bigl(\mathrm{median}_{\rm seed}\bigl(\mathrm{median}_{\rm unit}(n_{\rm surv})\bigr)\bigr)$

- $n_{\rm surv}$ は `L1w100_Aexact` の**全 499 境界窓**での生存境界数。
- ユニット中央値は**除外を掛ける前の全 100 ユニット**で取る。
- seed をまたぐ中央値を四捨五入して整数化する（端数 .5 は切り上げ）。**seed ごとに $K$ を変えない。**
- 本データでは per-seed が `[28.5, 32.5, 44.0, 39.5, 27.0, 46.0, 33.0, 38.0, 42.0, 34.5]`、その中央値 36.2 → **$K=36$**。$K$ は実装が算出し、36 との一致を非 gate の照合として記録する。

## 4. 登録する endpoint（**すべて `registered = 0`**）

$R' = |{\rm rate}_{\rm Aexact}| / |{\rm rate}_{\rm A1}|$ を seed ごとに作り、その中央値と bootstrap CI を出す。

| id | 量 | 判定 |
|---|---|---|
| **P1'-a**（主） | 全 499 境界での $R'$ | CI 下限 > 1 → `LAG_IS_PROTECTIVE` ／ CI 上限 < 1 → `LAG_IS_HARMFUL` ／ 1 を跨ぐ → `RATE_INCONCLUSIVE` |
| **P1'-b** | 最初の $K$ 境界に揃えた対照での $R'$ | 同上。**P1'-a と符号が割れたら `WINDOW_DEPENDENT` を出して両方報告する** |
| **P1'-c** | 復活件数（総数・タスク内） | Aexact が厳密に 0 → `ABSORPTION_EXACT_UNDER_ORACLE` |
| **P1'-d** REPORT_ONLY | 全滅到達 task、`unfit` の最終水準 | — |

- **復活**: `p_hat == 0` の記録点から次の記録点で `p_hat > 0` になった遷移。**タスク内**は追補1 の `internal_4500`（境界でない遷移）に限る。10 seed 合計で数える。
- **最初の 150 境界**は REPORT_ONLY として併記する（窓依存の目視用）。
- P1（総和版）の行は本 run では再計算して**照合にのみ使う**（§5 S_reproduce）。判定の上書きはしない。

## 5. サニティ

| id | 内容 | gate |
|---|---|---|
| **S_input** | 入力 21 ファイルの sha256 が `INPUT_SHA256` と一致 | **する** |
| **S_mask** | 追補1 の S9 を再実行（`boundary_499` = 499 = `flip_state` 変化数、`internal_4500` = 4500、`startup` = 1） | **する** |
| **S_reproduce** | 同じログから親の P1（`dbeta_boundary_499` の seed 中央値と $R_{499}$）を再計算し、`results/center_oracle_0831/verdict.csv` の記録値と 1e-9 以内で一致 | **する** |
| **S_known** | vault §2 の値（rate −0.0509 / −0.0211・$K$ 窓 −0.0489 / −0.0279・復活 34,222 / 25,723 / 0・$K=36$）と 5e-4 以内で一致 | しない（照合のみ） |

S_reproduce は「rate が同じログの同じマスクの厳密な細分であること」を担保する。**ここが落ちたら rate も信用しない。**

## 6. 交わらない線（vault §5 をそのまま運ぶ）

- §5 の可識別性の交絡はそのまま生きている。結論は「**EMA 遅れ窓を外すと悪化する**」までで、「$\mu$ が保護的である」ではない。
- 「centering を改善すれば LoP を防げる」と書かない。
- 0/10・10/10 は「観測しなかった」の強さ。`strict_dead = 1.0` が 10 seed で揃ったことは「必ず全滅する」ではない。
- **P1'-a のラベルで P1 の `BOTH_CONTRIBUTE` を上書きしない。** 併記の形は「総和では `BOTH_CONTRIBUTE`（$R_{499}=0.5167$）だが、これは生存境界数 36 対 387 の打ち切りを含む。率に直すと符号が反転する（P1'）」。

## 7. errata（vault 側の訂正）

vault `オラクル中心化_spec_0831_追補2` §1 は全滅到達を「task 154–454（中央値 ≈ 210）」と書くが、**per-seed は `[188, 269, 271, 154, 238, 158, 318, 210, 196, 454]` で中央値は 224**（range 154–454 は正しい）。210 は seed 7 の値である。P1'-d は REPORT_ONLY だが、**引用は 224 を使う**。
