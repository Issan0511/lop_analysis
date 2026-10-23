# spec: 担い手の交代を保存状態で確かめる（chargpt_handover_1023）

状態: 登録（走る前に量・集約・判定・予測を固定）/ 2026-10-23 / 読みと予測 Fable（V13）・spec と実装 Claude（Opus 5.5）/ 親: chargpt_seed10_1022
新しい学習はしない。保存済み checkpoint の順伝播と、既存の再生の出力の読み直しだけを行う。

## 1. 問い

C3-2（2 ブロック、pre-LN、FFN は ReLU、LN は γ のみ、d = 128、FFN 512）では、読み出しに隣接する FFN（L1）が課題 10 までにほぼ閉じ、後期に戻る。一つ手前の FFN（L0）は課題の中で沈み、課題 100 で多くが死ぬ。

Fable の読み「担い手の交代」は次の 4 文である。
- L1 が閉じている間、課題を学ぶ仕事は L0 に移る。
- L0 は働かされて育ち、回転が凍って死ぬ。
- L1 は最終 LN の分母を通る帰還で後から戻る。
- 戻る頃には L0 が使い潰されている。

この読みを、保存された課題末の状態で 4 つの検査にかける。

既知（重ねて主張しないもの）:
- L1 の入力平均 ȳ₁ を残差の成分に射影すると、序盤は FFN0、終盤の C3 は注意 1 が大きい（chargpt_l0_carrier_1021）。本 spec の検査 2 は最終 LN の前の残差 t の分担で、量が違う。
- L0 への装填の押しは、期待では浮かせる側に寄る（chargpt_seed10_1022 A の記録、沈める側 0.29〜0.34）。
- 帰還を塞ぐ腕 lnex1 では、L1 は対象 6 seed のうち 5 で生き返らない（chargpt_seed10_1022 C1）。

## 2. 材料

- 束: `obsidian-research-data/chargpt_seed10_1022/transfer/chargpt_seed10_1022_bundle.tar`（sha256 5729ec02…、manifest は同じ場所）。worktree の直下に必要な checkpoint だけ展開する。展開物は git に入れず、終わったら消す。
- 状態: 課題末 1〜10・25・39〜50・89〜100 の 35 点。腕は C3-2（主）と ReLUcap-2。seed 0〜9。合計 700 状態。init は使わない。
- 定義は `analysis/chargpt_seed10_1022/load10.py` を import して使う（`load`・`data`・`chunks`・`ce`）。
- probe: 状態ごとに、その課題の chain probe（`data(seed, t)` の x・y、64 系列 × 128 token）。
- CE: load10 の head の L と同じ式（8192 token すべての平均）。
- 生きた unit: probe のどこかで前活性 z > 0 になる unit（load10 の alive）。
- 網の source: `analysis/chargpt_latepush_1023/source/src`（033679e0 の blob と同じ。`git hash-object` で確かめた）。
- 検査 3 の入力: `results/chargpt_seed10_1022/revival/{C1,C2,C3}/seed*/<腕>/layer_stats.csv`（byakko の再生の出力）。

## 3. 量

### 3.1 検査 1 担い手の切除

FFN_l の出力は Block の `mlp(n2(x))` の戻り値（down(φ(up(·)))）とする。切除はこの戻り値を残差に足す前に置き換える。注意と LN は変えない。

| 記号 | 状態 |
| --- | --- |
| CE_a | そのまま |
| CE_b | FFN0 の出力を 0 |
| CE_c | FFN1 の出力を 0 |
| CE_d | 両方を 0 |

- L0 の担い carry0 = CE_b − CE_a。
- L1 の担い carry1 = CE_c − CE_a。
- 記録: carry01 = CE_d − CE_a。
- 記録: 平均で置く切除。FFN_l の出力を、そのままの forward での 8192 token の平均ベクトルで置き換える（mcarry0・mcarry1）。
- 記録: 位置 0 を除いた CE（訓練の損失と同じ範囲）でも同じ差を出す。

### 3.2 検査 2 残差への書き込みの分担

最終 LN の前の残差は t = e + a0 + f0 + a1 + f1 と書ける。
- e は token 埋め込み wte(x)。位置は注意の中の RoPE で入り、残差に足す位置ベクトルは無い。
- a_l = sc·attn_l(n1(·))、sc = 1/2。
- f_l は FFN_l の出力。

主の量は w_c = token 平均の ‖c_n‖ / ‖t_n‖ とする。

記録:
- 射影の分担 π_c = token 平均の ⟨c_n, t_n⟩ / ‖t_n‖²（5 成分の和は 1）。
- 中心化の比。各 token で d 方向の平均を引いてから同じ比を取る（LN が見る形）。
- ‖t_n‖ の token 平均。

### 3.3 層の統計（検査 4 と照合に使う）

同じ probe・同じ forward で、層ごとに次を出す。

| 量 | 定義 |
| --- | --- |
| alive | 生きた unit の割合（n_alive / 512） |
| dead005 | p_j < 0.005 の unit の割合（p_j は z_j ≥ 0 の token の割合。プロジェクトの「死」、C 判定の frac_p_lt005 と同じ式） |
| r_med | unit の μ_j / (σ_j + 1e−12) の中央値（σ は ddof 1、torch.quantile 0.5） |
| mu_med・m_median | μ_j の中央値（m_median は load10 と同じ torch.median で、照合に使う） |

### 3.4 検査 3 lnex1 の L0（既存の出力の読み直し）

- 対象: C1（C3-2、seed 0・1・2・4〜9）の control と lnex1。課題 S〜E は C1 の表のとおり。
- 量: layer_stats.csv の u = 10000、probe = uniform（C1 判定と同じ）の L0 の dead005 と r_med。
- seed ごとに、課題 S+1〜E の平均の差（lnex1 − control）を出す。
- R′（control で L1 の dead005 が初めて 0.5 以下になる課題）のある seed は、[R′, E] の平均の差も出す。
- seed 間は符号の数と両側 exact 符号検定（0 の差は除く）で記述する。
- 記録: chain probe の同じ量。nocommon の腕。C3（ReLUcap-2 の lnex1、10 seed）。C2（課題 25 末から γ を 0.3 倍にする gamma03f・gamma03、課題 26〜31、10 seed）。

読みの帰結は向きで 2 つある。予測の確率は無いので判定はしない。
- 「L1 が閉じている間は L0 が働かされる」なら、lnex1 は [R′, E] で L0 を control より沈める（dead005 が上、r_med が下）。
- 「L1 が戻るほど L0 が沈む」（検査 4 の予測の向き）なら、control の L0 の方が沈む。

### 3.5 検査 4 L1 の戻りと L0 の沈みの相関

- 対: 両端が保存された連続する課題末。1→2〜9→10、39→40〜49→50、89→90〜99→100 の 31 対。
- 増分: Δalive_L1 と Δr_med_L0（3.3 の量、自前の forward）。
- seed ごとに 31 対の Spearman ρ を出す。同順位は平均順位にする。片方の増分が定数なら ρ は未定義とする。
- 記録: Δ(1 − dead005_L1) と Δr_med_L0。Δalive_L1 と Δdead005_L0（正なら「L1 が戻るほど L0 が死ぬ」）。窓ごとの ρ（9・11・11 対）。
- 記録: 元の走の layer_stats.csv（訓練時の GPU float32、u = 10000、chain probe、全 100 課題）で 99 対の Δ(1 − dead)_L1 と Δr_med_L0 の ρ。

## 4. 集約

- 窓は W1 = 課題末 1〜9（9 点）、W2 = 39〜50（12 点）、W3 = 89〜100（12 点）。課題末 10・25 は軌跡に残すが窓には入れない。
- seed ごとに窓の中央値（numpy の中央値）を取る。
- seed 間は中央値と範囲 [最小, 最大] で示す。
- 数は seed ごとに条件を評価してから数える。unit・token・課題を独立な標本として数えない。

## 5. 予測の採点の条件（走る前に固定）

| 記号 | 腕 | ○ の条件 |
| --- | --- | --- |
| P1a | C3-2 | W2 で carry0 > carry1 の seed が 8/10 以上 |
| P1b | C3-2 | carry1 の W3 の中央値 > W2 の中央値の seed が 7/10 以上 |
| P1c | ReLUcap-2 | 3 窓のすべてで、carry1 ≥ min(carry0, carry0/2) の seed が 8/10 以上 |
| P2 | C3-2 | w_f0 が W1 → W2 で増える seed が 7/10 以上、かつ w_f1 が W2 → W3 で増える seed が 7/10 以上 |
| P4 | C3-2 | ρ < 0 の seed が 7/10 以上（未定義の seed は ○ に数えない） |

- 比べるのはどれも seed ごとの窓の中央値である（P4 を除く）。
- P1c の「同程度以上」は「L0 の担いの半分以上」と読んだ（同程度 = 2 倍以内）。carry0 ≤ 0 のときは carry1 ≥ carry0。
- P2 は片方だけ満たせば「半」、両方外れれば ×。
- 予測に seed 数が無かった P2 は、P1b・P4 と同じ 7/10 を置いた。
- 記録（採点外）: ReLUcap-2 の P1a・P1b・P2・P4 と同じ数。平均で置く切除での P1a〜P1c の数。射影の分担 π での P2 の数。

## 6. 予測（走る前）

Fable（1023、確率）:
- 検査 1: 課題末 39〜50 で L0 の担い > L1 の担いが 8/10 seed 以上（0.6）。課題末 89〜100 では L1 の担いが 39〜50 より増える seed が 7/10 以上（0.5）。ReLUcap-2 では L1 の担いが全窓で L0 と同程度以上（0.5）。
- 検査 2: FFN0 の書き込みの分担が課題末 1〜9 → 39〜50 で増え、89〜100 で FFN1 の分担が戻る（0.5）。
- 検査 4: L1 の生存の増分と L0 の位置の増分の Spearman が負（L1 が戻るほど L0 が沈む）の seed が 7/10 以上（0.55）。
- 検査 3: 予測なし。

Issa の予測: （Issa が書く）

## 7. 検査（gate が通らなければ全状態へ進まない）

| 記号 | 内容 | 許容 |
| --- | --- | --- |
| K0 | 束の tar の sha256 が manifest と一致。展開した各 file の sha256 も manifest と一致 | 完全一致 |
| K1 | 全 700 状態で、CE_a が head.csv の L と、n_alive（L0・L1）と m_median が states.csv と一致 | CE・m_median は相対 1e−8、n_alive は完全一致 |
| K2 | 成分つきの自前の forward の logits と m(x) の logits の差（全状態） | 最大絶対差 ≤ 1e−12 × max\|logits\| |
| K3 | Σ_c c_n と、m(x) の中で最終 LN に入る t_n（forward hook で別に取る）の差（全状態） | token ごとの相対差の最大 ≤ 1e−12 |
| K4 | gate 状態（両腕 seed 0 の課題末 1・50・100）で、5 つの切除の CE を module の forward を差し替えた m(x) でも出す | 相対差 ≤ 1e−12 |
| K5 | 変異対照: (a) 課題 t+1 の checkpoint で K1 が落ちる。(b) 課題 t+1 の probe で K1 が落ちる。(c) 注意 0 の出力を 0 にした CE は CE_b と違う。(d) 恒等の置き換え（f → f）で差が 0 | (a)(b) は K1 の許容を超える。(c) 差 > 1e−6。(d) 完全に 0 |
| K6 | 記録: 元の走の layer_stats.csv（float32）と、35 状態の dead・dead005・r_med の差 | 判定に使わない。最大差を報告する |
| K7 | 検査 3 の入力: provenance の complete と課題の範囲が C1 表と一致。R′ を出し直して judge.json の Rp と一致 | 完全一致 |

gate の順: K0 → seed 0 の課題末 1 で K1・K2・K3（両腕）→ K4・K5 → 全状態。全状態でも K1〜K3 を毎回確かめる。

## 8. 言わないこと

- 切除は forward を変える介入である。担いは「その層を外したときの CE の悪化」で、寄与率ではない。carry0 は、L0 を外したときに注意 1 と L1 の入力が変わる分を含む。
- 2 つの担いを足しても全体（carry01）にはならない。
- 0 に置く切除は、FFN 出力のうち token によらない成分も消す。だから最終 LN の分母を通る効果を含む。平均で置く切除はそれを残す記録である。
- 書き込みの比は大きさの比で、何を書くか（情報）は測らない。
- 検査 4 の ρ は 31 対の記述である。窓をまたぐ時期の差を含み、因果の向きは言わない。
- 検査 3 は byakko の再生で、元の走とビット一致しない。対照は byakko の control である。
- 2 ブロック・合成 2 次マルコフ・この Adam と lr・seed 0〜9 の外へ一般化しない。

## 9. 置き場

- spec: `specs/spec_chargpt_handover_1023.md`。
- code: `src/chargpt_handover_1023.py`（展開・gate・全状態の計算）、`analysis/chargpt_handover_1023/`（集約・検査 3・退避）。
- results: `results/chargpt_handover_1023/`（rows.csv、test3_rows.csv、tables.md、summary.md、checks.json、provenance.json、backup_manifest.json）。
- 展開した checkpoint は終わったら消す。git に入れない生成物は `obsidian-research-data/chargpt_handover_1023/` へ移す。
- push しない。親（Fable）が確かめて main に入れる。
