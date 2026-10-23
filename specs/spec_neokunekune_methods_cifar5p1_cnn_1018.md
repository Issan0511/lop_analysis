# neokunekune_methods_cifar5p1_cnn_1018 — 実ラベル 5+1 CIFAR の CNN での手法比較 12 腕

状態: **登録（実装・較正・本走の前）**。起草 2026-10-18 夜、Claude（Opus 5.5）。
発端: Codex の 12 腕案（Issa 経由）。vault の案 `可塑性喪失/spec/Neokunekune_手法比較12腕_5p1CNN_spec案_1018.md` を Issa が承認した（「12 腕の spec 案を承認します。登録して、実装・較正・本走まで進めてください。」）。本書はその案に実装の細部を足した登録版で、問い・腕・格子・判定は案と同じ。
前段: `specs/spec_neokunekune_cifar5p1_cnn_1018.md`（登録 `0250b361`）、結果 `results/neokunekune_cifar5p1_cnn_1018/main/`（main `29c0c677`）。前段は「損失ゲートの効果」の検証として閉じた（D1 `INCONCLUSIVE`、D2 `INFERIOR_TO_FREEZE`）。

## 1. 目的と問い

本 spec は前段と判定を分け、**手法としての競争力**を問う。

**問い**: 実ラベル 5+1 CIFAR の CNN で、腕ごとに lr と形を較正したとき、NEO と、前段で最良だった SNAfrz1 は、既存の選択肢に対してどこに着地するか。

- 前段の結果を見たあとで設計した。前段の結果を独立に予言したことにはしない。
- 共通の lr だけで勝っても、調整済み手法への優越とは言わない。較正した格子の外は未測として扱う。
- 前段の登録判定（ゲートの効果）は再計算しない。

## 2. 箱

前段 §2 の箱をそのまま使う。CIFAR-100 fine label・チャネル標準化・30 課題（奇数 hard 5 クラス × 500 枚、偶数 easy 1 クラス × 500 枚、90 クラス再訪なし）・出力 100・RC CNN（c1 c2 f1 f2 の 4 site、全層 bias）・Adam（β1 .9、β2 .999、ε 1e−8、bias correction あり）・batch 32・780 更新／課題・課題間で全状態を継続・float32・AMP なし・TF32 なし（`cuda.matmul` と `cudnn.conv` の fp32_precision を `ieee`）・決定的アルゴリズム。課題列（`c51_classes`）・batch 列（`c51_batch`、全課題を訓練前に確定）・初期化（`init`、CPU）・probe（`neo_c51_probe`）は前段の関数をそのまま使い、seed だけで決まる。

前段との違いは、lr を腕ごとに較正すること、腕の種類が増えることの 2 点だけである。実装は前段のモジュール `src/neokunekune_cifar5p1_cnn_1018.py` を import し、改変しない。

## 3. 腕（12）と形の格子

| 腕 | 中身 | 形の格子（同点の優先順） | 中心の出所 |
|---|---|---|---|
| R | ReLU | なし | — |
| SNA | 前段の通常の適応 Snake（c 0.6、β 0.01、clip [0.005, 3]） | なし | 前段 |
| SNAfrz1 | 前段の課題 1 末固定 | なし | 前段 |
| NEO | 前段の損失ゲート（Lref の窓 703〜780・床 1e−4 も同じ） | なし | 前段 |
| SN06 | 固定 Snake α = 0.6（全 site・全 unit） | なし | 前段の c |
| SN3 | 固定 Snake α = 3（全 site・全 unit） | なし | RC CNN の SN3 |
| KKT1 | 従来の kunekune。α は SNA と同じ規則。θ = 2αz が [−2π, π] なら Snake、θ < −2π なら z、θ > π なら z + 1/α | なし | `rlcifar_mlp_battle_0918` の kkt1 |
| ELU | z > 0 で z、z ≤ 0 で a(e^z − 1) | a ∈ [3.6, 1.0] | Lillo & Cheney v4 表 E2（5+1）: 3.6 |
| LK | z > 0 で z、z ≤ 0 で a z | a ∈ [0.4, 0.1, 0.8] | 同表 E2: 0.4 |
| RSL | r z + (1 − r) z σ(c z / p)。訓練中は r ∼ U(l, u) を要素ごと・更新ごとに引き、評価は r = (l + u)/2 | (c, p, l, u) ∈ [(0.5, 0.5, 0.673, 2.673), (5, 3, 0.125, 0.333)] | 同表 E2 と原典コードの既定 |
| ReDo | ReLU + ReDo（§3.1） | (τ, 周期 F) ∈ [(0, 1560), (0.025, 1000), (0.1, 1560)] | Kumar 2024 表 4・Sokar 2023 表 1・`baselines_cifar5p1_1008` |
| L2I | ReLU + L2 Init: 全 10 テンソルの勾配に 2λ(θ − θ₀) を足す。θ₀ は初期値 | λ ∈ [1e−2, 1e−3, 1e−1] | Kumar 2024 表 4: λ 1e−2 |

- ELU の z ≤ 0 側は a × `F.elu(z, 1)` で計算する。PyTorch の elu の逆伝播は e^z を入力から作るので、float32 の床（e^z − 1 の結果に 1 を足して微分を作ると z < −16.6 で 0 になる）を踏まない。
- RSL の式と引き方は `rlcifar_mlp_battle_0918.RandSmoothLeaky`（Lillo & Cheney のコードの写し）に従う。乱数は run ごとの CUDA generator（鍵は腕の設定・seed・段階）で、CUDA graph の外で毎更新バッファに引く。run の乱数は束の組み方によらない。
- SN06・SN3 の V は記録として更新するが、α には使わない。ReLU 系（R・ReDo・L2I）・ELU・LK・RSL の V も記録だけ。

### 3.1 ReDo（CNN 版）

Sokar et al. 2023（Alg. 1）と Dopamine の NeuronRecycler に従い、`baselines_cifar5p1_1008` の 5+1 MLP 版の手順を 4 site へ移す。

- 検査の時刻: 全体の更新番号 n（課題をまたいで数える）が F の倍数になった更新の直後。その更新が課題の最後の更新なら、課題末の評価の後まで遅らせる。fresh 対照では fresh 自身の更新番号で数える。
- スコア: その課題の直近 2 batch（64 枚、更新 j−1 と j の batch）を通した各 site の活性 a = φ(z) の |a| の平均。conv は画像 × 位置で平均する。層の平均で割り（分母に 1e−9 を足す）、s_i ≤ τ を休眠とする。
- 休眠の unit について、まず 4 site の入力側を直す。入力重み（conv はそのチャネルのフィルタ全体、fc はその行）を初期化と同じ U(±1/√fan_in) から引き直し、bias を 0 にする。乱数は run ごとの CPU stream（鍵は `neo_m_redo_reinit`・設定・seed）から site 順・unit 番号の昇順に引く。
- 次に出力側を 0 にする。c1 のチャネル i は c2 の入力チャネル i、c2 のチャネル i は f1 の入力列 64i〜64i+63（pool 後の flatten はチャネル優先）、f1 の unit i は f2 の列 i、f2 の unit i は出力層の列 i。
- Adam は、直した入力重みと 0 にした出力重みの m・v を 0 にする。bias の moment と更新番号は変えない。
- 課題ごとに site ごとの再初期化の数を記録する。

## 4. 較正

- 較正 seed は 100、101、102 の 3 本。評価 seed（2000〜2009）とは分ける。同じ CIFAR-100 を使うので外部検証とは呼ばない。
- lr の格子（同点の優先順）: R と Snake 族（SNA・SNAfrz1・NEO・SN06・SN3・KKT1）は [1e−4, 3e−4, 1e−3]。ELU・LK・RSL・ReDo・L2I は [1e−3, 3e−4, 1e−4]。先頭が文献・前段の値である（R は Lillo & Cheney 表 E2 と Kumar の baseline、ELU・LK・RSL は表 E2、ReDo・L2I は Kumar 表 4）。
- 格子点は腕ごとに（形 × lr）。順序は形の優先順を外側、lr の優先順を内側にとる。計 60 点（固定形の 7 腕 × 3、ELU 6、LK 9、RSL 6、ReDo 9、L2I 9）、3 seed で 180 run。
- 選ぶ量は、較正 seed の主量 A（§6）の 3 seed 平均。最大の格子点を採る。同点なら格子の順序で先のものを採る。
- 1 seed でも非有限（CE・勾配・V・重み）になった格子点は落とし、数を記録する。腕の全格子点が落ちたら、その腕は `NO_VALID_CONFIG` とし、その腕を含む比較は `INCOMPLETE` にする。
- 選んだ lr が 1e−4 か 1e−3、または選んだ形が順序のある格子の端（ELU は 2 値なので常に端、LK の 0.1・0.8、L2I の 1e−3・1e−1）なら `EDGE` と記録する。結果を見てから格子を広げない。広げるときは追補を登録してから行う。
- 較正の run は束 R ≤ 40 にまとめる。(格子点, seed) を腕の順（表の順）・格子の順・seed の順に並べ、先頭から 40 本ずつ切る。束の形は較正の比較に関係しない。
- 較正では unit 統計（site_stats）を取らない。課題ごとの online・最初 100 更新・train / test 精度・制御の記録は取る。

## 5. 本走

- 各腕を較正で選んだ設定で、seed 2000〜2009 × 30 課題回す。120 run を R = 40 の束 3 つで回す。束 A = {R, SNA, SNAfrz1, NEO} × 10 seed（前段と同じ並び: 腕優先、seed 昇順）、束 B = {SN06, SN3, KKT1, ELU} × 10 seed、束 C = {LK, RSL, ReDo, L2I} × 10 seed。
- 束 A の 4 腕がすべて lr 1e−4 を選んだときは、束 A は前段の本走と同じ計算になる。前段の `per_task.csv` と記録の値がビットで一致することを **S-nochange-main** として確かめる。一致しなければ FAIL と記録し、判定は本走の値で行う。
- 課題 29 の fresh 対照を全腕で行う。fresh は初期値・Adam 0・更新番号 0・V = 1・未校正・未固定から、その腕の手法を掛けたまま 780 更新する（ReDo の検査は fresh 自身の更新番号で数えるので 780 更新の中には来ない。L2I の θ₀ は同じ初期値。RSL の乱数は fresh 用の鍵）。
- 本走では前段と同じ診断（unit 統計・制御・幅）を取る。

## 6. 判定（前段のゲートの効果とは分ける）

主量 A(腕, seed) は前段と同じで、後期 hard 課題 t = 21, 23, 25, 27, 29 の更新前 online 精度の平均。

| 族 | 比較 | 数 |
|---|---|---|
| 主 | D_X = A(NEO) − A(X)、X ∈ {SN06, SN3, KKT1, ELU, LK, RSL, ReDo, L2I} | 8 |
| 副 | D′_X = A(SNAfrz1) − A(X)、同じ 8 腕 | 8 |

- 区間は seed の percentile bootstrap。20,000 回、NumPy の `default_rng(1018)` から `integers(0, 10, size=(20000, 10))` で作った 1 つの添字行列を全比較で共有する。
- 族ごとに Bonferroni で両側 1 − 0.05/8、分位点 0.003125 と 0.996875（NumPy の既定の線形補間）。
- ラベル: 下端 > 0 で `BETTER_THAN_<X>`、上端 < 0 で `WORSE_THAN_<X>`、それ以外は `INCONCLUSIVE`。端が閾値と等しいときは満たさない。
- 副のラベルとして、同じ区間で非劣性（下端 > −0.005）を併記する。差がないことや同等とは書かない。
- 両腕の 10 seed がすべて完走していることが前提。欠けがあればその比較は `INCOMPLETE`。
- 記述（判定ではない）: 12 腕の主量の順位表、全 hard・早期・早期 − 後期、最初 100 更新、train / test 精度、fresh − 継続、選ばれた lr と形、EDGE、ReDo の再初期化の数、NEO の g、幅と 2αs。
- 言い方は「この箱・この較正格子で」に限る。

## 7. 実装と検査

新しいモジュール `src/neokunekune_methods_cifar5p1_cnn_1018.py` が前段の Engine と活性化を継承して腕を足す。

- lr は run ごとの値を Adam の更新に掛ける。L2I は勾配に 2λ_r(θ − θ₀) を足し、L2I の run がない束では項を作らない。
- 活性化は前段の ReLU と同じく、束の中で run ごとに torch.where で選ぶ。束に無い種類の分岐は作らない。前段の 4 腕だけの束では、前段と同じ演算列になる。
- ReDo は graph の外で eager に行う。RSL の乱数も graph の外で引く。
- 非有限になった run は、その run だけ発散と記録して訓練を続ける（他の run の計算には混ざらない）。
- 検査（各検査に、わざと壊した変異を付け、検出されることを確かめる）:
  - **S-nochange-smoke**: 前段の 4 腕を前段の smoke と同じ束（seed 9000・9001、R = 8、2 課題）で回し、前段の smoke の記録と課題末の状態（退避先の checkpoint）にビットで一致する。
  - **S-act**: 各活性化の値と微分を、独立に書いた float64 の参照と照合する（float64 の丸めから導いた許容）。診断用の φ′ が autograd の微分と一致する。
  - **S-rsl**: 訓練時の r が [l, u] に入り、要素ごと・更新ごとに変わり、run ごとに独立で、評価では中点を使う。
  - **S-redo**: 休眠を作った状態で、休眠の判定・入力側の引き直し（参照の乱数と一致）・bias 0・出力側 0・Adam の m と v の 0 化・それ以外の不変・検査の時刻と遅延を、手で書いた参照と照合する。
  - **S-l2i**: 勾配が loss + λ‖θ − θ₀‖² の autograd と一致し、L2I の無い run は変わらない。
  - **S-lr**: lr だけ違う 2 run の 1 更新の変位の比が lr の比に一致する（float32 の丸めから導いた許容）。
  - **S-indep**: 新しい種類を含む束で、1 slot を変えても他の slot がビットで変わらない。
  - **S-graph**: 全 12 種類を含む束で、CUDA graph と eager がビットで一致する。
  - **S-resume**: 課題 1 の後で止めて再開しても、連続実行とビットで一致する（RSL の generator と ReDo の stream も含む）。
  - **S-fresh**・**S-eval**: fresh が初期状態から始まり、記録した精度が独立の forward と一致する。
- smoke は seed 9000・9001、最初の 2 課題、12 腕の代表設定（各腕の格子の先頭）で行い、実装と計時の確認に使う。性能の根拠にはしない。smoke の性能で格子や規則を選ばない。

## 8. 事前予測（較正の前に固定）

- **Claude（2026-10-18 夜、較正の前、vault の案と同じ）**:
  - 主の族で NEO が `BETTER` を取るのは R 系の 2 腕（ReDo・L2I）だけ。確率 0.5。
  - NEO 対 KKT1・SN06 は `INCONCLUSIVE`。確率 0.6。
  - Snake 族（SNA・SNAfrz1・NEO・KKT1）で選ばれる lr は 1e−4 か 3e−4。確率 0.6。
  - 12 腕の最上位は Snake 族のどれか。確率 0.7。
- **Issa**: 承認のときに予測は書かれなかった。未記入のまま登録する。

## 9. 計算量・保存・来歴

- 較正 180 run、本走 120 run。どれも 30 課題 × 780 更新。fresh は本走の 120 run × 780 更新。
- 前段の実測は R = 40 の束で 17 ms/更新。ReDo と RSL の分は smoke で測る。
- 出力: `results/neokunekune_methods_cifar5p1_cnn_1018/calib/<束>/`・`calib/selected.json`・`main/<束>/`・`main/verdict.json`・`main/summary.md`・`checks/checks.json`。前段と同じく provenance に spec の sha256・登録 commit・source の hash・開始時の git 状態・環境を残す。
- 生データ（state・checkpoint・unit 配列・step 配列）は `obsidian-research-data/neokunekune_methods_cifar5p1_cnn_1018/` へ照合つきで退避し、manifest を commit する。
- 本 spec は実装の前に commit・push する。実装・検査の commit も較正の前に push する。結果を見たあとで本文を書き換えず、変更は追補で記録する。
