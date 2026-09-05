# 命題 1–5（前活性の釣り合いは「支持の上端」で決まるか — 線形の対称性・傾きの非対称・初期依存・局所性・ノイズ正則化 R）

親: [[前活性の力学_事後_0904]] §8 ／ [[負に押す力の正体_0902]] / 状態: **確定（批評後・事前登録・実装前 commit）** / 作成 2026-09-05 / 出典チャット: `活性化プロット_0904`（続き）
関連: [[ゲート硬さダイヤル_spec_0902]]（宿主）／[[謎関数ダイヤル_spec_0903]]（活性化を足す流儀・S-copy）／[[現象3_谷埋めと延長走結果_0902]]（`valley_clamp_0902`）／[[命題リスト]]／[[運用ルール]] §2・§3／[[引用禁止]]／[[並列実行のメモリ予算]]

> **run id: `edge_law_0905`**。腕表・フック・記録列・解析定数の正本は repo の **`configs/edge_law_0905.yaml`**（本 spec と同時に実装より先に commit）。宿主は `gate_dial_0902`（1 層・幅 100・オラクル用量 12.16 固定・10 seed・lr 0.01・腕プロセス並列）を `weird_act_0903` と同じ流儀で写す（`_run_arm` の写し＋recorder の列追加＋**init フック**＋**v 凍結フラグ**＋**full-batch フラグ**）。対照はすべて **committed 値**か**同一走の腕**。
> **本 spec は Issa の命題 1–4（2026-09-05 チャット）に Claude の仮説を命題 5 として足したもの。** 命題 1–4 の文言は Issa のものを逐語で写す（§1）。
> **repo 側は spec と config を実装より先に単独 commit する**（[[運用ルール]] §2）。
> **走らせる前の分析（§2）はすべて事後・未登録**。本 spec の判定に使うのは §4 だけ。
> **草案（`spec_draft.md`）からの主な変更**: 参照窓を **タスク 451–500** に統一して参照値を全部取り直した／除外規則を `zmin` から **`layer1_denom`** に変更（参照腕に `zmin` が無いため）／`sunk` の定義を $\bar z$ から **$z_{\max}$** に変更（$\bar z$ 版は棚の無い参照腕でも 0.35–0.98 になり恒真）／$C$ を閉形式定数 **11.497681** に確定（測定量ではない）／5-a を「$\alpha$ 不変」から「**lag 付き・完全沈水部分集団・数値平衡**」に組み替え（元の形は R が正しいと FAIL する）／「上端が折れ目に釘付け」を R のラベルから外して**別仮説**にした／退化 PASS 経路（進捗ゲート無し・空虚な足切り・摂動 < ノイズ）を塞いだ／腕を 21→**30** 本に組み直した（線形・奇関数・滑らか凸・曲率反転・v 凍結・full-batch を追加、ELU scale を削除）。

---

## 0. 一行

**線形網は前活性を 0 に釘付けにし（命題 1）、非線形網では「押し」$\kappa\,\partial_{\bar z}\mathbb E_W[\varphi'^2]$ と「戻し」$\partial_{\bar z}\mathbb E_W[\varphi^2]$ の支持窓平均が釣り合う位置で前活性が止まる（命題 5）。** これを (i) 鏡像活性化で符号を、(ii) 折れ目を深さ $d$ に置いた棚／傾きを反転した棚で「どこの曲率が効くか」を、(iii) ELU の $\alpha$ 掃引と **$v$ 凍結**で「$\kappa$ 依存が因果か」を、(iv) 初期分布（平均・スケール）と $\eta$ の操作で「落ち着き先は何に依るか」を、事前登録して判定する。**釣り合いの位置は $z_{\max}$ の閉形式ではなく、各腕の実測支持形状を入れて数値で解いた平衡点で予測する。**

## 1. 命題（Issa 1–4 は逐語・5 は Claude）

1. **（Issa）** ゲートがない線形なネットワークは前活性が対称。
2. **（Issa）** 初期の前活性の分布周りの曲率の符号が一致するとき前活性は曲率の方向と反対に移動する（曲率が定義できない輩がいっぱいいるのでその時は傾きが小さい方という意味）。
3. **（Issa）** 各前活性の平均の移動が落ち着く点は初期の分布（平均）に依存しない。
4. **（Issa）** 3 が正しいとするなら、異なる活性化であっても釣り合う点が同じ位置同じ傾きであった場合一致する（その前に釣り合う点が存在するとか、周りの分布が変わったら平均も変わるとかはまじでそうなんだが）。
   - 自信度は Issa の逐語で §6.1 に置く（§1 のタグは付けない）。
5. **（Claude・仮説）** 適合後、一次の力は厳密適合で勾配自体が消えるため残らない。残るのは残差ノイズのある SGD の**ノイズ誘起ドリフト**（Blanc et al. 2020 / Damian et al. 2021 / **Smith et al. 2021 の modified loss** $\eta\,\mathbb E_x[\delta(x)^2\lVert\nabla_\theta f(x)\rVert^2]$。ラベルノイズの $\sigma^2\mathbb E\lVert\nabla f\rVert^2$ はその特別な場合）で、ユニット $i$ は
   $$R_i(\bar z)=\mathbb E_W[\varphi(z)^2]+\kappa_i\,\mathbb E_W[\varphi'(z)^2],\qquad \kappa_i=v_i^2\,(\mathbb E\lVert x_c\rVert^2+1)=v_i^2\cdot C,\quad C=11.497681$$
   を下る。速度 $\propto-\partial_{\bar z}R_i=-2\mathbb E[\varphi\varphi']-2\kappa_i\,\mathbb E[\varphi'\varphi'']$。第 2 項が「曲率×傾き」（命題 2 の中身）、第 1 項が「出力を小さくする井戸」（$v^2$ が**付かない**が、**読み出し $v$ が学習されることが出所**）。
   - **明示の近似 1**: $\delta^2$ は期待値の**中**にある。実測ではドリフトは flip 直後に集中する（§2）ので、$\mathbb E_W$ は本来「flip 直後の残差で重み付けした 32 パターン平均」。本 spec は**一様重み近似**を仮定として登録する。
   - **明示の近似 2**: 真の停留条件は $\partial R_i/\partial\bar z_i=0$ ではなく $\nabla R\perp\ker J$。その補正項は $v_i$ に**奇**（$\kappa$ は偶）。事後値では有意でない（§2）ので per-unit 還元を仮定として登録する。
   - **閉形式の帰結が使えるのは完全沈水（$z_{\max}<0$）のときだけ**: ELU で $\alpha$ が消えるのは $R=\alpha^2\{\mathbb E[(e^z-1)^2]+\kappa\mathbb E[e^{2z}]\}$ と括れるからで、支持の一部が $z>0$ に出ていると正側枝（$\alpha$ 非依存）と競合して $\alpha$ は消えない。実測の完全沈水率は 35%（§2）。**したがって本 spec は「$\alpha$ 不変」を主判定にしない**（§4.5-a）。
   - 族ごとの定性予測（数値平衡の解は §4.5-b）:
     - leaky: 押しは支持が折れ目をまたぐ間だけ → 上端が折れ目付近（$z_{\max}\approx0$、$\kappa$・$a$ に依らない）
     - ELU: 完全沈水なら $(1+\kappa)\mathbb E[e^{2z}]=\mathbb E[e^{z}]$ → $z_{\max}^\ast=z_0-\ln(1+\kappa)$（傾き 1・切片 $z_0$ の理論値は $-\Lambda(W)\in[-0.8,0]$）
     - 棚 $d>0$: 折れ目より下で $|\varphi|\ge d$ なので $\mathbb E[\varphi^2]$ に $d^2$ の床ができ、深さに依らない上向き力 $\approx2ad$ が残る → **上端は $-d$ に釘付けにならず $-d$ より上に離れ、離れ幅は $d$ とともに増える**（§4.5-b の数値解）
     - SiLU/GELU: 谷の向こうで両項とも下向き → 釣り合いなし（逃走）
     - Snake: $\varphi'^2$ の極小＝零点が罠、$\varphi^2$ が 0 側へずらす
     - 線形: $R=\bar z^2+\mathrm{const}$ → 0 に釘付け
     - $v$ 凍結: $\partial f/\partial v_i=\varphi(z_i)$ が $\lVert\nabla_\theta f\rVert^2$ から抜けるので $\mathbb E[\varphi^2]$ の井戸が消える → **釣り合いが消えて逃走**（§4.5-c）

## 2. 走らせる前に手持ちのログで見えていること（**事後・未登録**・§4 の判定には使わない）

窓は特記なければ **タスク 451–500 の 50 記録平均**、除外は **`layer1_denom`（末尾窓平均）> 0.25**、CI は **seed bootstrap 2000 回**。

| 事実 | 出所 | 数値 |
|---|---|---|
| 線形（`LIN100`）は対称なだけでなく **0 に釘付け**。最初の 1000 step（0.1 タスク・最初の flip より前）で終わり、中身は $w\cdot\mu'\approx-b$ | `width5_gate_b_0901/LIN100` 20 seed | 初期 \|z̄\| 中央値 0.62 / p95 1.74 → 末尾 **0.03 / 0.13**、P(z̄>0)=0.506、歪度 −0.02、corr(w·µ′, b) = −0.88 |
| ただし `LIN100` は**本走と土俵が違う**（批評で確認）: `centered_layers: []`・`target_mu_norm null`・`target_dose null`（dose 11.49→12.49 と動く）・別 `generator_offset`・`layer1_zmax` 列も `layer1_v_unit` 列も**無い** | `configs/width5_gate_b_0901.yaml:68`・npz 列 | → 対照から外し、`LIN_1216` を新規に引く（§3.1） |
| leaky 0.1 の上端は折れ目付近で止まる | `p3_extend_0902/LR_1216` 5M | $z_{\max}$ 中央値 **+0.112** [0.060, 0.156]、$\bar z$ 中央値 **−3.598**、半幅 3.786、$\lVert w\rVert$ 中央値 4.035、死亡率 **0.0%** |
| leaky $a\le0.01$ は折れ目の下を漂う（ReLU 化） | `gate_dial_0902/LR_a0p01…` | $z_{\max}$ 中央値 −1.5〜−2.2、$\bar z\approx-5$ |
| ELU α=1 の上端則 $z_{\max}\approx z_0-\ln(1+\kappa)$。**傾きは窓と $v$ の測定時刻で大きく動く**（批評で確認） | `p3_extend_0902/E_1216` 5M・タスク 451–500・denom>0.25（n=826） | 同時刻 $v$: $B$ **1.637** [1.506, 1.796]・切片 0.597 ／ **lag（$v$ をタスク 351–400 で測る）: 1.127 [0.965, 1.334]** ／ 完全沈水部分集団（$z_{\max}<0$・n=349）同時刻: **1.029** [0.848, 1.267]・切片 −0.778 ／ **lag ＋ 完全沈水: 0.794 [0.652, 0.988]** |
| $B$ は地平線に沿って単調に減る（批評で確認・タスク 491–500 窓・(zmax−z̄)>0.5） | `E_1216` 5M / 10M / 15M | **1.455 / 1.309 / 1.209**（15M・n=822 は §2 旧行の 1.21 と一致）。§2 の旧参照値は 15M の量だった |
| 除外規則の実体（批評で確認）: 旧「支持幅>0.5 の 822 ユニット」は **`layer1_denom`>0.25** で 15M ちょうど 822 と再現。`layer1_zmin` は committed ログのどこにも無い | `E_1216` 15M | 5M・タスク 451–500 では死亡率 **17.4%**、除外群は \|v\| 中央値 0.0009・$\lVert w\rVert$ 0.253 |
| $C=\mathbb E\lVert x_c\rVert^2+1$ は**測定量ではなく定数**（批評で確認）: $3.041^2+5\times0.25+1$ | `LR_1216`・`E_1216` 全 seed・全記録 | `layer1_mu_norm²+20·layer1_sigma_rms²+1` の min=max=**11.497681** |
| $\bar z$ 基準の沈下 `z̄≤−d−1` は**棚の無い参照腕で恒真**（批評で確認） | `LR_1216` タスク 451–500 | d=0.5/1/2/3 で 0.976/0.945/0.754/0.349。半幅が 0.69→3.79 に育つのが原因 |
| $z_{\max}$ 基準なら判別力がある（批評で確認） | 同上 | `LR_1216`: 0.085/0.016/0.000/0.000、`E_1216`: 0.261/0.200/0.116/0.071 |
| $v$ は完全に創発（批評で確認） | `LR_1216`/`E_1216` | Spearman(初期 \|v\|, 末尾 \|v\|) = −0.002 / +0.002。lag を 100/200 タスク遡らせると $B$ は 1.64→1.13→（491–500 窓で 1.45→0.92→0.71） |
| ドリフトは flip 直後に集中（批評で確認） | `LR_1216`/`E_1216` 末尾 10 タスク | タスク内 10 分位の \|dz̄\| 中央値 3.2e−1（flip 直後）→ 6.3e−4（タスク終端）＝約 500 倍減衰（E も 3.7e−1→1.3e−3） |
| 非線形腕は支持幅が**育ち**、線形腕は**縮む**（批評で確認・R の符号と逆） | `LR/E_1216` 対 `LIN100` | $\lVert w\rVert$ 1.418→4.035（LR）/6.656（E）、対して LIN100 は 1.41→0.41 |
| 交絡候補「支持幅」は無罪（批評で確認） | `E_1216` | corr(ln(1+κ), 半幅) = −0.024、幅を入れた partial slope 1.456 vs 周辺 1.450、幅単独の R² 0.007 |
| per-unit 還元の $v$-奇な補正項は有意でない（批評で確認） | `E_1216` 15M | $z_{\max}\sim 1+\ln(1+Cv^2)+W+\mathrm{sign}(v)$ で sign(v) 係数 **+0.032 ± 0.054** |
| 同一腕内 seed 分割の KS 帰無分布（批評で確認） | `LR_1216` 10 seed を 5:5 | 末尾 $\bar z$ の KS $D$ 中央値 0.066・p90 0.088・最大 0.114 → 旧閾値 0.10 は帰無分布の中 |
| SiLU/GELU は釣り合いなし | `p3_extend_0902` | $z_{\max}$ −9 → −18（5M → 15M） |
| Snake の零点からのオフセットは 0 側（符号は R と整合）だが $1/\kappa$ 依存は**出ない** | `snake_lr_diag_0903`（未コミット npz） | +0.21（α=1、P>0 0.73）/ +0.09（α=3）。$\kappa$ 分割で平ら |
| 命題 4 の字義に合わない事例が 2 つある（**ただし Issa 自身が除外している場合に当たる**・§4.4-a） | committed | Snake の零点 −0.785 と GELU の谷底 −0.752（同位置・傾き 0）で片や罠・片や素通り／`Gc_b1`（$z>z_c$ で GELU と bit 一致）で深さ 11.1→7.4 |
| 実コード経路の符号対称性は実測で真（批評で確認） | 1000 step・OMP=1・CPU | $W,b,v$ がバイト厳密に符号反転・$c$ はバイト同一。`layer1_w_norm`/`denom`/`mob`/`absmob` は**不変**、`zbar`/`dzbar`/`zmean`/`v_unit`/`M`/`B` は**反転**、`p'+p` は厳密に 1.0、$z_{\max}'=-z_{\min}$ |

## 3. 設計

### 3.1 腕（新規 **30 本**・すべて 1 層・幅 100・`centered_layers: [1]`・`target_mu_norm 3.041`・`target_dose 12.16`・10 seed・`generator_offset 0`）

「活性化」列は**生の関数名**。config の `activation` マップに `setdefault(name, {"name": name})` で足す（`p3_runs_0902.build_cfg` の流儀。宿主の `validate_config`／`_s_dial`／`_geometry` は**呼ばない** — ラベル辞書が {relu, leaky_relu, elu} 固定で KeyError になる）。

| 腕 | 活性化 | dial | init フック / フラグ | step | 目的 |
|---|---|---|---|---|---|
| `LRnull_1216` | `leaky_relu` | 0.1 | なし | 5M | **S-null**（`LR_1216` と全 5001 記録 bit 一致）＋ leaky 参照（新列つき） |
| `Enull_1216` | `elu` | 1.0 | なし | 5M | **S-null-E**（`E_1216` と bit 一致）＋ ELU 参照（新列つき） |
| `LIN_1216` | `leaky_relu` | **1.0**（＝線形） | なし | 5M | **命題 1 の主語**（土俵を揃えた線形網）・命題 4 の局所性ラダーの基準線 |
| `FL_1216` | `flip_leaky` | 0.1 | なし | 5M | 1-c アンサンブル鏡像（REPORT） |
| `FLn_1216` | `flip_leaky` | 0.1 | `negate` | **5M** | **S-mirror**（bit 鏡像・全地平線） |
| `TH_1216` | `tanh_b` (β=1) | 1.0 | なし | 5M | 命題 1 の奇関数一般化（対称と釘付けの分離） |
| `SP_1216` | `softplus_b` (β=1) | 1.0 | なし | 5M | **命題 2 の主節の前件が成り立つ唯一の腕**（滑らか・$\varphi''>0$ 一定符号） |
| `SH_d0p5_1216` | `shelf_leaky_d0p5` | 0.1 | なし | 5M | 局所性ラダー（A/B 判別には使わない） |
| `SH_d1_1216` | `shelf_leaky_d1` | 0.1 | なし | 5M | 局所性ラダー・曲率反転の対 |
| `SH_d2_1216` | `shelf_leaky_d2` | 0.1 | なし | 5M | 命題 2 判別・4-c・5-b |
| `SH_d3_1216` | `shelf_leaky_d3` | 0.1 | なし | 5M | 命題 2 判別（折れ目が初期支持の外）・4-c・5-b |
| `SH_d30_1216` | `shelf_leaky_d30` | 0.1 | なし | 5M | **折れ目に絶対届かない対照**（棚族の中の「線形」基準線） |
| `ST_d1_1216` | `steep_shelf_d1`（下側傾き 2） | 1.0 | なし | 5M | **曲率の符号だけ反転した対照**（折れ目位置・上側傾きは棚と同一） |
| `ST_d2_1216` | `steep_shelf_d2` | 1.0 | なし | 5M | 同上（$d$ 依存） |
| `E_a0p5_1216` | `elu` | 0.5 | なし | 5M | 5-a・5-b（$\alpha$ 掃引） |
| `E_a2_1216` | `elu` | 2 | なし | 5M | 同上 |
| `E_a4_1216` | `elu` | 4 | なし | 5M | 同上 |
| `LRbp5_1216` | `leaky_relu` | 0.1 | `b_offset` **+5** | 5M | 命題 3（上から） |
| `Ebp4_1216` | `elu` | 1.0 | `b_offset` **+4** | 5M | 命題 3（上から） |
| `LRbm5_1216` | `leaky_relu` | 0.1 | `b_offset` **−5** | **15M** | 命題 3（下から・遅い側なので 3 倍の地平線） |
| `Ebm4_1216` | `elu` | 1.0 | `b_offset` **−4** | **15M** | 同上 |
| `LRs0p5_1216` | `leaky_relu` | 0.1 | `scale` 0.5 | 5M | 5-f（leaky は正斉次なので**厳密に関数保存**） |
| `LRs2_1216` | `leaky_relu` | 0.1 | `scale` 2 | 5M | 同上 |
| `Evf1_1216` | `elu` | 1.0 | `v_freeze` ×1 | 5M | **5-c（$\kappa$ の外生化・本 spec 唯一の因果検定）** |
| `Evf4_1216` | `elu` | 1.0 | `v_freeze` ×4 | 5M | 同上（$\kappa$ を 16 倍で固定） |
| `LRlr0p005_1216` | `leaky_relu` | 0.1 | `lr` 0.005 | **10M** | 5-d（$\eta\cdot$step を揃える） |
| `LRlr0p02_1216` | `leaky_relu` | 0.1 | `lr` 0.02 | 5M | 5-d（$\eta\cdot$step 一致点は step 2.5M の記録で読む） |
| `Elr0p005_1216` | `elu` | 1.0 | `lr` 0.005 | **10M** | 5-d（ELU は動的レンジが広い） |
| `Elr0p02_1216` | `elu` | 1.0 | `lr` 0.02 | 5M | 同上 |
| `FBLR_1216` | `leaky_relu` | 0.1 | `full_batch`（32 パターン厳密勾配） | 500k | **ノイズ帰属**（REPORT_ONLY・S-fb が落ちたら NOT_RUN） |

対照（committed・再走しない）: `LR_1216`・`E_1216`（`results/p3_extend_0902/logs`）、`E_a0p1_1216`・`E_a0p01_1216`・`LR_a0p01_1216`（`gate_dial_0902`）。**`LIN100` は土俵が違うので対照から外す**（§2 第 2 行）。

### 3.2 新規活性化（`src/nets.py`・真の導関数・乱数消費なし）

**逐語で登録する（等価な別形に書き換えない）。** `act_fn`・`act_grad`・**`act_curv`（新設）**の**三方向すべて**に分岐を置く。名前は `ACTIVATIONS` と**ガード用タプルの両方**に足す（片方だけだと `set_activation` の域ガードが効かず、分岐の書き忘れは例外にならずに黙って ELU の式に落ちる — `act_fn` の if 連鎖の最後は ELU）。

- **`flip_leaky`**（leaky の奇鏡像 $\tilde\varphi(z)=-\varphi(-z)$）
  - `act_fn`: `torch.where(pre < 0, pre, self.act_alpha * pre)`
  - `act_grad`: `torch.where(pre < 0, torch.ones_like(pre), torch.full_like(pre, self.act_alpha))`
  - **述語は `<0`**。`>0` で書くと $\varphi'(\pm0.0)$ が 1 になり、鏡像の要請 $\tilde\varphi'(u)=\varphi'(-u)=a$ と食い違う（実測で `>0` 形は S-flip がバイト不一致、`<0` 形は $\pm0.0$ 込みで一致）。
  - 既存 `mirror_leaky`（負側の傾きを $-a$ にする V 字）とは別物。
- **`shelf_leaky_d`**（折れ目を深さ $d$ に置いた leaky・`SHELF_DEPTH = {"shelf_leaky_d0":0.0, "…d0p5":0.5, "…d1":1.0, "…d2":2.0, "…d3":3.0, "…d30":30.0}`）
  - `act_fn`: `torch.where(pre > -d, pre, self.act_alpha * (pre + d) - d)`
  - `act_grad`: `torch.where(pre > -d, torch.ones_like(pre), torch.full_like(pre, self.act_alpha))`
  - $z\ge-d$ で恒等、折れ目より下で傾き $a$。折れ目ちょうどで恒等枝と厳密に連続（`a*pre + (a-1)*d` は丸めが 2 回入り $-d$ を厳密に返さない）。
  - **`shelf_leaky_d0` は leaky と bit 一致**（S-limit 専用・本走に使わない）。**既知の例外**: `pre = -0.0` のとき shelf は `+0.0`、leaky は `-0.0` を返す。S-limit ではこの 1 点を**明示の既知差**として登録する。
  - `shelf_leaky_d30` は本走で折れ目に絶対届かない（初期 \|z\| p95 1.74、末尾半幅 p90 4.7）→ 棚族の中の線形基準線。
- **`steep_shelf_d`**（折れ目位置と上側傾きを棚と揃え、**下側の傾きだけ 2**・`STEEP_DEPTH = {"steep_shelf_d1":1.0,"steep_shelf_d2":2.0}`・`STEEP_SLOPE = 2.0` はクラス定数）
  - `act_fn`: `torch.where(pre > -d, pre, self.STEEP_SLOPE * (pre + d) - d)`
  - `act_grad`: `torch.where(pre > -d, torch.ones_like(pre), torch.full_like(pre, self.STEEP_SLOPE))`
  - 傾き 2 は `WEIRD_SLOPE_ACTIVATIONS` の [0,1] ガードに触れるので**傾きは `act_alpha` に入れない**。`dial` は 1.0 固定（`UNIT_ALPHA_ACTIVATIONS` ガードで `act_alpha == 1.0` を要求）。
- **`softplus_b`**（$\beta=1$・`SOFTPLUS_BETA` クラス定数・`UNIT_ALPHA_ACTIVATIONS`）
  - `act_fn`: `torch.nn.functional.softplus(pre, beta=1.0)`、`act_grad`: `torch.sigmoid(pre)`、`act_curv`: `s*(1-s)`（`s=torch.sigmoid(pre)`）
- **`tanh_b`**（$\beta=1$・`TANH_BETA` クラス定数・`UNIT_ALPHA_ACTIVATIONS`）
  - `act_fn`: `torch.tanh(pre)`、`act_grad`: `1 - t**2`、`act_curv`: `-2*t*(1-t**2)`
- **`act_curv`（新設・$\varphi''$）**: 名前→式の**明示辞書**で引き、未登録名は `NotImplementedError`（if 連鎖の最後に落とさない）。区分線形族（leaky・flip・shelf・steep・線形）は**恒等的に 0**（折れ目は測度 0）と登録する。ELU は `where(pre>0, 0, alpha*exp(pre))`。
- **線形**は既存流儀どおり `leaky_relu` の `dial = 1.0`（`width5_gate_0901` の `LIN*` と同じ）。新規実装は要らない。

### 3.3 init フック / 実行フラグ（runner・`setup_arm_dial` 直後・恒等式検査より前・乱数消費なし・既定は恒等）

**すべて in-place で書く**（`net.Ws[0] = -net.Ws[0]` の形だと `VecMLPL.__init__` が張った別名 `net.W`/`net.b`（`src/nets.py:304-305`）が古いテンソルを指したまま残り、`lop_metrics.py`・`freeze.py` 等が黙って初期値を読む）。

- `negate`: `net.Ws[0].neg_(); net.bs[0].neg_(); net.v.neg_()`（$c$ は不変）。命題 1 の定理: これと `flip_leaky` の組で軌道は `LR_1216` の厳密な符号反転。
- `b_offset` $c_0$: `net.bs[0].add_(c0)`。
- `scale` $s$: `net.Ws[0].mul_(s); net.bs[0].mul_(s); net.v.div_(s)`。**関数保存は正斉次な活性化（leaky・relu・線形）に限る。** $s\in\{0.5,2\}$ は 2 冪なので float32 で誤差なし。ELU では $\varphi(sz)\ne s\varphi(z)$ で関数が変わるため **ELU の scale 腕は置かない**。
- `lr` $\eta$: `st["lr"] = torch.full_like(st["lr"], eta)` **かつ** `for r in st["runs"]: r["lr"] = eta`。腕ブロックのキー `lr:` から読み、logs の payload に **`lr_used`** 列を書く（宿主の `arm_runs` は `common.lr_main` を読むので、書かないと走った $\eta$ が腕名の文字列からしか復元できない）。
- `v_freeze` $m$: `net.v.mul_(m)` ＋ **学習ループで $v$ の更新を止める**（`st["freeze_v"]=True` で `sgd_step_layers` の $v$ 更新をスキップ）。logs に `freeze_v` 列。
- `full_batch`: 学習ループを `env.full_support()` の 32 パターン厳密勾配に切り替える（`VecMLPL.forward_batch`/`grads_batch` を使い、`grads_centered_elu` のバッチ版を新規に書く）。logs に `batch_mode` 列。
- logs の payload に `init_hook`・`init_hook_arg`・`lr_used`・`freeze_v`・`batch_mode` を必ず書く（S-null は共通列だけを見る実装なので新列を足しても壊れない）。

### 3.4 記録列

宿主の per-unit 5 列（`mob/absmob/zmax/zmean/v_unit`）に加えて:

| 列 | 形 | 頻度 | 用途 |
|---|---|---|---|
| `layer1_zmin` | (rec, h) | 1000 step | $t_{\rm reach}$（支持が折れ目に届いたか）。$\bar z$・$z_{\max}$ から復元できる冗長列（S-support で検算） |
| `layer1_w_free` | (rec, h, 5) | **タスク終端のみ**（10,000 step） | 支持の厳密形状。$\mathbb E[e^{tz}]=e^{t\bar z}\prod_j\cosh(tw_j/2)$ で per-unit の厳密予測が書ける |
| `layer1_m_phi2` / `m_dphi2` / `m_phidphi` / `m_dphiddphi` | (rec, h) 各 1 列 | タスク終端 ＋ **末尾 20 タスクは 1000 step** | $\mathbb E_W[\varphi^2],\mathbb E_W[\varphi'^2],\mathbb E_W[\varphi\varphi'],\mathbb E_W[\varphi'\varphi'']$。停留残差 $G_i=2\mathbb E[\varphi\varphi']+2\kappa_i\mathbb E[\varphi'\varphi'']$ を**上端・窓・閉形式の代理なしに**直接測る（§4.5-g） |

recorder は既に 32 点厳密支持の $z$ を float64 で作っているので追加コストは掛け算だけ。モーメント列は float32。

### 3.5 窓・単位・$C$

- **主窓 $W_{\rm tail}$ = タスク 451–500**（`_window_indices(step, 10000, [451, 500])`・各 seed 50 記録）。ユニット別の値は窓内 50 記録の平均。15M 腕は $W_{\rm tail}$ = タスク **1451–1500** を主とし、451–500 も併記する。
  - 草案の 10 記録窓から広げた理由: flip 状態は 1 タスクに 1 ビットの Markov 連鎖で相関時間 7.5 タスク、10 記録では独立標本が実質 1〜2 個。実測の窓内 $z_{\max}$ の時間 sd は中央値 0.851（LR）/ 1.13（E）。
- **lag 窓 $W_{\rm lag}$ = タスク 351–400**（5M）／1351–1400（15M）。$\kappa$ の説明変数はここで測る（§4.5-a）。
- **初期値** = step 0 の記録。
- **除外規則（全判定に共通）**: `ALIVE` $=$ `layer1_denom`（$W_{\rm tail}$ 平均）$>0.25$。**理由: `layer1_zmin` は committed 参照ログのどこにも無く、`denom` なら参照腕でも同じ規則で再計算できる**（15M で n=822 と §2 の旧値が厳密に一致）。副次規則として半幅 $(z_{\max}-\bar z)>0.25$ を REPORT し、$n$ が 3% 以上違ったら記録する。
- $\kappa=C v^2$、$C=\mathbb E\lVert x_c\rVert^2+1=3.041^2+5\times0.25+1=\mathbf{11.497681}$（**閉形式定数**・測定量ではない）。
- 「支持幅」$=z_{\max}-z_{\min}=2\times$「半幅」、「半幅」$=z_{\max}-\bar z=\tfrac12\sum_{j\in\rm free}|w_j|$。

### 3.6 判定に共通のゲート・CI（**すべての登録判定に前置**）

| 記号 | 内容 | 破ったときのラベル |
|---|---|---|
| **CI** | seed bootstrap: 腕内 10 seed を復元抽出（15M/10M 腕も同じ）・**2000 回**・percentile 95%・`rng = np.random.default_rng(20260905)`。すべての登録統計量に付ける。PASS/FAIL は点推定でなく **CI と閾値の関係**で定義する（同等性なら CI が許容帯に内包、差なら CI が 0 を除く） | — |
| **G1 進捗ゲート** | 腕の $W_{\rm tail}$ の ALIVE 中央値 $\lVert w\rVert$ が同族参照（leaky 族: `LRnull` 4.035／ELU 族: `Enull` 6.656／それ以外: 自分の初期値 1.418 の 1.5 倍以上）の 1/2〜2 倍、**かつ** \|median $z_{\max}$($W_{\rm tail}$) − median $z_{\max}$(初期 0.753)\| $\ge0.3$ | `FROZEN` → その腕の位置判定は全部 `NOT_DETERMINED` |
| **G2 定着ゲート** | 判定統計量をタスク 301–350 / 376–425 / 451–500 の 3 窓で出し、単調ドリフトの大きさが $W_{\rm tail}$ の CI 幅を超える | `NOT_SETTLED` → PASS/FAIL を出さない |
| **G3 除外一致** | 判定を `ALIVE` と `ALL`（除外なし）の両方で計算し、**ラベルが一致すること**を PASS の条件にする。死亡率（`denom≤0.25` の割合）は `verdict.csv` の必須列 | 不一致 → `NOT_DETERMINED` |
| **G4 比較可能性** | 腕間比較で死亡率が 10 ポイント以上違う | その比較は `NOT_COMPARABLE` |
| **G5 ペア** | 活性化もフックも乱数を消費しない（S-stream）ので、`(seed, unit index)` は全腕で対応する。腕間比較は**対応するユニットの差** $\Delta_i$ の中央値＋seed bootstrap を主とし、プール分布の差を副次にする | — |
| **G6 発散** | NaN が出た seed は落として数を `verdict.csv` に必須記載。落とした seed が **2/10 を超えたら腕は `NOT_RUN`**。NaN でない逸走（$W_{\rm tail}$ の \|median $\bar z$\| > 50）は除外せず **FAIL 側の証拠**として扱い、`ALPHA_LIMITED`／`ARM_RUNAWAY` を別に立てる | — |

**多重性**: 命題ごとに**確証的な統計量を 1 つだけ**指名する（下表）。残りは副次または `REPORT_ONLY` と明記し、`verdict.csv` の `role` 列に `confirmatory` / `secondary` / `report` を入れる。

| 命題 | 確証的統計量（1 つ） |
|---|---|
| 1 | S-mirror の bit 一致（列別パリティ） |
| 2 | $\Delta_3=\mathrm{median}_i[z_{\max,i}(\mathrm{SH\_d3})-z_{\max,i}(\mathrm{SH\_d30})]$ |
| 3 | 保持率 $\rho$（$z_{\max}$・下からの腕） |
| 4 | 局所性半径 $d^\ast$（`SH_d` が `LIN_1216` に一致する最小 $d$） |
| 5 | 停留残差 $G_i$ の直接検定（§4.5-g） |

## 4. 判定（事前登録）

### 4.1 命題 1（対称性）

- **1-a（定理・走なし・REPORT）**: ユニット別 $(w_i,b_i,v_i)\to-(w_i,b_i,v_i)$ は関数・損失・SGD 写像を不変にし init は対称 → 奇関数 $\varphi$（線形を含む）で $z$ の集団分布は任意の時刻・任意の入力分布で厳密に対称。条件は「ゲートがない」ではなく**奇関数**（Snake は奇でない）。**この一般化は Claude の補題であり Issa の文言ではない。**
  - 実装上の前提（壊れたときに最初に疑う場所）: (i) オラクル用量 `_refresh_fixed_offset` は `st["env"].flip_state` と target だけを読み net に触らない（`src/dose_const_5m.py:185-192`）、(ii) `full_support_ro` は env に書かず RNG も消費しない、(iii) 発散ガード `_numeric_divergence_event` は `torch.isfinite` だけで符号盲、(iv) 縮約順は `OMP_NUM_THREADS=1` で固定。
- **1-b S-mirror（bit・確証的）**: `FLn_1216`（5M）対 `LR_1216`（先頭 5M・5001 記録）を**列別のパリティ**で比較する（比較は `arr.tobytes()` のバイト比較。`torch.equal` は符号盲なので単独では使わない）。
  - **符号反転**: `layer1_zbar`・`layer1_dzbar`・`layer1_zmean`・`layer1_v_unit`・`layer1_M`・`layer1_B`
  - **不変**: `layer1_w_norm`・`layer1_denom`・`layer1_mob`・`layer1_absmob`
  - **等式**: `p_hat'(FLn) + p_hat(LR) == 1.0` を**厳密等号**で（$\le1$ ではない。$p=p'=0$ の死ユニットが自動的に通ってしまう）。例外（$z$ が厳密に 0 の (unit, record) 数）を数え、0 であることを要求。
  - **対象外**（除外理由つきで登録）: `layer1_zmax` は参照に `zmin` が無いので張れない（代わりに `FLn` 対 `LRnull_1216`（新列つき）で $z_{\max}'=-z_{\min}$ を検査する）。`eff_rank` 系（LAPACK svdvals）と `quantile` 系は演算順が符号で変わりうるので対象外。
  - PASS = 10/10 seed で全列。**ラベル**: `MIRROR_EXACT` / `MIRROR_BROKEN` / `MIRROR_NOT_RUN`（排他・網羅）。
- **1-c アンサンブル鏡像（REPORT_ONLY）**: `FL_1216` $W_{\rm tail}$ の $\bar z$ 対 `LR_1216` の $-\bar z$。統計量は **seed 水準**にする: seed ごとに中央値を取り、`FL` の 10 値と $-$`LR` の 10 値の対応差の中央値＋seed bootstrap CI、および seed ごとの KS $D_s$ の中央値。閾値は §2 の帰無分布（同一腕内 5:5 分割で $D$ 中央値 0.066・p90 0.088・最大 0.114）ではなく、**S-KSnull で作る帰無分布の q95** を使う。
  - **ラベル**（1-b とは別に出す）: `ENSEMBLE_SYMMETRIC`（対応差の CI が $\pm0.5$ に内包 かつ $D$ 中央値 < q95）／`ENSEMBLE_ASYMMETRIC`（CI が 0 を除き \|中央値\| > 0.5）／`ENSEMBLE_NOT_RESOLVED`（それ以外）。**1-b が通って 1-c が落ちても「鏡像が壊れた」ではない**（10 seed で分布を分解できないだけ）。
- **1-d 線形網（`LIN_1216`・確証的でない主要判定）**: $W_{\rm tail}$ の ALIVE $\bar z$ について
  - 対称: \|median $\bar z$\| の CI が $[-0.1,+0.1]$ に内包 かつ $P(\bar z>0)$ の CI が $[0.45,0.55]$ に内包 かつ \|歪度\| < 0.2
  - 釘付け: median \|$\bar z$\| の CI 上端 < 0.3（`LIN100` 事後値 0.03 に対して余裕を取る）
  - **ラベル**: `LINEAR_PINNED`（対称かつ釘付け）／`LINEAR_SYMMETRIC_NOT_PINNED`（対称だが釘付けでない）／`LINEAR_ASYMMETRIC`（対称でない）／`NOT_DETERMINED`（G1–G3 で落ちた）。
- **1-e 奇な非線形（`TH_1216`）**: 定理は**対称性しか言わない**ので、対称と釘付けを分けて判定する。対称の条件は 1-d と同じ。釘付け: median \|$\bar z$\|($W_{\rm tail}$) の CI 上端 < 1.0、逃走: median \|$\bar z$\| がタスク 100→300→500 で単調増加し CI が重ならない。
  - **ラベル**: `ODD_SYMMETRIC_PINNED` / `ODD_SYMMETRIC_RUNAWAY` / `ODD_SYMMETRIC_INTERMEDIATE` / `ODD_ASYMMETRIC` / `NOT_DETERMINED`。
- **命題 1 の総合**: 上の 4 ラベルを併記する（単一ラベルに畳まない）。

### 4.2 命題 2（曲率／傾きの非対称の向き）

冒頭に置く注記: **読み A も読み B も Issa の言う「曲率／傾きの非対称」であり、争点は「どこで評価するか（初期分布の上か、後から届く先か）」＝局所性である。**

- **2-a 符号（sanity・REPORT）**: `FL_1216` の $W_{\rm tail}$ の median $\bar z$ の CI が $>+2$（`LR_1216` は $<-2$）。定理の帰結なので FAIL なら実装の異常。
- **2-b 判別（確証的）**: `SH_d3_1216`（折れ目 $-3$ が初期分布の外。初期支持が $-3$ に届くユニットは 1000 中 5 ＝ 0.5%）と `SH_d30_1216`（折れ目に絶対届かない・同一 runner・同一 seed・同一 5M）の**対応ユニット差**
  $$\Delta_3=\mathrm{median}_i\big[z_{\max,i}(\mathrm{SH\_d3})-z_{\max,i}(\mathrm{SH\_d30})\big]\quad(\text{ALIVE}\cap\text{両腕で ALIVE})$$
  を seed bootstrap CI つきで出す。**条件付けをしない**（`sunk_d` で選抜してから $z_{\max}$ を見ると選抜と検定が同じ軸に乗る）。
  - `CURVATURE_NONLOCAL`（読み B）: CI 上端 $<-0.3$
  - `CURVATURE_AT_INIT`（読み A）: CI が $[-0.3,+0.3]$ に内包（TOST 型の同等性）
  - `MIXED`: CI がどちらでもない
  - `NOT_DETERMINED`: G1–G4 のいずれかで落ちた
  - 同じ量を $d=2$（`SH_d2` − `SH_d30`）でも出し、$d$ に対する並びを REPORT。**$d=0.5$ と $d=1$ は 2 仮説の距離（0.5〜1.0）が許容幅と同程度なので判別に使わない。**
- **2-c 滑らかな腕（`SP_1216`・命題 2 の主節の前件が成り立つ唯一の腕・主要判定）**: softplus は全域で $\varphi''>0$。命題 2 は「曲率の方向と反対＝**下へ**」を予測。統計量 $=\mathrm{median}_i[\bar z_i(W_{\rm tail})-\bar z_i(\text{初期})]$、seed bootstrap CI。
  - `SMOOTH_DOWN`（CI 上端 $<-0.3$）／`SMOOTH_UP`（CI 下端 $>+0.3$）／`SMOOTH_STATIONARY`（CI が $\pm0.3$ に内包）／`NOT_DETERMINED`。
- **2-d 曲率反転の対照（`ST_d1`/`ST_d2`・主要判定）**: 折れ目の位置も上側の傾きも棚と同一で、**曲率の符号だけ逆・傾きの小さい側が上**。命題 2 の 2 つの読みはどちらも「**上へ**」を予測する。統計量 $\Delta_{\rm st}(d)=\mathrm{median}_i[z_{\max,i}(\mathrm{ST\_d}d)-z_{\max,i}(\mathrm{SH\_d}d)]$、$d\in\{1,2\}$。
  - `ASYMMETRY_SIGN_OK`（両 $d$ で CI 下端 $>+0.3$）／`ASYMMETRY_SIGN_BROKEN`（どちらかで CI 上端 $<-0.3$）／`ASYMMETRY_SIGN_NULL`（両方 $\pm0.3$ に内包）／`NOT_DETERMINED`。
- **2-e 届いたか（REPORT）**: 各ユニットの $t_{\rm reach}(d)=\min\{$タスク$: z_{\min}<-d\}$（届かなければ NaN）。reach 率、reach したユニットの中の $z_{\max}\le-d$ 率、not-reached 群の \|$\bar z$\| 中央値（線形挙動なら小さい）を出す。`SH_d3` で reach 率が 0.05 未満なら 2-b を `NOT_DETERMINED` に落とす（「棚に届かなかった」と「届いたが沈まなかった」を混ぜない）。
- **命題 2 の総合ラベル**: 2-b のラベルを主とし、2-c・2-d を修飾子として併記。

### 4.3 命題 3（初期依存）

参照: `LRnull_1216`（$z_{\max}$ median 0.112・$\bar z$ median −3.598）、`Enull_1216`（同 0.101 / −6.348）。摂動は $\pm5$（leaky）／$\pm4$（ELU）。理由: `LR_1216` の $z_{\max}$ 中央値は初期 0.753 → タスク 5 で 0.098 と **5 タスクで緩和**し、末尾の seed 間 SE は 0.036。旧 $\pm2$ ＋ 許容幅 $\pm0.5$ では上からの腕が落ちようがなく、下からの腕も分解能が無かった。

- **3-a 保持率（確証的）**: 位置の絶対差でなく保持率
  $$\rho=\frac{\mathrm{median}_{\rm arm}(z_{\max},W_{\rm tail})-\mathrm{median}_{\rm ref}(z_{\max},W_{\rm tail})}{\mathrm{median}_{\rm arm}(z_{\max},\text{初期})-\mathrm{median}_{\rm ref}(z_{\max},\text{初期})}$$
  を ALIVE・seed 対応（同じ seed 番号で arm と ref を組む）で計算し、seed bootstrap CI を付ける。
  - `MEAN_INDEPENDENT`: CI 上端 $<0.15$ **かつ** ペア条件（$\mathrm{median}_i\Delta_i$ の \|CI\| $<0.3$ かつ Spearman $\rho_s(z_{\max}^{\rm arm},z_{\max}^{\rm ref})\ge0.5$）
  - `MEAN_DEPENDENT`: CI 下端 $>0.40$
  - それ以外 → 3-b へ
- **3-b 緩和フィット（3-a が中間だったとき）**: seed ごとに median $z_{\max}$ の軌跡へ $z(t)=z_\infty+(z_0-z_\infty)e^{-t/\tau}$ を当てる（タスク 100–500、15M 腕は 100–1500）。
  - `MEAN_INDEPENDENT_SLOW`: $z_\infty$ の seed bootstrap CI が参照の $W_{\rm tail}$ 中央値を含む
  - `MEAN_DEPENDENT`: CI が含まない
  - `NOT_DETERMINED`: CI 幅が摂動幅（5 または 4）より広い、または G1–G3 で落ちた
  - $\tau$ は REPORT。
- **下からの腕は 15M**（`LRbm5`・`Ebm4`）。理由: 折れ目より下では $\varphi'=a=0.1$ で戻りは 10 倍遅く、上からの緩和 ≈5 タスクに対して 50–150 タスク（memory の $\tau_{\rm ens}$ 73–149 タスクと整合）。$-5$ から 500 タスクでは 3〜7$\tau$ しかない。
- **ラベル（命題 3・排他かつ網羅）**: `MEAN_INDEPENDENT` / `MEAN_INDEPENDENT_SLOW` / `MEAN_DEPENDENT` / `NOT_DETERMINED`。leaky と ELU で別々に出す。
- **3-c 下からの戻り道（REPORT・機構の判別）**: `LRbm5`・`Ebm4`・棚の沈下ユニットについて、median $z_{\max}$ がタスク 100→300→500(→1500) で単調上昇するか、$P(z_{\max}<-d-1\mid \text{sunk})$ が小さいか。R なら上昇・小、Itô ラチェット（片側可動度の吸収境界）なら停滞・大。
- **スケール（`LRs0p5`/`LRs2`）は命題 3 の判定に入れない**（Issa の文言は「初期の分布（平均）」）。§4.5-f へ移す。

### 4.4 命題 4（局所性）

- **4-a 字義（REPORT_ONLY・ラベル `LITERAL_OUT_OF_SCOPE`）**: §2 の 2 事例（Snake の零点 vs GELU の谷底／`Gc_b1`）は Issa 自身が括弧で除外した場合に当たる — GELU の谷底は「釣り合う点」ではない（前件不成立）、`Gc_b1` は「周りの分布が変わったら平均も変わる」ケース。**`REFUTED_BY_COMMITTED` とは書かない。** 「点だけを合わせる読みでは反例がある」という REPORT に留める。
- **4-b 局所性半径（確証的）**: `shelf_leaky_d` は「$z\ge-d$ で線形と厳密に一致し、その下だけ違う」活性化なので、$d$ を動かした棚と `LIN_1216` を比べれば「一致するために合わせる必要のある窓の広さ」が測れる。
  $$d^\ast=\min\{d\in\{0.5,1,2,3,30\}:\ |{\rm med}\,z_{\max}({\rm SH\_}d)-{\rm med}\,z_{\max}({\rm LIN})|\le0.5\ \wedge\ |{\rm med}\,\bar z({\rm SH\_}d)-{\rm med}\,\bar z({\rm LIN})|\le0.5\}$$
  （ALIVE・$W_{\rm tail}$・両方の中央値差に seed bootstrap CI を付け、CI の点推定で判定して CI 幅を併記）。
  - `LOCALITY_MONOTONE`: 中央値差が $d$ について単調非増加で $d^\ast$ が存在する（$d^\ast$ を報告）
  - `NONLOCAL`: $d=30$ でも一致しない、または単調でない
  - `NOT_DETERMINED`: G1–G3 で落ちた棚が 2 本以上
- **4-c 上端は折れ目に追随するか（主要判定・R とは独立の仮説）**: **無条件**の $z_{\max}$ で判定する（旧 4-b の「`sunk_d` で選抜してからその $z_{\max}$ を見る」は循環、かつ「全 $d$ で $n<20$ なら空虚に真」だった）。$d\in\{2,3\}$（判別に使う）と $d\in\{0.5,1\}$（REPORT）。
  - `EDGE_AT_KINK`: 両 $d$ で \|median $z_{\max}$ $+d$\| の CI が $\pm0.3$ に内包
  - `EDGE_DETACHED_UP`: 両 $d$ で median $z_{\max}$ の CI 下端 $>-d+0.3$（棚の上に離れる）
  - `EDGE_DETACHED_DOWN`: 両 $d$ で CI 上端 $<-d-0.3$
  - `EDGE_MIXED`: $d$ で結論が違う
  - `EDGE_NOT_DETERMINED`: 判定可能な $d$ が 2 本のうち 1 本以下
  - 併せて 3 状態（$z_{\max}$ と半幅の**同時**の言明）を REPORT: **状態 A**（$z_{\max}\approx0$・半幅 小＝線形）／**状態 B**（$z_{\max}\approx-d$・半幅 大）／**状態 C**（$z_{\max}\approx0$・半幅 大＝参照 leaky が実際にいる場所）。
  - **この判定は命題 5 のラベルには入れない**（可動度ラチェット＝折れ目下で有効 lr が $a^2$ 倍になる吸収境界でも同じ予測が出るので、R の証拠にならない）。

### 4.5 命題 5（R 仮説）

- **5-a $\kappa$ 依存（主要判定・$\alpha$ 掃引）**
  - 統計量 $B$: ALIVE・$W_{\rm tail}$ の $z_{\max}$ を $\ln(1+Cv^2)$（$C=11.497681$ **固定**）へ OLS した傾きの符号反転値。**$v$ は $W_{\rm lag}$（タスク 351–400）で測る**（同時刻 $v$ は応答と同時決定で完全に内生。初期 \|v\| は末尾 \|v\| を全く予測せず Spearman ≈ 0.00、lag を 100/200 タスク遡らせると $B$ は 1.64→1.13→0.70）。**母集団は完全沈水部分集団**（$W_{\rm tail}$ で $z_{\max}<0$）— 閉形式が成り立つのはここだけ（§1）。
  - **主判定は $\alpha$ 間のコントラスト**: $\Delta B(\alpha)=B(\alpha)-B(1)$（$\alpha=1$ は `Enull_1216`）を、seed を**共通に**復元抽出して 2000 回。
    - `ALPHA_CONTRAST_CONSISTENT`: $\alpha\in\{0.5,2,4\}$ すべてで $\Delta B$ の CI が $\pm0.3$ に内包（＝閉形式どおり $\alpha$ が消える）
    - `ALPHA_CONTRAST_MONOTONE`: $\Delta B$ が $\alpha$ に単調増加で、$\alpha=4$ の CI 下端 $>+0.3$（＝§4.5-b の数値解が予測する向き）
    - `ALPHA_CONTRAST_INCONSISTENT`: どちらでもない
    - `NOT_DETERMINED`: G1–G4／G6 で 2 腕以上落ちた
  - 副次（REPORT）: $B$ の 4 通り（同時刻／lag × 全体／完全沈水）と lag プロファイル（lag 0/100/200 タスク）。参照値は §2。R が正しければ lag 依存は小さいはず、という反証可能な言明。
  - 副次（REPORT）: 切片。理論は完全沈水・大 $W$ 極限で $z_{\max}^\ast=-\ln(1+\kappa)$、有限 $W$ 補正 $\Lambda(W)\in(0,0.8]$ なので median$(z_{\max}+\ln(1+\kappa))$ が $[-1.0,+1.0]$ に入るか。
  - 副次（REPORT）: 信頼性比 $\rho=1-\sigma^2_{\rm within}/\sigma^2_{\rm between}$（窓内 50 記録から）で減衰補正した $B$（EIV 補正）。
  - **自由 $C$ フィットは `REPORT_ONLY`**（旧 5-e の PASS/FAIL は検出力ゼロだった: $C$ を 5.5/11/22 と変えても $R^2$ は 0.285/0.282/0.273 しか動かず、自由 $C$ の seed bootstrap CI は 15M で [2.05, 24.66]）。profile RSS 曲線と $C$ の CI を出すだけにする。
- **5-b 数値平衡（主要判定・閉形式の代わり）**: 各腕・各ユニットについて、**その腕の実測支持形状**（`layer1_w_free` から作る 32 点厳密支持）と実測 $\kappa_i$（lag 窓の $v$）と実測 $\alpha$／$a$／$d$ を入れて $\partial R_i/\partial\bar z=0$ を数値で解き、予測 $z_{\max,i}^\ast$ を作る。統計量 $=\mathrm{median}_i[z_{\max,i}(W_{\rm tail})-z^\ast_{\max,i}]$、seed bootstrap CI。
  - `EQUILIBRIUM_PREDICTED`: ELU 4 腕（$\alpha$=0.5,1,2,4）と棚 4 腕のすべてで CI が $\pm1.0$ に内包、かつ腕を跨いだ**並び**（$\alpha$ 昇順／$d$ 昇順の予測順序）が観測順序と一致（Kendall $\tau=1$）
  - `EQUILIBRIUM_ORDER_ONLY`: 並びは一致するが $\pm1.0$ を外す腕が 1 本以上
  - `EQUILIBRIUM_OFF`: 並びが一致しない
  - `NOT_DETERMINED`: 判定できる腕が 8 本中 5 本未満
  - この解法は事後に既存 3 腕で検算済み: `E_a0p01`/`E_a0p1`/`E_1216` の実測 $z_{\max}$ −1.88/−2.12/+0.10 に対し予測 −2.50/−2.64/+0.69（誤差 ±0.6）。**§2 の「$\alpha\le0.1$ は釣り合いに届いていない」という事後解釈は誤りの可能性が高い**（この 2 腕は $\kappa$ 中央値 21〜25 で、平衡点そのものが −2.6 にある）。
- **5-c $v$ 凍結（確証的な因果検定）**: `Evf1_1216`（$v$ を初期値で凍結）・`Evf4_1216`（$v$ を 4 倍にして凍結・$\kappa$ は 16 倍）。$v$ が凍ると $\partial f/\partial v_i=\varphi(z_i)$ が $\lVert\nabla_\theta f\rVert^2$ から抜け、R の井戸 $\mathbb E[\varphi^2]$ が消える。
  - `WELL_FROM_READOUT`: `Evf1` の median $z_{\max}$($W_{\rm tail}$) が `Enull` より低く CI 下端の差 $<-1.0$、**かつ** タスク 100→300→500 で median $z_{\max}$ が単調減少して CI が重ならない（逃走）
  - `WELL_INDEPENDENT_OF_READOUT`: `Evf1` − `Enull` の CI が $\pm0.5$ に内包
  - `WELL_PARTIAL`: どちらでもない
  - `NOT_DETERMINED`: G1／G6 で落ちた、または `unfit`（末尾窓中央値）が `Enull` の 3 倍を超える（$v$ 凍結で適合できていない）
  - 副次（REPORT）: `Evf4` − `Evf1` の差。$\kappa$ が厳密に外生なので、R が残っていれば $-\ln((1+16\kappa)/(1+\kappa))$ の中央値ぶんのシフトが出るはず。
- **5-d $\eta$（主要判定は揺らぎのスケーリング）**: 位置の $\eta$ 不変は決定論的 ODE の固定点とも両立するので識別力が弱い。**核心は揺らぎの $\eta$ スケーリング**。
  - 統計量: $W_{\rm tail}$ 内の $z_{\max}$ の**時間 sd**（ユニット別・窓内 50 記録）の中央値を $\eta\in\{0.005,0.01,0.02\}$ で取り、両対数の傾きを seed bootstrap。ELU（`Elr0p005`/`Enull`/`Elr0p02`）を主、leaky（`LRlr0p005`/`LRnull`/`LRlr0p02`）を副次。
  - `FLUCT_SQRT_ETA`: 傾きの CI が $[0.3,0.7]$ に内包
  - `FLUCT_LINEAR_ETA`: CI が $[0.8,1.2]$ に内包
  - `FLUCT_OTHER`: どちらでもない
  - `NOT_DETERMINED`: G1／G6 で 2 腕以上落ちた
  - 位置の $\eta$ 不変は **REPORT**: $\eta\cdot$step を揃えて（0.005 腕は 10M、0.02 腕は step 2.5M の記録）median $z_{\max}$ の差の CI が $\pm0.3$ に入るか。
  - **時定数の読み替えは撤回する**: この系の駆動は flip 直後の残差パルスで、パルスの緩和時間が $\propto1/\eta$、$\sum_t\delta_t^2\propto1/\eta$ なので二次のドリフト量は $\eta^1$ になる。$\tau\propto\eta^{-1}$ は二次機構と両立し、一次機構の証拠にならない（§8）。REPORT として、タスク境界からの経過 step 別の \|dz̄\|（§2 で 500 倍の減衰）とその $\eta$ 依存を出す。
- **5-e $C$（S-check へ格下げ・判定から外す）**: $C=11.497681$ は用量固定介入のもとで厳密に定数。S-C は「実装が返す $C$ が閉形式と 1e−12 一致」＋「logs の `dose_relative_error` が 0」に置き換える。
- **5-f スケール（`LRs0p5`/`LRs2`・主要判定・命題 3 の文言外）**: leaky は正斉次なので**厳密に関数保存**で、balance 不変量 $\lVert w\rVert^2+b^2-v^2$ が入れた不均衡は勾配流で保存される。
  - `P5_SCALE_LAW_INVARIANT`: 両腕で median $z_{\max}$ − `LRnull` の CI が $\pm0.5$ に内包（上端則は不変）**かつ** 少なくとも一方で median $\bar z$ − `LRnull` の CI が 0.5 を超えてずれる
  - `P5_SCALE_FULLY_INVARIANT`: $z_{\max}$ も $\bar z$ も $\pm0.5$ に内包
  - `P5_SCALE_LAW_BROKEN`: $z_{\max}$ が $\pm0.5$ を外す
  - `NOT_DETERMINED`: G1（進捗ゲート）で落ちた。**$b$ の初期値は 0 なので $s$ は $z$ を初期に厳密に $s$ 倍し、`s=0.5` 腕は 1 step も学習せずに初期 median $z_{\max}$ 0.377 で $\pm0.5$ の中にいる。G1 が無いと凍結ネットが PASS する。**
  - 定量予測（REPORT）: 正斉次で不均衡が保存されるなら半幅は $s$ 倍のまま残り $\bar z\approx-3.6s^{\pm1}$（$s$=2 で ≈ −7.4、0.5 で ≈ −1.85）。
- **5-g 停留残差の直接検定（確証的）**: 新列（§3.4）から仮定なしの登録量
  $$G_i=2\,\mathbb E_W[\varphi\varphi']+2\kappa_i\,\mathbb E_W[\varphi'\varphi'']$$
  を作る（$\kappa_i$ は lag 窓の $v$）。上端・窓・閉形式の代理を一切使わない。
  - (i) **$W_{\rm tail}$ で $G_i\approx0$**: median \|$G_i$\| の CI 上端が、同じ腕の $W_{\rm lag}$ における median \|$G_i$\| の 1/3 未満（＝末尾で停留に近づいている）
  - (ii) **$dz̄$ と $-G_i$ の対応**: 末尾 20 タスクの 1000-step 記録で、per-unit の $\mathrm{d}\bar z_i$ と $-\eta\,G_i$ の Spearman 順位相関の CI 下端 $>0.3$、かつ両者の中央値の比の CI が $[0.3,3]$（桁）に内包
  - `STATIONARITY_DIRECT_PASS`: (i)(ii) とも成立し、それが **ELU 族・棚族・leaky・softplus・tanh の 5 族すべて**で成立
  - `STATIONARITY_DIRECT_PARTIAL`: 3 族以上で成立
  - `STATIONARITY_DIRECT_FAIL`: 2 族以下
  - `NOT_DETERMINED`: S-moment が落ちた、または判定できる族が 3 未満
  - この判定は活性化を跨いで同じ式なので、命題 4′（活性化間の一致）を初めて同じ土俵に乗せる。
- **5-h 沈下の順序（REPORT_ONLY）**: 旧 5-c。**R は棚族について平衡の $\kappa$ 依存を予測していない**（§1 の leaky 予測は「上端は $\kappa$・$a$ に依らない」）ので、これを R の反証に数えない。$C$ が定数なので $\kappa$ の AUC は \|v\| の AUC と**厳密に同一**（$\kappa=Cv^2$ は $v^2$ の単調変換）。したがって「$\kappa$ の AUC」ではなく「**\|v\| の AUC**」と書く。登録は「過渡の主張」として: 各ユニットの沈下時刻（$z_{\max}$ が $-d$ を最初に下回るタスク）と**その 200 タスク前**の \|v\| の順位相関、seed bootstrap CI つき。較正値: `LR_1216` で同じ手続き（$\kappa$@100k/500k/1M/5M → 末尾の深さ）の AUC は 0.528/0.546/0.520/0.514。
- **5-i full-batch（REPORT_ONLY・帰属）**: `FBLR_1216`（500k step ＝ 50 タスク・32 パターン厳密勾配）。予測: ノイズ誘起なら leaky の $z_{\max}\approx0$ が**消える／大幅に弱まる**、一次のあてはめならそのまま残る。S-fb（full-batch 勾配 = 32 個の単標本勾配の平均、相対 1e−6）が落ちたら `NOT_RUN` とし、§8 に「R の帰属は本 spec では検証されない」と名指しで残す。

- **命題 5 の総合ラベル（排他かつ網羅・決定木）**:
  1. 5-g または 5-b が `NOT_DETERMINED` → **`R_NOT_DETERMINED`**
  2. 5-g PASS（`STATIONARITY_DIRECT_PASS`）かつ 5-b `EQUILIBRIUM_PREDICTED` → **`R_SUPPORTED`**
  3. 5-g FAIL かつ 5-b `EQUILIBRIUM_OFF` → **`R_REFUTED`**
  4. それ以外 → **`R_PARTIAL`**
  - 5-c の結果を修飾子として必ず併記する（`R_SUPPORTED+CAUSAL` / `+NONCAUSAL` / `+UNDETERMINED`）。5-a・5-d・5-f・5-h・5-i は総合ラベルに入れず併記する。
  - **4-c（上端が折れ目に追随）は R のラベルに入れない**（別仮説 `EDGE_*` として §4.4-c で独立に出す）。

### 4.6 数値発散・データ依存の除外

- G6（§3.6）を適用。**発散の定義を事前に数値で書く**: NaN、または $W_{\rm tail}$ の \|median $\bar z$\| $>50$。
- NaN による除外は `NOT_RUN`、NaN でない逸走は 5-a の **FAIL 側の証拠**として扱い `ALPHA_LIMITED` を別に立てる（「発散した $\alpha$ は外して残りで判定」は PASS 方向にしか働かない事後選択）。
- 落とした seed 数はすべての判定について `verdict.csv` に列で残す。

## 5. 検査（本走前に PASS が要る）

| 検査 | 内容 | 捕まえるバグ |
|---|---|---|
| S-limit | `shelf_leaky_d0` が `leaky_relu` と forward/backward で**バイト**一致（格子 [−30,30]・24001 点 **＋ 明示の追加点** $\{+0.0,-0.0,\pm d\ (\forall d\in{\rm SHELF\_DEPTH\cup STEEP\_DEPTH}),\pm(d\pm1\,\rm ulp),\pm5\rm e{-}324,\pm1e{-}38\}$）。`pre=-0.0` で shelf が `+0.0`・leaky が `-0.0` を返すのは**既知の除外**として登録 | 棚の式が leaky に退化していない（`linspace(-30,30,24001)` の中央は −6.245e−16 で、0 も折れ目も格子に入っていない）。`torch.equal` は符号盲なのでバイトで比べる |
| S-flip | `flip_leaky(z) == -leaky_relu(-z)` と `flip_leaky'(z) == leaky_relu'(-z)` がバイト一致（同じ追加点込み） | 述語を `>0` で書くと $\varphi'(\pm0.0)$ が鏡像要請と食い違う（`<0` 形なら $\pm0.0$ 込みで一致するのを実測済み） |
| S-fd | 新族が自分の forward の真の導関数（float64 中心差分・折れ目 ±1e−3 除外。**S-limit/S-flip の追加点はここでは除外する**） | 代替勾配の混入・微分の書き間違い |
| S-curv | `act_curv` が `act_grad` の真の導関数（float64 中心差分・折れ目除外）。区分線形族は恒等的に 0 であることを別に検査 | $\varphi''$ の式ミス（§4.5-g の $G_i$ が全部ずれる） |
| S-fallthrough | 新 11 名の**それぞれ**について、`act_fn`・`act_grad`・`act_curv` の**三方向すべて**で、出力が同じ $\alpha$ の ELU 分岐の出力と**一致しないこと** | if 連鎖の最後が ELU なので、分岐の書き忘れは例外にならず黙って ELU として学習する（片方だけの検査だと `act_grad` の書き忘れを見逃す） |
| S-const | `SHELF_DEPTH`・`STEEP_DEPTH`・`STEEP_SLOPE`・`SOFTPLUS_BETA`・`TANH_BETA` と config の第 2 母数を突き合わせ（`weird_act_0903._s_const` の流儀） | 名前に埋めた深さとクラス定数の食い違い |
| S-guard | 新 11 名が `ACTIVATIONS` と、傾き族は `WEIRD_SLOPE_ACTIVATIONS`、$\alpha$ 不使用族は `UNIT_ALPHA_ACTIVATIONS` に**両方**入っていること。`steep_shelf`/`softplus_b`/`tanh_b` は `dial != 1.0` で例外 | `ACTIVATIONS` にだけ足すと `set_activation` の域ガードが一切かからない |
| S-hook-inplace | フック適用後に `net.W is net.Ws[0] and net.b is net.bs[0]` | 別名 `net.W`/`net.b` が古いテンソルに取り残され、`lop_metrics`/`freeze` 等が初期値を読む |
| S-hook-noop | **独立に 2 つの st を作る**: `gate_dial_0902.setup_arm_dial(cfg, blk, "cpu")` と 新 runner の `_setup_with_hook(cfg, blk, "cpu", hook=None)`。`net.Ws[0]/bs[0]/v/c`・`running_mean`・`layer_means[0]`・`env.flip_state` を**バイト**比較。**同時に「hook=negate にすると同じ比較が FAIL する」ことも記録** | 「テンソルを自分自身と比較する空虚な S 検査」（本プロジェクトで一度やった失敗）。negate 側の FAIL が検査の生存証明 |
| S-copy | `difflib.SequenceMatcher` で新 runner の学習ループ本体を `weird_act_0903._run_arm_weird` と比較し、opcodes が `equal` と登録済みの `insert` だけであること。挿入行の文字列を §5 に逐語で登録する | 雛形の `_s_copy` は `len(mine)==len(theirs)` と `len(diff)==1` を要求する zip 実装（`src/weird_act_0903.py:330-334`）で、**行を挿入すると必ず FAIL する** |
| S-null | `LRnull_1216`（5M）が `LR_1216` と、共通列 `set(a.files)&set(b.files)`（除外 `{arm, run_id, state_hash_final}`）で `np.array_equal(..., equal_nan=True)`、`state_hash_1m` は JSON 文字列一致 | 「全列 bit 一致」は原理的に不可能（`arm`・`run_id` は必ず違う、`state_hash_final` は参照が 15M 終端、新列 `zmin` 等は参照に無い、`dzbar` 先頭は NaN）。腕ブロックは `family: leaky`/`activation: leaky_relu`/`dial: 0.1`/`hidden: [100]`/`centered_layers: [1]`/`target_mu_norm: 3.041`/`target_dose: 12.16` を逐語で揃える |
| S-null-E | `Enull_1216`（5M）が `E_1216` と同じ規則で一致 | ELU 側の写しにも同じ保証を与える（5-a/5-b の参照腕になる） |
| S-mirror | §4.1-b（列別パリティ・バイト比較・`p'+p==1.0` の厳密等号） | 「全ユニット列を符号反転で比較」も「全列を一致で比較」も誤り。$\le1$ 形は死ユニットが自動的に通る |
| S-stream | 全 30 腕の step 0 の `net.Ws[0]/bs[0]/v/c` と最初の 100 入力バッチが、同じ seed どうしでバイト一致（フック適用**前**で比較） | 活性化・フック・新列が乱数を消費していない（G5 のペア判定はこれに依存） |
| S-support | `layer1_zmin == 2*layer1_zbar - layer1_zmax`（相対 1e−5）、かつ半幅 $=0.5\sum_{j\in\rm free}|w_j|$（`layer1_w_free` から） | 支持の列挙／中心化のズレ。zmin は冗長列なので、合わなければ recorder のどこかが壊れている |
| S-moment | 4 モーメント列を、同じ記録の $z$ から float64 で独立に再計算した値と相対 1e−5 一致（3 族 × 3 seed × 10 記録の抜き取り） | §4.5-g の全判定が乗る列の計算ミス |
| S-C | 実装が返す $C$ が閉形式 11.497681 と 1e−12 一致、かつ logs の `dose_relative_error` が 0 | 旧 S-C（2 通りで 1% 以内）は同じ定数を 2 回計算する恒真検査だった |
| S-vfreeze | `Evf*` の `layer1_v_unit` 列が全記録で厳密に定数（`np.ptp == 0`）、かつ `Evf1` の初期 $v$ が `Enull` と bit 一致 | $v$ の更新が実際に止まっているか（5-c の因果性がここに乗る） |
| S-lr | `lr_used` 列が腕ブロックの `lr` と一致し、`st["lr"]` と `st["runs"][i]["lr"]` の両方が書き換わっている | $\eta$ がどこにも記録されず腕名の文字列からしか復元できない（`arm_runs` は `common.lr_main` を読む） |
| S-taut | 参照腕（`LRnull`）で `sunk` 定義（$z_{\max}\le-d$）の率が $d=2,3$ で 0.05 未満、かつ旧定義（$\bar z\le-d-1$）の率が 0.7 超であることを記録 | 判定が恒真でないことの証拠を走の中に残す（旧定義は棚の無い腕で 0.35–0.98 だった） |
| S-KSnull | `LR_1216` の 10 seed を復元抽出で 2 つの独立 10-seed アンサンブルにし、$\bar z$ の KS $D$ の分布を 2000 回作って q95 を 1-c の閾値として**登録前に固定** | 旧閾値 $D<0.10$ は帰無分布の中にあった（同一腕内 5:5 分割で中央値 0.066・最大 0.114）。実効 $n$ は 1000 でなく seed 数の 10 |
| S-fb | full-batch 勾配が 32 個の単標本勾配の平均と相対 1e−6 一致（3 seed × 5 step） | バッチ版 `grads_centered_elu` の縮約ミス。落ちたら `FBLR_1216` は `NOT_RUN` |
| S-par | 1 腕を直列／並列で回して logs bit 一致（短縮走行で可・`OMP_NUM_THREADS=1` を揃える） | 並列化がストリームを動かしていない |
| RSS | 短縮走行（1 腕 10 seed・50k step）で VmRSS / ru_maxrss を実測し、並列数 $=\min(20,\ \lfloor\text{空きメモリ}/(1.5\times\text{peak})\rfloor)$ | [[並列実行のメモリ予算]]（コア数で並列数を決めると OOM でデスクトップごと落ちる） |

## 6. 事前予測

### 6.1 Issa（チャット 2026-09-05・逐語）
命題 1「自信流石にあるし多分検証しなくていい」／命題 2「自信あり」／命題 3「知らんけど」／命題 4「もっと分からん」。

### 6.2 Claude（実行前・§2 の事後分析と批評での再計算だけから）

- **1**: `MIRROR_EXACT` 95%。`LINEAR_PINNED` 85%（`LIN_1216` の median \|z̄\| < 0.15）。`TH_1216` は `ODD_SYMMETRIC_RUNAWAY` 55% / `ODD_SYMMETRIC_PINNED` 30%。1-c は `ENSEMBLE_SYMMETRIC` 60% / `ENSEMBLE_NOT_RESOLVED` 35%（10 seed では分解できない可能性が高い）。
- **2**: `CURVATURE_NONLOCAL` 65% / `MIXED` 25% / `CURVATURE_AT_INIT` 10%。$\Delta_3$ の点予測 **−1.5**（95% 予測区間 [−4, +0.3]）。`SH_d3` の reach 率 0.6（届きはする）。`SP_1216` は `SMOOTH_DOWN` 75%（$\Delta\bar z$ 点予測 −2.0）。`ST_d1/d2` は `ASYMMETRY_SIGN_OK` 80%（$\Delta_{\rm st}$ 点予測 +1.5）。
- **3**: 上から（+5/+4）`MEAN_INDEPENDENT` 85%（保持率 $\rho$ 点予測 0.05）。下から（−5/−4・15M）leaky `MEAN_INDEPENDENT` 45% / `_SLOW` 35% / `MEAN_DEPENDENT` 20%（$\rho$ 点予測 0.25、$\tau$ 点予測 120 タスク）。ELU は `_SLOW` 45%。
- **4**: `LITERAL_OUT_OF_SCOPE`（確定）。局所性は `LOCALITY_MONOTONE` 70%、$d^\ast$ 点予測 **3**（$d=30$ は必ず一致・$d=2$ は不一致）。4-c は **`EDGE_DETACHED_UP` 50% / `EDGE_AT_KINK` 30% / `EDGE_MIXED` 15%**（草案では `EDGE_FOLLOWS_KINK` 70% と書いたが、棚は折れ目より下で $|\varphi|\ge d$ ＝ $\mathbb E[\varphi^2]$ に $d^2$ の床ができ、深さに依らない上向き力 $2ad$ が残るので、R が正しければ上端は $-d$ より上に離れる）。$d=2,3$ の median $z_{\max}$ 点予測 **+0.35 / +2.20**（$d=0.5,1$ は −1.45 / −1.00）。
- **5**: `R_PARTIAL` 45% / `R_SUPPORTED` 30% / `R_REFUTED` 15% / `R_NOT_DETERMINED` 10%。
  - 5-a: `ALPHA_CONTRAST_MONOTONE` 55% / `ALPHA_CONTRAST_CONSISTENT` 25%。$B$（lag ＋ 完全沈水）の点予測 $\alpha=0.5/1/2/4$ で **0.75 / 0.79 / 1.20 / 1.75**（$\alpha=1$ の参照実測は 0.794 [0.652, 0.988]）。`E_a4` が G6 で落ちる確率 25%。
  - 5-b: `EQUILIBRIUM_ORDER_ONLY` 45% / `EQUILIBRIUM_PREDICTED` 30% / `EQUILIBRIUM_OFF` 20%。ELU 腕の median $z_{\max}$ 点予測 $\alpha=0.5/1/2/4$ で **+0.16 / +0.69 / +2.41 / +4.93**（実測 $\alpha=1$ は +0.10 なので、この予測自体が 0.6 ずれている）。
  - 5-c: `WELL_FROM_READOUT` 60% / `WELL_PARTIAL` 25% / `WELL_INDEPENDENT_OF_READOUT` 15%。`Evf1` の median $z_{\max}$ 点予測 **−4**（500 タスクで単調に沈み続ける）。
  - 5-d: `FLUCT_SQRT_ETA` 45% / `FLUCT_OTHER` 40%。位置の $\eta$ 不変（$\eta\cdot$step 一致）は 75%。
  - 5-f: `P5_SCALE_LAW_INVARIANT` 55%（$s$=2 で $\bar z$ 点予測 **−7.4**、$s$=0.5 で **−1.85**。草案の「0.5〜1.5 深い」は 2〜3 倍甘かった）。
  - 5-g: `STATIONARITY_DIRECT_PARTIAL` 45% / `_PASS` 25% / `_FAIL` 20%。
  - 5-i: full-batch で leaky の $z_{\max}$ が $+0.1$ から動かない確率 45%（＝一次のあてはめ）／$-2$ より下へ動かない確率 80%。
- **外れたときに第一に疑うもの**: (i) $\mathbb E_W$ の一様重み近似（実際は flip 直後の残差重み・§2 で 500 倍の差）、(ii) 5M で棚に届いていない（2-e の reach 率で切り分く）、(iii) per-unit 還元（$\nabla R\perp\ker J$ の補正）、(iv) 支持幅の成長が R と符号が逆（§8-6）。

## 7. コスト・実行

- 1 腕 1 プロセス（10 seed をベクトル化）・`OMP_NUM_THREADS=1`・5M ≈ 17.5 分（1M ≈ 3.5 分の実績）。
- 内訳: **5M 腕 25 本**（437.5 分）／**10M 腕 2 本**（70 分）／**15M 腕 2 本**（105 分）／**500k 腕 1 本**（full-batch は 1 step が約 4 倍なので ≈ 7 分）＝ CPU 合計 ≈ **620 分**。
- 並列数は §5 RSS で決める（1 プロセス ≈ 0.5 GiB 見込み・上限 20）。16 並列で理想 39 分、臨界パスは 15M 腕の 52.5 分 → **壁時計 ≈ 60–70 分・1 バッチ**。ピーク RSS ≈ 8 GiB（空き 21 GiB）。
- checkpoints: `common.checkpoints = [0, 1000000, 5000000]`（`_run_arm` は `<= total` で絞る）。10M/15M 腕は `[0, 1000000, 5000000, 10000000, 15000000]`。1 本 ≈ net+env+teacher×10 seed。**500k 腕は `checkpoints: []`** に落とす（S-null の比較対象は logs なので検査は成立する）。
- **生ログ**: 新列（`zmin` 1000-step、`w_free` タスク終端、モーメント 4 列 タスク終端＋末尾 20 タスクの 1000-step）込みで ≈ **100 MB/腕**、合計 ≈ **3.2 GB**。`.gitignore` 対象なので、repo には `summary.md`・`verdict.csv` と、**登録判定が読む腕の末尾窓だけを抜いた縮約 npz**（`results/edge_law_0905/logs_tail/`・合計 < 100 MB 見込み）を `git add -f` する。生ログ全体は NPZ_ARCHIVE に退避する。

## 8. 引用上の注意

1. §2 は全部事後・未登録。判定は §4 のみ。
2. 命題 4 の「反例」は Issa 自身が括弧で除外した場合（釣り合い点が存在しない GELU の谷底、周りの分布が変わる `Gc_b1`）に当たる。**`REFUTED` と書かない**（`LITERAL_OUT_OF_SCOPE`）。
3. R は**仮説**。しかも「上端が折れ目に溜まる」は**可動度ラチェット**（折れ目の下で有効 lr が $a^2$ 倍になる吸収的境界）でも同じく予測されるので、4-c の PASS は R の証拠にならない。R を他の説明から分けるのは **5-c（$v$ 凍結の因果）・5-g（停留残差の直接検定）・5-d（揺らぎの $\eta$ スケーリング）**の 3 つだけ。
4. **$\tau\propto\eta^{-2}$ は「二次機構の指紋」ではない**。この系の駆動は flip 直後の残差パルスで、パルスの緩和時間が $\propto1/\eta$ なので二次のドリフト量は $\eta^1$。草案 §8-3 の読み替えは撤回した。
5. $C$ は定数（11.497681）なので $\kappa=Cv^2$ は $v^2$ の単調変換。**「$\kappa$ 依存」と「$|v|$ 依存」は本走の観測だけでは原理的に分離できない**。分離できるのは $v$ を外生に固定した `Evf*` だけ。
6. **R は上端の位置を説明するが、支持幅の成長は説明しない（符号が逆）**。線形腕は $\lVert w\rVert$ 1.41→0.41 と縮み（R と整合）、非線形腕は 2.8–4.6 倍に育つ。LoP に効く「どれだけ深く沈むか」は $\bar z=z_{\max}-$半幅 の**半幅側**が支配しており、そこは R の外。
7. §1 の「$\alpha$ が消える」は**完全沈水（$z_{\max}<0$）のときだけ**。実測の完全沈水率は 35%。
8. 力場分解（self/rest）は使わない（[[命題リスト]] Q12）。

## 9. 作業表

1. spec を vault と repo に commit（実装より先）
2. 実装（`nets.py` の 11 名 ＋ `act_curv`・runner の写し＋フック＋`v_freeze`＋`full_batch`・recorder の新列・tests・解析）＋ §5 の全検査 → commit
3. 本走（1 バッチ・並列）→ S-ext 相当の付随検査 → 解析 → `results/edge_law_0905/summary.md`・`verdict.csv`（列: `arm, judgment, role, statistic, window, exclusion, n, death_rate, point, ci_lo, ci_hi, gate_G1..G6, label`）
4. 結果ノート `測定/命題1-5_上端則結果_0905.md`・[[走の履歴]] 1 行・[[現在地]] 反映・命題リストへの転記

## Log

- 2026-09-05 18:1x: 草案（`spec_draft.md`）を Opus 4 レンズ（統計・実装・理論・忠実性）で批評（79 件・blocker 19）→ 改訂。批評で確認した数値は §2 に「批評で確認」と付けて入れた。
- 2026-09-05: 改訂が残した 7 つの未決を Claude が裁定（Issa の裁量許可 [[proj-004-operating-latitude-0902]] による・Issa 未確認）: (1) 腕 30 本（ELU scale と ±2 オフセットを落とし、線形・奇関数・滑らか凸・曲率反転・v 凍結・full-batch を入れる）を採る。(2) 停留残差 $G_i$ の直接検定（5-g・新 4 列＋`act_curv`）を採る（実装量は増えるが、上端則の代理量に頼らない唯一の判定）。(3) 棚の事前予測は R の数値解に合わせて `EDGE_DETACHED_UP` 側に置く（Claude 自身の一様支持での再計算でも、$\kappa\le1$ では上端が折れ目より 1〜3.7 上、$\kappa\ge3$ で折れ目付近と、$\kappa$ 依存が出た。5-b の per-unit 数値平衡がこれを吸収する）。(4) `Evf*` の unfit 足切り 3 倍を採る。(5) full-batch 腕は最後に実装し、S-fb が落ちるか時間切れなら `NOT_RUN`。(6) `LRnull`/`Enull` の再走（5M×2）を採る。(7) 生ログは `logs_tail/` だけ `git add -f`、全体は NPZ_ARCHIVE へ。
- 実装前 commit: vault `spec/命題1-5_上端則_spec_0905.md`・repo `specs/spec_edge_law_0905.md` ＋ `configs/edge_law_0905.yaml`。
