# 活性化キメラの 2×2 spec 0913 —— ELU と leaky の引き分けは二つの利点の相殺か
状態: **未実行（事前登録・起草・改訂 3）** / 更新: 2026-09-14（改訂 2: 批評を反映し、仮説判定を単純効果で定義し直し、co-primary・null 双子・分離腕・用量ゲート・段 1/段 2 を追加。改訂 3: 数値と規則の検証で見つかった食い違いを反映。SE の推定量の固定、S16 の区間を模擬から導出、ラベルを対比の状態から排他に組み直し、co-primary の合わせ方、分離腕の用量ゲート、箱 B H の用量の錨、層局所腕を段 2 から外した） / 起草: Claude（仮説は Issa・Issa の予測欄は未記入）
親: [[ELUの幅成長と沈下_結果_0909]]（「LR ≈ ELU1」の出所）/ [[高ゲート帯の幅ダイヤル_結果_0911]]（**大キメラ SMAXH ≡ ELUF を箱 B で既に走らせている**）/ [[ゲート被覆か非線形性か_結果_0911]] / [[浅い帯は独立のレバーか_結果_0912]] / [[RandomLabelMNIST結果_0906]] / [[ゲート硬さダイヤル結果_0902]] / [[命題1-5_上端則結果_0905]] / [[前活性オフセット結果_0906]] / [[現象3_非ReLU戻り道の対応づけ_0902]]
姉妹（別の軸で相殺を切る Issa の仮説・混ぜない）: [[ELUの沈下と学習余力を分ける_spec_0913]] / [[ELUの沈下と学習余力を分ける_結果_0913]]
run id: `act_chimera_0913`。
- **段 1（中核）** を先に回す（§2.1）。
- **段 2**（`act_chimera_0913/stage2`: RL の V・S 家族、箱 B の lr 感度）は同じ規則で今登録し、段 1 の後に Issa の承認で回す。層局所 H 腕は写したループの登録ブロックでは実装できないので、段 2 から外した（§2.1）。
- SCR の lr 退避枝 `act_chimera_0913/scr_lr0p005` は §2.4 の条件でのみ回す。

実装（予定）: branch `claude/act_chimera_0913`（origin/main b50f127 から）
- `src/act_chimera_0913.py`（活性化 8 種・LR02・分離腕 4 種・h・φ″・`ChimeraMLPL`・`make_act`）
- `src/act_chimera_pmnist_0913.py` / `src/act_chimera_rlmnist_0913.py` / `src/act_chimera_scr_0913.py` + `configs/act_chimera_scr_0913.yaml`
- `src/act_chimera_report_0913.py` / `src/test_act_chimera_0913.py`
- `src/act_chimera_launch_0913.py`（§7 のジョブキュー。各ジョブの起動直前に MemAvailable から J を計算し直し、PID・起動時刻・コマンド行を記録し、段の前後に `ps` と `nvidia-smi` の出力を provenance に取る）
- 33a0cab から同一 bytes で持ち込む 7 ファイル（§2.3）
- スモークの出力先は `results/_smoke_act_chimera_0913/{pmnist,scr,rlmnist}`（本走の `results/act_chimera_0913/` とは分ける。edge_law の runner は outdir に完全なログがそろった腕を飛ばし（`_complete_arm_logs`・`src/edge_law_0905.py:609`）、判定器は outdir のファイルを読むので、出力先を共有するとスモークと本走を取り違えうる）

**順序は §6・§7 の通り**: G−1（未決への回答）→ 実装 → G0 → 3 環境のスモーク → G0.5 → 1 回だけ commit・push → 本走。その commit を provenance の `git_hash` にする。
環境: Permuted MNIST（箱 B）・Random-Label MNIST（RL-MNIST）・SCR condA（箱 A）。CIFAR は Issa の判断で入れない。

## 0. 問い

### 0.1 観察と仮説（Issa）
箱 B で leaky（a=0.1、以下 LR）と ELU（α=1、以下 ELU1）はほぼ同じ劣化を示す（`results/elu_growth_0909/summary.md`: 精度の低下 t2–6 → t116–120 が LR −2.05/−1.67/−2.57 pt、ELU1 −2.09/−1.33/−2.09 pt）。
**Issa の仮説**: 両者が並ぶのは、逆向きの利点が二つ打ち消し合っているからである。
- (a) ELU は z≈0 で φ′ = e^z ≈ 1 を保つ。0 近傍に留まる前活性は強く生きている。
- (b) leaky は深部で飽和しない（z<0 の全域で φ′ = 0.1）。深く沈んだユニットも動ける。

### 0.2 2×2
因子は「0 近傍の傾き」∈ {1, 0.1} と「深部の傾き」∈ {飽和, 0.1} の二つ。

| | 深部 飽和 | 深部 0.1 |
|---|---|---|
| **近傍 1** | ELU1（親） | **大キメラ**（(a) と (b) を両方持つ） |
| **近傍 0.1** | **小キメラ**（どちらも持たない） | LR（親） |

キメラは 3 家族 × 2 = 6 本作る（§1）。家族は、継ぎ目の深さと鋭さを変えても箱の効果が残るかを確かめるために置く。
- 硬い継ぎ目 H（z_c = ln 0.1）: SMAXH / SMINH
- 値の min/max V（z_v ≈ −10）: VMIN / VMAX
- 滑らか S: SMAXS / SMINS

**家族を並べても、ゲートと前向きの値は分けられない。** どの家族でも近傍因子は E[φ] を下げ、E[h] を上げる向きに同時に動く（§1.4）。そこで箱 B と SCR に、前向きと逆向きで別の関数を使う**分離腕 4 本**（§1.5）を置く。

### 0.3 仮説が当たったときの予言と、外れ方の意味
損失的な量 Y（小さいほど良い・§3.1）について、seed ごとに次の対比を取る。

- N = ½[(Y_LR + Y_small) − (Y_ELU1 + Y_big)]（>0: 近傍の傾き 1 が損失を下げる・主効果）
- D = ½[(Y_ELU1 + Y_small) − (Y_LR + Y_big)]（>0: 深部の非飽和が損失を下げる・主効果）
- I = ½[(Y_big + Y_small) − (Y_LR + Y_ELU1)]（0: 加法的。<0: 両方そろうと足し算以上に効く）

**仮説が名指ししているのは主効果ではなく単純効果である。**

| 単純効果 | 恒等式 | 意味 | 仮説の予言 |
|---|---|---|---|
| Y_small − Y_ELU1 | N + I | 深部が飽和しているときの (a) の利点（(a) 単独） | > 0 |
| Y_small − Y_LR | D + I | 近傍が 0.1 のときの (b) の利点（(b) 単独） | > 0 |
| Y_LR − Y_big | N − I | 深部が 0.1 のときの (a) の上乗せ | > 0 |
| Y_ELU1 − Y_big | D − I | 近傍が 1 のときの (b) の上乗せ | > 0 |
| Y_ELU1 − Y_LR | D − N | 両親の差 | 上の 1・2 行目のどちらよりも小さい |

- 上の 4 つが揃うと、**大キメラは両親に勝ち（`BIG_BEATS_BOTH`）、小キメラは両親に負ける（`SMALL_BELOW_BOTH`）**。
- 「相殺」は、(a) 単独と (b) 単独の利点がどちらも実在し、両親の差がどちらの利点よりも小さいことを言う。
- small が親と並び、big だけが勝つ（I < 0 で N + I ≈ 0 や D + I ≈ 0）なら、それは**相乗であって相殺ではない**。

| 結果 | 意味 |
|---|---|
| 4 つの単純効果がすべて > 0、両親の差が (a)(b) 単独の利点より小さい、SUPERADDITIVE でない（`CANCELLATION_SUPPORTED`） | 相殺が立つ |
| 4 つとも > 0 だが、両親の差が一方の利点と同程度か、SUPERADDITIVE（`PARTIAL_CANCELLATION`） | 二つの利点は実在するが釣り合っていない、または相乗が乗っている |
| N ≈ 0・D > 0（`DEEP_ONLY`。I ≈ 0 のとき big ≈ LR・small ≈ ELU1） | (a) は不活性。引き分けは (b) と別の何かの相殺 |
| N > 0・D ≈ 0（`NEAR_ONLY`。I ≈ 0 のとき big ≈ ELU1・small ≈ LR） | (b) は不活性 |
| N ≈ 0・D ≈ 0（`NEITHER`） | 二つは単に似た関数 |
| small 側の単純効果が無く big だけが勝つ | 相乗。相殺の物語は誤り（`CANCELLATION_REFUTED`） |
| どちらかが < 0 | その「利点」はむしろ害 |
| 分離腕で近傍の単純効果が前向き側に乗る（`N_VIA_FORWARD`） | 効果はゲートでなく前向きの値（床・平均出力・h）を通る。(a) の機構は立たない |
| H と S で食い違う（`SHARPNESS_DEPENDENT`） | 継ぎ目の鋭さ（C⁰/C¹ の角・h の形・帯の漏れ）が効いている |
| 因子の用量が足りない（`NOT_TESTABLE_WEAK_MANIPULATION`・`NOT_TESTABLE_JOINT_UNVISITED`） | その因子の「効果なし」は読まない |

### 0.4 走る前に既に分かっていること（開示）
数値はすべて committed の行からの事後計算で、m は §3.2 の SE だけ（σ_traj は双子を走らせるまで無い）。

1. **大キメラ SMAXH は箱 B で走り済み。** `src/gate_shape_0911.py:52-66` の `ELUFloor(0.1)`（band_dial_0911 の `ELUF` 腕）と φ・φ′・autograd が float32/float64 の 2M 点で bit 一致する。

   | 対比（窓平均・seed 0/1/2） | ΔL [pt]（1 SE） | −ΔA_late [pt]（SE_late） |
   |---|---|---|
   | LR − ELUF | +1.36 / +1.12 / +0.64（0.17 / 0.26 / 0.16） | +1.11 / +0.85 / +1.08（0.14 / 0.09 / 0.08） |
   | ELU1 − ELUF | +0.94 / +0.81 / +0.44（0.22 / 0.31 / 0.20） | +0.74 / +0.56 / +0.39（0.11 / 0.12 / 0.18） |

   - SE は §3.2 で固定した推定量（ddof = 1・lag-1 は隣接対の Pearson 相関）による。全体平均で中心化した ACF を使うと、ELU1 − ELUF の seed 2 は 0.19 / 0.17 になる。
   - co-primary の両方で 3/3 が m を超えるので、**`BIG_BEATS_BOTH` は既知**である（最も際どいのは seed 2 の ELU1 − ELUF の約 2.2 SE。対比の σ_traj が SE の 1.9 倍を超えない限り崩れない）。
   - **箱 B の H 家族の大キメラは検定ではなく再現（G1 の錨）として扱い、採点しない。**
2. **箱 B の仮説判定は SMINH 一つで決まる。**
   - G1 で LR・ELU1・SMAXH の 3 セルが committed 軌道に固定される。big を含む 2 つの単純効果と両親の差は、走る前に決まっている。採点するのは Y_small を含む 2 つの単純効果と、相殺検定だけである。
   - 旧版の判定（`BOTH_FACTORS_HELP` かつ D-par ∈ {TIE, UNRESOLVED}）は、small := LR（SMINH が LR と同じ軌道）でも 3/3 で成立した（L で N = 0.89/0.72/0.42、D = 0.47/0.40/0.22、I は SUPERADDITIVE）。SMINH の L が LR より ε 良いと D は ε/2 下がるので、旧判定は ε < 2(D − m) のあいだ成立し続けた。最も際どい seed 2（D = 0.221、m = 0.102）で **ε = 0.24 pt** のとき初めて崩れる（ε = 0.2 pt なら成立）。これは「SMINH が LR より大きく良くはない」ことしか言っていなかったので、§4 D3 で定義を改めた。
   - **改めた定義の損益分岐**: small を「LR の系列から late 窓の精度を δ 下げたもの」とする。この small − LR の系列は両窓で定数になり、それ自身の m は定義できない（§3.2 は sd = 0 で例外を投げる）。そこで (i) の small 側の 2 項は成り立つとし、(ii) だけを両親の対比 Y_ELU1 − Y_LR の m で評価した。L と −A_late の両方で (ii) が初めて成立する δ = max_endpoint(|Y_ELU1 − Y_LR| + m) は、seed 0/1/2 で **0.62 / 0.43 / 0.87 pt**（旧記載の 0.65/0.45/0.90 は 0.05 刻みの切り上げ）。つまり SMINH は late 水準で LR より約 0.45–0.9 pt 悪くないと支持にならない。σ_traj を足すと δ はさらに増える。
3. **両親は「引き分け」ではない（主窓）。**
   - ELU1 − LR の ΔL は −0.42/−0.31/−0.20（SE 0.20/0.12/0.21）。向きは 3/3 で ELU1 が良いが、seed 2 は分解できない。
   - A_late では ELU1 が +0.37/+0.29/+0.69（SE_late 0.13/0.11/0.18）高く、3/3 で分解する。
   - Issa の窓（t2–6 → t116–120）では並ぶ。本 spec は両親の差を「0 であること」ではなく「各利点より小さいこと」として相殺検定に使う（§4 D3）。
4. **小キメラの近縁は、両親より悪い側にはっきり出てはいない（旧版の読みを訂正）。**
   - 箱 B の ELU03（φ′ = 0.3e^z・(0.3, 飽和)）: L では LR に対して +0.31/−0.08/+0.35（SE 0.46/0.16/0.23）、ELU1 に対して +0.73/+0.23/+0.55（SE 0.60/0.12/0.22）。
   - ところが A_late では ELU03 が LR より +0.18/+0.21/+0.41 高く（3/3 で SE_late 超え）、ELU1 より 0.19/0.08/0.29 低い（seed 1 の 0.082 は SE_late 0.090 に届かない）。**late 水準では両親の間に寄っているが、`SMALL_BETWEEN` には届かない。**
   - D2 の規則（§4.0 の対比の状態と co-primary の合わせ方）を字義どおり当てると、σ_traj = 0 で次になる。
     - L: small − LR は 3/3 で |c| ≤ 2m（0.31 ≤ 0.91、0.08 ≤ 0.31、0.35 ≤ 0.46）で方向でないので EQ。small − ELU1 は seed 2 で 0.55 > 2m = 0.43 なので EQ でない。→ `SMALL_EQUALS_LR`。
     - −A_late: small − LR は 3/3 で DIR（small が良い）なので EQ ではない。small − ELU1 は seed 1 で m に届かず（DIR でない）、seed 2 で 0.29 > 2m = 0.26（EQ でもない）。BETWEEN・EQUALS のどれにも入らない。→ `SMALL_UNRESOLVED`。
     - 合わせると **`SMALL_EQUALS_LR_L_ONLY`**（下流では UNRESOLVED。代理なので用量ゲートは計算しない）。旧記載の `SMALL_UNRESOLVED` は合わせ方の規則を当て損ねていた。
   - ELU03 を small とした N・D は L だけなら 3/3 で > m だった。−A_late では N は 3/3 で > m、D = +0.28/+0.18/−0.01（SE 0.10/0.10/0.11。seed 2 で方向にならず、seed 0 は 0.28 > 2m = 0.21 で EQ でもない）なので D は UNRESOLVED。合わせると N = `HELPS`、D = `HELPS_L_ONLY` で、見出しは `FACTORIAL_OTHER` になる。
   - LR001（a = 0.01）を small の代理にすると、LR との差は L +0.38/+0.51/+0.33（SE 0.23/0.18/0.27）、−A_late +0.55/+0.72/+0.28（0.12/0.12/0.14）で `SMALL_BELOW_BOTH` になる。ただし相殺検定 (ii) は、L で seed 0 と seed 2（seed 2: 0.198 + 0.207 = 0.405 > 0.332）、−A_late で seed 2 が落ち、`PARTIAL_CANCELLATION` になる。
   - CELU03: band_dial_0911 の窓の中央値で定義した L で 3.00/1.96/2.68（本 spec の窓平均の L では 3.02/2.13/2.25）・死 14 ユニット。LR とは混在。
   - 箱 A の E_a0p1_1216（φ′ = 0.1e^z）は onset 5/10・log10 U −1.33 で、LR −2.65・E −2.73 より約 1.3 dex 悪い（`results/gate_dial_0902/summary.md:45,55-56`）。こちらは両親より悪い側にいる。
5. **RL-MNIST では親が並ばない。**
   - Codex の `elu_environment_0913`（80 epoch/タスク・別の乱数列）では、ELU1 の online が t11–20 で 11.7%、t41–50 で 10.2%。層 2 の z̄ は t10 で −43、t50 で −154。LR は 51.8%/48.8%。
     - 出所は main に無い。Codex の worktree `~/Projects/claude/elu_response_anchor_0913`（branch `codex/elu-response-anchor-0913`、両ファイルを最後に変えた commit 46becd9）の `results/elu_three_studies_0913/summary.md:76,78` と `results/elu_environment_0913/levels.csv`。
   - 0906 の 400 epoch 系には ELU 腕が無い。したがって RL は非対称な場になりうる（§4.0 の床規則）。
6. **平均ゲートの交絡は、箱 B の大キメラについては既に「分解できない」側にある。**
   - gate_shape_0911 の 13 腕で Spearman(Ḡ, L) = −0.81。実現した Ḡ（層 1・t101–120 の窓平均）は LR 0.246/0.243/0.219、ELU1 0.261/0.255/0.274、ELUF 0.362/0.336/0.350。
   - 同じ seed の 4 段の leaky の梯子（LR001・LR・LR03・LIN）で Ḡ から補間した線からの残差は、L で ELUF −0.81/−0.58/−0.01（SE 0.14/0.31/0.25。3 seed とも LR と LR03 の間で λ = 0.57/0.46/0.58）、ELU1 −0.35/−0.24/+0.07。−A_late では ELUF −0.74/−0.61/−0.65（SE 0.11/0.08/0.07）で 3/3 が線の下にある。ELUF は L の seed 2 でだけ線の上にいる。
   - 4 段の梯子なら、D6 の big 側は L で UNRESOLVED・−A_late で BELOW、合わせて `SMAXH_BELOW_LEAKY_LINE_A_ONLY` になる。
   - **ただし登録した梯子は LR02 を足した 5 段である。** LR02 の Ḡ は LR（0.22–0.25）と LR03（0.44–0.45）の間に来る見込みで、ELUF（0.34–0.36）の上側の相手はおそらく LR03 から LR02 に替わる。したがって L の残差は走る前には分からない。seed 2 の L の残差が −m を下回れば、SMAXH は両 endpoint で BELOW になり、H 家族の `LOCATION_MATTERS` にも届きうる（SMINH が ABOVE のとき）。
   - 反実仮想の Γ で割る版（旧 D6）は、D − N ≡ Y_ELU1 − Y_LR を含む。既知の両親の差（N − D = +0.42/+0.31/+0.20）でほぼ決まるので、報告だけにする。
7. **前向きの値の交絡は、どの家族でも近傍因子と共線である。**
   - 親の late 状態（層 1・LR ユニット・seed 0/1/2）で、Φ_N は H −.527/−.533/−.551、V −.335/−.340/−.352、S −.634/−.642/−.664。Ψ_N は H +.497/+.504/+.523、V +.586/+.607/+.638、S +.574/+.584/+.606。
   - 前向きの値は大きく効くことが分かっている。SCR の act_offset_0906 では、定数 c = −0.5 だけで z̄ が +1.50 [1.44, 1.57]、zmax が +0.47 動いた（SUB_ endpoint と上端の位置そのもの）。
   - 箱 B の ELUF は、LR より約 1.1 浅い（z̄_inv −2.77/−3.15/−3.06 対 −4.08/−3.97/−4.23、差 1.31/0.82/1.17）が、幅は同じ（cnorm 8.02/8.07/8.05 対 8.06/8.07/8.13）。実現した Ḡ の上乗せ（+0.116/+0.093/+0.131）は、反実仮想の Γ_N^H（0.042–0.046）の 2–3 倍ある。前向きのオフセットが駆動しうる内生的な経路そのものである。
   - → 分離腕（§1.5・D5a）を段 1 に入れる。
8. **V 家族の近傍対比は、箱 B ではゲートの予言と前向きの予言が重なる。**
   - V の近傍は B1 = [z_c, 0] で H と同じゲート（+.046/+.044/+.042）を足し、B2 = [z_v, z_c) で −.047/−.050/−.053 を引く。合計の Γ_N^V は LR ユニットで −.002/−.006/−.011、ELU1 ユニットで +.005/+.001/+.006 と、seed で符号が変わる。
   - 近傍帯のゲートが機構なら、V の近傍は H の近傍をほぼ再現するはずである。ELU03 代理の seed 0（N = 1.04、D = 0.62）で、seed 0 の LR ユニットの値（Γ_N^V = −0.0016、Γ_N^H = 0.0456）に揃えて帯ごとの価値を別々に置くと、帯別のゲートの予言は 0.52 になる。前向きの予言区間 [0.64, 1.18]·N = [0.66, 1.23] からの距離は 0.15 で、m（0.30）より近い。一方、ゲートの価値を場所に依らないとした全体版の予言は −0.04 で（3 seed 平均の Γ_N^V = −0.006 を混ぜると −0.14）、観測 X^V ≈ 0.6 は全体版の規則では `NEAR_VIA_FORWARD` と誤って読まれる。
   - VMIN ≡ ELU1・VMAX ≡ LR が成り立つ帯（B0∪B1∪B2）の質量は、LR ユニットで 91–93%、ELU1 ユニットで 83–85%。
   - → **V は箱 B と SCR では報告だけ**にし、採点は P(z < z_v) が大きい RL（段 2）に限る（§4 D5b）。
9. **SCR の両親も既知である。**
   - G1-SCR で両親は edge_law の null 腕に bit で固定される。log10 U(491–500) の E − LR は [0.95, −0.11, −1.15, −0.15, 0.30, 0.54, −1.46, −0.78, −0.33, −1.41]、sd 0.83、E < LR は 7/10。**D-par SCR = `PARENTS_UNRESOLVED` は既知**。
   - 水準で検出できる効果は約 1 dex。機構量（沈下率・z̄・mob）は 10/10 で分解する。
   - SCR には等価帯が無い（§4.0）ので、不活性による反証も相殺検定もできない。

## 1. 活性化と腕

### 1.1 定数（float64・50 桁 mpmath で照合済み）
| 記号 | 値 |
|---|---|
| a（leaky 傾き） | 0.1 |
| z_c = ln a | −2.3025850929940455（float32: −2.3025851249694824） |
| z_v（e^z − 1 = a z の負根） | −9.999545794446535（= −10 − W₀(−10e⁻¹⁰)、残差 0.0。float32: −9.99954605102539） |
| e^{z_v} | 4.542055534648271e−05 |
| C_SMAXH = 0.9 + 0.1 ln 0.1 | 0.6697414907005954 |
| C_SMINH = 0.1 − 0.1 ln 0.1 | 0.3302585092994046 |
| C_SMINS = 0.1 ln(1/11) | −0.23978952727983707 |

### 1.2 定義と箱
すべての腕で z > 0 のとき φ = z、φ′ = 1。z = 0 は負側の枝に入る（ELU の規約）。

| 腕 | 箱（近傍, 深部） | φ（z ≤ 0） | φ′（z ≤ 0） | 継ぎ目・連続性 | 負側の床/漸近 | h(−∞) |
|---|---|---|---|---|---|---|
| `ELU1` | (1, 飽和) | e^z − 1 | e^z | 0: C¹ | −1 | 1 |
| `LR` | (0.1, 0.1) | 0.1z | 0.1 | 0: C⁰ | 非有界 | 0 |
| `SMAXH` | (1, 0.1) | e^z − 1（z ≥ z_c）; 0.1z − C_SMAXH（z < z_c） | max(e^z, 0.1) | z_c: C¹, 0: C¹ | 0.1z − 0.670 | 0.670 |
| `SMINH` | (0.1, 飽和) | 0.1z（z ≥ z_c）; e^z − C_SMINH（z < z_c） | min(e^z, 0.1) | z_c: C¹, 0: C⁰ | −0.330 | 0.330 |
| `VMIN` | (1, 0.1) | min(ELU1, 0.1z): e^z − 1（z ≥ z_v）; 0.1z（z < z_v） | e^z（z ≥ z_v）; 0.1 | z_v: C⁰（φ′ が 0.1 → 4.5e−5 に跳ぶ・唯一の非凸）, 0: C¹ | 0.1z | **0** |
| `VMAX` | (0.1, 飽和) | max(ELU1, 0.1z): 0.1z（z ≥ z_v）; e^z − 1（z < z_v） | 0.1（z ≥ z_v）; e^z | z_v: C⁰, 0: C⁰ | −1 | 1 |
| `SMAXS` | (1, 0.1) | 0.1z + 0.9(e^z − 1) | 0.1 + 0.9e^z | 0: C¹ | 0.1z − 0.9 | 0.9 |
| `SMINS` | (0.1, 飽和) | 0.1 ln(0.1 + e^z) − 0.1 ln 1.1 = 0.1·softplus(z − z_c) + C_SMINS | 0.1·σ(z − z_c)（0⁻ で 1/11） | 0: C⁰ | −0.240 | 0.240 |

**恒等式**
- φ_SMAXH + φ_SMINH ≡ φ_ELU1 + φ_LR、かつ φ_VMIN + φ_VMAX ≡ φ_ELU1 + φ_LR（全 z）。H と V は関数空間で厳密な要因計画になる。
  - φ′・E[φ]・E[h] のように φ や φ′ に線形な量は、交互作用対比が厳密に 0（実測 ≤ 9e−16）。**結果に交互作用が出れば、それは学習力学から来ている。**
- S は加法的でない。φ′ の残差は 0.1e^z(0.9 − e^z)/(0.1 + e^z)（最大 +0.047・z = −1.53）、φ の残差は最大 0.140。S は正の mob 交互作用を内蔵する: 2Γ_I = E[φ′_SMAXS + φ′_SMINS − φ′_LR − φ′_ELU1] は、親の late 状態（gate_shape_0911 の層 1・ガウス近似）で LR ユニット +0.012、ELU1 ユニット +0.011（Γ_I は 0.0059 / 0.0052）。上限は点ごとの最大 +0.047。
- SMAXS = 0.1z + 0.9·ELU1(z)。SMAXS の φ′ は全域で SMAXH 以上、SMINS の φ′ は全域で SMINH 以下。S の操作は H より強い。
- **S の因子は帯について純粋でない。** S の近傍因子は B2 にもゲートを足し（LR の late 状態で Γ_B2 = +.0056/+.0057/+.0059、Γ_N .059/.057/.055 の約 10%）、S の深部因子は B1 にも足す（Γ_B1 +.0079/+.0077/+.0075、Γ_D .070/.071/.073 の約 11%）。
- SMINS では φ′(z_c) = 0.05 がちょうど成り立つ。`sat`（φ′ < 0.05）は SMINS で P[z < z_c]、SMINH・ELU1 で P[z < ln 0.05 = −2.996] と意味が変わる。

**h(z) = zφ′ − φ**（z > 0 では全腕 0。z<0 では接線の切片の符号反転なので、h の深部極限は負側の床/切片そのものになる）

| 腕 | h（z ≤ 0） | 値域 | h(−5) | h(z_c) | h(−1) |
|---|---|---|---|---|---|
| LR | 0 | {0} | 0 | 0 | 0 |
| ELU1 | e^z(z−1) + 1 | [0, 1) | 0.9596 | 0.6697 | 0.2642 |
| SMAXH | ELU1 の h（z ≥ z_c）; **定数 0.66974**（z < z_c） | [0, 0.670] | 0.6697 | 0.6697 | 0.2642 |
| SMINH | 0（z ≥ z_c）; e^z(z−1) + 0.33026 | [0, 0.330) | 0.2898 | 0 | 0 |
| VMIN | ELU1 の h（z ≥ z_v・z_v⁺ で z_v(e^{z_v} − 0.1) = 0.99950）; 0（z < z_v） | [0, 0.99950] | 0.9596 | 0.6697 | 0.2642 |
| VMAX | 0（z ≥ z_v）; ELU1 の h | [0, 1) | 0 | 0 | 0 |
| SMAXS | 0.9[e^z(z−1) + 1] | [0, 0.9) | 0.8636 | 0.6028 | 0.2378 |
| SMINS | 0.1zσ(z−z_c) − 0.1softplus(z−z_c) + 0.23979 | [0, 0.240) | 0.2017 | 0.0553 | 0.0069 |

**h の深部値は箱で揃わない。** (1, 0.1) の箱は VMIN 0 < SMAXH 0.670 < SMAXS 0.9、(0.1, 飽和) の箱は SMINS 0.240 < SMINH 0.330 < VMAX 1。

### 1.3 実装（float32 で安全な式・`src/act_chimera_0913.py` に逐語で置く）
`zc = z.clamp(max=0.)`、`LNA = z_c`、`ZV = z_v`（Python float）。

```python
ELU1 : where(z>0, z, expm1(zc))                                          dphi: where(z>0, 1, exp(zc))
LR   : H.ARMS['LR']（host の Activation をそのまま使う）
LR02 : H.Activation('LRx', 'leaky', .2)（GS:78-81 の LR03/LR001 と同じ作り方）
SMAXH: GS.ELUFloor(0.1) のインスタンスをそのまま使う（再実装しない。演算順が変わると committed 軌道との bit 一致を失う）
       φ = where(z>0, z, where(z>=LNA, expm1(zc), 0.1*(z-LNA)+(0.1-1.)))  dphi: where(z>0,1, clamp(exp(zc), min=0.1))
SMINH: where(z>0, z, where(z>=LNA, 0.1*z, exp(z.clamp(max=LNA)) - C_SMINH))   dphi: where(z>0,1, clamp(exp(zc), max=0.1))
VMIN : where(z>0, z, where(z>=ZV, expm1(z.clamp(ZV,0.)), 0.1*z))           dphi: where(z>0,1, where(z>=ZV, exp(z.clamp(ZV,0.)), 0.1))
VMAX : where(z>0, z, where(z>=ZV, 0.1*z, expm1(z.clamp(max=ZV))))          dphi: where(z>0,1, where(z>=ZV, 0.1, exp(z.clamp(max=ZV))))
SMAXS: where(z>0, z, 0.1*zc + 0.9*expm1(zc))                               dphi: where(z>0,1, 0.1 + 0.9*exp(zc))
SMINS: where(z>0, z, 0.1*softplus(zc - LNA) + C_SMINS)                     dphi: where(z>0,1, 0.1*sigmoid(zc - LNA))
```

**実装上の規則**
- ELU1 は箱 B では `EG.ELU(1.0)`（`src/elu_growth_0909.py:18-23`）、SCR では `VecMLPL` の `elu` を使う。
- **全クラスに `kind` 属性を置く**（`'chimera_<腕名>'`。`'adaptive_snake'` 以外）。RL.run_one:121 が読む（S7）。
- h は `_hneg_elu(z) = z·exp(z) − expm1(z)` を部品にし、腕名で引く表として置く（`isinstance` の分岐で黙って 0 を返さない）。
- **継ぎ目の述語**
  - φ の枝は定数の述語 `z>=LNA`／`z>=ZV` による `torch.where` で選ぶ。
  - φ′ は同じ述語による where か、clamp で書く。z が継ぎ目ちょうどのとき、解析 φ′ と autograd(φ) の食い違いは float32 の z_c で SMAXH（ELUFloor の clamp）2.2e−8、SMINH 7e−9、VMIN（where）3.0e−8 と実測した。SMAXH は 1 ulp × φ″ = 2.4e−8 とほぼ同じで、VMIN はそれを超える。したがって「≤ 1 ulp × φ″」では上限にならない。S3b の許容値 8·eps·|φ′| + 1 ulp·max|φ″|（float32 の z_c で 1.19e−7）がすべてを覆い、S3b がそれを検査する。
  - `torch.minimum/maximum` は同値点で勾配を 50/50 に割るので使わない（z = 0 で VMIN の勾配が 1 でなく 0.55 になる）。
  - 捨てる側の枝も exp/log の引数を定義域に clamp する（しないと backward が 0·inf = NaN を作る）。
- SMINS の `softplus` の引数は ≤ 2.3026 なので、threshold 20 による段差は起きない。
- **箱 B と RL の学習勾配は `phi` の autograd から来る**（`src/gate_shape_0911.py:201-202`、`src/pmnist_rlmnist_0906.py:154-160`）。`dphi` は読み出しにしか使わない。
- **SCR は解析的な `act_grad` で学習する**（`src/nets.py:576`）。φ′ の正しさが軌道を決める（S13b）。
- SCR の EdgeRecorder は φ″ を最初から書く（`src/edge_law_0905.py:71` の `m_dphiddphi`）ので、φ″ も置く。
  - SMAXH: e^z（z_c<z<0）, 0（z<z_c）
  - SMINH: 0（z_c<z<0）, e^z（z<z_c）
  - VMIN: e^z（z_v<z<0）, 0（z<z_v）
  - VMAX: 0（z_v<z<0）, e^z（z<z_v）
  - SMAXS: 0.9e^z
  - SMINS: 0.1σ(1−σ)、σ = σ(z − z_c)
  - ELU1: e^z、LR: 0
- 使ってはならないもの（S8）
  - `EG.make_act` は未知の腕名を**黙って leaky にする**（`elu_growth_0909.py:24-27`）。
  - `EG.hdefect` は未知クラスで**黙って 0 を返す**（:28-34）。`EG.measure(..., want_grad=True)` はその中で hdefect を呼ぶ（:62, :66）。
  - `GS.check_dphi` の格子は [−6, 6] で、z_v を含まない（GS:115-135）。
- 新しい `make_act` は未知名で KeyError を投げる。

### 1.4 実際の分布の上で、操作の大きさはどれくらいか（事後・参考値）
入力は gate_shape_0911 の late 窓（t101–120）の層 1 ユニット分布（`zcur_i`, `sdcur_i` のガウス近似。測定 Ḡ との一致は 0.01）。その上で 8 関数を評価した。キメラ自身の軌道は内生的に変わるので、**本走では §3.3 の定義で、各走の実分布の上で測り直す。**

| 分布 | 量 | H 近傍 Γ_N / 深部 Γ_D | V 近傍 / 深部 | S 近傍 Γ_N / 深部 Γ_D / 2Γ_I |
|---|---|---|---|---|
| LR のユニット | mob 対比 | +.044 / +.058 | **−.006** / +.008 | +.057 / +.071 / +.012 |
| ELU1 のユニット | mob 対比 | +.044 / +.056 | +.004 / +.016 | +.056 / +.068 / +.011 |
| LR のユニット | E[φ] 対比 Φ_N / Φ_D | −.537 / −.216 | −.342 / −.021 | −.647 / — |
| LR のユニット | E[h] 対比 Ψ_N / Ψ_D | +.508 / −.181 | +.610 / −.079 | +.588 / — |

帯別の内訳は §0.4-7・8 と §1.2 に書いた。
- 表の値はすべて 3 seed の平均で、§3.3 の ½ 対比の定義による（S は加法的でないので、Γ_N と E[φ′_SMAXS − φ′_LR] = Γ_N + Γ_I は一致しない。後者は LR ユニットで .063、Γ_D + Γ_I = E[φ′_SMAXS − φ′_ELU1] は .077。改訂 2 の表は S の列にこの Γ + Γ_I を載せていた）。
- **ゲートの操作量はどの家族でも E[φ]・E[h] の操作と同じ向き**に並ぶ。H・V・S を並べてもゲートと前向きの値は分けられない。分離腕（§1.5）を置く理由である。
- **V の近傍は近傍帯のゲートを H と同じだけ動かし、B2 で同じ量を引く。** 合計 Γ だけ見ると「ゲートを動かさない対照」に見えるが、帯で見るとそうではない（§4 D5b）。

### 1.5 追加の腕
**分離腕（前向きと逆向きで別の関数・surrogate）**

| 腕 | 前向き φ | 逆向き（学習に使う φ′） | 何を分けるか |
|---|---|---|---|
| `GN` | φ_LR | φ′_SMAXH | 近傍の単純効果（LR → SMAXH）のうち、ゲートだけ |
| `FN` | φ_SMAXH | φ′_LR | 同じ単純効果のうち、前向きの値だけ |
| `GD` | φ_ELU1 | φ′_SMAXH | 深部の単純効果（ELU1 → SMAXH）のうち、ゲートだけ |
| `FD` | φ_SMAXH | φ′_ELU1 | 同じ単純効果のうち、前向きの値だけ |

- **読み出しでの φ′ と h（全環境）**: 分離腕の `dphi` は φ′_bwd とする（学習の勾配が見るゲート）。gate_block（`gbar`・`off_i`・`hard_i`）、C.measure の `gate_mean`、EG の `sat`・`dead_units`、`mob_band`、実現 Ḡ はすべてこの `dphi` で計算する。帯分率は腕に依らない。`fn_mom` は 8 関数を分離腕の状態の上で評価する（腕自身の φ は使わない）。
- 分離腕の h は **h = z·φ′_bwd − φ_fwd** とする。§3.4 の欠損項の恒等式 E = Σ(∂L/∂a₁)h は、r_W1b1 = Σ(∂L/∂a₁)φ′_bwd(z)z、r_W2 = Σ(∂L/∂a₁)φ_fwd(z) なので、この h でだけ成り立つ（S4 と S17 がこの h を検査する）。

  | 腕 | h（z ≤ 0） | 値域 | h(−5) | h(z_c) | h(−1) |
  |---|---|---|---|---|---|
  | GN | z(max(e^z, 0.1) − 0.1)（z < z_c で 0） | [−0.280, 0] | 0 | 0 | −0.2679 |
  | FN | 0.1z − expm1(z)（z ≥ z_c）; C_SMAXH = 0.66974（z < z_c） | [0, 0.670] | 0.6697 | 0.6697 | 0.5321 |
  | GD | ELU1 の h（z ≥ z_c）; 0.1z − expm1(z)（z < z_c。z_v で 0 を切り、その下で負・非有界） | (−∞, 0.670] | 0.4933 | 0.6697 | 0.2642 |
  | FD | ELU1 の h（z ≥ z_c）; z·e^z − 0.1z + C_SMAXH（z < z_c。+∞ に発散） | [0, ∞) | 1.1361 | 0.6697 | 0.2642 |

- 箱 B（autograd で学習）: `torch.autograd.Function` で forward = φ_fwd(z)、backward = grad·φ′_bwd(z) にする。z は §1.3 の clamp 済みの式で評価する。S1 は腕名で除外し、S17 で検査する。
- SCR（解析勾配で学習）: `ChimeraMLPL` の名前 1 つにつき act_fn = φ_fwd、act_grad = φ′_bwd、act_curv = φ″_bwd を置く（S15 は分離腕の名前も含め、act_curv = φ″_bwd を act_grad の autograd と比べる）。
- RL では段 2（任意）。
- **前例の注意**: SCR の bwd_leak_0902 では、BL（ReLU 前向き + leaky 逆向き）が発散し、FL（leaky 前向き + ReLU 逆向き）は 10/10 onset だった。前向きと逆向きが食い違う学習は勾配流ではないので、発散しうる。発散した分離腕は §4.0 の打ち切りで記録し、救済しない。退避枝の引き金にもしない。
- **代替（未決 4）: 定数オフセット腕**
  - `LRpc`（φ_LR + c）と `SMAXHmc`（φ_SMAXH − c）。c = Φ_N^H を親の late 状態で測った値（≈ −0.53）。
  - φ′ は変わらず、E[φ] が c、h が −c 動くので、Φ_N（−0.53）と Ψ_N（+0.50）の両方に、ゲート変化 0 で合う。
  - 深部は `ELU1pc`（ELU1 + c_D）と `SMAXHmcD`、c_D = Φ_D^H。
  - 採る場合は D5a の規則の GN/FN を「オフセット腕」に読み替えて、同じ規則で判定する。

**leaky の梯子腕 `LR02`**（箱 B のみ・a = 0.2）
- D6 の leaky 線で、LR（Ḡ ≈ 0.24）と LR03（≈ 0.45）の間を補間するために置く。ELUF の Ḡ ≈ 0.35 がその間にある。
- 梯子の残りの段（LR001・LR・LR03・LIN）は gate_shape_0911 の committed 行を同じ seed で使う（同じプロトコル。LR は G1 で同一性が確認される）。

**null 双子 `LRtw{k}`**
- LR と全乱数列を共有し、初期値の 1 要素 W1[k, 0] だけを `nextafter(·, +∞)`（1 ulp）にした走である（追補 2 で b1[k] に +1e−6 に変更。生きているかは S18b で判定し、窓の最初のタスクまでに LR と食い違わない双子は σ_traj から除く）。
- 箱 B は k = 0..3 × seed 0–2、RL は k = 0 × seed 0–2 を置く。
- 窓内のタスク間ノイズに入らない、軌道ごとの持続的なずれ σ_traj を測る（§3.2）。

## 2. 環境

### 2.1 全環境に共通
- **腕は乱数に入らない**（S-init / S-pair）。同じ seed の全腕が、初期値・データ列・バッチ順（SCR では教師・入力列・flip）を bit で共有する対応のある設計である。双子は初期値の 1 要素が 1 ulp 違うだけ（S9・S18。追補 2 で b1[k] に +1e−6 に変更）。
- **段 1 と段 2**

  | 環境 | 段 1（中核・この順に回す） | 段 2（今登録・段 1 の後に Issa の承認で回す） |
  |---|---|---|
  | 箱 B | 8 腕 + 分離腕 4 + LR02 + 双子 4 = 17 腕 × seed 0–2 = 51 走 | lr 感度（H 家族 4 腕 × lr {5e−4, 2e−3} × 3 = 24 走・D9） |
  | SCR | 8 腕 + 分離腕 4 = 12 腕 × 10 系列 × 5M | — |
  | RL | H 家族 4 腕（LR, ELU1, SMAXH, SMINH）+ 双子 1 = 5 腕 × seed 0–2 = 15 走 | V・S 家族 4 腕 × 3 = 12 走（D5b・D4）、分離腕 GN・FN × 3（任意） |

  段 1 のラベルは D-par・D0・D1–D3・仮説判定・D5a・RL の床・D8。D4・D5b・D6・D7 の規則はここで凍結し、段 1 の保存配列から 2 回目の判定器の実行で計算する（§8 のスキーマが前提・G0.5 が保証する）。
- **層局所 H 腕（SMAXH を層 1 だけ・層 2 だけ）は段 2 から外した。** H.forward（`src/pmnist_boundary_host_0908.py:242-253`）は 1 つの活性化オブジェクトを両層に掛け、層の番号を渡すのは AdaptiveSnake だけである。AdaptiveSnake の部分クラスにすると、学習ループの `act.update()`（GS:211-212）と C.measure の `act.alpha()` が呼ばれる。層ごとの活性化には、H.forward の写し・C.measure の acc の経路・gate_block の層 2 の読み出しを置き換える登録ブロックが要り、P1–P6 には無い（S14 が落ちるか、未登録の変更が要る）。回すなら、それらのブロック（例: P7 = 写した `forward(params, x, act_l1, act_l2)`、act_l1 = act_l2 で H.forward と bit 一致する S 検査つき）を追補で登録してからにする。
- **lr の方針（明示）**
  - 主走は全腕でプロトコルの lr をそのまま使う: 箱 B Adam 1e−3、RL Adam 1e−3、SCR SGD 0.01。腕ごとの調整はしない。
  - 理由 (i): 仮説が説明しようとしている「引き分け」はこの lr での観察である。
  - 理由 (ii): 腕ごとに lr を選ぶと、結果で選んだ変数が一つ増える（最大値選択と同型）。
  - 理由 (iii): Adam は勾配の大域スケールを正規化するので、傾きの違いは一次ではステップ幅に入らない。ただしゲートのパターン（どのパラメータにどれだけ勾配が来るか）は入る。
  - 感度検査は段 2 で箱 B に限る（D9）。×2 は band_omega_0912 の前例、×0.5 は対数で対称にするため。
  - SCR は SGD で、傾きが実効ステップを直接変える。大キメラは lr 0.01 で安定限界の近くにいる（親の 5M 状態での反実仮想で lr·λ_max/2 = 0.98–1.96。E は lr 0.02 で 581k step に発散した前例）。**安定余裕を記録し、§2.4 の規則で退避枝を起動する。** 30k の予備走では遅い発散を排除できないので、予備走で lr を選ぶことはしない。
- **インタプリタ（参照を作ったものに固定する）**
  - 箱 B・RL: `/usr/bin/python3`（~/.local の torch 2.13.0+cu130・numpy 2.5.2）。gate_shape_0911 の provenance の numpy 2.5.2 と、0906 の `launch_rlmnist.sh` の `python3` に一致する。
  - SCR: `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`（torch 2.13.0+cu130・numpy 2.5.1）。edge_law の docstring の起動行である。edge_law の `arm_status` には git_head と時間しか無く、interpreter は記録されていない。G1-SCR が落ちたら、最初にこれを疑う。
  - G0 で `sys.executable`・torch・numpy のバージョンを assert し、全 provenance に書く。

### 2.2 Permuted MNIST（箱 B・`elu_growth_0909`／`gate_shape_0911` のプロトコルと同一）
**プロトコル**（H = `src/pmnist_boundary_host_0908.py`、GS = `src/gate_shape_0911.py`、C = `src/width_sink_clamp_0909.py`、T = `src/transport_common_0910.py`）

- **ネットと入力**
  - 784→100→100→10（H:38 の DIMS）、全層にバイアス（H:237）。同じ活性化オブジェクトを隠れ 2 層に掛け、出力は線形（H:242-253、φ は :250,252）。
  - 入力は pixel/255 の float32、正規化なし（H:172-192）。
- **データ**
  - タスクごとに新しい置換を引く（GS:192-194・`H.stream('perm')`）。
  - タスクごとに訓練 10,000 枚を Hamilton 配分で層化抽出する（H:200-218）。
  - batch 16 × 625 更新 = タスク内 1 周（H:39-40）。120 タスク。
- **学習**
  - loss は `F.cross_entropy`（mean）。
  - 手書き Adam: β = (0.9, 0.999)、eps = 1e−8 を sqrt の外に足す。m・v・tc はタスクを跨いで保持する（GS:191, 203-210）。
  - 初期値は U(±1/√fan_in)、CPU の `'init'` 系列（H:225-239）。
- **乱数と計算環境**
  - 乱数は sha256(`"pmnist_0905|{role}|{seed}"`) で系列を分ける（H:152-156）。
  - CPU 1 thread、`H.setup('cpu')` で deterministic（H:668-673）、`CUBLAS_WORKSPACE_CONFIG`（H:31）。
  - MNIST は `T.data_dir()`（T:44-54）が sha256 を照合する。worktree に `data/` は無いので、`boundary_groups_0908.py:10` の `H.DATA_DIR` 固定を経由する。

**腕と seed**: 段 1 は 17 腕 × seed 0,1,2 = 51 走（t1–120・無介入）。段 2 は §2.1。

**G1 の錨（white-san で bit 一致を要求）**
- `LR`・`ELU1`・`SMAXH`（= `ELUF`）× seed 0–2 を、`results/gate_shape_0911/{LR,ELU1,ELUF}_s{s}_units.npz` と照合する。
  - 対象は per-unit 16 配列（C.measure の 6 + gate_block の 10）× t1–120 = 1920 配列と、`_rows.csv` の `acc` 120 値。件数ガードは 1 走 2040 件。
  - **判定は全配列を `a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)` で行う。** `ref_hard_i` は bool なので、maxabs は数値配列にだけ併記する（bool の減算は TypeError になる）。
- 同じ C.measure と GS.gate_block を無改変で呼べば、演算は完全に同一になる（GELU/SILU 720 配列、R/LR03/LR001 600 配列で 0.0 の前例）。
- 副次の錨として、LR・ELU1 の `zbar_i`・`sd_i`・`cnorm_i` を `results/elu_growth_0909/*_none_s{s}_units.npz` と ≤ 1e−10 で照合する（GS:331 の前例。累積順の違いで 1e−15 の丸めが出る）。
- gate_shape_0911 は white-san で走った（commit d6f6876 のメッセージ「75 走すべて成功・white-san」）。
- 残りの腕（SMINH, VMIN, VMAX, SMAXS, SMINS, 分離腕, LR02, 双子）には錨が無い。

**実装経路（既存モジュールの sha256 を動かさない）**
- 活性化は新モジュール `src/act_chimera_0913.py` に置く。
- 学習ループは GS.train（GS:184-230）を `src/act_chimera_pmnist_0913.py` へ逐語で写し、次の登録ブロックだけを足す（S14 はこの一覧と照合する）。署名の変更は比較しない（本体は AST で取る）。

  | # | 種類 | 位置（GS の行） | 内容 |
  |---|---|---|---|
  | P1 | 挿入 | :189（`mutate == 'init'` のブロック）の直後 | 双子の 1 ulp 摂動（`twin` が None でないとき W1[k, 0] を nextafter）（追補 2 で b1[k] に +1e−6 に変更・`AC.perturb_twin_b1(p, twin)`） |
  | P2 | 置換 | :190 | `act = make_act(arm)` → `act = act_obj` |
  | P3 | 挿入 | :191（Adam 初期化）の直後、タスクループの前 | 読み出しの状態の初期化（W̃ の前回値・層 2 の前回値） |
  | P4 | 挿入 | :199（`_, _, g_start = gate_block(...)`）の直後 | タスク開始時の読み出し: 層 1・2 の `zstart_i`（プローブ平均）と帯分率（§3.4）。例: `zs1 = (probe.px[:, perm] @ p[0].detach().T + p[1].detach()).double()` |
  | P5 | 置換 | :210 | `.001` → `lr`（lr = 0.001 では同じ double なので主走の軌道は不変で、G1 がそれを証明する） |
  | P6 | 挿入 | :216（`r.update(gr_)`）の直後 | タスク終端の読み出し（§3.3–3.4・t20,40,…,120 の E 恒等式・params の sha256） |

- **発散の記録**: `T.finite_guard` は 120 タスクを走り切った後に assert で落ち、行も units も書かない。新ランナーはそれを try/except で囲み、失敗したら部分行と `pmnist/{ARM}_s{seed}_divergence.json`（最初に非有限になったタスク）を書いてから `DIVERGED` にする。この経路は S20 が検査する。
- 新しい `make_act` は未知名で KeyError を投げる。
- **GS.run の検査の扱い**（GS.run は写さない。新ランナーが自前の run を持つ）
  - `TIME_CAP`（GS:31, :377 の `assert wall < TIME_CAP`）は置かない（§7: 時間上限は assert にしない）。
  - G2（GS:306-307 の `check_dphi` の assert）は S1 に置き換える（GS の格子 [−6, 6] は z_v を含まない・§1.3）。ELUF の連続性の assert（GS:308-309）は S3 に置き換える。
  - G3（GS:313-315 の gate_block の分散の恒等式 ≤ 1e−9 と、:355-360 の対照）はそのまま残す（gate_block を無改変で呼ぶので、同じ許容値が成り立つ）。
  - EG の欠損項の恒等式の assert（EG:135-138）は §3.4 の規則で残す。

### 2.3 Random-Label MNIST（`pmnist_rlmnist_0906` のプロトコルと同一・400 epoch）
**プロトコル**（RL = `src/pmnist_rlmnist_0906.py`、H05 = `src/pmnist_0905.py`）

- **データとラベル**
  - seed ごとに訓練 1200 枚を `randperm(60000)[:1200]` で引く（`rl_subset` 系列、RL:53-60）。
  - 画像は全タスクで同じ。ラベルはタスクごとに iid 一様に引き直す（`rl_labels`、RL:63-65,135）。
- **学習**
  - 50 タスク × 400 epoch × 75 step = 30,000 step/タスク（RL:44,136）。epoch ごとに `randperm(1200)` を引く（`rl_batch`、RL:149-150）。
  - 784-100-100-10、host の init（RL:123）。
  - 手書き Adam lr 1e−3（β 0.9/0.999・ε 1e−8・跨いで保持、RL:130-131,172-180）。
  - 勾配は `autograd.grad` による（RL:154-160）。乱数はすべて CPU の生成器で引き、テンソルだけを device に送る（RL:116-195）。したがってラベル列とバッチ列は device に依らない。
- **記録**
  - `online_acc` = 各 batch の更新前正答の平均（RL:156-159,194）。
  - `memo_acc` = タスク終端に 1200 枚を現ラベルで評価（RL:192-194）。
  - タスク終端の診断は `evaluate_rl`（RL:90-109、層ごとの median だけ）。
  - `per_task.csv` は `H05.write_csv` が `f"{v:.10g}"`（有効 10 桁）で書く（H05:650-665）。
- **発散**: 0906 は「非有限の loss をタスク終端で検出して break、救済しない」（RL:162-164,186-191）。**本 spec では黙って落とさず `INCOMPLETE` と記録する**（§4.0。`rlmnist/{ARM}/s{seed}/divergence.json` に最初の非有限タスクを書く。この経路は S20 が検査する）。
- 判定窓は t31–50 の `online_acc` 平均（`analysis/pmnist_0905/verdict_rlmnist.py:16,38-44`）。

**400 epoch を短縮しない理由**
- 80 epoch（Codex）では LR 自身が予算不足になる（終端の訓練精度 84.7%・`BUDGET_LIMITED`）。予算と可塑性が交絡する。
- Codex の走はラベル/バッチの系列が 0906 と違い（`env_labels_0913` 等）、seed の対にできない。
- 400 epoch なら、写したループを 0906 の LR の行に 10 桁で錨づけできる（G1-RL）。

**腕と seed**: 段 1 は H 家族 4 腕 + 双子 × seed 0,1,2 = 15 走。V・S 家族は段 2（RL は両親が z_v を踏む唯一の環境なので、V の採点は RL でだけ行う・§4 D5b）。

**device（スモークの実測で決める・未決 5）**
- 箱 B と同じ 784-100-100-10・batch 16・手書き Adam は、CPU 1 thread で 75k step を 60–72 s で回す（gate_shape の wall_seconds）。1.5M step なら ≤ 1,200–1,440 s/走。GPU 単独の 0906 は 1,144 s/走。
- スモークで、CPU（`torch.set_num_threads(1)`。RL ランナーは設定していないので足す）と GPU で 1 タスクずつ時間を測る。
- **CPU の方が段 1 全体で速ければ、RL の全腕を CPU で回す**（腕の対を 1 つの device の中に保つ）。その場合、0906 の GPU の LR との比較は記述だけにし、G1-RL は GPU の別検査として行う。
- GPU の方が速ければ、全腕を GPU で回し、G1-RL を 3 seed の全 50 タスクに広げる。

**G1-RL（写したループの錨・GPU で行う）**
- 新ランナーの行を `H05.write_csv` と同じ `.10g` 整形で書き、`results/pmnist_rlmnist_0906/LR/per_task.csv` と**セルの文字列が完全一致**することを要求する（`1` と `1.0` の違いも不一致）。新ランナーは seed ごとに `rlmnist/{ARM}/s{seed}/per_task.csv` を書く（§8）ので、参照の同じ seed・task の行と比べる。
  - これは「有効 10 桁の文字一致」であって bit 一致ではない。ただし 1.5M step のカオス的な増幅で、1 bit の違いも途中で 10 桁に現れる。
  - 対象の列は数値 20 列（online_acc … w_norm_l3）。件数ガードは、必須のスモーク（LR s0 t1–2）で 40 セル、GPU の本走に広げた場合は 20 × 50 × 3 = 3000 セル。CPU の場合は任意で LR s0 の t1–50（1000 セル・単独で約 19 分）。
  - 変異対照: 初期値 +1e−3 で 1 タスク回し、同じ文字列比較で 1 セル以上が不一致になること（追補 2-1 で要素を b1[0] に指定）。
- **新しい側の決定性（bit）**: 選んだ device で LR s0 の t1–2 を 2 回回し、t2 終端の params の sha256 が一致すること。変異対照は、2 回目の初期値の 1 要素を nextafter にした走で sha256 が変わること（追補 2 で、双子 LRtw0（b1[0] +1e−6）の b1[0] を除いた sha256 が変わることに変更）。
- 0906 がこの機械で走ったことは、provenance に hostname が無いので確定していない。この機械の /tmp に 0906 の起動・監視ログ（session 2c202656 の scratchpad の `rlm_*.log`）が残っていることは傍証である。**G1-RL のスモークで検証し、落ちたら本走に入らず Issa に上げる。**
- ELU1 と全キメラには錨が無い。

**実装経路と阻害要因**
- **host と参照が main に無い。** proj_004_drift で未追跡、`exp/shell_l2_rlmnist_0913`@33a0cab（L2Init のセッション・親 6617a06）でだけ commit されている。本ブランチに 33a0cab と同一 bytes で持ち込み、sha256 を assert する。

  | ファイル | sha256 |
  |---|---|
  | `src/pmnist_0905.py` | 53e2c10288ff91e39148e81f5756769e1a9a4ff88b375e31120f87745eab26ec |
  | `src/pmnist_rlmnist_0906.py` | 6f3624b59fd70ad0638da201eddcd5c1ca91d057df83c167352074380d7debdd |
  | `results/pmnist_rlmnist_0906/LR/per_task.csv` | 3d67114c2962c441efb458dd7891eaff71e07b86cab58ea6825000b9b3006817 |
  | `results/pmnist_rlmnist_0906/LR/provenance.json` | 69f4ef6ff7770742352630731b435aae7cc83976ed5ed15fccd78525d3797149 |
  | `results/pmnist_rlmnist_0906/R/per_task.csv` | 7a1903614e589d525f1c7637e1c3154aaad7be07d4e9f384f58366c1d9021158 |
  | `results/pmnist_rlmnist_0906/R/provenance.json` | 45e4866c6c58490384f991d9332f256ad89a9152dd2cb095c8f21eaed3a219f4 |
  | `analysis/pmnist_0905/verdict_rlmnist.py` | d555f23ad80e44b9bd276db4aee093d125f23fb0c690b60b7caf932d9aebc977 |

  - 同一内容の add/add は、ブランチ同士のマージでは衝突しない。ただし proj_004_drift の未追跡コピーが pull を止める（§8 の片付け）。
  - `pmnist_boundary_host_0908.py` と `pmnist_0905.py` は byte 同一だが、RL は `src.pmnist_0905` という名前で import するので、別名のまま置く。
  - G1-RL・S11（0906 の `data_sha256`）・S16（R/LR seed 0–9 の床）・AT_FLOOR の 0.00026 の導出は、どれもこの持ち込みの後でしか worktree の ROOT から読めない。
- **データ**: `H05.DATA_DIR` はファイル位置から worktree の `data/mnist` を指す。そこには無いので、ランナーが main clone の `data/mnist` に固定する。4 つの gz の sha256 が gate_shape_0911 と 0906 の provenance の `data_sha256` に一致することを assert する（S11）。
- **ループ**: `RL.run_one`（RL:116-195）を `src/act_chimera_rlmnist_0913.py` へ逐語で写す。登録ブロックは次の 3 つだけ（S14）。

  | # | 種類 | 位置（RL の行） | 内容 |
  |---|---|---|---|
  | R1 | 置換 | :120 | `act = H.ARMS[arm]` → `act = act_obj` |
  | R2 | 挿入 | :123（`params = H.init_params(...)`）の直後 | 双子の 1 ulp 摂動と、読み出しの状態の初期化（W(0) の保存）（追補 2 で b1[k] に +1e−6 に変更・`AC.perturb_twin_b1(params, twin)`） |
  | R3 | 挿入 | :192（`m = evaluate_rl(...)`）の直後 | §3.3–3.4 の per-unit 読み出し。**`.cpu().double()` に写してから計算する**（`use_deterministic_algorithms(True)` の CUDA では histc・bincount・scatter_add 系が RuntimeError になる） |

  - `H.ARMS` への実行時登録はしない。

### 2.4 SCR condA（`gate_dose_0830` の `LR_1216`/`E_1216`、`edge_law_0905` の runner）
LR と ELU1 を同じ条件で並べた最も新しい系は、`edge_law_0905` の `LRnull_1216`/`Enull_1216` である（`configs/edge_law_0905.yaml:57-58`）。どちらも `gate_dose_0830` の `LR_1216`/`E_1216` の bit 複製で、per-unit の zmin・w_free・moments を持つ。

**プロトコル**
- **入力と教師**
  - 入力は m = 20（遅い flip 15 bit + 毎 step iid 5 bit）、1 タスク 10,000 step（`configs/gate_dial_0902.yaml:52`、`src/envs.py:48-84`）。
  - 教師は LTU 100、β = 0.7（`src/envs.py:12-45`）。
- **学習者**
  - 1 層 100 ユニット（`VecMLPL`、`src/nets.py:129`・:365-371）。
  - 二乗誤差、SGD、batch 1、**lr 0.01**（`configs/edge_law_0905.yaml:13`）。
  - dose 12.16 の oracle 固定オフセット（`src/dose_const_5m.py:139-191`、`src/gate_dose.py:212-238`）。
- **系列**
  - 10 系列を 1 プロセスでベクトル化する（`configs/edge_law_0905.yaml:14-15`・`generator_offset: 0`）。
  - **seed の部分集合は別の入力列になり、runner が拒否する**（`src/edge_law_0905.py:593-606`）。したがって SCR だけ 10 seed（系列）で判定する。
- **地平線と記録**
  - 5M step = 500 タスク。
  - 1000 step ごとに float64・32 パターンの厳密な支持で `unfit` を計る。
- **窓**
  - LoP の窓はタスク終端の記録だけを使う: 2–11 / 91–100 / 491–500（`src/gate_dose.py:889-908` の `_window`）。
  - 機構の窓はタスク 451–500（`configs/edge_law_0905.yaml:98` の `tail_window_tasks`）。

**腕**: 12 腕 × 10 系列 × 5M。
- 名前は edge_law の表の `LR_1216`/`E_1216`（`configs/edge_law_0905.yaml:89-90` で gate_dose/p3_extend の参照名として使われている）と衝突させない: `chLR_1216`, `chE_1216`, `chSMAXH_1216`, `chSMINH_1216`, `chVMIN_1216`, `chVMAX_1216`, `chSMAXS_1216`, `chSMINS_1216`, `chGN_1216`, `chFN_1216`, `chGD_1216`, `chFD_1216`。
- `u_fr`（φ′ < 1e−6 の凍結深さ）は、ELU1・SMINH・VMAX・SMINS で 13.8155、LR・SMAXH・SMAXS・VMIN で null。VMIN の帯内最小ゲートは 4.5e−5 > 1e−6 なので凍結しない。分離腕は学習が見る φ′_bwd で決める: GN・GD（φ′_bwd = SMAXH、下限 0.1）と FN（LR）は null、FD（ELU1）は 13.8155。
- 12 行の `family` / `activation` / `dial` / `u_fr`:

  | 行 | family | activation | dial | u_fr |
  |---|---|---|---|---|
  | chLR_1216 | leaky | leaky_relu | 0.1 | null |
  | chE_1216 | elu | elu | 1.0 | 13.8155 |
  | chSMAXH_1216 / chSMAXS_1216 / chVMIN_1216 | chimera | 腕名（SMAXH 等） | 0.1 | null |
  | chSMINH_1216 / chSMINS_1216 / chVMAX_1216 | chimera | 腕名 | 0.1 | 13.8155 |
  | chGN_1216 / chFN_1216 / chGD_1216 | chimera | 腕名 | 0.1 | null |
  | chFD_1216 | chimera | 腕名 | 0.1 | 13.8155 |

  - 親 2 行は `LRnull_1216`/`Enull_1216` と逐語で同じ（下の経路 5）。
  - キメラと分離腕の `dial` は leaky の傾き a = 0.1 を書く（ログの `act_alpha` にはこの値が入る）。式は §1.1 の定数で固定し、`ChimeraMLPL.set_activation` はキメラ名・分離腕名で `dial == 0.1` を assert する（別の値なら ValueError）。

**G1-SCR（white-san で bit 一致・edge_law はこの機械で走った）**
- 新ランナーの `chLR_1216`・`chE_1216` を、`proj_004_drift/results/edge_law_0905/logs/{LRnull,Enull}_1216_seed{0..9}.npz` と照合する。規則は edge_law の `s_null`（`src/edge_law_0905.py:1422-1480`）に従う。
  - `ref_arm='LRnull_1216'`/`'Enull_1216'` と `ref_dir` を**明示で渡す**（:1434-1440）。
  - 共通キーのうち `S_NULL_SKIP`（:82 = arm, run_id, state_hash_final）以外の全配列を `np.array_equal` で比べる。比べたキーの名前と件数を、実装前に参照 npz から作って固定した期待一覧と一致させる。
  - `state_hash_1m` も一致すること。
  - **`s_null` は両方のログを共通の記録数に切り詰める**（:1453-1458）ので、早く落ちた走も通ってしまう。10 系列すべてで `n_records == 5001` と `state_hash_1m` への到達を別に assert する。
  - 参照 npz 20 本の sha256 を provenance に記録する（proj_004_drift では未追跡）。
- 前提: edge_law の null ログは git_head 2a3c0ab の nets.py で作られた。その後 nets.py は 1b65dad（snake_flip_0906: snake1・snake_amp・snake の act_curv）/b6a380d/654c3c1/e15f573 で加法的に編集されている（`git log 2a3c0ab..HEAD -- src/nets.py` の 4 件）。leaky/elu の bit が変わっていないことは、G1-SCR 自体が現行の nets.py で検査する（smooth_kink_0907 の「S-null は offset_grid とバイト一致」は lab 上の検査で、この機械の edge_law 参照との一致ではない）。

**実装経路（`src/nets.py` の bytes を変えない）**
- nets.py を直接編集すると、`codex/zero-attraction-0913`（89521b3）の `SNAKE_PHASE_0913` 追加と同じ場所で衝突する。これが新モジュールにする理由である。
- 経路
  1. `src/act_chimera_0913.py` に `class ChimeraMLPL(VecMLPL)` を置く。`set_activation`/`act_fn`/`act_grad`/`act_curv` をキメラ名と分離腕名についてだけ上書きし、それ以外は `super()` に委ねる。
  2. `setup_arm_dial`（`src/gate_dial_0902.py:393-405`）を写した `setup_arm_chimera` の中で、`set_activation` の直前に `st["net"].__class__ = ChimeraMLPL` を 1 行入れる。これは乱数も状態も触らない（S13）。
  3. **runner**: `edge_law_0905.main --config` は edge_law 自身の `run_single_arm` を使い、それが `_run_arm_edge` を直に呼ぶので、写したループは呼ばれない。そこで新モジュールが自前の `run_single_arm`/`main` を持ち、起動時に `edge_law_0905.CONFIG = <新 yaml>`、`edge_law_0905._TABLE = None` を設定する（`table()`/`_hook_of()`/`registered()` がこの大域を読むため）。
  4. `_run_arm_edge` を写した `_run_arm_chimera` に、edge_law の登録挿入方式（`RUN_INSERTS`・`_copy_opcodes` :657）で `setup_arm_dial = setup_arm_chimera` を 1 行入れる。位置は本体の先頭、`st = setup_arm_dial(...)` より前。RUN_INSERTS の順序もそれに合わせる。
  5. 新 config の `chLR_1216`/`chE_1216` の行は、`LRnull_1216`/`Enull_1216` と family・activation・dial・checkpoints `[0, 1000000, 5000000]` まで逐語で揃える。s_null は `family`/`activation` の文字列も比べる。
  6. 旧経路の `mlp2_phase1.forward_centered` や `dose_const_5m.forward_const` は `torch.relu` 固定なので使わない。
- 追加の読み出し（帯分率・8 関数の Γ/Φ/Ψ・出力層の λ）は、タスク終端の `w_free` と z̄ から 32 パターンの支持を再構成して判定器で計算する。ランナーへの挿入は要らない。ckpt（0・1M・5M）から全 Hessian の λ も判定器で計算する。
- RSS は `edge_law_0905.rss_probe`（`src/edge_law_0905.py:1224-1258`。:1207 の `recorder_bytes` で記録器の本走サイズを外挿する）の方式で見積もる。短縮走行の RSS をそのまま使わない。
  - **edge_law の rss_probe をそのままは使えない。** 子プロセスを `python -m src.edge_law_0905 --arm <arm>` で `--config` 無しに起こすので、子の `table()` は `configs/edge_law_0905.yaml` を読み、キメラの腕名は拒否される。親の側の `table()` も本走の地平線を edge_law の表から取る。
  - そこで新モジュールが自前の `rss_probe` を持つ: 子を `python -m src.act_chimera_scr_0913 --arm <arm> --steps 50000 --outdir <smoke>/rss/run` で起こし（config は新モジュールが `configs/act_chimera_scr_0913.yaml` に固定する）、`recorder_bytes` はそのまま import して、地平線は新しい表（5M と退避枝の 10M）から取る。
  - 測る腕は `chLR_1216`・`chSMINS_1216`・`chGD_1216`（親・キメラ・分離腕の各経路から 1 つ）で、外挿したピークの最大を使う。

**安定余裕と lr の退避枝（事前登録）**
- **安定余裕**: 系列ごとに lr·λ_max/2 を記録する。λ_max は、出力層については各タスク終端で λ_max(2·E_x[φφᵀ])（32 パターンの厳密な支持・float64）、全 Hessian については 1M・5M の ckpt で冪乗法（S19）から取る。
  - 境界は算術で置く: 二次形式の上の SGD は lr·λ < 2、すなわち lr·λ/2 < 1 のときだけ安定である。
  - いずれかの系列で、窓（491–500）の lr·λ_out/2 の中央値、または 5M ckpt の lr·λ_full/2 が ≥ 1 なら、その腕を `NEAR_EOS` とする。
  - 前例: 初期状態では 1.15 で生き残り、≥ 1.37 で発散した（固定点の周りでは λ が学習中に動く）。境界の 1 は、その前例ではなく二次形式の算術から置いた。
- **起動条件**: 主 8 腕のいずれかで、(i) 系列が 1 本でも非有限になる（`NUMERIC_DIVERGENCE`。edge_law の G6 は 2 系列まで落とせるが、本 spec では落とさない）、または (ii) `NEAR_EOS` になる。分離腕だけの発散・NEAR_EOS は起動条件にしない（その分離は §4.0 の打ち切り）。
- **退避枝の中身**
  - 12 腕すべてを、edge_law と同じ `hook: {type: lr, value: 0.005}`・total_steps 10,000,000・checkpoints `[0, 1000000, 5000000, 10000000]` で回し直す（η·step を lr 0.01 × 5M に揃える）。窓は 991–1000、機構の窓は 951–1000。
  - **行は今、1 回だけ commit する yaml に登録する**（arm_table は名前で引き、行ごとに hook・total_steps・checkpoints を持つ。後から行を足すと config の sha256 と `git_hash` が変わり、「1 回だけ commit してその commit から回す」が崩れる）。名前は主行に `_lr0p005` を挟んだ 12 行: `chLR_lr0p005_1216`, `chE_lr0p005_1216`, `chSMAXH_lr0p005_1216`, `chSMINH_lr0p005_1216`, `chVMIN_lr0p005_1216`, `chVMAX_lr0p005_1216`, `chSMAXS_lr0p005_1216`, `chSMINS_lr0p005_1216`, `chGN_lr0p005_1216`, `chFN_lr0p005_1216`, `chGD_lr0p005_1216`, `chFD_lr0p005_1216`。family・activation・dial・u_fr は同名の主行と同じ。
  - `output.dir` は config に 1 つしか無いので、退避枝は `--outdir results/act_chimera_0913/scr_lr0p005` を明示して起動する（本走の `results/act_chimera_0913/scr` と混ざらない）。
  - 錨は `edge_law_0905` の `LRlr0p005_1216`/`Elr0p005_1216`（この機械・10,001 記録・`hook: {type: lr, value: 0.005}` を確認済み）。`chLR_lr0p005_1216`/`chE_lr0p005_1216` を、`ref_arm='LRlr0p005_1216'`/`'Elr0p005_1216'` と `ref_dir` を明示して、s_null と同じ規則で全 10,001 記録（`n_records == 10001` を別に assert）と `state_hash_1m` を照合する。
  - **退避枝を回したら、SCR の主ラベル（全家族の D1–D3・仮説・D5a・D8）はすべて lr 0.005・10M から取る。** lr 0.01 の値は副として報告する。家族ごとに lr を混ぜない。
  - lr 0.005 でも主腕が非有限か NEAR_EOS になったら、その腕を含む家族は最終的に `NUMERIC_DIVERGENCE` / `NOT_DETERMINED_NEAR_EOS` とし、lr をさらに半分にはしない。

## 3. 測定

### 3.1 endpoint（窓は固定。極値は選ばない）
| 環境 | Y（小さいほど良い）| 窓 | 補助 |
|---|---|---|---|
| 箱 B | **co-primary: L = mean acc(t16–20) − mean acc(t101–120) [pt] と −A_late = −mean acc(t101–120) [pt]**（test 1 万枚・現置換） | base t16–20・late t101–120（gate_shape 系） | Issa の窓 t2–6 → t116–120 を副 |
| RL | **co-primary: Y = −100·mean online_acc(t31–50) [pt] と Y_logit = −mean logit(online_acc)(t31–50)** | t31–50（0906 の判定窓） | memo_acc、早期 online（t1–10）、崩壊タスク |
| SCR | **Y = log10 U(491–500)**。U = タスク終端の unfit 10 記録の平均、床 1e−16 | 5M（退避枝では 991–1000） | 1M 窓（91–100）・onset（U ≥ 0.05）の数・submerged 率（SUB_） |

- **co-primary にする理由**: L は base の水準で動く。キメラは可塑性だけでなく初期の学習も変える（ELUF の base は seed 0–1 で LR より 0.25/0.27 低い）。base が高く late も良い small が、L だけで「悪い」と出うる（§0.4-4 の ELU03）。
- **RL の logit を co-primary にする理由**: online_acc は床（0.114）と天井（SNA 0.986）で圧縮される。交互作用は尺度で符号が変わりうる。
- **平均を使う理由**: gate_shape の L は窓の中央値だった。2×2 の対比は腕をまたぐ線形結合で、median(A) + median(B) ≠ median(A + B) だから、本 spec は窓平均で定義する（elu_growth_0909 spec §6 と同じ理由）。gate_shape 系の中央値版 L も併記するが、ラベルには使わない。

### 3.2 対比と、その分解能 m
N・D・I と 2 腕の差 Y_A − Y_B は、どれも係数 w_a の線形対比 c = Σ_a w_a·Y_a である。

**窓内ノイズ SE（seed ごと）**
- **箱 B**: タスク系列 c_t = Σ_a w_a·acc_a(t) を作る（同じ seed の腕は同じ置換・同じテスト集合を共有するので、タスク難度はここで相殺される）。
  - L_c = mean_base c_t − mean_late c_t、SE(L_c) = √(SE_base² + SE_late²)。−A_late の対比は SE_late だけを使う。
  - SE_W = sd_W(c_t) / √n_eff、n_eff = max(1, n(1 − ρ₁)/(1 + ρ₁))。
  - **推定量を固定する**（窓内の値 c_1, …, c_n、n = 5 または 20）:
    - sd_W は ddof = 1（`np.std(c, ddof=1)`）。
    - ρ₁ = max(0, r)、r = (c_1, …, c_{n−1}) と (c_2, …, c_n) の Pearson 相関。各部分列をそれぞれの平均で中心化する（`np.corrcoef(c[:-1], c[1:])[0, 1]`）。
    - 全体平均で中心化する標本 ACF（Σ(c_t − c̄)(c_{t+1} − c̄)/Σ(c_t − c̄)²）は使わない。base の 5 タスクでは両者で m が最大 1.5 倍違い、seed ごとのラベルが動く（ELU03 − LR の L seed 0 で 0.46 対 0.31、ELU03 − ELU1 で 0.60 対 0.39）。ddof = 0 も使わない（5 タスクの窓で SE が 1.118 倍変わる）。
    - §0.4 の SE と m はすべてこの推定量で計算した。S16 (vii) が §0.4-1 の SE 表を小数 2 桁で再現することを assert する。
  - sd が 0 または r が NaN（部分列の分散が 0）の系列では例外を投げる（m を 0 や NaN にしない・S16）。
- **RL**: c_t = Σ w_a·100·online_acc_a(t)（logit 版は Σ w_a·logit(online_acc_a(t))）、t31–50、同じ SE_W。同じ seed の腕はタスクごとのラベルを共有する。

**軌道間のばらつき σ_traj（null 双子から・箱 B と RL）**
- SE は 1 本の軌道の窓内のタスク間ノイズで、軌道ごとに持続するずれ（偶然拾った死にユニットなど）を含まない。gate_shape_0911 の 120 腕対（16 腕。ELU1 と CELU1 は同一軌道なので片方を除く）で、seed 間の対比の sd / RMS(SE) は、上の推定量で L の中央値 0.99（IQR 0.62–1.41）、−A_late で 1.30（0.76–1.88）だった。SE がすべてを含む場合の期待値は 0.83（0.54–1.18）なので、どちらも大きい。1 SE で数える両側の偽ラベル率（3/3）は 0.008 ではなく、L で 0.016、−A_late で 0.036 になる（2·(1 − Φ(0.833/比))³）。
  - 改訂 2 の「1.09（0.73–1.51）」は全体平均で中心化した ACF による L だけの値だった（ACF では −A_late が 1.31（0.77–1.89））。
- 双子の対比 c_tw = Y(LRtw_k) − Y(LR)（endpoint ごと・seed と k ごと）から、1 腕あたりの軌道分散を σ²_arm = ½·max(0, mean_{seed,k}(c_tw² − SE_tw²)) とする。箱 B は 12 個、RL は 3 個の平均。
- 対比 c の分解能: **m(c) = √(SE(c)² + (Σ_a w_a²)·σ²_arm)**。2 腕差なら Σw² = 2、N・D・I なら 1。
- 仮定: LR の双子のばらつきを全腕に使う（§9 限界）。
- 双子がラベルの前に揃わない（発散・未完）ときは σ_arm = 0 で計算し、全ラベルに `(M_SE_ONLY)` を付ける。（追補 2-2: 窓の最初のタスク（箱 B t16・RL t31）までに LR と食い違わない双子は除く。生きた対が 2 未満なら `NOT_DETERMINED_TWIN(M_SE_ONLY)`、切り詰めで σ² = 0 なら `SIGMA_TRAJ_OK` に `clipped` の注記。2-4 A10: LR と双子の device・git_hash・コード・インタプリタが違う対も除く）
- **SCR**: seed ごとの m は置かない（§4.0）。

### 3.3 操作の大きさ（その走自身の状態の上で測る）
家族 F ∈ {H, V, S}・seed ごとに、4 走（LR, ELU1, big_F, small_F）それぞれのタスク終端の前活性 z を集める。
- 箱 B: 固定プローブ 512 枚（`boundary_probe` 系列・現置換）、層 1 と層 2。
- RL: 1200 枚、層 1 と層 2。
- SCR: 32 パターン（z = z̄ + w_free·(x − 0.5) で厳密に再構成）。

その上で 8 関数の φ′・φ・h を評価し、帯 B_k ごとに (unit, 入力) の平均を取る（**帯別・層別に保存する**）。
- Γ_N = ½E[φ′_big + φ′_ELU1 − φ′_LR − φ′_small]（H・V では厳密に E[φ′_big − φ′_LR]）
- Γ_D = ½E[φ′_big + φ′_LR − φ′_ELU1 − φ′_small]（H・V では E[φ′_big − φ′_ELU1]）
- Γ_I = ½E[φ′_big + φ′_small − φ′_LR − φ′_ELU1]（H・V で 0、S で > 0）
- Φ_X・Ψ_X: 同じ係数を φ と h に掛けたもの（前向きの値の対比）
- 帯別の成分 Γ_{X,Bk} = 上の期待値に 1(z ∈ B_k) を掛けたもの。Σ_k Γ_{X,Bk} = Γ_X（S12）。
- 窓は D0 では late 窓（箱 B t101–120、RL t31–50、SCR 491–500）、4 走で平均する。層 1・層 2 を別々に出す。
- **SE_Γ はタスク系列から取る**: 窓内のタスクごとの Γ_t の sd / √n_eff（§3.2 と同じ n_eff）。入力方向の標本 SE は使わない（512 枚 × 100 ユニット × 20 タスクでは極小になり、存在の検査が空虚になる）。
- **実現した平均ゲート Ḡ**: 自腕の φ′ を自腕の状態で平均したもの（層 1 は gate_block の `gbar`、層 2 は新しい読み出し）。反実仮想の Γ とは別に出す（ELUF − LR で実現 Ḡ の差は反実仮想の 2–3 倍・§0.4-7）。

**占有**（腕に依らない z の帯。ユニット別・層別）
- B0 = z > 0
- B1 = z_c ≤ z ≤ 0
- B2 = z_v ≤ z < z_c
- B3 = z < z_v
- 加えて P[|z| < 1]
- 帯別のゲート寄与: mob_k = E[φ′(z)·1(B_k)]（自腕の φ′。Σ_k mob_k = mob）

### 3.4 機構の読み出し（(a) と (b) を直接見る・層別にラベル）
**箱 B・RL（ユニット i・層 l・タスク t）**
- **帯分率**: タスク開始時と終端の p_{Bk,i}（プローブ上の入力の割合。箱 B は挿入 P4 と P6、RL は前タスク終端 = 開始時）。
- **(b) の主読み出し: 深さ速度 v_i(t)**
  - 箱 B の層 1: v_i = S̄·Δm_i + Δb_i（m_i = 行平均、b_i = バイアス、S̄ = `probe.Sbar`。z̄_i ≈ S̄·m_i + b_i は C.measure の `eps_max`（C:218）の恒等式で、置換が変わっても不変な深さ成分）。Δ はタスク開始（= 前タスク終端、t = 1 は初期値）からの差。
  - RL の層 1: 画像が固定なので v_i = Δz̄_i（1200 枚平均）。
  - 層 2: v_i = Δ(W2_i·μ_φ1 + b2_i)、μ_φ1 = プローブ上の層 1 出力の平均（腕ごとに違う・終端の値で固定して前後に使う）。
- **(b) の戻り**: 深部層（下の層別）のユニットのうち、終端の z̄_i ≥ z_c になった割合。距離を含むので UP/DOWN だけにし、FLAT は付けない。
- **(a) の主読み出し**: 近傍層のユニットの沈下ハザード（終端の z̄_i < z_c になった割合。UP/DOWN だけ）と、近傍帯の自腕ゲート mob_{1,i}。
- **回転**: 歩幅の分子 ‖ΔW̃_i‖ をラベルに使い、‖W̃_i‖ を別に出す。ω = ‖ΔW̃‖/‖W̃‖ は幅で動くので、出すがラベルに使わない。
  - 層 1 の W̃ は中心化行（W − 行平均）。
  - 層 2 の W̃ は、腕自身の μ̂_φ1 = μ_φ1/‖μ_φ1‖ 方向を除いた行（W2 − (W2·μ̂)μ̂）。μ̂ 方向の成分は層 2 の「深さ」として別に出す。
- **Adam の状態**（タスク終端）: 行ごとの mean |m̂|/(√v̂ + ε) と、√v̂ < 10ε の要素の割合。Adam はゲートの大きさをパラメータごとにほぼ打ち消すので、飽和が効く場所（ε と標本の不均一）を見る。
- **層別（組成を揃える）**
  - タスク開始時の帯分率で層に分ける: 近傍層 = argmax_{k ∈ {B0, B1, B2∪B3}} p_{k,i}(start) が B1 のユニット、深部層 = B2∪B3 のユニット。
  - 層の中を支配分率 [⅓, ½) / [½, ¾) / [¾, 1] の 3 bin に分け、4 走で共通に 1 ユニット以上ある bin だけを使い、4 走をプールした件数で重み付けする。
  - **FLAT（等価）の分解能ガード**: 1–2 ユニットの bin は m が大きく、|c| ≤ 2m がほぼ自動で成り立つ。そこで同じ読み出し・同じ対比を層別しない層全体のユニットで取ったときの m_all を基準に、層別した対比の m が 2m ≤ 3·m_all を満たすとき（全 seed）だけ `*_FLAT` を付ける。3·m_all は層全体の対比が方向ラベルを 0.933 で得る効果の大きさ（§4.0 の到達表）で、この条件の下では、その大きさの効果が層別の等価帯に入る率は 2m の効果の行（0.085）以下になる。満たさなければ `NOT_TESTABLE_LOW_OCCUPANCY`。方向ラベル（UP/DOWN）にはこのガードを要らない。
  - D7 の m は §3.2 の式による。σ_arm は読み出しごと・層ごと・bin ごとに、双子の同じ読み出しから §3.2 と同じ手順で出す（双子の bin に 1 ユニットも無い seed・k は平均から除き、全部無ければ σ_arm = 0 で `(M_SE_ONLY)`）。（2-4 A5: 追補 2-2 の生存の規則を D7 にも当て、窓の最初のタスクまでに生きていない (双子, seed) を除く）
  - z̄ の幅 1 の bin は使わない。ユニット内の広がりが腕で違う（band_dial の σ_inv: ELU1 4.36・ELUF 2.37・LR 2.56。late 層 1 の 28–38% のユニットで sdcur_i > |zcur_i|）ので、同じ z̄ の bin でも B0/B1/B3 への露出が違う。
- 対比: 近傍層で X(big) − X(LR) と X(ELU1) − X(small)、深部層で X(big) − X(ELU1) と X(LR) − X(small)。窓内タスク系列から §3.2 と同じ m を作る。

**SCR**
- 深さは zmax で定義する（zcoord_0903 の教訓）。
- **復元の読み出し: 腕ごとの緩和率 `relax_rate`**（新規の道具。edge_law に同じものは無い）
  - 使う記録: タスク t ∈ 451–500 の終端記録 k（step = t·10,000）と、その直前の記録 k − 1（step = t·10,000 − 1,000。`step[k] − step[k−1] == 1000` を assert）。
  - 系列 s（seed ファイル）ごとに、ユニット i とタスク t をプールして、y = `layer1_dzbar[k, i]`（記録 k − 1 → k の 1,000 step の増分）を x = `layer1_zmax[k−1, i]` − zmax*_{s,i} に OLS で回帰する（切片あり）。zmax*_{s,i} = そのユニットの `layer1_zmax[k, i]` の t ∈ 451–500 の中央値（ユニットごと・系列ごと）。`relax_rate[s]` = −傾き。
  - x を増分の前の記録から取るのは、y と同じノイズを x に入れないため（gate_dose の q2 の x0/inc と同じ組み方・`src/gate_dose.py:926-947`）。位置をタスク終端の 1,000 step に固定するのは、増分がタスク内の位置で 2〜3 桁減衰するため（zcoord_0903 の教訓）。
  - 全ユニットの版を主にし、生きているユニット（`layer1_denom` > edge_law の `ALIVE_DENOM`）だけの版を併記する。
  - 引用の訂正: edge_law の §4.3-b `relax_fit`（`src/edge_law_analyze_0905.py:1439`）は中央値 zmax の時間への指数当てはめ、§4.5-b `equilibrium_zmax`（:2034）は数値平衡で、どちらも dzbar を距離に回帰しない。`edge_law_analyze_0905.act_numpy`（:259）は活性化名で引くのでキメラ名には使えず、φ は `src/act_chimera_0913.py` の numpy 版から取る。
  - 生の dzbar は固定点までの距離を含み、前向きの値が固定点を動かす（act_offset）ので、それ自体はラベルに使わない。
- 帯分率・mob_k（32 パターンの厳密支持）、安定余裕 lr·λ/2（§2.4）。

**初期の過渡（箱 B・RL）**
- 初期化直後は層 1 の z の sd が約 0.19 で、z < −1 の点は無い。SMAXH は最初のサンプルが z_c を越えるまで ELU1 と同じ、SMINH は LR と同じ軌道を辿る。近傍 1 の腕は層 1 の出力分散を 2.25 倍通す（0.031 対 0.014）ので、初期の N には初期ゲインの成分が乗り、D は恒等的に 0 である。
- t1 からのタスク系列 N_t・D_t・I_t を出す。
- SMAXH が ELU1 から、SMINH が LR から初めて離れるタスク（タスク終端の params の sha256 が初めて異なる t）を出す。継ぎ目を最初に踏んだ時点の無料の読み出しになる。

**既存の量（ラベルに使わず報告する）**
- 箱 B:
  - C.measure の行（`zbar_inv`, `sigma_inv`, `cnorm`, `pos_frac`, `rowmean`, `bias`, `w2col` …、C:167-229）
  - gate_block（`gbar` = mob, `cov`, `q10_med`, `hard_dead`, `gcorr` …、GS:137-168）
  - EG の `sat`（φ′ < 0.05）と `dead_units`。腕ごとに意味が変わる（§1.2）ことを表に明記する
  - 幅の指数 p_late = ln cnorm を ln t に回帰した t60–120 の傾き（elu_growth_0909 V1 と同じ）
  - 欠損項の恒等式 E = Σ(∂L/∂a₁)h(z)。t20,40,…,120 に訓練 4096 枚（`manual_seed(20260909+seed)`、学習系列と独立、EG:121-123）で計る。式は EG:56-68 を写し、h だけ §1.2 と §1.5 の腕別の表にする（`EG.measure` は呼ばない）
    - **誤った h（変異対照 E_pred_wrong）は腕ごとに固定する。** EG:66 の `wrong = LR if ELU/Snake else ELU(1.0)` はクラスで分岐するので、キメラには意味を持たない。占有された範囲で自腕の h と違う h を選ぶ:

      | 腕 | E_pred_wrong に使う h |
      |---|---|
      | LR・LR02・双子・VMAX・SMINH・SMINS・GN・FN | ELU1 の h |
      | ELU1・SMAXH・VMIN・SMAXS・GD・FD | LR の h（≡ 0。E_pred_wrong = 0 なので対照は \|E\|/r_scale） |

    - 走ごとに EG:137-138 と同じく `identity_rel < 1e−6` と `identity_mutctl > 1e−4` を assert する。LR・LR02・双子では EG:139-141 と同じく E ≈ 0（`leaky_zero < 1e−6`）も assert する。
    - 誤った h でも `identity_mutctl ≤ 1e−4` になる走（自腕の h と誤った h が、占有された範囲でほとんど同じ値になる）は、pass にせず「対照を実証できない」と provenance に記録する。
- RL: `evaluate_rl` の全列
- SCR: submerged 率（zmax ≤ 0）、z̄ の中央値、mob の中央値、alive、strict_dead、zmin、w_free、moments（edge_law の記録器が既に書く）
- 前向き値の交絡表: 全腕の late 窓の E[φ]・E[h]・床・h(−∞)

## 4. 判定（事前登録）

### 4.0 共通規則
- **seed ごとに算出し、束ねない。** CI も有意差も出さない。
- **箱 B と RL（3 seed）**
  - **1 つの対比 c の状態**（endpoint ごと）。4 つは定義で排他で、**D-par・D1–D3・仮説判定・D5a・D6・D7・D9 のラベルはすべてこの状態から組む**。ラベルの条件に「|·| ≤ 2m が 3/3」とあれば EQ を、「> m が 3/3」とあれば DIR を指し、帯の条件を単独では使わない。
    - `DIR+`: 3/3 seed で c > m(c)
    - `DIR−`: 3/3 seed で c < −m(c)
    - `EQ`: 3/3 seed で |c| ≤ 2m(c)、かつ DIR± でない
    - `UNRES`: 他
  - **方向ラベル**（HELPS/HARMS/BEATS/BELOW など）は DIR± から、**等価ラベル**（INERT/EQUALS/TIE/ADDITIVE/ON_LINE/FLAT）は EQ から作る。1 m 帯では真の効果 0 でも 0.683³ = 0.32 しか出ず、反証側の結果（NEITHER など）が構造的に出にくくなるので、許容する誤りから帯を導いた（下の到達表）。
  - **co-primary の合わせ方**: 箱 B は L と −A_late、RL は pt と logit で、ラベルを endpoint ごとに別々に計算してから合わせる。
    - 合わせる単位は**状態ごと**である: 各対比の状態、N・D・I の状態、D1・D2 のラベル、4 つの単純効果の状態、仮説判定のラベル、D5a のチャネルの状態、D6 の腕ごとのラベルを、それぞれ合わせる。見出し（D3 の BOTH_FACTORS_HELP 等、D6 の家族見出し、D4、D8）は、合わせた後の状態から組む。
    - 両 endpoint が同じ → その状態
    - 一方が決まり（UNRES・`*_UNRESOLVED` 以外）、他方が UNRES → `<状態>_L_ONLY` / `<状態>_A_ONLY`（RL は `_PT_ONLY` / `_LOGIT_ONLY`）
    - 両方が決まり、違う → `CO_PRIMARY_CONFLICT(L=…, A=…)`（RL は `PT=…, LOGIT=…`）
    - 下流（見出し・仮説判定・D4・D6 の家族見出し・D8）では、`_*_ONLY` と `CO_PRIMARY_CONFLICT` をすべて UNRESOLVED として扱う。
  - **用量ゲート**（D0）: 用量が足りない対比の EQ は、注記ではなく `NOT_TESTABLE_*` に置き換える。DIR は残し、`(WEAK_DOSE)` を付ける。
- **到達表**（m が正しいノイズ尺度で、seed が独立な正規のとき）

  | ラベルの型（3/3） | 1 seed の規則 | 真の効果 0 | 1 m | 2 m | 3 m | 4 m |
  |---|---|---|---|---|---|---|
  | 方向 | c > m | 0.004 | 0.125 | 0.596 | 0.933 | 0.996 |
  | 等価（本 spec・EQ） | \|c\| ≤ 2m かつ DIR± でない | 0.865 | 0.553 | 0.085 | 0.0015 | 0.000 |
  | 帯の条件だけ（参考） | \|c\| ≤ 2m | 0.870 | 0.593 | 0.125 | 0.004 | 0.000 |
  | 等価（旧 1 m 帯・参考） | \|c\| ≤ m | 0.318 | 0.109 | 0.004 | 0.000 | 0.000 |

  - 等価ラベルは 3 m 以上の効果を 0.0015 で否定する。2 m の効果は 0.085 の確率で等価に入る。
  - 組み合わせ: 両因子が真に 0 のとき `NEITHER` は約 0.75（0.865²）、N = 3m・D = 0 のとき `NEAR_ONLY` は約 0.81（0.933 × 0.865）。
  - **`CANCELLATION_SUPPORTED` の検出力**（4 つの単純効果がすべて同じ大きさ δ のとき。相関 0 と完全相関の間）: δ = 2m で 0.13–0.60、3m で 0.76–0.93、4m で 0.98–1.00。箱 B では big 側の 2 つが既知なので、small 側の 2 つだけで 2m 0.36–0.60、3m 0.87–0.93。相殺検定と co-primary の一致がさらに下げる。
  - 推定 SE そのものの揺れ（base 窓は 5 タスク）と ρ₁ の 0 での切り捨てで、実際の率は正規近似から数ポイントずれる。§3.2 の推定量そのものを LR の実測の窓 sd で模擬すると（S16 (ii)）、+3 SE の方向は L 0.892・−A_late 0.914（正規近似 0.933）、真の効果 0 の等価は L 0.828・−A_late 0.856（正規近似 0.865）になる（両窓の sd を 1 にそろえた模擬では、1 m 帯の 3/3 は 0.33–0.34）。
- **SCR（10 系列）**
  - 方向ラベルは ≥ 9/10 の符号一致（両側で null 下 2·11/1024 = 0.021）。
  - 系列内のノイズ（log10 U_5M の窓平均のデルタ法 SE: LR 中央値 0.28・E 0.23 dex）より、系列間の軌道のばらつき（E − LR の対の sd 0.83 dex）がはるかに大きく、複製検定の側が律速するので、系列ごとの m は置かない。
  - **等価ラベルは SCR では付けない**（それを言える帯は 1.28 × 0.83 ≈ 1.06 dex 以上になり、無意味）。したがって SCR には INERT が無く、NEITHER・NEAR_ONLY・DEEP_ONLY は起こりえない。**不活性による反証はできない。**
  - m を要る判定（D0 の用量、D4、D5b、D6）は `NOT_APPLICABLE_SCR`。
  - 機構量（submerged の対 sd 0.037、z̄ 0.30）には同じ 9/10 規則を当てる。
- **打ち切り**
  - 箱 B: `DIVERGED`（§2.2 の try/except）。その腕を含む家族のラベルはすべて `NOT_DETERMINED_DIVERGED`。
  - RL: `INCOMPLETE`（発散で break）。その腕を含む家族は `NOT_DETERMINED_INCOMPLETE`。
  - SCR: `NUMERIC_DIVERGENCE`・`NEAR_EOS`（→ §2.4 の退避枝）。窓の全記録が床 1e−16 なら `INCONCLUSIVE_EXACT_FIT` とし、その腕を含む対比を打ち切る。分離腕の発散はその分離だけを `NOT_DETERMINED_DIVERGED` にする。
- **RL の床**
  - **`AT_FLOOR`**: O_arm(t31–50) ≤ F_seed + 2.86 × 0.00026。
    - F_seed = 各タスクのラベルの最多クラス比率の t31–50 平均（`rl_labels` 系列から厳密に計算。seed 0/1/2 で 0.11450 / 0.11400 / 0.11383）。
    - 0.00026 = 0906 の崩壊腕 R（10 seed × 20 タスク）の (online_t − 最多比率_t) のタスク sd 0.00116 / √20。R の online は自分の seed の最多比率に相関 0.94–1.00 で追随し、他の seed の最多比率とは −0.59〜0.58（隣の seed だけなら −0.26〜0.25）。
    - 2.86 = Φ⁻¹(1 − 0.05/24) = 2.865。族の 24 = RL で床規則を当てる 8 腕（3 家族の LR・ELU1・キメラ 6）× 3 seed。
      - 段 1 だけの族は H 家族 4 腕 × 3 = 12 腕seed で 2.64（双子を入れた 15 走なら 2.71）になる。段 2 を足しても段 1 のラベルが変わらないように、段 1 から 24 で固定する。双子と分離腕（RL 段 2 の任意の GN・FN）は 2×2 の家族に入らず床規則を当てないので、族に数えない。
      - 固定の影響は閾値の 0.00026 × (2.865 − 2.638) = 0.00006 で、LR の O − F（0.68–0.70）に比べて無視でき、向きは床と判定しやすい側（保守側）である。
  - **4 腕のうちどれかが、どれかの seed で AT_FLOOR なら、その家族の N・D・I とそれらの等価・交互作用ラベルはすべて `NOT_TESTABLE_FLOOR`。** 床は切り詰めなので、I = ½(Y_big − Y_LR) が見かけの SUPERADDITIVE を作る。
  - 床にある腕を含む 2 腕の差には「床より上」の向きしか付けない（`<A>_ABOVE_FLOOR_ARM`。A が 3/3 で床より上であることを要求）。

### D-par（両親・環境ごと・記述）
Δ = Y_ELU1 − Y_LR（co-primary）:
- `PARENTS_TIE`: Δ が EQ
- `PARENTS_ELU_BETTER`: Δ が DIR−
- `PARENTS_LR_BETTER`: Δ が DIR+
- 他は `PARENTS_UNRESOLVED`（endpoint の合わせ方は §4.0。一方の endpoint だけ決まれば `_L_ONLY`・`_A_ONLY` 等、食い違えば `CO_PRIMARY_CONFLICT`）
- RL で ELU1 が床なら `PARENTS_LR_ABOVE_FLOOR_ARM` か UNRESOLVED。
- SCR は符号 9/10 で ELU/LR_BETTER、それ以外は UNRESOLVED。

**D-par は仮説判定に使わない**（UNRESOLVED を「並んでいる」に数えない）。両親の差は D3 の相殺検定の中で使う。既知の値: 箱 B は L で UNRESOLVED・A_late で ELU_BETTER（§0.4-3）、SCR は UNRESOLVED（§0.4-9）。

### D0 操作チェック（家族 × 因子 × 環境 × 層）
各因子が操作する帯: H 近傍 = B1、H 深部 = B2∪B3、V 近傍 = B1∪B2、V 深部 = B3、S = z < 0 全体。

- **存在**
  - 箱 B・RL: Γ_X^F > 2·SE_Γ（§3.3 のタスク系列の SE）が全 seed で成り立つ。
  - SCR: 操作する帯の質量が 32 パターン × 100 ユニットのうち 1 セル以上（≥ 1/3200）ある系列が ≥ 9/10、かつ Γ_X^F > 0。
  - 成り立たなければ、深部因子は **`NOT_TESTABLE_JOINT_UNVISITED`**、近傍因子は `NOT_TESTABLE_NO_CONTRAST`。V の深部には P[z < z_v]（家族 4 走の平均）を併記する。
- **用量の妥当性**（箱 B・RL。等価ラベルの前提・H を含む全家族）
  - **用量は、EQ を読む対比ごとに、その対比に入る走の状態の上で測る**（層 l ごと・late 窓・seed の中央値）。
    - 主効果 X ∈ {N, D, I}: 4 走平均の Γ_X^{F,e,l}（§3.3）。
    - 単純効果（2 走の差 Y_P − Y_Q。D1・D2 の EQUALS、D3 の仮説判定の等価側、RL の N|deep）: Γ_{PQ}^{e,l} = E[φ′_P − φ′_Q] を P と Q の 2 走の状態で平均したもの（近傍 1 か深部 0.1 を持つ側を P にとり、正にそろえる）。4 走平均の用量は使わない。SMINH が沈むと、small を含む対の用量だけが小さくなりうるからである。
  - **ρ_min は co-primary の endpoint ごとに作る。**
  - **基準（箱 B の H 以外: e ≠ 箱 B、または F ≠ H）**: ρ = max_l Γ^{F,e,l} / Γ^{H,箱B,l}（同じ対比・同じ層）。
    - 箱 B の H でその対比が DIR± なら ρ_min = 2·m̃ / |c̃|（箱 B H の seed 中央値）。箱 B の効果を用量に比例して縮めても、まだ 2m を超える用量である。EQ か UNRES なら ρ_min = 1（箱 B と同じ用量以上）。
    - 箱 B の他家族は endpoint ごとに（L の状態には L の ρ_min、−A_late には −A_late の ρ_min）当てる。RL は endpoint が箱 B と対応しないので、max(ρ_min^L, ρ_min^{A}) を pt と logit の両方に当てる。
  - **箱 B の H**: 上の比は ρ ≡ 1 で自己参照になり、小さいが安定な Γ でも存在の検査だけで等価を読めてしまう。そこで、G1 で既知の軌道に固定される big 側の単純効果を錨にする。
    - 錨: K_N = Y_LR − Y_big（近傍）、K_D = Y_ELU1 − Y_big（深部）と、それぞれの 2 走の状態で測った用量 Γ_{K_N} = E[φ′_big − φ′_LR]、Γ_{K_D} = E[φ′_big − φ′_ELU1]。
    - 同じ因子の small 側の単純効果（近傍 Y_small − Y_ELU1・深部 Y_small − Y_LR）と主効果 X について、ρ = Γ̃ / Γ̃_{K_X}、ρ_min = 2·m̃ / |K̃_X|（m̃ は判定する対比の m、K̃_X は endpoint ごとの seed 中央値。I は N と D の大きい方の ρ_min を使う）。K_X がその endpoint で DIR でなければ ρ_min = 1。
    - 意味は「既知の単純効果を用量に比例して縮めた予言が、判定する対比の 2m を超えるときだけ EQ を読む」で、閾値は m と既知の効果からの算術で決まる。§1.4 の事前の反実仮想値（H 近傍 .044・深部 .058）は ρ の横に併記するが、ゲートには使わない（キメラ自身の状態で用量が動くことが問題の中心なので）。
    - 既知の単純効果 K_N・K_D 自身の EQ は採点しない（§0.4-1）。
  - ρ ≥ ρ_min でなければ、その対比の EQ を `NOT_TESTABLE_WEAK_MANIPULATION`（深部因子は `NOT_TESTABLE_JOINT_UNVISITED`）に置き換える。DIR は残し `(WEAK_DOSE)` を付ける。
  - 仮定: 効果が用量に比例し、環境をまたいで効果/ノイズ比が移る（§9）。
  - SCR は `NOT_APPLICABLE_SCR`（等価ラベルが無い）。
- 層ごとの状態を verdict.csv に出す。因子の判定は「どれかの層で通る」で行う。
- 通れば `MANIPULATION_OK`。
- 変異対照（S16 (iv)。committed の gate_shape_0911 の LR・ELU1・ELUF の late 状態（ガウス近似）と合成の Y の上で）
  - 継ぎ目を z = −40 に置いた擬キメラ（占有された範囲では親と同一）が `NOT_TESTABLE_*` になること。この状態では ELU1 のユニットの深い裾のため存在は通り（Γ ≈ 0.002 > 2·SE_Γ）、落とすのは用量である（ρ ≈ 0.03–0.04）。
  - 深部因子の継ぎ目を z = −8 に置いた擬 big（z < −8 だけ傾き 0.1、他は ELU1）が、存在は通るが（Γ ≈ 0.016）用量で `NOT_TESTABLE_JOINT_UNVISITED` になること（ρ ≈ 0.30–0.32 に対し、ELU1 − ELUF の既知の効果からの ρ_min は L で約 0.5、−A_late で約 0.4）。存在が落ちたら「対照を実証できない」と記録し、pass にしない。

### D1 大キメラ（家族 × 環境・co-primary）
2 つの対比 a = Y_big − Y_LR、b = Y_big − Y_ELU1 の状態（§4.0）から組む。下の 7 つは状態の定義で排他である。
- **`BIG_BEATS_BOTH`**: a が DIR− かつ b が DIR−（Y_big < Y_LR − m かつ Y_big < Y_ELU1 − m が 3/3）。
- `BIG_BELOW_BOTH`: a が DIR+ かつ b が DIR+。
- `BIG_BETWEEN`: (a, b) = (DIR−, DIR+) または (DIR+, DIR−)。DIR は 3 seed で同じ符号を要るので、両親の並び（どちらの親の側に近いか）は全 seed で同じ向きになる。seed ごとに親の順序が入れ替わる場合は BETWEEN にしない。
- `BIG_EQUALS_BOTH`: a が EQ かつ b が EQ（近傍・深部の両方の対の用量ゲート）
- `BIG_EQUALS_LR`: a が EQ（対 (big, LR) の用量ゲート）で b が EQ でない／`BIG_EQUALS_ELU1`: b が EQ（対 (big, ELU1) の用量ゲート）で a が EQ でない
- 他 `BIG_UNRESOLVED`
- 例: (small, LR, ELU1, big) = (0, 0, 0, −2)、m = 1 は a・b が DIR−（EQ にはならない）なので `BIG_BEATS_BOTH` だけになる（改訂 2 の字義では EQUALS_BOTH も同時に満たした）。
- **RL**: ELU1 がどれかの seed で AT_FLOOR なら、`BIG_BEATS_BOTH` は `BIG_BEATS_BOTH_FLOOR_PARENT`（Y_big < Y_LR − m が 3/3 かつ big が 3/3 で床より上）に置き換える。

### D2 小キメラ（co-primary）
D1 と同じ組み方で、a = Y_small − Y_LR、b = Y_small − Y_ELU1 の状態から作る（7 つは排他）。
- **`SMALL_BELOW_BOTH`**: a・b がともに DIR+（Y_small > Y_LR + m かつ > Y_ELU1 + m が 3/3）。
- `SMALL_BEATS_BOTH`（ともに DIR−）／`SMALL_BETWEEN`（(DIR+, DIR−) か (DIR−, DIR+)）／`SMALL_EQUALS_BOTH`（ともに EQ）／`SMALL_EQUALS_LR`（a が EQ・b が EQ でない。対 (small, LR) の用量ゲート = 深部因子）／`SMALL_EQUALS_ELU1`（b が EQ・a が EQ でない。対 (small, ELU1) の用量ゲート = 近傍因子）／`SMALL_UNRESOLVED`
- endpoint の合わせ方は §4.0（例: L だけで `SMALL_BELOW_BOTH` なら `SMALL_BELOW_BOTH_L_ONLY`、−A_late だけなら `_A_ONLY`）。
- **RL の床**: small **または** ELU1 がどれかの seed で AT_FLOOR なら `NOT_TESTABLE_FLOOR`（床の腕を含む差には §4.0 の `<A>_ABOVE_FLOOR_ARM` しか付けないので、small が床なら `SMALL_BELOW_BOTH` にも到達しない）。

### D3 要因分解と仮説の判定（家族 × 環境・co-primary）
**状態**: N・D・I をそれぞれ、HELPS（DIR+）、INERT（EQ・用量ゲート）、HARMS（DIR−）、UNRESOLVED に分ける（§4.0。endpoint ごとに作ってから状態ごとに合わせ、見出しは合わせた N・D から組む）。SCR は HELPS/HARMS（9/10）と UNRESOLVED の 3 状態。

- **見出し（記述・主効果）**
  - `BOTH_FACTORS_HELP`（HELPS, HELPS）
  - `NEAR_ONLY`（HELPS, INERT）
  - `DEEP_ONLY`（INERT, HELPS）
  - `NEITHER`（INERT, INERT）
  - 他は `FACTORIAL_OTHER(N=…, D=…)`
- **交互作用**: `ADDITIVE`（I が EQ・両因子の用量ゲート）／`SUBADDITIVE`（I が DIR+: どちらか一方で足りる・冗長）／`SUPERADDITIVE`（I が DIR−: 両方そろって足し算以上に効く）／`INTERACTION_UNRESOLVED`。SCR は SUB/SUPER（9/10）か UNRESOLVED。

**仮説の判定（H 家族で決める。H は厳密な要因計画だから）**

箱 B・RL（endpoint ごとに下の表でラベルを出し、§4.0 の規則で合わせる。SUPPORTED と REFUTED は両 endpoint で一致したときだけ立つ）:
- (i) **4 つの単純効果**: Y_small − Y_LR、Y_small − Y_ELU1、Y_LR − Y_big、Y_ELU1 − Y_big がすべて DIR+（= `SMALL_BELOW_BOTH` ∧ `BIG_BEATS_BOTH`）。
- (ii) **相殺検定**: 各 seed で |Y_ELU1 − Y_LR| + m(Y_ELU1 − Y_LR) < min(Y_small − Y_ELU1, Y_small − Y_LR)。両親の差が、(a) 単独と (b) 単独の利点のどちらよりも、m 以上小さい。
- (iii) 交互作用が `SUPERADDITIVE`（I が DIR−）でない。

| ラベル | 条件 |
|---|---|
| **`CANCELLATION_SUPPORTED`** | (i) ∧ (ii) ∧ (iii) |
| `PARTIAL_CANCELLATION` | (i) は成り立つが、(ii) がどれかの seed で落ちるか、(iii) が落ちる（どちらかを注記） |
| **`CANCELLATION_REFUTED`** | 4 つの単純効果のどれかが、EQ（方向でなく \|·\| ≤ 2m が 3/3・その対の用量ゲートを通る）か DIR−（逆向き） |
| `CANCELLATION_UNRESOLVED` | 他（合わせた結果の `_L_ONLY`・`_A_ONLY`・`_PT_ONLY`・`_LOGIT_ONLY`・`CO_PRIMARY_CONFLICT` を含む） |

- **排他性**: SUPPORTED と PARTIAL は (i)（4 項すべて DIR+）を要り、REFUTED はどれかの項が EQ か DIR− であることを要るので、状態の定義で同時には立たない。改訂 2 の字義では、4 つの単純効果がすべて 1.5 m で両親が等しい結果が、(i)–(iii) を満たし（SUPPORTED）、同時に「|·| ≤ 2m が 3/3」で REFUTED も満たしていた。
- 批評の例 Y = (small, LR, ELU1, big) = (0, 0, 0, −2) は、small の単純効果が EQ なので `CANCELLATION_REFUTED` になる（旧版では SUPPORTED だった・S16）。
- **箱 B の注記**: big を含む 2 項と両親の差は既知（§0.4-1〜3）。判定は `(SMINH_ONLY)` を付けて出し、採点は Y_small を含む項だけで行う。
- **分離腕の注記**: D5a が `N_VIA_FORWARD` なら `_NEAR_VIA_FORWARD`、`D_VIA_FORWARD` なら `_DEEP_VIA_FORWARD` を付ける。2×2 の並びは立っても、その因子について Issa の機構（ゲート）は立たない。
- **RL の床**: 4 腕のどれかがどれかの seed で AT_FLOOR なら `CANCELLATION_NOT_TESTABLE_RL_FLOOR`。
  - ELU1 と small が 3/3 で AT_FLOOR、かつ big と LR が 3/3 で床より上なら、見出しを **`DEEP_DOMINATES_FLOOR`** とし、近傍を非飽和の行だけで読む: N|deep = Y_LR − Y_big で `NEAR_HELPS_GIVEN_DEEP`（DIR+）／`NEAR_INERT_GIVEN_DEEP`（EQ・対 (big, LR) の用量ゲート）／`NEAR_HURTS_GIVEN_DEEP`（DIR−）／`NEAR_GIVEN_DEEP_UNRESOLVED`。
  - 仮説判定は `DEEP_DOMINATES_FLOOR` でも `CANCELLATION_NOT_TESTABLE_RL_FLOOR` とする。
- **SCR**（相殺検定は m が無いので置けない）
  - `CANCELLATION_DIRECTIONAL_SUPPORT`: 4 つの単純効果がすべて予言の向きに ≥ 9/10。
  - `CANCELLATION_DIRECTIONAL_AGAINST`: どれかの単純効果が逆向きに ≥ 9/10。
  - 他は `CANCELLATION_UNRESOLVED`。
  - SCR の判定は最大でも「向きの支持」で、相殺（両親の差が小さいこと）も不活性による反証も言えない。

### D4 箱の一貫性（H 対 S・箱 B と RL・2 回目の判定器）
**因子ごと**（X ∈ {N, D}）に、H と S の D3 の合わせた状態 (s_H, s_S) から決める。
- 前提: X が H と S の両方で D0（存在と用量）を通る。一方でも通らなければ `BOX_CONSISTENCY_NOT_TESTABLE(X)`。
- 因子ごとの真理表（UNRESOLVED には `_*_ONLY`・`CO_PRIMARY_CONFLICT` を含む）:

  | s_H ＼ s_S | HELPS | HARMS | INERT | UNRESOLVED |
  |---|---|---|---|---|
  | HELPS | `CONSISTENT(X)` | `SHARPNESS_DEPENDENT(X)` | `SHARPNESS_DEPENDENT(X)` | `UNRESOLVED(X)` |
  | HARMS | `SHARPNESS_DEPENDENT(X)` | `CONSISTENT(X)` | `SHARPNESS_DEPENDENT(X)` | `UNRESOLVED(X)` |
  | INERT | `SHARPNESS_DEPENDENT(X)` | `SHARPNESS_DEPENDENT(X)` | `CONSISTENT_INERT(X)`（S の用量 ≥ H の用量を D0 で確認した場合だけ。でなければ `UNRESOLVED(X)`） | `UNRESOLVED(X)` |
  | UNRESOLVED | `UNRESOLVED(X)` | `UNRESOLVED(X)` | `UNRESOLVED(X)` | `UNRESOLVED(X)` |

  - 方向と INERT の組を食い違いに数えるのは D8 の `ENV_DEPENDENT` と同じ扱い（両方決まっているときだけ比べる）。
- 家族の見出し（因子ごとのラベルを併記する）:
  - どちらかの因子が `SHARPNESS_DEPENDENT` → **`SHARPNESS_DEPENDENT`**（一方の因子が一致していても、食い違いを優先する）
  - 両因子が `CONSISTENT` か `CONSISTENT_INERT` → **`BOX_CONSISTENT`**（両因子とも `CONSISTENT_INERT` なら `BOX_CONSISTENT_INERT`）
  - 他（一方が一致し他方が UNRESOLVED か NOT_TESTABLE を含む） → `BOX_CONSISTENCY_UNRESOLVED`
- S の因子は帯について純粋でない（§1.2 の漏れ約 10%）。`SHARPNESS_DEPENDENT` には、帯別 Γ（B1/B2/B3）で H と S の用量を並べた表を必ず付け、「鋭さ」と「帯の漏れ」を区別しないまま読まない。
- SCR は `NOT_APPLICABLE_SCR`。RL は段 2。

### D5 経路 —— ゲートか前向きの値か
#### D5a 分離腕（箱 B・SCR・段 1）
近傍: 端点 A = LR、B = SMAXH、ゲート腕 G = GN、前向き腕 F = FN。深部: A = ELU1、B = SMAXH、G = GD、F = FD。

箱 B（endpoint ごとに作り、§4.0 の規則で合わせる）:
- 前提: 単純効果 Y_A − Y_B > 2m が 3/3。成り立たなければ `NOT_TESTABLE_NO_SIMPLE_EFFECT`。
- **分離腕の操作ゲート**（分離腕自身の走の late 状態で測る。D0 は 4 走の家族にしか当たらないので、ここで別に置く）
  - G（ゲート腕）: Γ^G = E_{G の状態}[φ′_bwd,G − φ′_A]（層ごと。近傍では B1、深部では B2∪B3 にしか差が無い）。存在は Γ^G > 2·SE_Γ（§3.3 のタスク系列の SE）が全 seed。用量は ρ_G = Γ̃^G / Γ̃^B、Γ^B = E_{B の状態}[φ′_B − φ′_A]（B 端点が実際に届けたゲートの差）。ρ_min = 2·m̃(Y_G − Y_A) / |Ỹ_A − Ỹ_B|（endpoint ごと・seed 中央値）。
  - F（前向き腕）: Φ^F = E_{F の状態}[φ_fwd,F − φ_A] で、存在は |Φ^F| > 2·SE_Φ、用量は ρ_F = Φ̃^F / Φ̃^B（Φ^B = E_{B の状態}[φ_B − φ_A]）、ρ_min は G と同じ形。Ψ は報告する（F の h = zφ′_A − φ_fwd なので、Ψ^F = −Φ^F で別の情報を持たない）。
  - 例: GN（前向き LR・逆向き SMAXH）が LR と同じ深さまで沈むと、ゲートの差は B1 にしか無く、そこに質量がほとんど無ければ、ゲートの差を届けないまま LR に並び、見かけの `N_VIA_FORWARD` を作る。このゲートがそれを止める。
  - ゲートが落ちたチャネルの STAYS は `NOT_TESTABLE_WEAK_MANIPULATION` に置き換える（MOVES は残し `(WEAK_DOSE)` を付ける）。
- **チャネルの状態**（ch ∈ {G, F}。対比 Y_A − Y_ch の §4.0 の状態から）
  - MOVES: Y_A − Y_ch が DIR+（B の側に動く）
  - STAYS: Y_ch − Y_A が EQ（方向でなく |·| ≤ 2m が 3/3）かつそのチャネルの操作ゲートを通る
  - 他（逆向きの DIR、UNRES、ゲート落ち）: その他
- ラベル（チャネルの状態の組で決まるので排他）:
  - **`X_VIA_GATE`**: G が MOVES、F が STAYS
  - **`X_VIA_FORWARD`**: F が MOVES、G が STAYS
  - `X_BOTH_CHANNELS`: G と F がともに MOVES
  - `X_NEEDS_BOTH`: G と F がともに STAYS（どちらか単独では出ない）
  - 他は `X_CHANNEL_UNRESOLVED`（ゲート落ちが原因なら `NOT_TESTABLE_WEAK_MANIPULATION` を併記）
  - 改訂 2 の字義では、m 単位で A = 0、B = −3、G = −1.5、F = 0 が VIA_GATE と NEEDS_BOTH を、G = F = −1.5 が 4 つすべてを同時に満たしていた。STAYS が方向でないことを要るので、今は前者が `X_VIA_GATE`、後者が `X_BOTH_CHANNELS` だけになる。
- ゲート側の取り分 (Y_A − Y_G)/(Y_A − Y_B) と前向き側の取り分を seed ごとに併記する。
- 到達: 真にゲートだけで効果が 3m のとき（腕ごとの独立なノイズ、m = √2·σ_arm の模擬）、前提と合わせた `X_VIA_GATE` は約 0.51（前提の Φ(1)³ = 0.60 が律速する）。前提を満たした条件付きなら約 0.81（(0.977 × 0.954)³）。
- 変異対照（S16 (iv)）: 逆向きの差を z < −40 にだけ置いた擬 GD（φ′_bwd = e^z（z ≥ −40）、0.1（z < −40））の Γ^G がゲートで `NOT_TESTABLE_WEAK_MANIPULATION` になること。

SCR（log10 U と SUB_ の両方で・向きだけ）:
- 前提: sign(Y_B − Y_A) が ≥ 9/10 で揃う。揃わなければ `NOT_TESTABLE_NO_SIMPLE_EFFECT`。
- 分離腕の存在: G・F それぞれの走の状態（32 パターンの厳密支持）で、操作する帯の質量が ≥ 1/3200 の系列が ≥ 9/10、かつ Γ^G > 0（F は Φ^F ≠ 0 で B と同符号）。落ちたチャネルを「動かない」側に数えず、そのラベルを `NOT_TESTABLE_WEAK_MANIPULATION` にする。
- `X_GATE_DIRECTIONAL`: G が A から B の向きに ≥ 9/10 動き、F は動かない（< 9/10）
- `X_FORWARD_DIRECTIONAL`: 逆
- `X_BOTH_DIRECTIONAL`／他 `X_CHANNEL_UNRESOLVED`

#### D5b V 家族の帯別予言（RL のみ・段 2）
- 箱 B と SCR の V は報告だけ（`REPORT_ONLY_V`）。登録時点で、箱 B の V 近傍は `NOT_TESTABLE_COLLINEAR`（§0.4-8）、SCR の V 深部は LR が z_v を踏まないので `NOT_TESTABLE_JOINT_UNVISITED` と分かっている。
- H の効果から、帯ごとの価値を推定する。H の近傍はゲートを B1 だけで、H の深部は B2∪B3 だけで動かすので:
  - e_{B1} = N^H / Γ_{N,B1}^H
  - e_{deep} = D^H / Γ_{D,B2∪B3}^H
- ゲートの予言（帯別）: p_g = e_{B1}·Γ_{X,B1}^V + e_{deep}·(Γ_{X,B2}^V + Γ_{X,B3}^V)。「ゲートはどこにあっても同じ価値」とする全体版 X^H·Γ_X^V/Γ_X^H も seed_contrasts.csv に出すが、ラベルには使わない。
- 前向きの予言: 区間 P_f = [min, max]{X^H·Φ_X^V/Φ_X^H, X^H·Ψ_X^V/Ψ_X^H}
- **前提は H の分母だけに置く**: Γ^H・Φ^H・Ψ^H が D0（存在と用量）を通り、|N^H| > 2m かつ |D^H| > 2m。V 自身の D0 は要求しない（V の近傍の合計 Γ ≈ 0 は設計どおりである）。
- 全 seed で dist(p_g, P_f) ≥ 2m でなければ `NOT_TESTABLE_COLLINEAR`。
- 3/3 で |X^V − p_g| ≤ m かつ dist(X^V, P_f) > m → **`X_VIA_GATE_V`**
- 3/3 で dist(X^V, P_f) ≤ m かつ |X^V − p_g| > m → **`X_VIA_FORWARD_V`**
- 他は `X_CHANNEL_UNRESOLVED_V`。RL の床で H の N・D が NOT_TESTABLE なら `NOT_TESTABLE_FLOOR`。

### D6 ゲートの価値 —— 総ゲート量で足りるか（箱 B のみ・H と S・2 回目の判定器）
**主: leaky 線からの残差**
- 梯子 = 同じ seed の {LR001, LR, LR02, LR03, LIN}。Ḡ_arm = 層 1 の実現ゲート `gbar` の t101–120 窓平均。
- Ḡ_arm を挟む隣り合う 2 段 (lo, hi) と λ = (Ḡ_arm − Ḡ_lo)/(Ḡ_hi − Ḡ_lo) で、系列 c_t = acc_arm(t) − (1−λ)·acc_lo(t) − λ·acc_hi(t) を作る。残差 r_arm を co-primary の両方で §3.2 の規則で出す（Σw² = 1 + (1−λ)² + λ²）。
- Ḡ_arm が梯子の範囲外（どれかの seed）なら、その腕は `NOT_TESTABLE_OUTSIDE_LADDER`。
- 腕ごと（endpoint ごとに作り §4.0 で合わせる）: `<ARM>_ON_LEAKY_LINE`（r が EQ）／`<ARM>_BELOW_LEAKY_LINE`（r が DIR−: 同じ総ゲート量の leaky より損失が少ない）／`<ARM>_ABOVE_LEAKY_LINE`（r が DIR+）／`<ARM>_LINE_UNRESOLVED`
- 家族の見出し（合わせた後の腕ラベルから）: **`GATE_MASS_ONLY`**（big と small がともに ON_LEAKY_LINE）／**`LOCATION_MATTERS`**（big が BELOW かつ small が ABOVE）／他 `GATE_WORTH_UNRESOLVED`
- 既知（4 段の梯子 LR001・LR・LR03・LIN の場合だけ）: 箱 B の SMAXH（= ELUF）は L で r = −0.81/−0.58/−0.01（UNRESOLVED）、−A_late で −0.74/−0.61/−0.65（BELOW）で、合わせると `SMAXH_BELOW_LEAKY_LINE_A_ONLY`。
- **登録した 5 段の梯子（LR02 を含む）では、H 家族の見出しは走る前には決まっていない。** ELUF の上側の相手が LR02 に替わると L の残差が動き、seed 2 の L で r < −m になれば SMAXH は両 endpoint で BELOW になり、`LOCATION_MATTERS` にも届きうる（§0.4-6）。S 家族は盲検。

**報告（ラベル無し）**
- 反実仮想版: c_t = N_t/Γ_N − D_t/Γ_D のタスク系列と §3.2 の SE。D − N ≡ Y_ELU1 − Y_LR を含むので、既知の両親の差でほぼ決まる（§0.4-6）。
- 実現版: 同じ式の Γ を実現 Ḡ の差に置き換えたもの（層別）。

### D7 機構（H 家族・箱 B と RL は層ごと・SCR・2 回目の判定器）
§3.4 の読み出しと層別で、読み出し R ごとに 2 つの対比 (c₁, c₂) の状態（§4.0。m は §3.4 の D7 の m）から次の規則でラベルを作る。
- 対比の向き: 近傍層では c₁ = R(big) − R(LR)、c₂ = R(ELU1) − R(small)（どちらも「近傍の傾き 1 − 0.1」）。深部層では c₁ = R(big) − R(ELU1)、c₂ = R(LR) − R(small)（どちらも「深部 0.1 − 飽和」）。
- 規則（どの読み出しでも同じ・排他）:
  - c₁・c₂ がともに DIR+ → `<R>_UP`、ともに DIR− → `<R>_DOWN`
  - ともに EQ → `<R>_FLAT`（FLAT を許す読み出しで、§3.4 の分解能ガードを通るときだけ。通らなければ `NOT_TESTABLE_LOW_OCCUPANCY`。FLAT を許さない読み出しでは EQ を UNRES と同じに扱う）
  - ともに決まっている（DIR± か、FLAT を許す読み出しでの EQ）が状態が違う → `<R>_MIXED`
  - どちらかが UNRES → `<R>_UNRESOLVED`
  - 共通 bin が無ければ `NOT_TESTABLE_NO_COMMON_BIN`
- 読み出し R（UP の意味と FLAT の可否）
  - (a) 近傍層
    - `NEAR_BAND_SINKING`（沈下ハザード。**`NEAR_BAND_SINKING_DOWN`** = 近傍 1 で沈みにくい）。FLAT なし。
    - `NEAR_BAND_STEP`（‖ΔW̃‖。**`NEAR_BAND_STEP_UP`** = 近傍 1 で歩幅が大きい）。FLAT あり。Adam の比（`adam_ratio`）は同じ規則で `NEAR_BAND_ADAM_*` として併記する（FLAT あり）。
  - (b) 深部層
    - `DEEP_BAND_VELOCITY`（深さ速度 v。**`DEEP_BAND_VELOCITY_UP`** = 深部 0.1 で上向きの速度が大きい）。FLAT あり。
    - `DEEP_BAND_RETURN`（戻り確率。`_UP` = 深部 0.1 で戻りやすい）。FLAT なし。
- **FLAT は距離や幅の因子を含まない読み出し（v・‖ΔW̃‖・Adam の比）にだけ付ける。**
- SCR: 緩和率の対比を 9/10 の符号で `DEEP_RELAX_FASTER`／`SLOWER`／`UNRESOLVED`。
- 読み方: D3 で N = HELPS なのに (a) が近傍層でどれも動かないなら、近傍の利点は近傍帯ユニットを経由していない（層 2・前向きの値・D5a を見る）。D についても同様。

### D8 環境をまたぐ一貫性（因子ごと）
H 家族の D3 の co-primary 状態を、因子（N, D）ごとに 3 環境で並べる。
- ある環境で因子が**決まっている**のは、合わせた状態が HELPS/INERT/HARMS（SCR は HELPS/HARMS）で、打ち切り・床（`NOT_TESTABLE_FLOOR`・`DEEP_DOMINATES_FLOOR` の D）・`NOT_TESTABLE_WEAK_MANIPULATION` でないとき。`(WEAK_DOSE)` の付いた HELPS/HARMS は決まっているに数える（用量が弱くても効果が出たことは言える）。`_*_ONLY` と `CO_PRIMARY_CONFLICT` は決まっていない。
- `CROSS_ENV_CONSISTENT`: 両因子とも ≥ 2 環境で決まり、決まった状態がすべて同じ。
- `ENV_DEPENDENT(X)`: ある因子 X が、ある環境で HELPS、別の環境で INERT か HARMS（両方決まっている）。**到達できるラベルの違い（SCR に INERT が無いなど）だけでは付けない。**
- 他は `CROSS_ENV_NOT_DETERMINED`。
- RL の N|deep（= N − I）は N と別の量なので、表に併記するが数えない。

### D9 lr への頑健性（段 2・箱 B・H 家族・因子ごと）
lr 5e−4 と 2e−3 のそれぞれで D3 の状態を計算し直す。
- 両因子が 3 つの lr すべてで決まり、同じ状態なら `PATTERN_LR_ROBUST`
- ある因子が、ある lr で HELPS、別の lr で INERT か HARMS なら `PATTERN_LR_DEPENDENT(X, lr)`
- 他は `LR_ROBUSTNESS_NOT_DETERMINED`

### 副（同じ規則で計算し、ラベルは付けるが主判定にしない）
- 箱 B: Issa の窓（t2–6 → t116–120）の D1–D3。主窓と食い違えば `WINDOW_DEPENDENT` を併記する。
- RL: 早期 online（t1–10）の D1–D3（初期ゲインの過渡と、蓄積した可塑性の喪失を分ける）。
- 箱 B・RL: t1 からのタスク系列 N_t・D_t・I_t と、キメラが親から離れる最初のタスク（§3.4）。
- SCR: 1M 窓の D1–D3。submerged 率と z̄ を Y にした D3 と D5a（見出しに `SUB_`・`ZBAR_` を前置）。

### 4.x 空虚になる条件とガード
| ラベル | 空虚になる条件 | ガード |
|---|---|---|
| 方向・等価 | 分解能 m が壊れている（0・NaN・巨大）、SE が軌道のずれを含まない | S16（合成系列で到達率を実測・NaN で例外）、null 双子の σ_traj（§3.2） |
| 仮説判定 | 既知のセルだけで決まる、並びの前提が必ず満たされる | 単純効果と相殺検定で定義（§4 D3）、箱 B は `(SMINH_ONLY)`、D-par を使わない |
| L のラベル | base の水準の違いだけで L が動く | co-primary（−A_late）の一致 |
| SMALL_BELOW_BOTH・N/D/I（RL） | 飽和列が床に張り付いて自明に悪い、切り詰めが見かけの交互作用を作る | 床の腕を含む家族は `NOT_TESTABLE_FLOOR`、logit と pt の一致 |
| D3・I | どこかのセルが打ち切り | 家族ごと `NOT_DETERMINED_*` |
| 等価ラベル（INERT・EQUALS・NEITHER） | 継ぎ目の先に質量が無い、操作が弱い | D0 の用量ゲート（全家族・層別）、z = −40 の擬キメラの対照 |
| D0 の存在 | 入力標本 SE が極小、SCR で SE = 0 | タスク系列の SE、SCR は 1 セルの質量の床 |
| 全ラベル | 方向と等価の条件が同時に成り立ち、2 つのラベルが並ぶ | §4.0 の対比の状態（DIR±/EQ/UNRES は排他）から組む・co-primary は状態ごとに合わせ、`_A_ONLY`・`CO_PRIMARY_CONFLICT` を定義 |
| 箱 B H の等価 | 用量の比が自己参照（ρ ≡ 1）で、tiny だが安定な Γ でも読める | 既知の big 側の単純効果と、その対の用量を錨にした ρ_min（D0）、z = −8 の擬 big の対照 |
| D5a | 単純効果そのものが小さい、分離腕が操作帯を踏まずに端点に並ぶ | 前提 Y_A − Y_B > 2m、分離腕自身の状態での存在・用量ゲート（STAYS を NOT_TESTABLE に置換）、z = −40 の擬 GD の対照 |
| D5b | 予言が重なる、V の合計 Γ ≈ 0 で分母が壊れる | 帯別予言、H の分母だけに D0、`NOT_TESTABLE_COLLINEAR` |
| D6 | 補間の範囲外、既知の big で見出しが決まる | `NOT_TESTABLE_OUTSIDE_LADDER`、腕ごとのラベル（5 段の梯子では big の L の残差も走る前には未知・§0.4-6） |
| D7 | 帯の組成が腕で違う、距離や幅が混ざる、1–2 ユニットの bin で m が大きく FLAT が自動で出る | 帯分率で層別・プール件数で重み、FLAT は距離・幅の無い量だけ、2m ≤ 3·m_all の分解能ガード（`NOT_TESTABLE_LOW_OCCUPANCY`）、MIXED/UNRESOLVED の明示 |
| 分離腕の読み出し・E 恒等式 | φ′ と h をどちらの関数から取るか曖昧、誤った h が自腕の h と同じで対照が空虚 | `dphi` = φ′_bwd・h = zφ′_bwd − φ_fwd（§1.5）、腕ごとの誤った h の表と「対照を実証できない」の記録（§3.4） |
| 打ち切りの経路 | 本走前に一度も通らない try/except | S20（非有限を注入して DIVERGED/INCOMPLETE と divergence.json・部分行数・G0.5 の行を確認） |
| D8/D9 | 検出力の差だけで「食い違い」になる | 因子ごと・両方決まっているときだけ比べる |
| G1 | 比較件数 0 で一致を報告する、切り詰めで早死にが通る | 件数ガード（箱 B 2040/走・RL 40 または 3000 セル・SCR はキー一覧と n_records = 5001、退避枝は 10001） |
| 判定器 | 本走の後に配列が足りないと分かる | G0.5（スモーク出力で全行を計算） |

## 5. 予測（記名・走る前）

| 判定 | 環境 | Claude の予測（確信） | 根拠 | Issa |
|---|---|---|---|---|
| D-par | 箱 B | **既知** L: `PARENTS_UNRESOLVED`、A_late: `PARENTS_ELU_BETTER`（採点しない） | G1 で軌道が固定される（§0.4-3） | |
| D-par | SCR | **既知** `PARENTS_UNRESOLVED`（E < LR 7/10）（採点しない） | G1-SCR（§0.4-9） | |
| D1 H | 箱 B | **既知** `BIG_BEATS_BOTH`（採点しない） | SMAXH ≡ ELUF（§0.4-1） | |
| D2 H | 箱 B | `SMALL_UNRESOLVED` 40%、`SMALL_BELOW_BOTH` 35%、`_L_ONLY` 15%、`SMALL_EQUALS_LR` 10% | 代理が割れている。ELU03（近傍 0.3）は L で LR と EQ・late 水準では両親の間に寄り、D2 を当てると `SMALL_EQUALS_LR_L_ONLY`（§0.4-4）。LR001（Ḡ が SMINH に近い約 0.19）は両 endpoint で両親より悪い。SMINH の近傍は ELU03 より弱く、Ḡ は LR001 に近い（LR の状態で .186） | **`SMALL_BELOW_BOTH`**（仮説どおり・Claude の最頻と異なる） |
| D3 H 見出し | 箱 B | `FACTORIAL_OTHER` 45%、`BOTH_FACTORS_HELP` 40%、他 15% | co-primary の一致が D を削る（ELU03 代理で D の −A_late は seed 2 で等価帯） | |
| 仮説 | 箱 B | `CANCELLATION_UNRESOLVED` 40%、`PARTIAL_CANCELLATION` 30%、`CANCELLATION_SUPPORTED` 15%、`CANCELLATION_REFUTED` 15% | 支持には SMINH が late で LR より 0.45–0.90 pt 悪いことが要る（§0.4-2）。LR001 代理は PARTIAL（両親の差 0.2–0.7 pt が small 側の利点と同程度） | **`CANCELLATION_SUPPORTED`**（仮説どおり・Claude の最頻と異なる） |
| D5a 近傍 | 箱 B | `N_VIA_GATE` 30%、`N_BOTH_CHANNELS` 25%、`N_CHANNEL_UNRESOLVED` 30%、`N_VIA_FORWARD` 15% | 回転率は ḡ だけに従う（elu_turn_0912 `GATE_ONLY`）。一方で ELUF は同じ幅で約 1.1 浅く、実現 Ḡ の上乗せは反実仮想の 2–3 倍（前向きのオフセットが駆動しうる経路） | **`N_VIA_GATE`**（仮説 (a) は「勾配 1 を保つ」なのでゲート経路。Claude は 30/30 で割れている） |
| D5a 深部 | 箱 B | `NOT_TESTABLE_NO_SIMPLE_EFFECT` 40%、`D_VIA_GATE` 30%、`D_CHANNEL_UNRESOLVED` 30% | ELU1 − ELUF は seed 2 で約 2.2 SE、σ_traj を足すと 2m を割りうる | |
| D6 SMINH | 箱 B | `NOT_TESTABLE_OUTSIDE_LADDER` 40%、`SMINH_ON_LEAKY_LINE` 30%、`SMINH_ABOVE_LEAKY_LINE` 20%、他 10% | SMINH の Ḡ は梯子の下端 LR001（0.18–0.22）付近 | |
| D6 S 家族（SMAXS・SMINS・見出し） | 箱 B | 予測なし（盲検） | S の実現 Ḡ を見積もる材料は §1.4 の反実仮想しかなく、H で実現 Ḡ の上乗せが反実仮想の 2–3 倍ずれた（§0.4-7）ので、梯子のどの段の間に入るかも決められない | |
| D0（因子 × 層） | 箱 B | H・S は近傍・深部とも `MANIPULATION_OK` 75%。V の近傍は合計 Γ が ≈ 0 で存在が落ち `NOT_TESTABLE_NO_CONTRAST` 60% | 親の late 状態で P(z < z_c) = 0.65–0.77、P(\|z\| < 1) = 0.06–0.12 と両帯に質量があり、H の用量は既知の単純効果の錨でも足りる見込み。V の近傍の合計は −.002〜+.006（§0.4-8） | |
| D0（因子 × 層） | RL | 深部因子 `MANIPULATION_OK` 80%、近傍因子 `NOT_TESTABLE_WEAK_MANIPULATION` か `NOT_TESTABLE_NO_CONTRAST` 55% | 床に向かう腕の層 2 の z̄ は −43〜−154 で深部帯は埋まるが、近傍帯の質量は P(\|z\|<1) = 0.02–0.04 と小さい | |
| D0 | SCR | 近傍・深部とも存在 `MANIPULATION_OK` 70%（V の深部は `NOT_TESTABLE_JOINT_UNVISITED` 既知） | 32 パターン × 100 ユニットで 1 セル以上の質量という床は低い。LR は z_v を踏まない（§4 D5b） | |
| D-par | RL | `PARENTS_LR_ABOVE_FLOOR_ARM` 65%、`PARENTS_UNRESOLVED` 20%、`PARENTS_LR_BETTER` 15% | ELU1 の AT_FLOOR の予測（70%）から。Codex の 80 epoch で ELU1 11.7% 対 LR 51.8%（§0.4-5） | |
| D2 H | RL | `NOT_TESTABLE_FLOOR` 70%、`SMALL_UNRESOLVED` 20%、他 10% | small（SMINH）と ELU1 のどちらかが床なら `NOT_TESTABLE_FLOOR`（D2 の床規則は「または」） | |
| D4 | 箱 B | `BOX_CONSISTENT` 55% | S は H より強い同じ向きの操作。帯の漏れは約 10% | |
| D7 (b) 深さ速度 | 箱 B 層 1 | `DEEP_BAND_VELOCITY_UP` 55% | z < z_c でゲートは 0.1 対 e^z（1〜10 倍）。ただし Adam がゲートの大きさを打ち消す | `DEEP_BAND_VELOCITY_UP`（仮説 (b) そのもの。Claude と同じ → 起草側の予測（Issa 承認）） |
| D7 (a) 沈下 | 箱 B 層 1 | `NEAR_BAND_SINKING_DOWN` 45% | 近傍 1 の腕は浅い（ELUF z̄_inv が LR より約 1.1 浅い） | |
| D7（全読み出し） | RL | 予測なし | RL の層別読み出しに前例が無く、ELU1・SMINH が床に張り付くと層の組成が 1 帯に潰れて、共通 bin の有無すら見積もれない | |
| D7 緩和率 | SCR | 予測なし | 緩和率（dzbar を zmax の距離に回帰）はこの形では初めて使う道具で、SCR の親どうしの値も未測。L = 0 の速度場は平均への回帰だった前例（ビー玉仮説の棄却）があり、傾きの向きの差を見積もる根拠が無い | |
| ELU1・SMINH の `AT_FLOOR` | RL | 2 腕とも 3/3 で AT_FLOOR（70%） | Codex 80 epoch の ELU は約 48k step で崩壊した。400 epoch では t2 前後にあたる。層 2 の z̄ は −43 → −154 | |
| D1 H | RL | `BIG_BEATS_BOTH_FLOOR_PARENT` 40%、`BIG_EQUALS_LR` は用量不足で NOT_TESTABLE 30%、他 30% | SMAXH の深部は LR と同じ 0.1。近傍帯の質量は P(\|z\|<1) = 0.02–0.04 と小さい | **`BIG_BEATS_BOTH`**（床に落ちる親があれば `_FLOOR_PARENT`。仮説どおり。床に落ちるか自体は仮説が言わないので、床の行は起草側の予測（Issa 承認）） |
| D3 H・仮説 | RL | 見出し `DEEP_DOMINATES_FLOOR` 65%、仮説 `CANCELLATION_NOT_TESTABLE_RL_FLOOR` 75% | 上の床予測から | |
| N\|deep | RL | `NEAR_HELPS_GIVEN_DEEP` 35%、等価側は用量ゲートで `NOT_TESTABLE_WEAK_MANIPULATION` 35%、`NEAR_HURTS_GIVEN_DEEP` 20%、UNRESOLVED 10% | (a) が働く質量がほとんど無い | |
| 退避枝の起動 | SCR | 起動する（SMAXH・SMAXS のどちらかが NUMERIC_DIVERGENCE か NEAR_EOS）55% | 親状態での lr·λ/2 = 0.98–1.96。境界 1 は算術 | |
| 分離腕の発散 | SCR | GN・GD のどちらかが発散 30% | BL の前例（前向きと逆向きの食い違い） | |
| D2 H | SCR | `SMALL_UNRESOLVED` 50%、`SMALL_BELOW_BOTH` 35% | SMINH の φ′ は E_a0p1（−1.33 dex、5/10 onset）より各点で大きい。悪化は 1 dex に届かない見込み | |
| D1 H | SCR | `BIG_UNRESOLVED`（60%、生存した場合） | 検出可能な効果は約 1 dex。親どうしは 7/10 | |
| 仮説 | SCR | `CANCELLATION_UNRESOLVED` 70%、`DIRECTIONAL_SUPPORT` 15%、`DIRECTIONAL_AGAINST` 15% | 上の二つから | **`CANCELLATION_DIRECTIONAL_SUPPORT`**（仮説どおり・Claude の最頻と異なる） |
| SUB_ D3 H | SCR | `SUB_BOTH_FACTORS_HELP` 55% | ゲートが高いと浅くなる。負の切片（c < 0）は浅くする方向（offset_grid）で、交絡も同じ向き | |
| D5a 深部（log10 U） | SCR | 前提が立たず `NOT_TESTABLE_NO_SIMPLE_EFFECT` 60%、`D_CHANNEL_UNRESOLVED` 25%、他 15% | 前提は ELU1 → SMAXH の符号が ≥ 9/10 で揃うこと。D1 H の SCR の予測（`BIG_UNRESOLVED` 60%）と、水準で検出できる効果が約 1 dex であること（§0.4-9）から、揃わない見込み | |
| SUB_ D5a 近傍 | SCR | `N_FORWARD_DIRECTIONAL` 35%、`N_BOTH_DIRECTIONAL` 25%、`N_GATE_DIRECTIONAL` 20%、UNRESOLVED 20% | SCR では定数オフセットだけで z̄ が 1.5 動く | |
| D8 | — | `CROSS_ENV_NOT_DETERMINED` 60%、`ENV_DEPENDENT` 25%、`CROSS_ENV_CONSISTENT` 15% | RL は床で N・D が切り詰められ、SCR は INERT を持たない | |

**Issa の欄の記入（追補 1・2026-09-14 03:10 JST）**: Issa の指示は「仮説どおり」。仮説が直接言う 5 行（箱 B の D2・仮説・D5a 近傍、RL の D1、SCR の仮説）と D7 (b) に、仮説から一意に決まるラベルを起草側が書き入れた。書いた時点で、実装は進行中・スモークも本走も未着手で、結果の数値は 1 つも読んでいない。仮説が言及しない行（D0・D-par・RL の床・D6・D4・D8・発散など）は空欄のままとし、規則どおり「起草側の予測（Issa 承認）」として独立に数えない。Issa と Claude の最頻が異なるのは箱 B の D2・仮説・SCR の仮説の 3 行（D5a 近傍は Claude が 30/30 で割れているので、独立に数える）。

**予測の重み**: Claude の通算は約 2.5/10、Issa の仮説は 9/5・edge_law・sign_ladder（`VALLEY_CAUSAL`）と当ててきた。**本 spec は Issa の仮説を主線として回す。** Issa の欄は結果を読む前に埋め、書いた時点（走の前か後か・何を見たか）を追補に記す。Issa の予測が Claude と同じなら「起草側の予測（Issa 承認）」と書き、独立に数えない。

**外れたときに第一に疑うもの**
1. **`SMINH_ON_LEAKY_LINE` と `N_VIA_FORWARD` が同時に出る** → 「二つの利点」は総ゲート量と前向きの値で説明がつく。平均ゲートの交絡（ρ = −0.81）とオフセットの前例がそのまま説明する。相殺の物語は弱い意味でしか立たない。
2. **`SMALL_BELOW_BOTH` が出ない** → SMINH が別の「そこそこ良い活性化」に着地しただけかもしれない（sign_ladder §5-1 の読み方）。floor −0.33 は ELU1 の −1 より浅い。late の z̄・σ・cnorm・dead を 4 腕で並べる。
3. **分離腕が発散する・両方の分離腕が端点のどちらにも並ばない** → 前向きと逆向きの食い違いそのものが害になっている。オフセット腕（§1.5 の代替）を追補で登録する。
4. **G1 が落ちる** → 機械・インタプリタ・torch の差（2.13.0+cu130、numpy 2.5.2 / 2.5.1 を確認）。**比較の許容を緩めるのは、endpoint を読む前に追補を書いた場合だけ**（sign_ladder §5-3 の前例）。G1-RL はもともと 10 桁の文字一致なので、「数値の許容誤差に緩める」追補は、10 桁一致が落ちた原因を 1 行も読む前に書く。

## 6. 検算（変異対照つき）と進捗ゲート
すべての検査に変異対照を付け、対照が実際に検査を落とすことを assert する（空虚な S 検査を通算 6 回作った履歴があるため）。許容値は演算回数と eps から導く（eps32 = 1.19e−7、eps64 = 2.22e−16）。対照は同じ許容値を超えなければならない。

| # | 検査 | 許容値の導出 | 変異対照（落ちなければならない） |
|---|---|---|---|
| S1 | 解析 φ′ と autograd（f32/f64、z ∈ [−30, 10] の 400,001 点、継ぎ目 ±4 ulp は除く）。8 関数と LR02。分離腕は腕名で除外（S17） | 枝あたり ≤ 4 演算 × 各 eps·\|φ′\| × 演算順で 2 倍 = 8·eps → f32 9.5e−7・f64 1.8e−15（実測 5.96e−8 / 1.1e−16） | SMAXH の深部傾き 0.11（0.0100）、SMINS の φ′ = 0.1σ(z)（0.052）、VMIN の継ぎ目を z_c に置く（0.100） |
| S2 | φ と ∫₀ᶻφ′（mpmath quad、[−40, 5] の 21 点。**積分区間を {0, z_c, z_v} で分割し、各小区間を quad にかける**。autograd には切片の誤りが見えないので必要） | 8·eps64·(1 + 0.1·40) = 8.9e−15（実測 2.2e−16。分割しないと角と跳びで 1e−10 級になる） | SMAXH の切片を ln 0.11 で作る（0.0095）、SMINH の深部を expm1 で書く（0.67）、VMIN の継ぎ目を z_c に置く（この 21 点格子 `linspace(−40, 5, 21)` の上で最大 0.58。z = z_c ちょうどでの上限は 0.67） |
| S3 | 継ぎ目の連続性（Z⁻ = nextafter(Z, −∞)、Z⁺ = nextafter(Z, +∞)。Z ∈ {0, z_c, z_v}、dtype ごと） | C¹: \|Δφ\| ≤ max\|φ′\|·(Z⁺ − Z⁻) + 4eps·max(1, \|φ\|)、\|Δφ′\| ≤ max\|φ″\|·(Z⁺ − Z⁻) + 4eps。C⁰: \|Δφ′ − J\| ≤ 同じ bound。J = 0.9（0: LR/SMINH/VMAX）、10/11（0: SMINS）、∓0.0999546（z_v: VMIN/VMAX）。**GS:309 の ELUF 連続性（±1e−9 の探針・<1e−8）は置き換える**（実測 2.0e−10 は 0.1 × 2e−9 そのもので、旧 spec の 1e−12 は正しい実装を落とす） | SMAXH の切片を ln 0.11 で作る（Δφ が bound を超える） |
| S3b | 継ぎ目ちょうど: 各 Z ∈ {0, z_c, z_v} の float32 表現と float64 表現で、z = Z での φ′ と autograd(φ) を比べる（φ と φ′ の述語が一致すること） | ≤ 8·eps·\|φ′\| + 1 ulp·max\|φ″\|（§1.3 の clamp の規則） | VMIN の φ′ の `>=` を `>` に替える: float32 の z = f32(z_v) で、正しい φ′ = e^z = 4.54e−5 に対し変異は 0.1 を返し、差が約 0.0999 になって落ちる（S3 の Z⁻/Z⁺ ではどちらの実装も同じ値を返すので、この対照は S3 ではなく S3b に付ける） |
| S4 | h の実装と z·φ′ − φ（f64 格子）。分離腕 4 本は z·φ′_bwd − φ_fwd（§1.5 の表）と比べる | 8·eps·(1 + \|z\|·\|φ′\| + \|φ\|)（実測 ≤ 1.3e−15） | SMAXH の h に ELU1 の h を使う（0.33）。GN の h に z·φ′_fwd − φ_fwd（≡ 0）を使う（z = −1 で 0.27）。未知の腕名で h 表が KeyError を投げること（EG.hdefect の 0 返しは使わない） |
| S5 | 極端な z（±1e4, ±1e3, −800, ±746, −745, ±200, −104, −103, −90, −50, ±1e−30, 0, +50）で φ・autograd 勾配・φ′・h が有限（f32/f64） | 有限であること | clamp を外した式（ELU `where(z>0, z, expm1(z))`、SMINH の深部 `exp(z)`、SMINS `0.1*log(0.1+exp(z))`、VMAX の深部 `expm1(z)`）が f32 z ≥ 200・f64 z ≥ 746 で NaN を出すこと |
| S6 | 要因計画の恒等式 φ_SMAXH + φ_SMINH − φ_ELU1 − φ_LR = 0（φ′ も）、V も同様（[−40, 10]） | 8·eps·(1 + \|z\|) | S 家族の残差（φ 最大 0.140・φ′ 最大 0.047）が許容値を超えること、C_SMINH を ln 0.11 で作った SMINH で落ちること |
| S7 | SMAXH の腕が `GS.ELUFloor(0.1)` の型そのものであり、§1.3 の参照式と 2M 点で bit 一致すること（φ・φ′・autograd、signbit も比べる）。ELU1 ≡ `EG.ELU(1.0)`、LR ≡ `H.ARMS['LR']`。**全クラスが `kind`（`'adaptive_snake'` 以外の文字列）を持つこと** | bit 一致・属性あり | SMAXS を ELUFloor と比べて不一致になること。`kind` を消した部分クラスで検査が落ちること |
| S8 | 新しい `make_act` が未知名で KeyError を投げ、新モジュールの AST に `EG.make_act`・`EG.train`・`EG.run`・`EG.hdefect`・**`EG.measure`** の参照が無いこと | — | `EG.make_act('SMINH', 0.)` が leaky を返すこと（罠が実在する実証）。`EG.measure` を 1 行呼ぶ一時ファイルを走査器が検出すること |
| S9 | 乱数の共有。箱 B: init・t1–3 の置換・層化 idx・バッチ順・プローブ idx の sha256 が全腕で一致（双子は init の 1 要素だけが 1 ulp 違い、他は一致。追補 2 で b1[k] に +1e−6 に変更）。RL: init・subset・t1–3 のラベル・t1 の epoch 置換。SCR: init W/b/v・教師・flip 系列・先頭 30k の入力の sha256 | 一致 | seed+1（箱 B・RL）、`generator_offset: 1`（SCR）で不一致になること |
| S10 | 測定が学習を変えない。箱 B: 新読み出しの有無で **2 タスク後**の params が maxabs 0.0。RL: **2 タスク後**（読み出しは task 1 の学習がすべて終わった後に入るので、効くのは task 2 から） | 0.0 | (i) 測定の後に W1 += 1e−9 する読み出しで > 0。(ii) RL: 読み出しの中で `torch.randperm(2, generator=g_batch)` を 1 回呼ぶ読み出しで、task 2 終端の差 > 0（乱数を消費する誤りの検出） |
| S11 | MNIST 4 ファイルの sha256 = gate_shape_0911 と 0906 の provenance の `data_sha256`。持ち込み 7 ファイルの sha256 = §2.3 の表 | 一致 | 1 byte を変えた scratch のコピーで不一致になること |
| S12 | H・V で Γ_N（½ 対比）= E[φ′_big − φ′_LR] と Γ_I = 0。帯分解で Σ_k Γ_{X,Bk} = Γ_X、Σ_k mob_k = mob（ユニット別） | 8·eps64·4（Γ）、n_inputs·eps64（帯） | S 家族の Γ_I（親の late 状態で 0.0052–0.0059。2Γ_I = 0.011–0.012）が許容値を超えること。B2 を落とした和が B2 の質量 > 0 のときに落ちること（B2 が空なら「対照を実証できない」と記録し、pass にしない） |
| S13 | SCR のクラス差し替え: 差し替えの前後で state_dict の bytes が一致し、既存名（leaky_relu, elu）の act_fn/act_grad/act_curv が `VecMLPL` と signbit まで bit 一致（[−30, 30]） | bit 一致 | 傾きを nextafter(0.1) にした部分クラスで不一致になること |
| S13b | SCR のキメラ 6 名と LR/ELU1: `ChimeraMLPL` の act_fn/act_grad/act_curv が、§1.3 の参照（SMAXH は `GS.ELUFloor(0.1)`、φ″ は §1.3 の表）と、float32（学習 dtype）と float64（記録器 dtype）の [−30, 10] 格子と S5 の極端点で signbit まで bit 一致（同じ式を逐語で置く） | bit 一致 | SMINH の act_grad の上限を 0.11 にした部分クラスで不一致になること |
| S14 | ループの写し: 本体を AST（`ast.get_source_segment` で関数本体の文）で取り出し、宿主との差分が**登録した置換（旧行 → 新行の対）と登録した挿入だけ**であること。箱 B は §2.2 の P1–P6、RL は §2.3 の R1–R3、SCR は `_run_arm_chimera` の 1 挿入。抽出した行数が、実装前に宿主から数えて固定した期待値に等しいこと（edge_law の `_body` は署名の終わりを `->` で探すので、`->` の無い GS.train では 0 行になり空虚に通る。`_copy_opcodes` は replace をすべて unregistered にする） | 差分 0・行数一致 | 写しの 2 つの登録ブロックの**間**の 1 文字を変えた source で差分が出ること。行数の期待値を 0 にすると落ちること |
| S15 | SCR の φ″（act_curv）と act_grad の autograd（f64、継ぎ目 ±4 ulp を除く） | 8·eps64 | SMAXS の φ″ = e^z（係数 0.9 を落とす） |
| S16 | 判定器の単体検査。(i) 陽性: committed の gate_shape_0911 の ELUF − LR が両 endpoint で `BEATS` 3/3（σ_traj = 0）。(ii) 合成系列: 各 seed・各窓（base 5・late 20）で LR の実測の窓 sd（seed 0/1/2 の base 0.226/0.378/0.516 pt・late 0.604/0.561/0.404 pt）の iid 正規ノイズを、腕ごとに独立に、全腕で共通の基準系列に足した 4 腕 × 3 seed を 4000 組作り、2 腕差の対比で、真の効果 0 のときの等価ラベル（EQ）の 3/3 の率と、+3 真の SE のときの方向ラベルの 3/3 の率を、endpoint ごとに測る。(iii) 床: 0906 の R seed 0–9 は全部 AT_FLOOR、LR seed 0–9 は全部そうでない。(iv) 用量と存在（D0 の変異対照）: 継ぎ目を z = −40 に置いた擬キメラと、深部の継ぎ目を z = −8 に置いた擬 big が `NOT_TESTABLE_*`（後者は存在を通って用量で落ちること）。D5a の分離腕ゲートでは z < −40 にだけ差を置いた擬 GD が `NOT_TESTABLE_WEAK_MANIPULATION`。SCR では z̄ をずらして操作する帯を空にした再構成で、存在の検査が落ちること。(v) 仮説の写像: 合成で Y = (small, LR, ELU1, big) = (0, 0, 0, −2) が REFUTED。small := 「LR の系列 + LR の実測の窓 sd の独立な正規ノイズ」が SUPPORTED にならない（ノイズを足さない small := LR は small − LR の系列が 0 になり (vi) の例外で止まるので、それを使わない）。(vi) m の計算で sd = 0 や NaN が例外を投げる。(vii) §0.4-1 の SE 表（LR − ELUF・ELU1 − ELUF の 3 seed × 2 endpoint）と §0.4-3 の ELU1 − LR の SE を、§3.2 の推定量で小数 2 桁まで再現する | (ii) の区間は、§3.2 の推定量そのもの（ddof = 1・Pearson の ρ₁ を 0 で切る・n_eff ≥ 1・上の窓 sd）の 10⁶ 組の模擬（`numpy.random.default_rng(20260914)`）の率を中心に、4000 組の ±3 二項 SE で広げたもの: **L の等価 0.828 [0.810, 0.846]・方向 0.892 [0.878, 0.907]、−A_late の等価 0.856 [0.839, 0.872]・方向 0.914 [0.900, 0.927]**。正規近似（0.865 / 0.933）と自由度 4 の t 近似はどちらも ρ₁ の切り捨てによる SE の水増しを表さず、正しい判定器を落としうるので区間に使わない。(vii) は 2 桁の一致 | LR と ELUF の腕名を入れ替えた入力でラベルが逆になること。(ii) で m を SE/2 にした変異が区間の外に出ること（模擬で等価 0.33–0.34・方向 0.976–0.979）。(iv) で用量ゲートを外した変異が INERT を返すこと。(vii) で ρ₁ を全体平均で中心化した ACF にした変異が ELU1 − ELUF の seed 2 の L で 0.19 を返して一致しないこと |
| S17 | 分離腕: 前向きが φ_fwd と bit 一致し、autograd の勾配が上流 × φ′_bwd と一致（f32/f64・S1 と同じ格子と許容値）。`dphi` が φ′_bwd を返し、h 表が z·φ′_bwd − φ_fwd を返すこと。SCR では `ChimeraMLPL` の分離腕名の act_fn = φ_fwd・act_grad = φ′_bwd・act_curv = φ″_bwd が bit 一致 | S1 と同じ（h は S4 と同じ） | 逆向きに φ′_fwd を使う変異（GN なら (z_c, 0) で差 e^z − 0.1、最大 0.9）で落ちること。`dphi` に φ′_fwd を返させた変異で落ちること |
| S20 | 打ち切りの経路（本走の前に一度は通す）。箱 B: t1–3 の短い走で task 2 の学習の後に W1[0, 0] = inf を注入し、`DIVERGED`・`pmnist/{ARM}_s{seed}_divergence.json` の最初の非有限タスク = 3・部分行 2 行・G0.5 の該当家族の行が `NOT_DETERMINED_DIVERGED` になること。RL: t1–2 の走で task 1 の後に同じ注入をし、`INCOMPLETE`・`rlmnist/{ARM}/s{seed}/divergence.json`・部分行 1 行・`NOT_DETERMINED_INCOMPLETE` | 状態・タスク番号・行数の一致 | 注入しない同じ走が `COMPLETE` になり divergence.json を書かないこと |
| S18 | 双子: init が LR と 1 要素だけ、ちょうど 1 ulp 違い、他の全配列と全乱数列が一致すること（追補 2 で「b1[k] の 1 要素だけ、float32 で +1e−6 に最も近い表現値」に変更。生存の判定 S18b は追補 2-2） | 一致・差は 1 要素 | 摂動 0 の「双子」を作ると検査が「差が 0 要素」で落ちること。本走後に、双子の acc 系列が LR と 1 タスク以上で異なることを assert する（揃っていたら σ_traj を `NOT_DETERMINED_TWIN`）。S18b の対照は、RL で死んだ画素の重み W1[0, 0] に +1e−6 を置いた「双子」が 1 タスク後に生きていないと判定されること（追補 2-2） |
| S19 | SCR の安定余裕: w_free から再構成した出力層の λ_out が、1M ckpt の状態から直接組んだ 2·E[φφᵀ] の固有値と一致。冪乗法の λ_full が小さい系列の `eigvalsh` と一致 | 8·eps64·λ（再構成）、冪乗法は反復の収束幅を記録 | 対角（m_phi2）だけから λ を取る近似が、非対角の効く状態で一致しないこと |

**進捗ゲート**
- **G−1**: 未決事項 1–12 に Issa が答えるか、既定を明示で受け入れるまで、実装の commit も起動もしない。
- **G0**: S1–S20 がすべて通るまで、どの環境も起動しない。インタプリタ・torch・numpy のバージョンを assert する（§2.1）。
- **スモーク（3 環境とも本走の前に済ませる）**
  - 箱 B: 全 17 腕 × t1–3。G1-PM の t1–3（LR・ELU1・SMAXH × s0–2 の 3 タスク分の配列と acc）。
  - SCR: 全 12 腕 × 30k step。G1-SCR の陽性の短縮比較（`chLR_1216`・`chE_1216` の 30k を、`ref_arm='LRnull_1216'`/`'Enull_1216'` と `ref_dir` を明示した s_null の規則で参照の先頭 31 記録と比べ、比べたキーの一覧が固定した期待一覧と一致し、両ログが 31 記録であることを別に assert する）。G1-SCR の変異対照（`generator_offset: 1` の 30k で `unfit[:31]` が不一致）。§2.4 の自前の rss_probe。
  - RL: G1-RL（GPU・LR s0 t1–2 の 40 セル）と新しい側の決定性、H 家族と双子の 1 タスク、CPU と GPU の 1 タスクの時間、並列 1・2・4・J での処理量（§7）。
- **G0.5（判定器の通し）**: `src/act_chimera_report_0913.py` を 3 環境のスモーク出力に端から端まで掛け、verdict.csv の全行（D-par, D0–D9, 副）が値を持つこと（`NOT_DETERMINED_*` は可）、KeyError も欠損配列による NaN も出ないこと、§8 のスキーマ（キー・dtype・形・層）と一致することを assert する。変異対照: 読み出しのキーを 1 つ消した出力で G0.5 が落ちること。
- **commit・push**: G0.5 を通った作業木を 1 回だけ commit・push し、本走はすべてその commit から回す。
- **G1-PM（bit）**: 本走で LR・ELU1・SMAXH × s0–2 を gate_shape_0911 と照合（1 走 2040 件）。対照（seed 0）: 初期値 W1[0,0] += 1e−3 で t1 まで学習し、不一致になること。
- **G1-SCR（bit）**: chLR_1216・chE_1216 × 10 系列が edge_law の null 腕と s_null の規則で一致し、10 系列すべてで n_records = 5001・state_hash_1m に到達。
- **G1-RL（10 桁の文字一致）**: §2.3。
- **G1 の失敗の扱い**
  - **G1-PM が落ちたら、原因が分かるまで SCR と RL を起動しない**（共通の活性化モジュールとループの写し方を疑う）。
  - G1-SCR・G1-RL の失敗は、その環境だけを無効にする。
- **G2**: 操作チェック（D0）。止めずに、ラベルの可否にだけ使う。
- **G3**: 発散・未完・床・NEAR_EOS。止めずに §4.0 の打ち切りに使う。SCR の NUMERIC_DIVERGENCE・NEAR_EOS（主 8 腕）だけは §2.4 の退避枝を起動する。
- **本走の後にコードを変える必要が出たら**: 新しい commit を作り、コード経路が変わった環境をすべて回し直し、`git_hash` を環境ごとに provenance に記録する。

## 7. 計算量・マシン・並列・順序・停止規則

**マシン**: すべて white-san で走らせる（i7-14700K 28 thread・RAM 30 GiB・RTX 5060 Ti 16 GB・torch 2.13.0+cu130）。箱 B と SCR の bit 参照はこの機械で作られている（RL は §2.3 の傍証）。lab・node・GCP では G1 が成立しないので使わない。

**時間の見積もり**（既存の実測から）
- **待ち行列の模型を 1 つに揃える。** CPU のジョブは並列 J の枠を埋めながら順に回るので、同じ長さ d のジョブ n 本は ⌈n/J⌉ 巡 × d で終わる（n·d/J は下限で、見積もりには使わない）。GPU は処理量が飽和するので、0906 の実測の処理量（単独の 1.93 倍）で割る。
- RL の CPU の 1 走の長さ d は、LR の 1,200 s（下端）から、1,440 s × キメラの上乗せ 1.16 = 1,670 s（上端）。

| 段 | 走 | 実測の根拠 | 見積もり |
|---|---|---|---|
| G0・スモーク | 各環境 | 箱 B 3 タスク × 17 腕、SCR 30k × 12、RL の G1-RL と 1 タスクずつ | 約 20–30 分 |
| 箱 B 段 1 | 51 走 | GS の腕 67–71 s/走（6 並列）。読み出しの追加で約 90 s | J = 6: ⌈51/6⌉ = 9 巡 × 90 s ≈ 13.5 分 |
| SCR 段 1 | 12 腕 × 10 系列 × 5M | Codex zero_attraction_learning_0913（この機械・9/13・edge_law runner・12 並列・19 ジョブ）で 899–1,751 s/腕、ピーク RSS 0.755–0.771 GiB（`~/Projects/claude/zero_attraction_0913/results/zero_attraction_learning_0913/launch_status.json` の `peak_rss_kib` 791,868–808,704） | 12 並列が J の範囲に入れば 1 巡 ≈ 15–29 分 |
| SCR 退避枝（条件付き） | 12 腕 × 10M | 上の約 2 倍、RSS 約 1.2 GiB（J が下がる） | 約 30–60 分 × 2 巡 |
| RL 段 1（CPU の場合） | 15 走 × 1.5M step | 箱 B の CPU 1 thread の step 単価から 1,200–1,440 s/走、キメラの φ の上乗せは CPU 実測で ELUF/LR 1.10・ELU1/LR 1.16 | J = 8: 2 巡 × 1,200–1,670 s ≈ 40–56 分、J = 6: 3 巡 ≈ 60–84 分 |
| RL 段 1（GPU の場合） | 15 走 | 0906: 単独 1,144 s/走、7 並列で処理量は単独の 1.93 倍に飽和（同時刻に offset_grid の SCR 28 腕が走っていたので CPU 競合で過小かもしれない）。キメラの上乗せ +0–30%（GPU は未測） | 15 × 1,144 / 1.93 × 1.0–1.3 ≈ 2.5–3.2 h（J = 6 でも J = 8 でもほぼ同じ） |
| 段 2（参考） | RL の V・S 12 走、箱 B lr 感度 24 走 | 上と同じ | RL CPU は J = 6・8 とも 2 巡 ≈ 40–56 分／GPU 12 × 1,144 / 1.93 × 1.0–1.3 ≈ 2.0–2.6 h、箱 B は ⌈24/6⌉ = 4 巡 × 90 s ≈ 6 分 |

段 1 の合計（各行の和・退避枝を除く）は、RL を CPU で回せば約 89–157 分 = **1.5–2.6 h**（下端は J = 8・SCR 15 分・スモーク 20 分、上端は J = 6・SCR 29 分・スモーク 30 分）、GPU なら約 3.3–4.4 h。

**並列数（コア数では決めない・OOM でデスクトップごと落とした履歴）**
- 各段のスモークで、プロセスごとのピーク RSS を `/usr/bin/time -v` で実測する。RL は GPU メモリも実測する。SCR は `edge_law_0905.rss_probe` で本走サイズに外挿する（短縮走行の RSS は桁で過小評価になる）。
- **ジョブのキューで回す**: 1 ジョブ = 1 (腕, seed)（SCR は 1 腕）。**各ジョブを起動する直前に** `MemAvailable` を読み、J = ⌊MemAvailable / (1.5 × RSS_peak)⌋ − 1 を計算し直し、実行中の数が J 未満のときだけ起動する。
  - 1.5 は edge_law の `RSS_HEADROOM`（`src/edge_law_0905.py:85`）。−1 は見積もりを外したジョブが 1 本あっても溢れないための余裕。
  - 腕ごとに seed を順に回す固定のプロセス配置はしない（J が 1 本足りないと 1 腕が枠を得られず、手で J を上げる誘惑が OOM の型になる）。
- **RL の並列数はスモークの処理量で決める**: 並列 1・2・4・J で 1 タスクの処理量（step/s の合計）を測り、最良の処理量の 90% 以内に入る最小の並列数を使う。0906 では 7 並列でも単独の 1.93 倍にしかならなかったので、GPU では 3–4 を超えて増やしても RAM（1.2–1.6 GB/proc）と OOM の危険が増えるだけになりうる。壁時計の見積もりはこの実測で置き直す。
- 上限: 箱 B 6（GS の前例）、SCR 12（9/13 のこの機械の前例）、RL は上の規則。
- 参考実測: 箱 B 914 MB/proc（Mnist 読込後）、RL 1.2–1.6 GB/proc（0906 の monitor）、SCR 0.755–0.771 GiB（= 0.81–0.83 GB。5M・12 並列・上の launch_status.json）。
- 起草時点（9/14 00:45）: MemAvailable 19.3 GiB、SwapFree 2.0 GiB。RSS 1.6 GiB なら J = ⌊19.31/2.4⌋ − 1 = 7、J = 8 には RSS ≤ 1.43 GiB が要る。
- 改訂 3 の時点（9/14 02:02・`/proc/meminfo`）: MemAvailable 18.0 GiB、**SwapFree 2.1 GiB（SwapTotal 8.0 GiB 中 5.9 GiB 使用）**。この値では RSS 1.6 GiB で J = ⌊18.03/2.4⌋ − 1 = 6、J = 8 には RSS ≤ 1.34 GiB が要る。（改訂 2 に併記した「7.6 GiB 中 5.3 GiB 使用」は SwapFree 2.0 と算術が合わず、SwapTotal も 8.0 GiB だったので削除した。）
- swap がほぼ尽きているので、起動時のガードだけでは実行中の他セッションやデスクトップの増加を防げない。
- **段は同時に走らせない**（箱 B と SCR の同時で約 15 GB、SCR と RL の同時で約 23 GB > 空き 19 GB）。

**他セッション**
- 各段の前に、他セッションの学習プロセス（`ps` の PID とコマンド行）と `nvidia-smi` の compute プロセスを provenance に書く。
- 起草時点では学習プロセスは無く（nvidia-smi はデスクトップ描画 1.4 GB のみ・load 0.35）、Codex の zero_attraction_learning_0913 は 9/13 23:15 頃に完了、zero_attraction_analysis_0913 は 00:43 に書き込み中だった。run id もファイルパスも act_chimera_0913 と衝突しない。
- **SCR・RL の時間帯を事前に Issa に伝え、その間に Codex が 12 並列の走を起動しないようにしてもらう。**

**順序**
1. G−1（未決への回答）→ 実装
2. G0（静的検査、数十秒）
3. 3 環境のスモーク（箱 B → SCR → RL の順・同時にしない）
4. G0.5 → commit・push
5. 箱 B 段 1（51 走）。G1-PM を確認する。箱 B を先に置くのは、**ループの写し・読み出しの挿入・`make_act` の振り分け**を約 13 分で検証するためである（錨のある 3 腕はどれも host のクラスを使うので、新しい 5 キメラの式そのものは G1 を通らない。式は S1–S7・S13b が検査する）。
6. SCR 段 1（→ 条件付きで退避枝）
7. RL 段 1（キュー）
8. 判定器の 1 回目（段 1 のラベル）→ 2 回目（D4・D5b・D6・D7）
9. 段 2 は Issa の承認後に同じ commit から（コードを変えるなら §6 の規則）

**停止規則**
- S 検査・G0.5 の失敗 → 起動しない。
- G1-PM の失敗 → 箱 B を無効とし、SCR と RL も原因が分かるまで起動しない。G1-SCR・G1-RL の失敗 → その環境だけを停止し、無効として報告する。
- 発散 → §4.0 の打ち切りで記録し、救済しない。SCR は §2.4 の条件で退避枝へ進む。
- `MemAvailable` < 1.5 × RSS_peak → 次のジョブを起動しない。実行中のジョブは殺さない。止めるときは PID を指定し、`pkill -f` は使わない（自分のシェルを殺す）。
- 時間上限は assert にしない（固定の上限が正しい走を殺した教訓）。スモークから外挿した時間の 3 倍を超えたら、監視役が警告を出すだけにする。

## 8. 出力

```
results/act_chimera_0913/
  checks/static_checks.json            S1–S20 の値・許容値・対照値、G0.5 の結果
  pmnist/{ARM}_s{seed}_rows.csv / _units.npz / _provenance.json    段 1（units は既存の 16 キーだけ・将来の bit 錨として commit・約 1.4 MB/走）
  pmnist/{ARM}_s{seed}_readout.npz     新しい読み出し（退避・下のスキーマ）
  pmnist/{ARM}_s{seed}_divergence.json （DIVERGED の走だけ・§2.2）
  rlmnist/{ARM}/s{seed}/per_task.csv / provenance.json / divergence.json    1 ジョブ = 1 (腕, seed) で並列に書くので seed ごとに分ける。readout_s{seed}.npz は退避
  rlmnist/{ARM}/per_task.csv           判定器が seed ごとの csv を結合したもの（行数を seed ごとに 50、INCOMPLETE なら打ち切りまでと照合する）
  scr/arm_status/ / provenance.json / s_null.json / s_pair.json    logs/*.npz・ckpts/・readout_{ARM}.npz は退避
  scr_lr0p005/                         （退避枝を回した場合のみ）
  stage2/                              （段 2 を回した場合のみ）
  verdict.csv                          ラベル（D-par, D0–D9, 副）× 環境 × 家族 × 層
  seed_contrasts.csv                   seed ごとの対比・SE・σ_traj・m・Γ/Φ/Ψ（帯別・層別・全体版と帯別版の p_g）・占有・床・lr·λ/2
  summary.md / fig_*.png / report_provenance.json
  backup_manifest.json                 source・backup・bytes・sha256
```

**読み出しのスキーマ（G0.5 が照合する・凍結）**

| ファイル | キー | 形 | dtype | 層 |
|---|---|---|---|---|
| 箱 B `_readout.npz`（T = 120, U = 100） | `fn_mom`（8 関数 × 帯 B0–B3 × {φ′, φ, h} の (unit, 入力) 平均） | (T, 2, 8, 4, 3) | float64 | 1, 2 |
| | `occ_start`, `occ_end`（帯分率） | (T, 2, U, 4) | float32 | 1, 2 |
| | `mob_band`（自腕の帯別ゲート） | (T, 2, U, 4) | float32 | 1, 2 |
| | `pabs1`（P[\|z\|<1]） | (T, 2, U) | float32 | 1, 2 |
| | `zbar_start`, `zbar_end` | (T, 2, U) | float64 | 1, 2 |
| | `depth_vel`（v_i） | (T, 2, U) | float64 | 1, 2 |
| | `step_num`（‖ΔW̃_i‖）, `wt_norm`（‖W̃_i‖）, `mu_comp`（層 2 の μ̂ 成分） | (T, 2, U) / (T, 2, U) / (T, U) | float64 | 1, 2 / 2 |
| | `adam_ratio`, `adam_epsfrac` | (T, 2, U) | float32 | 1, 2 |
| | `gbar_l2`, `mu_phi1` | (T, U) | float64 | 2 |
| | `E_ident`（E, E_pred, E_pred_wrong @ t20…120） | (6, 3) | float64 | 1 |
| | `param_sha`（タスク終端 params の sha256） | (T,) | `<U64` | — |
| RL `readout_s{seed}.npz`（T = 50, 入力 1200） | 箱 B と同じキー（`E_ident` を除く） | 同上 | 同上 | 1, 2 |
| SCR `readout_{ARM}.npz`（判定器が logs から作る・T = 500, S = 10） | `fn_mom` (T, S, 8, 4, 3) f64、`occ` (T, S, U, 4) f32、`mob_band` (T, S, U, 4) f32、`lam_out` (T, S) f64、`lam_full` (3, S) f64、`relax_rate` (S,) f64 | | | 1 |

- 分離腕の走も同じキーを持つ（`mob_band`・`gbar_l2` は `dphi` = φ′_bwd で計算・§1.5。D5a の操作ゲートは `fn_mom` から計算する）。
- **サイズ予算**（非圧縮）: 箱 B の readout ≤ 4 MB/走（見積もり 2.90 MB）、RL ≤ 2 MB/走（1.21 MB）、SCR ≤ 24 MB/腕（見積もり 19.9 MB = `fn_mom` (500, 10, 8, 4, 3) f64 3.84 MB + `occ` と `mob_band` (500, 10, 100, 4) f32 各 8.0 MB）。改訂 2 の SCR 12 MB はこのスキーマで守れなかった。SCR の readout は退避するだけで commit しないので、dtype を落とさずに予算の方を上げる。
- commit するのは rows・provenance・既存 16 キーの units・集計済みの csv だけで、readout は退避する（新しい読み出しを units に足すと、段 1 の箱 B 51 走で約 150 MB になる）。
- **provenance**（全走）: run id、arm、seed、env、lr、`git_hash`（走る前に push した commit。rebase・amend しない。本走後にコードを変えたら環境ごと）、code/spec/host の sha256（箱 B は H・GS・C・EG、RL は H05・RL と持ち込み 7 ファイル、SCR は nets・edge_law・gate_dial、config、参照 npz 20 本）、data の sha256、`sys.executable`・torch/numpy のバージョン、**hostname・CPU 型番・GPU 名・device**（EG と GS の provenance には無かったので足す）、wall 秒、ピーク RSS、他セッションのプロセス、checks（G1 の値と件数、対照値）。
- **`--analyze-only` は provenance を書き直さない。** 判定器は自分の `report_provenance.json`（report の sha256・読んだ入力の sha256）を別に書く。
- **片付け**（CLAUDE.md §4・結果を commit した日のうちに）
  1. `git -C <worktree> status --porcelain --ignored --untracked-files=all` で git 外のファイル（readout、SCR の logs/ckpts、スモーク）を洗い出し、`__pycache__` 以外を `~/Projects/obsidian-research-data/act_chimera_0913/` へ移す。`results/act_chimera_0913/backup_manifest.json` を commit する。
  2. `git -C <worktree> fetch origin` → `merge origin/main` → `push origin HEAD:main`（断られたら fetch からやり直す）。
  3. **main に入った後の proj_004_drift**: main に `src/pmnist_0905.py`・`src/pmnist_rlmnist_0906.py`・`results/pmnist_rlmnist_0906/{LR,R}/*`・`analysis/pmnist_0905/verdict_rlmnist.py` が入ると、proj_004_drift にある同名の未追跡コピーのせいで `git pull` が止まる（内容が同一でも git は未追跡ファイルを上書きしない。scratch で確認済み）。
     - 未追跡コピーの sha256 が commit した blob と一致することを確かめる。
     - Issa の OK を得てから（他セッションがそこから import している）、コピーを退避して proj_004_drift を fast-forward できるようにする。
     - L2Init のセッション（33a0cab は同じ未追跡の木の `analysis/pmnist_0905`・`results/pmnist_rlmnist_0906`・`results/_checks_pmnist_rlmnist_0906` をさらに持つ）と調整する。
  4. `git -C ~/Projects/claude/proj_004_drift merge-base --is-ancestor claude/act_chimera_0913 origin/main && echo OK` で OK が出てから、`worktree remove`・`branch -D`・`push origin --delete`。
- vault には結果ノートを 1 本新しく作り、関連ノートへは 1 行リンクだけ足す（並走セッション衝突の規則。編集前に `git log` を見る）。

## 9. 限界（結果の前に明記）
- **箱 B の仮説判定は SMINH 一つで決まる。** H 家族の big・両親は既知結果の再現で、検定ではない。箱 B で新しいのは SMINH・S 家族・分離腕・V（報告だけ）である。RL と SCR は、SCR の D-par（既知・§0.4-9）を除いて盲検である。
- 3 seed（箱 B・RL）の検出力は §4.0 の到達表の通りで、2 m の効果は方向ラベルを 0.60 でしか得ない（推定 SE の揺れを入れた模擬ではさらに数ポイント低い・S16 (ii)）。「支持されない」は「効果が無い」を意味しない。SCR の水準は約 1 dex 未満の差を見分けられず、等価も相殺検定も持たない。
- σ_traj は LR の双子から推定し、全腕に同じ値を使う。腕ごとに軌道の混沌の強さが違えば m を誤る。RL の σ_traj は 3 個の双子の対比からの粗い推定である。
- 分離腕は前向きと逆向きが食い違う学習で、勾配流ではない。「ゲートだけ」「前向きだけ」の腕が悪いことには、食い違いそのものの害が混ざりうる。
- V は箱 B と SCR では報告だけ。V の採点は RL の段 2 だけで、そこでも床に阻まれうる。
- D5b と D0 の用量ゲートは、効果が操作量（Γ・Φ・Ψ）に比例するという一次近似に立つ。比例しない応答（閾値・飽和）は UNRESOLVED か NOT_TESTABLE 側に落ちる。用量ゲートは、効果/ノイズ比が箱 B から他の環境に移ると仮定する。
- D6 の leaky 線は実現 Ḡ（内生的）に対する補間で、梯子の段が粗い（LR02 を足しても 5 段）。範囲外の腕は判定しない。LR02 を足すと、既知の ELUF の残差も補間の相手が替わって動く。
- 箱 B の H の用量ゲートは、既知の big 側の単純効果が small 側の対の効果/用量比を代表すると仮定する（交互作用 I があると崩れる）。
- 前向きの値（E[φ]・E[h]・床）は、H・V・S のすべてで近傍因子と同じ向きに動く。分けるのは分離腕だけで、分離腕は箱 B と SCR にしか無い（RL は段 2 で任意）。
- 全環境で lr は段 1 では一点。感度は段 2 の箱 B の H 家族にしか無い。RL と SCR の結論は、その lr に条件付く。
- RL は ELU1 に錨が無い。400 epoch での崩壊は未測。CPU で回した場合、RL の本走は 0906 と bit で比べられない（写したループは G1-RL で GPU 上だけ錨づけする）。
- 占有と操作量の事前値（§1.4、§0.4）は親の軌道の反実仮想で、キメラ自身の分布ではない。
- 箱 B は層 1 の既存の読み出しに錨があるが、層 2 の新しい読み出しには錨が無い（非侵襲性は S10 と層 1 の G1 で担保する）。
- 本走は相関でなく活性化関数への介入だが、φ の形を変えると幅・深さ・h・平均出力が同時に動く。媒介の分離（クランプ）は本 spec の範囲外で、分離腕が分けるのは「ゲート対前向きの値」の一段だけである。

## 未決事項（実装の前に Issa に決めてほしいこと・G−1）
1. **leaky の傾き a = 0.1 と ELU α = 1 で固定してよいか。** 継ぎ目 z_c = ln a と z_v は a で決まる。箱 A の gate_dose の標準値も a = 0.1（`configs/gate_dose_0830.yaml:51`「単一値。掃かない」）。
2. **seed 数と規則。** 既定は箱 B・RL とも 3 seed・3/3。
   - 検出力（1 対比・§4.0 の到達表）: 3 seed・3/3 は 2 m で 0.60、3 m で 0.93（null 下の偽ラベル 0.008）。5 seed・≥ 4/5 にすると 2 m で 0.82、3 m で 0.995（null 下 0.0055）、等価ラベルは 3 m の効果を 0.003 で否定する。
   - 対応のある設計なので、seed を増やすなら同じ seed の全腕が要る。箱 B で seed 3–4 を足すと +34 走（J = 6 で ⌈34/6⌉ = 6 巡 × 90 s ≈ 9 分。LR・ELU1・SMAXH の seed 3–4 には錨が無い）。RL は +10 走（CPU は J = 6・8 とも 2 巡 ≈ 40–56 分／GPU 10 × 1,144 / 1.93 × 1.0–1.3 ≈ 1.7–2.1 h）。
3. **主 endpoint。** 既定は箱 B で L と −A_late の一致を要求する co-primary、RL で pt と logit の一致。L 単独に戻すか。Issa の窓（t2–6 → t116–120）は副。
4. **ゲートと前向き値の分離の方法。** 既定は surrogate の分離腕 4 本（箱 B・SCR）。SCR では BL（ReLU 前向き + leaky 逆向き）が発散した前例がある。代替は定数オフセット腕（§1.5）、または両方（箱 B +12 走・SCR +4 腕）。
5. **RL の device。** 既定は「スモークで CPU と GPU の 1 タスクの時間を測り、段 1 全体で速い方に全腕を載せる」。CPU なら約 40–84 分（J = 8 で 2 巡、J = 6 で 3 巡）で済むが、0906 とは bit で比べられない（G1-RL は GPU の別検査・任意で LR s0 の t1–50 まで広げる約 19 分）。GPU なら LR を 0906 に 10 桁で錨づけでき、約 2.5–3.2 h かかる。§2.3 の「400 epoch にする理由」のうち「LR を 0906 の行で錨にできる」の意味がこれで変わる。
6. **段 1 と段 2 の線引き。** RL の V・S 家族（D5b・D4 の RL）、箱 B の lr 感度（D9）を段 2 に回した。どれかを段 1 に戻すか。層局所 H 腕は写したループの登録ブロックで実装できないので段 2 から外した（§2.1）。回すなら追補で登録ブロックと S 検査を足すか。
7. **null 双子の本数。** 既定は箱 B 4 本 × 3 seed、RL 1 本 × 3 seed。RL を 2 本にすると +3 走。
8. **SCR の地平線。** 5M だけにした。15M は ELU が LR を 10/10 で抜く地平で、費用は約 3 倍、RSS は約 2 GB/腕。
9. **SCR の退避枝**（lr 0.005・10M・12 腕すべて）を、非有限だけでなく安定余裕（lr·λ/2 ≥ 1）でも起動する規則で事前に承認するか。起動したら SCR の主ラベルは全家族で lr 0.005 から取る。
10. **箱 B の扱い。** SMAXH を採点外にし、仮説判定を「SMINH で決まる」と明記する扱いでよいか。箱 B の Issa の予測欄は D2・仮説・D5a・D6 の SMINH から採点を始める。
11. **RL の持ち込み 7 ファイル**（host 2 本・0906 の LR/R の per_task.csv と provenance・verdict_rlmnist.py。33a0cab と同一 bytes）を本ブランチで commit してよいか。L2Init 側のセッション（`exp/shell_l2_rlmnist_0913`）に知らせるか。main に入った後、proj_004_drift の同名の未追跡コピーを退避してよいか（§8 片付け 3）。
12. **Issa の予測欄**（§5）。結果を読む前に。

**知らせ（決定は要らない）**
- `~/Projects/claude` 直下に、CLAUDE.md §5 に載っていない Codex の worktree が 3 つある（`elu_response_anchor_0913`・`zero_attraction_0913`・`zero_attraction_analysis_0913`）。`wt/` の外にある。本 spec では触らない。
- edge_law の provenance（arm_status）には interpreter が記録されていない。SCR は docstring の `.venv/bin/python` を使う（§2.1）。

## 追補 2（2026-09-14 08:45 JST・スモークの後・本走の前・endpoint は 1 つも読んでいない）

スモーク（`results/_smoke_act_chimera_0913/`）で G0・箱 B・SCR・G0.5 は通り、**RL の G1 変異対照が空虚で落ちた**。原因を調べる途中で、null 双子の摂動も空虚または丸めで吸収されることが分かった。以下を本走の前に変える。

### 2-1 RL の G1 変異対照（停止の理由）
- **事実**: 対照は初期値 W1[0, 0] に +1e−3 を置いていた（§2.3 は要素を指定していなかった）。MNIST の画素 0 は訓練 60,000 枚すべてで 0（常に 0 の画素は 67 個）で、RL は入力を置換しないので、W1[0, 0] は出力にも勾配にも効かない。対照の走は params の sha が変わったのに、per_task.csv の 20 セルがすべて一致した（0/20）。
- **変更**: RL の対照の摂動は **b1[0] に +1e−3** とする。層 1 のバイアスは全標本で生きている。合格条件（1 セル以上の不一致）は変えない。
- 箱 B の対照（W1[0, 0] +1e−3）は入力置換で生きており、実測で 8/11/11 腕が不一致になったので変えない。

### 2-2 null 双子の摂動
- **RL**: W1[k, 0] は 2-1 と同じ理由で死んでいる。LRtw0 の per_task.csv は t1–2 で LR と 20 列すべて一致した。
- **箱 B**: W1[k, 0] は生きているが、1 ulp は float32 の丸めで吸収されうる。t3 終端で双子 12 本のうち 5 本の params の sha が LR と一致した。学習は決定論なので、一度一致した双子は以後ずっと一致する。
- **変更**（箱 B・RL 共通）: 双子 k の摂動を **b1[k] に +1e−6（絶対値）** とする（W1 は触らない）。
  - 1e−6 は、初期値 |b1| ≤ 1/√784 = 0.0357 での float32 の ulp（≤ 3.7e−9）の約 270 倍、|b| が 1 まで育っても ulp（6e−8）の約 16 倍で、丸めでは消えない。
  - Adam の 1 step（lr 1e−3）の 0.1% なので、null の摂動として扱える。
- **S9・S18 の文言**: 「init が LR と 1 要素（b1[k]）だけ違い、差が float32 で +1e−6 に最も近い表現値であること」。対照（摂動 0 の双子で「差が 0 要素」になり落ちる）はそのまま。
- **生きているかの判定（新・S18b）**: 各タスク終端で、摂動した要素を除いた params の sha を LR と比べ、最初に食い違ったタスクを `twin_live_task` として記録する。endpoint の窓の最初のタスク（箱 B t16・RL t31）までに食い違わない双子は「死んだ双子」として σ_traj から除く。
- **σ_traj のラベル**: 生きた対が 2 未満なら `NOT_DETERMINED_TWIN(M_SE_ONLY)`。スモークの報告は、生きていない対を含む n_pairs = 2・σ² = 0 で `SIGMA_TRAJ_OK` を出していた。生きた対が 2 以上で推定量の切り詰めにより σ² = 0 になった場合は、`SIGMA_TRAJ_OK` のまま備考に `clipped` を書く。

### 2-3 RL の device と並列（§7・未決 5 の既定規則の適用）
- スモークの実測（1 タスク）: CPU 15.1–18.1 s、GPU 20.0–25.9 s。50 タスクに直すと CPU 755–905 s、GPU 単独 1,000–1,300 s。
- CPU の処理量は並列 1/2/4 で 2,039 / 4,014 / 7,318 step/s（4 で 3.59 倍）。→ **RL の全腕を CPU で回す**。G1-RL は §2.3 のとおり GPU の別検査として回す。
- 並列数は、測った {1, 2, 4} の中で最良の 90% 以内に入る最小の 4 とし、上限は J 規則で、RSS は CPU の実測 1.02 GiB を使う（GPU の 1.73 GiB は使わない）。
- 見込み: 箱 B 51 走 ≈ 15 分、SCR 12 腕 ≈ 20–30 分、RL 15 走 ≈ 65 分、G1-RL の GPU 検査 ≈ 25 分。

### 2-4 実装で spec の字面から変えたところ
実装のレビュー後の報告にある 27 項（1–27）と、スモークの報告の 4 項（S4–S7）を、下の表に記す（記入は 2-1〜2-3 の修正と同時に行う）。どれも判定規則の意味は変えず、数値の許容・配列の形・検査の置き場所を実測に合わせたものである。2-1・2-2 を実装するときに決めたこと（A1–A8）も同じ表に足す。その実装のレビューで直したこと（A9–A11・A7 と A8 の追記）も足す。行番号は A9–A11 を直した後の作業木のもの（commit 前）。

| # | 項目 | spec の字面 | 実装 | 理由 | file:line |
|---|---|---|---|---|---|
| 1 | S3b の許容値 | 8·eps·\|φ′\| + 1 ulp·max\|φ″\| | 同じ式に絶対値の eps を 1 つ足す | z_v で torch の expm1 の backward の丸めが相対の規則を超え、ELU1・VMIN が両 dtype で落ちる | `src/test_act_chimera_0913.py:442` |
| 2 | S12 を走の中で: 帯の和の許容値 | n_inputs·eps64 | n_inputs·eps64·max(1, max\|v\|) | 深い状態では φ（0.1z）や分離腕の h が非有界。実測の比は ≤ 0.002 | `src/act_chimera_pmnist_0913.py:405-416` |
| 3 | §3.4 の E 恒等式の対照 | 誤った h でも ≤ 1e−4 の走は pass にせず「対照を実証できない」と記録 | `ctl_demonstrated` に記録するだけで、走の検査（failed_checks）には入れない | 対照の大きさは状態に依る（SMINS の t1 で 6.7e−4） | `src/act_chimera_pmnist_0913.py:707` |
| 4 | 箱 B の `E_ident` の形 | (6, 3) | (len(e_tasks), 3)。スモークは (3, 3)。DIVERGED の走の配列は完了したタスクまでに切る | スモークは t1–3 で計る | `src/act_chimera_pmnist_0913.py:267` |
| 5 | 箱 B の S10 (ii)・EG:59 の写し・Adam の読み出し | S10 (ii) の乱数消費の対照は RL の項。E 恒等式は EG:56-68 の写し。Adam の比の対象は明記なし | 箱 B の S10 (ii) は `H.stream` を差し替えて生成器を捕まえる。EG:59 の写しに `detach` を足す（値は同じ）。Adam の読み出しは重みの行だけでバイアスを含めない（箱 B・RL） | 箱 B のループは生成器を外に出さない。autograd の警告を出さない | `src/act_chimera_pmnist_0913.py:352, 388`・`src/act_chimera_rlmnist_0913.py:146`・`src/test_act_chimera_0913.py:1674` |
| 6 | sat・dead_units | EG の式（EG は double の z を float に落として評価） | gate_block と同じ float32 の前向きの z で評価 | gate_block・mob_band と同じ z に揃える | `src/act_chimera_pmnist_0913.py:325` |
| 7 | 錨の腕の発散 | G1-PM は 1 走 2040 件の一致 | 錨の腕が DIVERGED なら G1-PM を落として例外。判定器は腕の打ち切り（`NOT_DETERMINED_DIVERGED`）を環境全体の `NOT_DETERMINED_G1_PM_FAILED` より先に出す | 切り詰めで早死にが通らないように（§4.x） | `src/act_chimera_report_0913.py:492, 528` |
| 8 | G1-PM の適用範囲 | LR・ELU1・SMAXH × seed 0–2 | 双子と lr ≠ 1e−3（段 2）の走には当てない | `LRtw*` は活性化名が LR なので、全双子が G1 で落ちていた（実装のバグ） | `src/act_chimera_pmnist_0913.py:631` |
| 9 | I の非錨の ρ_min | 明記なし | 箱 B H の max(ρ_min_N, ρ_min_D) | I は N と D の両方の操作を含む | `src/act_chimera_report_0913.py:1129, 1215` |
| 10 | 段 2（D9）の σ²_arm | §3.2「LR の双子のばらつきを全腕に使う」 | 段 2 には双子が無いので、段 1（lr 1e−3）の双子の σ²_arm を借りる | 段 2 に双子を置いていない | `src/act_chimera_report_0913.py:2463` |
| 11 | D7 の bin ごとの σ² | §3.2 の σ²_arm | bin ごとに全 seed・全 k でプールし、層別の系列には Σŵ_b²σ²_b で合わせる。双子のデータの無い bin は 0 とし `(M_SE_ONLY)` を付ける | 層別の系列の軌道分散を bin ごとの独立なずれとして組む | `src/act_chimera_report_0913.py:2199, 2270` |
| 12 | D5b の前提 | 層・用量の比の置き方は明記なし | 層 1 だけ。Φ/Ψ の用量は ρ = \|Φ̃^{H,e}\|/\|Φ̃^{H,箱B}\| を Γ の ρ_min と比べる | spec に規則が無いので固定した | `src/act_chimera_report_0913.py:1968` |
| 13 | SCR の D5a | ゲートの落ちたチャネルの扱いは明記なし | 動くがゲートで落ちたチャネルは `MOVES(WEAK_DOSE)`。`NOT_TESTABLE_WEAK_MANIPULATION` は動かない側のチャネルがゲートで落ちたときだけ | 落ちたチャネルを「動かない」に数えない | `src/act_chimera_report_0913.py:1838-1846` |
| 14 | D1・D2 の EQ 側 | D0 の用量ゲートは EQ を `NOT_TESTABLE_*` に置き換える | EQ の状態がゲートで落ちたらラベルは `NOT_TESTABLE_*` の文字列。`PARENTS_TIE` は用量でゲートしない。co-primary の合わせ方は UNRESOLVED より NOT_TESTABLE を残す | 両親の同点は操作の比較でない | `src/act_chimera_report_0913.py:1067` |
| 15 | `se_window`・Φ⁻¹・RL の FLOOR_Z | Φ⁻¹(1 − 0.05/24) = 2.86 | n < 3 で例外（スモークの窓は `NOT_DETERMINED_SHORT_WINDOW`）。Φ⁻¹ は二分法。RL の `FLOOR_Z` = 2.8653 | 3 点未満では ρ₁ が定義できない。scipy を使わない。2.86 は丸め | `src/act_chimera_report_0913.py:81, 136`・`src/act_chimera_rlmnist_0913.py:130` |
| 16 | S16 (iv) の継ぎ目 −40 と −8 | 擬キメラ・擬 big が `NOT_TESTABLE_*`（後者は存在を通って用量で落ちる） | 継ぎ目 −40 では近傍の因子が存在で（Γ の符号が定まらない）、深部の因子が用量で（ρ 0.036）落ちる。端から端の状態は ELUF・ELU1。継ぎ目 −8 の検査は small の枠に合成の φ′ −0.2 を置き、4 走の D の用量を通して対のゲートだけを試す | 判定器の経路を端から端まで通して試す | `src/test_act_chimera_report_0913.py:264, 295` |
| 17 | SCR のスモークの G1 | s_null の規則で参照の先頭 31 記録と比べ、両ログが 31 記録 | `s_null_prefix`（独自 step 列の 7 列は step の値で揃える）。主走の G1 は `EL.s_null` のまま。「両ログが 31 記録」は「自分のログが 31 記録」 | s_null の切り詰めではその 7 列を揃えられない | `src/act_chimera_scr_0913.py:522, 631` |
| 18 | S19 の許容値 | 8·eps64·λ（再構成） | 事前の Weyl の上限。float64 の経路は 8·eps64·λ を満たす | log z̄ が float32 で保存される | `src/act_chimera_scr_0913.py:763` |
| 19 | SCR の readout | §8 の凍結の 6 キー・`lam_full` (3, S) | 追加のキーを持つ。`lam_full` はログが届いた登録の ckpt ごとに 1 行（10M は (4, S)）、無ければ NaN。`lam_full_ok`・`lam_full_rho`・`lam_full_method`・`lam_full_status` を足す | 退避枝の 10M と λ_full の収束を記録する | `src/act_chimera_scr_0913.py:1177` |
| 20 | 分離腕の λ_full | 全 Hessian の冪乗法 | 分離腕は代替勾配場のヤコビアン（Hessian ではない）。密な計算に落ちたら最大実部（スペクトル半径も記録）。主腕は対称化した J の eigvalsh | 分離腕の場は勾配流でなく J が非対称 | `src/act_chimera_scr_0913.py:817, 879, 909` |
| 21 | mob の再構成の assert | 明記なし | 継ぎ目から δz 以内のパターンを持つユニットを除く | float32 の z̄ の丸めで継ぎ目の述語が反転しうる | `src/act_chimera_scr_0913.py:1135` |
| 22 | `relax_rate` の生存の規則 | 明記なし | 機構の窓で `denom` の平均 > 0.25。S9-SCR は b = 0 を assert する | b は乱数を使わない | `src/act_chimera_scr_0913.py:795` |
| 23 | SCR の provenance | §8 `scr/provenance.json`・`s_pair.json` | 腕ごとのファイルを `--provenance` で結合する。`s_pair.json` は書かず、S9 は static_checks に置く | 腕ごとに並列のジョブで書く | `src/act_chimera_scr_0913.py:1218` |
| 24 | RL の INCOMPLETE の走 | 0906 は `acc = nan` の行を足す | 完了した行だけを書く（0 行なら見出しだけ）。readout は params から float64 で組み直す。`g_batch` は読むだけで消費せず、S9 の epoch 置換は状態の sha256 で見る | merge が行数を照合できるように。乱数を消費しない | `src/act_chimera_rlmnist_0913.py:723, 730` |
| 25 | G1-RL の走ごとの適用 | G1-RL（GPU） | 走ごとの G1-RL はプロトコルの lr と 400 epoch の LR だけ。`g1_smoke` は GPU だけで、CPU の決定性は別の `determinism` | 0906 の参照は GPU で作られた | `src/act_chimera_rlmnist_0913.py:684, 763, 864, 893` |
| 26 | launcher | §7 | スモークは §7 の参考 RSS を使う。箱 B のスモークは 1 プロセス（`--smoke`）。RL の G0.5 の配置は `--device`（既定 cpu）で回す。失敗が 1 つでもあれば以後を起動しない。commit 対象が未追跡の間は `backup --execute` を拒む | スモークの時点で実測の RSS が無い。OOM と未記録の退避を避ける | `src/act_chimera_launch_0913.py:82, 98, 316, 732` |
| 27 | S16 の置き場所と出力 | §6 S16 | S16 は `test_act_chimera_report_0913.py` に置き、主テストファイルが import する。出力は `seed_contrasts.csv`。SCR の Y_ZBAR = −median z̄ | 3 ランナーの検査が同じファイルに並行して足されていた | `src/test_act_chimera_report_0913.py:682`・`src/act_chimera_report_0913.py:1560, 1666` |
| S4 | SCR のスモークの G1 の行数 | §6 スモーク: 参照の先頭 31 記録と比べる | `layer1_moment_step`・`layer1_w_free_step` と、それに結び付く列（計 7 列）は共有する step の値で揃えるので 4 行だけ比べる | edge_law の s_null の切り詰めではその列を揃えられない | `src/act_chimera_scr_0913.py:522-581` |
| S5 | スモークの回し方（コード変更なし） | §6・§7: J 規則、RL は並列 1・2・4・J の処理量 | SCR は `--rss-peak-gib 1.5`（J = 6）。RL は scratch のドライバ（launcher の `run_queue`・`rl_job`・`post_job`・`rl_throughput` をロックの下で呼ぶ）で回した。処理量は CPU 1/2/4・GPU 1/2/3（J は未測。GPU 4 は 4 × 1.73 GiB が 6 GiB の予算を超えるので省いた）。CPU の `chosen_parallel` = 4 は測った中での最良 | 他セッションと RAM を分け合う予算 | `results/_smoke_act_chimera_0913/launch/` |
| S6 | G0 の記録 | §6: G0.5 を通った作業木を commit し、その commit から回す | `static_checks.json` は HEAD b50f127・コード未追跡で記録した。commit の後に G0 を回し直す | 実装とスモークは commit 前に行った | `results/act_chimera_0913/checks/static_checks.json` |
| S7 | 箱 B の V 家族の早期の一致（予期どおり） | §2.3（V の採点は RL でだけ） | VMAX の params の sha256 が seed 1–2 の t1–3（seed 0 は t1–2）で LR と一致し、VMIN は t1 で ELU1 と一致した | z_v を踏まない間、V は親と同じ関数 | `results/_smoke_act_chimera_0913/pmnist/` |
| A1 | `twin_live_task` の置き場所 | 2-2「各タスク終端で…LR と比べ、最初に食い違ったタスクを `twin_live_task` として記録する」 | 走は b1[k] を 0 にした写しの sha256 の列を provenance の `checks.s18b.sha_excl_b1` に書く（箱 B の LR は k = 0..3・RL の LR は k = 0、双子は自分の k）。`twin_live_task` は判定器が同じ seed の LR の列と比べて出し、verdict.csv の `TWIN_LIVE` 行（`TWIN_LIVE` / `NOT_DETERMINED_TWIN_DEAD`）、seed_contrasts.csv の `twin_live_task_{双子}`、report_provenance.json の `extra.twin_live` に書く | LR と双子は並列のジョブなので、双子の走の中では LR の列がまだ無いことがある。走の provenance は後から書き直さない（§8） | `src/act_chimera_pmnist_0913.py:303, 722`・`src/act_chimera_rlmnist_0913.py:443, 758`・`src/act_chimera_report_0913.py:728, 2372` |
| A2 | 「摂動した要素を除いた」 | 摂動した要素を除いた params の sha | 要素を除く代わりに、その要素を 0 にした CPU の写しで sha256（LR と双子で同じ規則）。S18 は、init でこの sha256 が LR と一致することも要る | 配列の形を変えず、全 params の sha256 と同じ組み方にする | `src/act_chimera_0913.py:411, 434` |
| A3 | S18 の「最も近い表現値」 | 差が float32 で +1e−6 に最も近い表現値 | b1[k] の新しい値が、float64 で b1[k] + 1e−6 を計算して float32 に丸めた値に等しく、両隣の表現値が目標により近くないこと。実現した差 `realised_delta` と ulp の数 `n_ulps` を provenance に書く | 差そのものの表現は b1[k] の ulp に依るので、値で比べる | `src/act_chimera_0913.py:384, 434` |
| A4 | RL の走の中の S18 | §6 S18（RL は pytest だけだった） | `run_job` が双子の走そのものの W(0)（R2 の後）を host の init と比べ、摂動 0 の対照とともに `checks.s18` に書く。落ちたら `failed_checks` に S18 | 箱 B と揃える | `src/act_chimera_rlmnist_0913.py:749` |
| A5 | D7 の双子 | §3.4（D7 の σ² も双子から） | 2-2 の生存の規則を D7 にも当て、窓の最初のタスクまでに生きていない (双子, seed) を除く。生きた対が 2 未満なら `(M_SE_ONLY)` | σ_traj と同じ双子の規則 | `src/act_chimera_report_0913.py:2199, 2250` |
| A6 | RL の決定性の変異対照 | §2.3「2 回目の初期値の 1 要素を nextafter にした走で sha256 が変わる」 | 双子 LRtw0（b1[0] +1e−6）の t2 終端の、b1[0] を 0 にした sha256 が LR と変わること | 全 params の sha256 は摂動した要素そのものを含むので、軌道が同じでも変わる（スモークで W1[0, 0] の双子が空虚に通った） | `src/act_chimera_rlmnist_0913.py:914` |
| A7 | launcher の `require_gates` | スモークの `pass_`・G0 と G0.5 の `pass_` | 箱 B のスモークの `LRtw0_s0` に +1e−6 の S18 と S18b の記録、G1-RL のスモークに b1[0] の対照、決定性に b1[0] を除いた sha256 の対照があることも要る。static_checks.json は `pass_` に加えて、G0 の `code_sha256`（`act_chimera_0913.py`・`act_chimera_*_0913.py`・2 テストファイル）がいまのファイルと、G0 の `git_head` が HEAD と、G0.5 の `report_sha256` がいまの判定器と一致することを要る（S6） | 追補 2 より前のコードのスモーク（`pass_` = True の空虚な決定性を含む）や、追補 2 より前のコード・別の commit で回した G0・G0.5 で本走を起動させない（`pass_` だけ見ると通っていた） | `src/act_chimera_launch_0913.py:507, 516, 560, 579, 726`・`src/test_act_chimera_0913.py:1438` |
| A8 | `clipped` の注記 | 生きた対が 2 以上で推定量の切り詰めにより σ² = 0 | σ² = 0 のとき、`twin_sigma2` の注記の先頭に `clipped` と mean(c_tw² − SE_tw²) の値を書く。verdict.csv の `SIGMA_ARM_*` の note でも `clipped (…)` を先頭に置き、`n_live_pairs`・`sigma2` をその後、残りの注記を最後に書く | ラベルは `SIGMA_TRAJ_OK` のまま | `src/act_chimera_report_0913.py:812, 2381` |
| A9 | 走ごとの G1-RL の gating と変異対照の走 | §2.3（G1-RL と、その変異対照を同じ文字列比較で） | gating の述語を `g1_gating` に出す。gating（落ちたら `failed_checks` に G1_RL）は GPU・プロトコルの lr と 400 epoch・seed 0–2・双子でない LR で、**読み出しが登録の `RLReadout` の走だけ**。変異対照の読み出し（`PerturbInitReadout`）の走は比較を記録するが落とさない | 2-1 で対照が生きたので、対照の走が `run_job` の中で G1_RL で落ち（AssertionError）、`g1_smoke.json` が書かれず RL が永久に拒否されていた。旧 W1[0, 0] の対照は 20/20 一致だったので表に出なかった | `src/act_chimera_rlmnist_0913.py:684, 763, 885`・`src/test_act_chimera_0913.py:1385` |
| A10 | S18b の比較の前提 | 2-2「摂動した要素を除いた params の sha を LR と比べる」 | 判定器は LR と双子の provenance の走の同一性（device・ホスト・CPU・スレッド・インタプリタ・torch・numpy・git_hash・code_sha256）を `sha_excl_ident` に置き、記録が無いか食い違う対を `NOT_DETERMINED_TWIN_DEAD`（why = `LR/twin provenance mismatch: …`）にする | CPU と GPU は t1 で 20 セル中 14 が食い違うので、別の device・コードで回った LR と組むと、摂動に依らず t1 で「生きている」になる | `src/act_chimera_report_0913.py:456, 459, 482, 758`・`src/test_act_chimera_0913.py:1161, 1874`・`src/test_act_chimera_report_0913.py:566` |
| A11 | RL の RSS_peak と device の混入 | 2-3「RSS は CPU の実測 1.02 GiB を使う」 | `rss_peak_from_smoke('rlmnist', device=…)` は `machine.device` が一致する provenance と、ジョブ名の device が一致する /usr/bin/time だけを使う（CPU 1.019 GiB・GPU 1.730 GiB）。stage1・stage2 は `--device` を渡す。`rl_state` は provenance の device が `--device` と違う走を DONE にせず拒否する。RL ランナーの CLI は `--device cuda` に `--outdir` を要る | 全 device の最大を取ると CPU の段 1 が GPU の 1.73 GiB で J を決めていた。`--arm LR --seed 0 --device cuda` を outdir なしで回すと本走の `rlmnist/LR/s0` に落ち、CPU の段 1 がそれを DONE として飛ばしていた | `src/act_chimera_launch_0913.py:453, 465, 590, 601, 907, 961`・`src/act_chimera_rlmnist_0913.py:976` |

## 追補 3（2026-09-14 09:35 JST・本走の前・endpoint は 1 つも読んでいない）RL の CPU 本走を GCP の VM で回す

**理由**: 別セッション（`snake_phase_mnist_0914`）が white-san で PMNIST の 120 タスク走を約 14 本同時に回していて、MemAvailable は 7.9 GiB、swap は 8 GiB 中 7.1 GiB 使用だった。この状態で RL を回すと 80〜100 分かかり、OOM の危険も残る。Issa の判断（09:30 頃）で、RL の CPU 本走だけを GCP に出す。

**変えるもの（§7 の「すべて white-san」を RL の CPU の走についてだけ改める）**
- **場所**: RL 段 1 の 15 走（LR・ELU1・SMAXH・SMINH・LRtw0 × seed 0–2）を、**すべて同じ 1 台の VM** で回す。VM は cloud-computer-507713（Issa の指定）・asia-northeast1-b・c2d-standard-32（AMD EPYC・AVX2）・Ubuntu 24.04 LTS・`--max-run-duration=4h`（期限で DELETE）。腕を 2 台に分けない（対応のある設計は同じ機械の上でだけ bit で成り立つ）。
- **環境の固定**: VM でも `/usr/bin/python3`（3.12.3）・torch 2.13.0+cu130・numpy 2.5.2 を使い、`assert_environment()` をそのまま通す。provenance に `hostname`・`/proc/cpuinfo` の model name を足す。
- **決定性**: §2.3 の CPU 決定性検査（LR s0 の t1–2 を 2 回、t2 終端の sha 一致・双子の対照）を **VM の上で回し直し**、それを RL 段 1 の起動条件にする。white-san の CPU で測った決定性の記録は RL の本走のゲートに使わない。
- **変えないもの**: G1-RL（0906 との 10 桁一致）は white-san の GPU の別検査のまま（0906 は white-san の GPU の参照で、スモークで 40/40 一致した）。G1-RL は RL の CPU 本走と比べる検査ではないので、VM に移しても役割は変わらない。箱 B と SCR は bit 参照が white-san にあるので white-san から動かさない。
- **並列**: VM では 15 走を同時に起動する（1 走 1 スレッド・RSS 約 1 GiB × 15 ≪ 128 GB。32 vCPU = 16 物理コアに 15 本）。J 規則（MemAvailable）はそのまま適用する。
- **結果の回収**: VM の `results/act_chimera_0913/rlmnist/` を `gcloud compute scp` で white-san の worktree に戻し、sha256 の一覧を `results/act_chimera_0913/rlmnist/transfer_manifest.json` に残す。VM では commit・push しない。
- **git_hash**: VM は push 済みの本走 commit を `git clone --branch claude/act_chimera_0913` で取得し、`git rev-parse HEAD` が white-san の本走 commit と一致することを起動前に assert する。
