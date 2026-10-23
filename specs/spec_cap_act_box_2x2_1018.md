## 2. cap12 の活性化 × 箱 2×2 と分離腕

**run id:** `cap_act_box_2x2_1018`  
**推奨ファイル:** `specs/spec_cap_act_box_2x2_1018.md`  
**継承:** `cap_cifar_ee_0920`、`cap_cifar5p1_1007`、F51/RC の各宿主。

### 2.1 問い・腕

各セルの `ref` と `cap12` を seed0–9 で対応比較する。

| 活性化 | RC/std・50課題・lr1e−3 | F51/std・30課題・lr1e−4 |
|---|---|---|
| ELU(α=1) | 既存 ref/cap12 を資格確認後に再利用 | **新規 ref/cap12** |
| ReLU | **新規 ref/cap12** | 既存 ref/cap12 を資格確認後に再利用 |

両活性化は各箱内で初期重み・画像・ラベル・バッチ列が共通。ELU は各箱の宿主の native 実装を使う。**箱間では出力次元・課題・lr・更新数も違う**ので BOX_SPLITS はこの箱の組合せを指す。

半径は各セル自身の task1 最終 Adam 更新後の W1/W2 の unit 別行ノルム。task2 の最初の更新から、毎 Adam 更新後に `c_i=min(1,rho_i/||w_i||)`、`w_i←c_i w_i`。上限以下は bit 不変、小さい行を膨らませず、bias/W3/b3/moment/時刻は不変。半径・射影前後ノルム・発火数・除去量を保存。F51 で ELU を足す際も元の ReLU 専用分岐へ名前だけ流し込まず、S-act で訓練経路を確認する。

### 2.2 主指標と2×2の判定（m=8）

主値 W は RC で t31–50、F51 で t∈{21,23,25,27,29} の online 平均。各セル `D=W_cap12−W_ref`。判定用区間は4セルのDと、各箱内の `D_ELU−D_ReLU` 2本、各活性化内の `D_RC−D_F51` 2本。計8本。

セルは `HELP` ⇔ POS(D)、`NO_RESCUE` ⇔ D の上端≤0、それ以外 `UNRESOLVED`。NO_RESCUE は「正の改善が支持されず、区間も非正側にある」で、bias機構の不在を意味しない。悪化は `HARM`（上端<0）も併記する。区間が0をまたぐだけならNO_RESCUEにせず未決とする。EQ(D)を満たすセルは別に `EQUIVALENT_WITHIN_0.005` と記録し、HELP/NO_RESCUEの方向ラベルを同等性だけで上書きしない。

| 主ラベル | 4セルに要求するパターン |
|---|---|
| `ACTIVATION_SPLITS` | ELUの両箱がHELP、ReLUの両箱がNO_RESCUE |
| `BOX_SPLITS` | RCの両活性化がHELP、F51の両活性化がNO_RESCUE |
| `MIXED` | その他。未決セルの数と名前も記録 |

連続量の交互作用と全4効果も示す。二つの既知セルだけを新しい独立成功として数えない。t1 の学習が床付近なら `T1_IMPAIRED` と online/memo/最多クラス率を開示する。**cap は t2 以降なので t1 ですでに死んだ網を予防できない**。その場合の NO_RESCUE を活性化一般の不能性と読まない。

副: 各課題の native 微分、m/s/r、死亡率、W/bノルム、射影の頻度と量、RCのt1→後期低下。F51は手法なしfresh(t29)とgapも保存するが、共通freshによるgap差はt29精度差の符号反転で独立の証拠ではない。

### 2.3 分離腕（F51/ReLU、既定で spec に含める）

| 腕 | task2 からの操作 |
|---|---|
| `cap12b` | cap12 と同じ W 半径・係数 c_i で、対応する b1/b2 も `b_i←c_i b_i`。W3/b3とmomentはそのまま |
| `lr_only` | capなし。下記の事前較正で決めた層別 Adam lr の schedule のみ |

cap12b は**連結した (w,b) のノルム射影ではない**。Wだけから係数を作り、同じ係数をbiasにも掛ける。固定された上流入力なら各unitの前活性全体をc_i倍し、その瞬間の符号を保つ。層を重ねた軌道でr不変を保証する主張はしない。

**lr_only の操作的定義。** capは候補ベクトルを半径方向へ写すので、多次元でその効果を一つの lr に厳密変換できない。較正seed100–109でref/cap12を一組ずつ走らせ、各課題t・隠れ層lについて実変位（Adam＋射影後−更新前）の相対ノルム `U_l=RMS_k(||ΔW_l,k||_F/||W_l,k,before||_F)` を測る。`q_l,t=median_s(U_cap12/U_ref)` を一度固定し、評価seedの当該課題の W_l/b_l のlrを `η·q_l,t` にする。W3/b3はηのまま、t1は全腕η、半径上限は付けない。表・schedule hashを評価前に保存する。

qを≤1へclipしない。q>1なら「lrを下げる対照」ではなく「歩幅を合わせるため増やした対照」と報告する。分母0は `SCHEDULE_UNDEFINED` とし、別の都合のよい値に置換しない。これは較正での**相対実変位の層別RMS**を合わせる対照で、方向・半径成分・評価軌道の一致を保証しない。

分離の区間族は主2×2と別で m=4: cap12b−cap12、cap12b−ref、lr_only−cap12、lr_only−ref。前者がPOSかつcap12b−refもPOSなら `BIAS_SCALING_HELPS`。lr_only−refがPOSかつEQ(lr_only−cap12)なら `STEP_CONTROL_REPRODUCES`。両方なら両方を併記し、一つの原因に強制帰属しない。その他は `SEPARATION_UNRESOLVED`。実測歩幅の一致誤差も示し、「実効学習率は完全に除外」とは書かない。

### 2.4 S 検査

| S | 本物で要求 | 必ず落とす変異 |
|---|---|---|
| S-reuse | 既存2セルの主データ/hash/seedを照合。新runnerのrefと宿主、task1のcapとrefが一致 | 5+1とRCの半径/seed/窓を交換 |
| S-act | ELU/ReLU のforwardとnative勾配がそれぞれの宿主と一致、深い負側と0点を含む | 診断dphiだけELUへ変更／ReLU forwardを残す |
| S-cap | 半径が各runのt1値、t2以降だけ発火、行別上限を丸め上界内で満たす。上限内・ゼロ行もfixtureに含む | 初期化半径／全行を球面へ膨らませる／毎課題半径更新 |
| S-capb | cはWのみから生成、bも同じc、他のtensor/moment不変 | (w,b)連結norm／biasだけ縮め忘れ |
| S-lr | scheduleが較正seedだけから独立再計算でき、実lrが表どおり、capの書込なし | 評価seedからq計算／qを1にclip／隠れてcap |
| S-verdict | 人工の4パターン/未決/悪化、欠測を正しく分類 | p非有意をNO_RESCUEとする／hard窓にeasy混入 |

### 2.5 Issa の事前予測欄

| 記入項目 | Issa 記入欄 |
|---|---|
| 主ラベルと確率 | 未記入 |
| 新2セルのmean D（pt）・範囲 | 未記入 |
| cap12b / lr_only の予測、qが1より大か小か | 未記入 |
| 外れた場合に柱3のどの文を限定するか | 未記入 |
| 記名・時刻・署名対象 spec hash | Issa ／ 未記入 ／ 未記入 |

費用は「足す活性化×箱の組合せ2つ」と「参照を含めた実走腕数」を分けて数える。新2セルはref/capで4腕×10seed、加えて分離2腕とその較正。RCは各50課題。既存参照を再利用できなければ必要な再走を明記し、比較が成立しないままラベルを出さない。

---


## 共通規約と実行予測

`spec_six_runs_common_1018.md`を本仕様の一部として継承する。特に同ファイル§8のIssaの実行指示・論文由来の採用予測を正本とする。
