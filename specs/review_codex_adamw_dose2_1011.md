# adamw_dose2_1011 本走前監査（Codex）

## 0. 要約

**判断: 直してから。** 固有指摘は high 3 件、medium 5 件、low 3 件。実装の AdamW/coupled L2 経路と既存照合は堅いが、主判定 `RULE_PREDICTED` と A2 の読みラベルに、実現値から許されない強い結論を出せる論理穴がある。特に、(i) `RESCUED` でも Z の閾値だけで「救わなかった」として FAILS 予測を的中扱いできる、(ii) `IMPAIRED_EARLY` を SAVED から除外しない、(iii) 有意差なしを「L2 Init 並み」「R 並み」と読む、の 3 点は本走後の中心的解釈を誤らせる。

## 1. 零点の訂正の妥当性

**指摘（medium）**

### 1.1 native `F.elu` の局所微分の零点は妥当だが、実際の逆伝播積の零点ではない

- 根拠: `F.elu` は入力を backward 用に保存し、宿主も `F.elu(q).sum()` の autograd を局所ゲートとして再計算している（`src/cifar_interventions_0920.py:66-69`、`src/rlcifar_mlp_battle_0918.py:187-198`）。CPU・torch 2.13.0+cu130・float32・flush なしで 0.01 刻みに測ると、dy=1 は z=-103.98 で 0、-103.97 で最小 subnormal 1.40129846e-45。`EluBackward0` は `_saved_self` を保持し、`aten::elu_backward(..., is_result=False, self)` の値と一致した。よって z₀=ln(2^-150)=-103.972… は「局所微分 exp(z) の丸め零点」として正しい。
- [再現] 正規近似を退避済み unit 配列から 1 ファイルずつ再計算した。最大絶対誤差は ref .0705、cap1 .0684、cap2 .0875、cap12 .00129 で、spec の ±.09 に収まった（読み出し実装 `analysis/adamw_dose2_1011/rule.py:86-119`）。148 状態の帯も ρ1=[4.4791687, 9.6086497]、f=[.6666718, 1.1602063] と定数を再現した（同 `:93-104`）。
- [再現] dy を掛けると零点は移動した。CPU で dy=1e-4 は最初の非零が z=-94.76、dy=1e-8 は -85.55 だった。ところが Z/`zero2` は dy を含まない局所ゲートだけを数える（`src/cifar_interventions_0920.py:223-236`、`specs/spec_adamw_dose2_1011.md:181-184`）。「対の9割以上が勾配を運ぶ」という説明は実際の `dy*exp(z)` については成立しない。
- 修正案（元 → 新）: 「訓練の微分が厳密に0／勾配を運ぶ」→「活性化の局所微分が厳密に0／局所ゲートが非零」。実際の backward 積を機構主張に使うなら、上流勾配との積の零率を別に記録し、Z とは別指標にする。
- 本走への影響: 数値としての Z と RULE の実装は一致するが、Z を実パラメータ勾配の輸送と解釈すると誤る（medium）。

### 1.2 CUDA 閾値はこの監査では未実測

- 根拠: GPU 使用禁止のため CUDA カーネルは再実行していない。[未検証] repo には RTX 5060 Ti で前進線 -16.982107、exp 零点 -103.972 とした既存記録がある（`specs/spec_eps_rejudge_0917.md:32-33`）。subnormal 保持の provenance は CPU/CUDA とも true（`results/cifar_ledger_0920/provenance.json:24-25`）。ただし今回の run 固有の生の二分法出力は保存されていない。
- [再現] CPU の前進飽和線はカーネル経路依存だった。1要素では最後の -1 が約 -17.32868、16要素以上の vector 経路では約 -16.636 であり、どちらも backward の -103.97 とは別線だった。
- 修正案（元 → 新）: CUDA 実測を「確認済み」とだけ書く → 閾値・torch/CUDA/driver・テンソル形状・生出力を provenance/check に保存する。
- 本走への影響: CUDA の Z は既存記録と解析的下あふれに支えられるが、今回の監査単独では再検証不能（low）。

## 2. 深さの規則

**指摘（medium）**

### 2.1 算術は再現するが、帯の移植仮定が本走前の S-ctrl ですでに外れている

- [再現] `rule.py` の float32 c、‖w‖*、3角を独立実行し、wd3e-2=`FAILS`、wd1e-1=`BORDERLINE`/binary `FAILS`、wd1e0=`SAVES`、中心 Z=.9147/.5528/~0 を再現した（`analysis/adamw_dose2_1011/rule.py:50-75`）。境は浅い角 λ≈.064、中心 λ≈.18、深い角 λ≈.51 で一意。
- [読解] κ=[1.6,2.3] は P180、ρ1・f は ref/cap12 系148状態、r は ref からの移植で、WD 下に同定された量ではない（`analysis/adamw_dose2_1011/rule.py:34-40,93-104`、`specs/spec_adamw_dose2_1011.md:98-113`）。
- [再現] S-ctrl の wd1e0 は task 4 で s2=1.253、登録帯 [2.119,16.343] の下。τ=0.033課題なので単なる初期緩和だけでは説明しにくい。測定は seed 100・1時点だが、規則の因子帯がこの箱の WD 状態を被覆しない直接証拠である（`results/adamw_dose2_1011/checks.json:566-591`）。同じ縮小方向が λ=.1 に及べば s2 は小さくなり、Z は下がるため、際どい腕は `FAILS` から `SAVES` 側へ動く。λ=.03 も同方向だが境までの余裕は大きい。
- 修正案（元 → 新）: P180 κ と非WDのρ1/f/rの直積を「帯」→ 外挿であることを主判定名に明示し、S-ctrl の帯外を事前の反証として扱う。規則を変えないなら `RULE_PREDICTED` の結論を「登録した外挿規則の的中」に限定する。
- 本走への影響: 採点自体は可能だが、外れを偶然や境界誤差ではなく既知の外挿失敗として読む必要がある（medium）。

### 2.2 τ と主窓

- [再現] τを 1/(1-c) とする登録算術は 33,354、9,998、1,000更新 = 1.112、.333、.0333課題で一致した（`analysis/adamw_dose2_1011/rule.py:64-68`）。厳密な振幅 e-fold は -1/log(c) だが差は1更新未満の桁で、結論に影響しない。t31までには3水準とも初期振幅の緩和は十分進む。
- [読解] λ=1 の早期有害性は `IMPAIRED_EARLY` で測るが、規則の SAVED には反映されない。この論理穴は観点3で high とする。

## 3. 判定

**指摘（high）**

### 3.1 `RULE_PREDICTED` は「機能を救ったか」の予測として非対称で、RESCUED を失敗予測の的中にできる

- 根拠: `SAVED = RESCUED ∧ DEPTH_ABOVE_LINE`、それ以外を全て NOT_SAVED とし、binary FAILS は NOT_SAVED なら一致する（`specs/spec_adamw_dose2_1011.md:159-165`、`analysis/adamw_dose2_1011/verdict.py:252-275`）。したがって、例えば λ=.1 が `RESCUED` かつ Z/Zref=.12 なら、機能ラベルでは救っているのに登録予測 FAILS が「的中」する。λ=.03 と .1 の両方がこの形でも、λ=1だけ RESCUED+DEPTHなら `RULE_HOLDS` 3/3 になりうる。
- [再現] 実際の固定 call（FAILS, FAILS, SAVES）に対し SAVED 真偽8通りを列挙すると、3/3・2/3・≤1/3 は排他的かつ網羅的。ただしこれは論理の全域性であって、機能ラベルとの意味一致ではない。3値 class の `BORDERLINE` 自体は採点せず、中心値で FAILS に潰す。
- 修正案（元 → 新）: 主名 `RULE_PREDICTED`/「救うλを当てる」→ 機能ラベルの二値一致を主判定にし、現行 conjunction は `RULE_DEPTH_CONJUNCTION` 等の機構副判定へ分離する。少なくとも `RULE_HOLDS` から「救うλ」を主張しない。
- 本走への影響: 中心的な成功ラベルが、実際の救済予測を外しても成功になり、結論を誤る（high）。

### 3.2 `IMPAIRED_EARLY` が SAVED/RULE から除外されない

- 根拠: λ腕の早期成績が10 seed中6以上で中点未満なら `IMPAIRED_EARLY` だが、腕ラベルは変えない（`specs/spec_adamw_dose2_1011.md:150-157`、`analysis/adamw_dose2_1011/verdict.py:204-220`）。SAVED は RESCUED と DEPTH だけを見る（同 `:258-260`）。一方、読み方は IMPAIRED_EARLY なら「可塑性を守った」と書かない（`specs/spec_adamw_dose2_1011.md:196-203`）。
- 修正案（元 → 新）: `SAVED = RESCUED ∧ DEPTH` → `SAVED = RESCUED ∧ DEPTH ∧ ¬IMPAIRED_EARLY`、または `RULE_HOLDS_BUT_IMPAIRED` を成功と別扱いにする。
- 本走への影響: λ=1 が早期に有害でも主判定の成功数に入り、同じ文書内の読み方と矛盾する（high）。

### 3.3 cap 判定の移植と副判定の一意性

- [再現] cap1/cap2/cap12 を代入する既存 S-verdict は SPLIT/SPLIT/RESCUED と endpoint/floor count を再現し、変異7種を検出している（`analysis/adamw_dose2_1011/checks.py:1252-1375`）。登録版から §7 を除いて比較すると、判定・腕の変更はなく §13 の追補だけだった。
- [再現] DOSE 4ラベル、A1腕ラベル、DEPTH、EQUILIBRIUM、WIDTH は if/elif 上排他的。RULE の3ラベルも一致数0〜3を全て被覆する（`analysis/adamw_dose2_1011/verdict.py:269-340`）。
- [読解] EQUILIBRIUM の×1/2〜×2と WIDTH の0.1は、装置から導いた誤差境界ではなく実務的な固定帯である（`specs/spec_adamw_dose2_1011.md:171-184`）。WIDTH の「緩和後は傾き0」は中心予想を与えるが、±.1の同等性幅そのものは導いていない。
- 修正案（元 → 新）: `MATCH`/`STOPPED` を機構確認 → 登録した記述的帯への適合と明記し、中心結論の成否に使わない。
- 本走への影響: 副ラベルは計算可能だが、帯の物理的必然性まで主張すると解釈が強すぎる（medium）。

## 4. A2 の設計

**指摘（high）**

### 4.1 読みラベルが「有意差なし」を同等性に置き換える

- 根拠: `NORM_MATTERS` は RESCUED かつ L2 Init より「有意に下でない」、`TARGET_MATTERS` は NO_RESCUE と定義される（`specs/spec_adamw_dose2_1011.md:186-192`）。実装も L2との差の両側符号検定が p<.05 で下向きの時だけ `below_L=True` とし、それ以外を L2並みにする（`analysis/adamw_dose2_1011/verdict.py:445-478`）。n=10 の符号検定では8/10がL2より下でも p=.109で、RESCUEDなら `NORM_MATTERS`。同様に Holm 後に R より有意に上でなければ、相当な正の効果でも `NO_RESCUE`→`TARGET_MATTERS` になりうる。
- [再現] ラベルの if/elif は排他的・網羅的だが、ノイズや低検出力は `UNRESOLVED` ではなく3つの因果的な読みのどれかに強制される。
- 修正案（元 → 新）: 「有意に下でない＝並み」「NO_RESCUE＝R並み」→ 事前の実質差マージンによる paired equivalence/non-inferiority を使い、成立しない場合は `READING_UNRESOLVED`。`BOTH` も残余カテゴリではなく、R超・L未満の両方をマージン付きで示す。
- 本走への影響: A2 の主目的である NORM/TARGET/BOTH が、効果ではなく検出力で決まり、誤った機構結論を許す（high）。

### 4.2 「強さを揃えた」は全座標では成立しない

- [読解] λW=2λ/sqrt(vhat) は損失勾配がL2項より大きく、mhat≈2λθ、vhatを固定視できる近似。較正は R の全327,500座標の中央値で、W1が307,200座標を占める。spec 自身が W2では対応3.1/1.6、W3では62/31と開示する（`specs/spec_adamw_dose2_1011.md:56-61`）。coupled 本走ではL2項自身がm,vを変えるため、Rで測った vhat は介入後の実効率ではない。
- 修正案（元 → 新）: 「adamw10/5と強さを揃える」→「Rの全座標中央値で名目対応させた。層・時刻・介入後vhatでは未整合」。matched AdamW差を純粋な通し方の効果と断定しない。
- 本走への影響: 走は有効だが、層ごとの強さの違いと正規化経路を切り分けられず、因果読みを弱める必要がある（medium）。

### 4.3 coupled 実装と検査

- [再現/読解] coupled は全6テンソルで `gr + 2λp` をm,vより前に置く（`src/adamw_dose2_1011_c51.py:74-92`）。S-coupled-host は別実装の宿主 `iv='l2:<λ>'` を同seedで走らせ、per_taskをバイト比較するため自己比較ではない（`analysis/adamw_dose2_1011/checks.py:895-919`）。係数半分・bias除外・vだけrawの3変異を検出する（同 `:922-930`）。S-mut は after と decoupled の丸め帯0.27倍以内、coupledは99.99%帯外を記録している（`results/adamw_dose2_1011/checks.json:804-825`）。ここは PASS。

## 5. 変異対照 S-ctrl

**指摘（medium）**

- [再現] 数字の定義は seed 100、task1はWDなし、task2〜4を400 epoch、task4終端の各隠れ層 z の画像方向SDの unit中央値。ro1はW3/b3だけλ=1、wd1e0は全6テンソルλ=1（`analysis/adamw_dose2_1011/checks.py:625-645`）。結果は ref s2=281.51、ro1=306.49、wd1e0=1.253（`results/adamw_dose2_1011/checks.json:579-591`）。
- 意味: S-ctrl のPASSは「隠れ層へのWDが幅を縮めた」だけを検査し、規則の帯やonlineを検査していない。wd1e0の1.253は登録浅い角2.119より下で、κ・ρ1・fの外挿が少なくともこのseed/時点で過大。規則のs2が小さく外れると Z は小さくなり、wd1e-1の際どい予測は救う側へ、wd3e-2も同方向へ動く。
- 修正案（元 → 新）: S-ctrl PASSを規則の補強と扱う → 幅を縮める経路の変異対照だけとし、帯外は独立の事前警告として本文・最終解釈に残す。
- 本走への影響: 1 seed・1時点なので規則を反証し切らないが、境界腕の予測信頼度を大きく下げる（観点2と同一の medium 指摘）。

## 6. A1 の起動方式

**PASS**

- [再現/読解] `WDEngine` は task1 prefix の P,m,v、Adam更新番号、label/batch generator状態を復元し、task2からWDを掛ける（`src/adamw_dose2_1011.py:221-289`、宿主状態 `src/cifar_interventions_0920.py:180-203`）。task2のラベル・順序継続と最初の更新は S-branch が検査する。
- S0 は現ドライバで task1の状態・行・unit、task2〜10の各行・unit、t10状態を bit 一致させた（`results/adamw_dose2_1011/checks.json:12-47`）。t11以降だけ異なるコード分岐はなく、ref再利用は妥当。P180と違いWD開始がtask2だが、主窓にはτ上の初期条件が残らない。早期差は `IMPAIRED_EARLY` で別に見る。

## 7. 予算

**指摘（low）**

- [読解] S-cost のA1は wd1e0・task2・20 epoch・単独1本で、後期腕や3本同時ではない（`analysis/adamw_dose2_1011/checks.py:1537-1557`）。記録値はA1 904 MiB/RSS 1.657 GiB、A2 1958 MiB/RSS 1.750 GiB（`results/adamw_dose2_1011/checks.json:1059-1082`）。subnormalが多い後期のλ=.03/.1や同時実行の処理速度は代表しない。
- [読解] 現在値（GPU使用3702 MiB、MemAvailable約14 GiB、CNN RSS計約4.6 GiB）に対し、A1 3本とA2 1本を既存peakで足してもGPU/RAMには余裕がある。launchはA1ごとにGPU/RAMを再評価する（`analysis/adamw_dose2_1011/launch.sh:22-47`）。ただしA2開始時には同じGPU/RAM gateを再評価しない（同 `:49-57`）。swap満杯なので瞬間的な外部負荷には弱い。
- 修正案（元 → 新）: task2単独のS-costだけで全期間を代表 → メモリgateの根拠にだけ使い、速度の代表とはしない。A2の各起動前にもA1と同じ空きGPU/RAM gateを適用する。
- 本走への影響: 現状の瞬間値ではOOMの蓋然性は低いが、外部負荷変化時のA2起動に防御がない（low）。

## Codex の予測（§7 を見る前に確定）

以下は規則のラベルを写したものではなく、S-ctrlの帯外、P180、capの既知結果を含めた本走結果の予測。確率は各行で合計1。

| 項目 | ラベルごとの確率 | 根拠 1 行 |
|---|---|---|
| A1 `RULE_PREDICTED` | `RULE_HOLDS` .38 / `RULE_PARTIAL` .47 / `RULE_FAILS` .15 | λ=.1の境界がS-ctrlの縮小方向で反転しやすく、明確側2腕より境界腕で1点落とす公算を最大にした。 |
| A1 λ=.03 ラベル | `RESCUED` .15 / `SPLIT` .45 / `COLLAPSED` .40 | P180では戻りがあるが、この箱の規則幅は深く、cap1/cap2型のseed分裂を最頻にした。 |
| A1 λ=.03 `DEPTH_ABOVE_LINE` | true .20 / false .80 | s2の登録下端でもZ=.65だが、WD下のr浅化・帯下振れだけを反転余地として残した。 |
| A1 λ=.1 ラベル | `RESCUED` .55 / `SPLIT` .35 / `COLLAPSED` .10 | 規則中心は境界直下だが、λ=1のS-ctrlが帯より狭く、両層縮小はcap12に近づく。 |
| A1 λ=.1 `DEPTH_ABOVE_LINE` | true .58 / false .42 | 境界腕であり、s2帯の下振れならZ<.1、登録中心ならZ>.1になる。 |
| A1 λ=1 ラベル | `RESCUED` .65 / `SPLIT` .10 / `COLLAPSED` .25 | 深さは安全だが、強い全層縮小による早期・絶対性能の毀損を無視できない。 |
| A1 λ=1 `DEPTH_ABOVE_LINE` | true .97 / false .03 | task4のs2=1.25は零点から十分浅く、rが大幅に深くならない限りZはほぼ0。 |
| A2 λ=1e-2 | `RESCUED` .45 / `PARTIAL` .45 / `NO_RESCUE` .10 | coupled L2の既知のReLU改善は強いが、0への標的と中点閾値のため完全救済と部分救済を同率にした。 |
| A2 λ=5e-3 | `RESCUED` .28 / `PARTIAL` .52 / `NO_RESCUE` .20 | 弱いcoupled項はRを越えてもL2 Initとの中点を安定して越えにくい。 |
| A2 読み | `NORM_MATTERS` .28 / `TARGET_MATTERS` .07 / `BOTH` .65 | coupled 0はAdamWより改善する一方、L2 Initには届かず両要因が残ると予測する。 |

stamp: **2026-10-11 16:07:48 JST**。予測固定直後のファイル sha256 は **`0d57ddcfbfb4178ed7d373caee2dd790c552bd3f50ffccc5df6954ef2dba211e`**（`$W/predictions_stamp.txt`）。この時刻以後、上の予測表は変更していない。

## 9. 落とし穴

### 9.1 主判定が当たっても「救うλを当てた」とは限らない

- シナリオ: λ=.03と.1がともに `RESCUED` だが Z/Zref=.12、λ=1が `RESCUED` かつZ/Zref=.01。現行コードは前2腕を NOT_SAVED として FAILS予測の的中、λ=1をSAVEDとしてSAVES予測の的中とし、`RULE_HOLDS` を出す（`analysis/adamw_dose2_1011/verdict.py:256-275`）。しかしラベルだけの予測は2腕で外れている。`n_label_only_match` が併記されても、主ラベル名と §5.8 の見出し文が強すぎる。
- シナリオ: λ=1がtask2で大きく損ない `IMPAIRED_EARLY`、後期だけrefより良く `RESCUED`+DEPTH。現行はSAVEDとして数える一方、本文は「可塑性を守ったと言わない」。成功判定と結論文が衝突する。

### 9.2 A2 の読みは検出力と未測定の第4セルに依存する

- シナリオ: l2c1e-2が10 seed中8 seedでL2 Initより低いが、残り2 seedの差が逆向き。符号検定p=.109なので「有意に下でない」。中点は越えていれば `NORM_MATTERS` になるが、「L2 Init並み」は示していない。
- シナリオ: coupled 0がRを8/10 seedで上回るがHolm後に非有意なら `NO_RESCUE`→`TARGET_MATTERS`。実質改善があっても「0へ引くのは効かない」と読める。
- 2×2は (初期値,coupled)、(0,decoupled)、(0,coupled) の3セルで、(初期値,decoupled) がない（`specs/spec_adamw_dose2_1011.md:126-128`）。中間値を `BOTH` と呼んでも、交互作用や「両方が必要」を同定できない。さらにmatched strengthは層別に不一致なので、差の原因を標的・正規化・層別強度に分けられない。

### 9.3 seed、窓、複数比較

- A1は10 seedで各腕2 endpointを97.5%にしたが3腕をまたぐ補正はない（`specs/spec_adamw_dose2_1011.md:152-155,319-320`）。`RULE_HOLDS` は3腕全ての一致なので成功側の積集合には保守的でも、個別腕・DOSE・WIDTH・DEPTHを同時に列挙して機構物語を選ぶと選択性が残る。
- A2のn=10符号検定は離散的で、8/10はp=.109、9/10はp=.0215。Holm族内のもう一方のpとの順序で境界の扱いが変わる。効果量区間ではなく有意/非有意だけで因果ラベルを付ける設計はseedノイズで割れる。
- A1主窓t31–50は登録済みで改変の余地はないが、崩落→戻り→再崩落の腕を20課題平均へ潰す。`RESCUED` と戻り事象、窓内AT_FLOOR割合、時系列を必ず併記しないと「持続的に救った」と誤読する。

### 9.4 ref再利用と再現性

- S0のbit一致は強く、同じコード経路なのでref再利用は妥当。ただし照合はt10までであり、将来driver/kernelが変わった状態で本走だけ再開した場合はt11–50の対が保証されない。provenanceのdriver/source/checks hashを最終集約時にも照合し、S0実行時と本走時の環境差があれば `CHECK_FAILED` にする必要がある（現在の verdict は `checks.json` のS0真偽だけを読む: `analysis/adamw_dose2_1011/verdict.py:751-776`）。

### 9.5 効かなかった時の切り分け

- wd1e-1がCOLLAPSEDなら、κ/ρ/f/rのどれが外れたかは事後表で記述できるが、それらは相関し、単独原因は同定できない。wd1e0がCOLLAPSEDかつDEPTH_ABOVE_LINEなら「局所ELU零点」仮説自体が不十分だが、上流勾配、出力層縮小、早期障害のどれかは分離できない。
- A2がNO_RESCUEでも、標的0が悪い、coupledが強すぎる、層別実効率が悪い、早期障害が残る、の区別は2腕だけではできない。§10の一般化制限は妥当だが、`TARGET_MATTERS` という因果名だけがその制限を越えている。

## §7 を見た後の所見（Step B）

**指摘（medium）**

### B.1 Claude列の主要ラベルは採点可能

- [再現] `CLAUDE_P` と層別表は全項目で確率和1、binaryは全て[0,1]。A1腕ラベルの `RESCUED_FUNCTION_ONLY`/`ALIVE_UNRESOLVED`、A2の3ラベルと読み3ラベル、まとめ3ラベルは §5・§12 と完全一致し、`OTHER`桶も適用外ラベルを受ける（`analysis/adamw_dose2_1011/verdict.py:546-579`）。固定call 2値、腕ラベル6種、DEPTH 3種の36組合せを列挙し、SAVEDと一致真偽に未定義・二重適用はなかった。
- 表の略記 `HOLDS/PARTIAL/FAILS` と `ABOVE_LINE` はコードでは `RULE_HOLDS/...`、`DEPTH_ABOVE_LINE` だが、見出し付きの略記として一意に対応する（`specs/spec_adamw_dose2_1011.md:241-255`、`analysis/adamw_dose2_1011/verdict.py:593-617`）。

### B.2 A1-11 と「未採点」の規則が表とコードで一致しない

- 根拠: A1-11は「ラベルの項目」に置かれ、Claude列は残りを `AT_LINE`:`UNRESOLVED`=3:1 と明示する（`specs/spec_adamw_dose2_1011.md:237-251`）。しかし verdict は `DEPTH_ABOVE_LINE` か否かだけのbinary Brierを計算し、ATとUNRESOLVEDの予測を採点しない（`analysis/adamw_dose2_1011/verdict.py:611-617`）。また §7.3 はWIDTHがALIVEでない場合などを未採点とするが、コードは `WIDTH_FROZEN|WIDTH_SPLIT` を通常採点し、`WIDTH_INAPPLICABLE`/`DOSE_INAPPLICABLE` もp=0のOTHERとして採点する（同 `:598-609`、桶 `:575-587`）。
- 修正案（元 → 新）: A1-11をラベル多クラスとして登録 → 3ラベルで採点するか、表を明示的にbinary項目へ移し3:1記述を削る。「未採点」の条件は項目ごとにコード化し、FROZEN/SPLITを採点するならその例外文を直す。
- 本走への影響: 科学判定ラベルは変わらないが、事前予測のBrierと的中集計が登録文どおりでなくなる（medium）。

### B.3 Fable列は独立予測・完全な確率予測として扱えない

- 根拠: §7自身がFableはS-ctrlと「Claudeの予測」の要約を見た後に記入したと開示する（`specs/spec_adamw_dose2_1011.md:235`）。A1-4は `RESCUED .75` だけで残り.25のラベル配分がなく、`IMPAIRED_EARLY .15` は別事象なので補完にならない（同 `:244`）。空欄項目も多い。verdictの自動採点はClaude列だけを走査し、Fable列を読まない（`analysis/adamw_dose2_1011/verdict.py:543-565,598-604`）。
- 修正案（元 → 新）: Fable列を第2の独立確率予測 → 「情報を見た後の参考予測」と明記し、採点する項目は確率和1に補完して別表/別コードにする。Codex列は上のstamp時点で§7未閲覧なので独立列として保持できる。
- 本走への影響: 本走の科学結果には影響しないが、予測者間比較や独立予測の成功数にFableを含めると誤る（low）。

### B.4 §7を見ても独立予測表は変更しない

- Claude/Fableより、Codexはλ=.1のDEPTHとA2 coupled L2を高め、λ=1の崩壊/障害を高めに置いた。これはS-ctrlの帯外と「非有意=同等」の問題を重く見た差である。Step Aの確率は変更していない。

## 最後に

**本走に進んでよいか: 直してから。**

直す項目（severity順）:

1. **high**: `RULE_PREDICTED` の主結果を機能救済の一致と現行の RESCUED∧DEPTH 機構一致に分離し、`RESCUED`をFAILS的中にできないようにする。
2. **high**: `IMPAIRED_EARLY` の腕をSAVED成功から除外するか、成功とは別ラベルにする。
3. **high**: A2の `NORM_MATTERS/TARGET_MATTERS/BOTH` に同等性・非劣性マージンと `READING_UNRESOLVED` を導入し、非有意を「並み」にしない。
4. **medium**: Zを局所微分零率と明記し、実backward積の輸送を主張しない。
5. **medium**: S-ctrlの帯外を規則外挿の事前警告として扱い、規則成功の一般化を制限する。
6. **medium**: A1-11とWIDTH等の予測採点規則を表とコードで一致させる。
7. **medium**: A2の強さ対応を全座標の一致と呼ばず、層別・時刻別の不一致を結論に反映する。
8. **medium**: EQUILIBRIUM/WIDTHの固定帯を記述的な登録帯と位置付ける。
9. **low**: CUDA閾値の生検算をprovenanceへ保存し、A2起動前にも資源gateを適用する。Fable列は参考予測として扱う。
