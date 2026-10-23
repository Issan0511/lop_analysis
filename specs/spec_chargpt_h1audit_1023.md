# spec: H1 のどこが初期から違い、どこが後期に変わるか

登録: 2026-10-23 / Codex / 親 chargpt_loadsign_1023。実装・新規測定前に commit。
依頼: 「初期はどこが成立してて、後期はどこがだめかちゃんとしらべて」。新しい学習・API ワーカー・サブエージェントなし。

## 1. 対象と区別

C3-2 L1 主、ReLUcap-2 L1 対照。10 seed、課題末35点
(1..10,25,39..50,89..100)。初期1..9、中期39..50、後期89..100。
各 seed 内で窓の中央値、その10 seedの範囲を報告する。
load10.py の probe、alive、中心化head、F、κ、SGD own-weight pushを保持する。
本解析は F を作る現課題入力と勾配入力を一致させた probe_control.py の続き。
元の一様入力での結果と混ぜない。Adam・上流の更新・有限学習窓は測らない。

H1 の旧ラベル平均は入力 index m≠n を消す。この操作は unit index j≠i の除去と異なる。
実際の learned feature を固定して独立旧ラベルを平均する仮想計算は可能だが、
学習し直した網を含む旧ラベル平均ではない。固定状態から後者を同定したと言わない。

## 2. 予測誤差と旧ラベルを分離する恒等式

ξ_n = onehot(y_n)-u、e_n=p_n-u、δ_n=p_n-onehot(y_n)。
固定読み出し V に対して S(V,e)=S(V,ξ)+S(V,δ) が厳密に成立する。
oracle と呼ぶ S(V,ξ) は誤差信号だけを差し替える帳簿で、forward や重みは変えない。
V=V^c, κF, V^perp の3つで exact LN と direct (LN の分散微分のみ除去) を記録。

確率的ラベルのため、真の条件付き分布 q_n を誤差信号 q_n-u にした値も記録する。
最初の位置は定常2文字分布 π の条件付き分布、それ以外は ChainM.P[前2文字]。
F は元の抽選ラベルで作ったものを保持する。p=q は一様な旧ラベルや onehot 正答を意味しない。
CE(p,q)-H(q)=KL(q||p)、accuracy、p_true を記録する。

## 3. unit の装填と残り（同じ旧正解との内積を実測）

r=FFN前の残差、a=ReLU(Wx)、t=r+Ua、σ²=mean(Pt)^2+eps、qLN=Pt/σ、
Γ=diag(final LN gamma)、P=中心化射影、T=ΓPU、G=T^T T。
B=Σ_n ξ_n(a_n/σ_n)^T、F_r=Σ_n ξ_n(ΓPr_n/σ_n)^T。
F=B T^T+F_r。
direct の unit i への読み出しは
κF T_i = κ B_i G_ii + κΣ_{j≠i}B_j G_ji + κF_r T_i。
自己装填・他unit装填・固定残差装填に分け、LN radial も前解析と同じ厳密式で各源に配賦。
actual e、oracle ξ、true q-u の各誤差信号で記録。
head 残り V^perp は別記する。自己装填以外の和を unit 側の残りと呼ぶ場合も、
H1 の v_i^0 と同一とは言わない（σと最終LNの位置依存がある）。
旧ラベルで重み付けした押しが小さいかを測るのであって、未観測の旧ラベル平均の証明にはしない。

## 4. H1 が消した入力間の項を明示する

oracle ξ による S_load は m,n の二重和。m=n の対角 D と m≠n の O に厳密分割する。
J_n=Γ(P-qLN_n qLN_n^T/Ddim)/σ_n を最終LNのJacobianとすると
D_i = κ(1-1/C) mean_n[K_n gate_ni h_n^T J_n U_i]、O=S_load(ξ)-D。
direct と exact、自己装填・他unit装填・固定残差装填それぞれで D を計算。
direct 自己装填の対角は
D_self,i=κ(1-1/C)G_ii mean_n[K_n gate_ni a_ni/σ_n²]。
これは K≥0 なら正で、H1 に最も近い単独unitの自己形。
actual 自己装填 = D_self + O_self + 予測誤差補正、という同一スケールの分解をする。
独立一様旧ラベルを、κ・feature・J・K・gate固定で平均したoracle loadingは厳密にD。
これは実データにその仮定が成立するという主張ではない。

## 5. 指標と事前の判定

各成分で沈める割合 s=P_alive(S_i>0)、中央値|S|。
O と D、予測誤差補正とoracle、unit残りと自己装填は
median_alive(|X|/(|X|+|Y|)) を記録（両方0は除き、除外数も記録）。
符号相殺は positive-to-nonpositive の alive に対する割合も記録。

Q1 自信だけで復元: 後期oracle exact load の s≥.75、actual より≥.20増が8/10 seed以上。
true条件付き分布にも同じ判定を別に出す。満たさなければ「自信だけでは復元しない」。
Q2 単独unit自己形は保たれる: 後期 D_self,direct の s≥.95 が8/10以上。
K<0の影響を分けるため |K| 版も同時に計算する。
Q3 初期から非対角が無視できない: 初期 O_self,direct の上記 share>.5 が8/10以上。
Q4 旧正解でも他unitの符号が悪化: oracle other,direct の s の初期→後期低下が≥.20のseed8/10以上。
Q5 unit側残りがランダム程度という記述: oracle direct の「other+fixed-residual+head-perp」の
sが.40..60に入るseed数、自己装填に対するabs shareを初期・後期とも記録する。
これは ensemble の無相関性の検定ではなく、同じ状態での実測押しの診断。

全判定は機構の診断で、因果判定・有意性検定ではない。
予測: Q1は満たさない、Q2は満たす。Q3,Q4,Q5は向きを固定しない。
前解析を既知とした追試であり、独立再現とは呼ばない。

## 6. 入力次元・配置の記録

入力次元128・FF512は全状態で不変。
FFN入力の未中心化 second moment と中心化 covariance の participation rank、
TのGramのparticipation rank、列間cos²の平均、K>0の割合を測る。
次元の介入・再学習はしないので「入力次元が原因」と断定しない。
配置の数値と他unit装填の符号を別々に報告し、GとBのどちらが原因かも単独では断定しない。

## 7. 検証と保存

入力は既存bundle SHA256 5729ec0279f25d884cfec0aef8c4234d2f9788d57208fed99a5c7eff4ad23e61。
tarからメモリに読む（展開複製なし）。CPU float64・8 threads。開始前 python本数/GPU確認。
seed0両arm全35状態で probe_control_states.csv の s、κ、n_aliveを照合。
8状態(両arm×task1,9,50,100)で actual/oracle のautogradと式を照合。
全状態で F、成分和、e=ξ+δの押しの恒等式を相対誤差1e-10以下で確認。
独立な小型テンソルテストで入力二重和を明示しD/O式を照合。
一致しなければ本走を止める。新しい学習もoptimizerも起動しない。
code: analysis/chargpt_h1audit_1023/。results: results/chargpt_h1audit_1023/。
states.csv, windows.csv, judge.json/md, summary.md, gate.json, provenance.jsonをgitへ。
unit配列はrawに保存し、最後にobsidian-research-dataへSHA検証して退避。
結果ノートは既存vault可塑性喪失/測定/にMCPで作成。親は前解析の結果ノート。
終了時はCLAUDE.md §4に従いmainへmerge/pushし、自分のworktree/branchを消す。
