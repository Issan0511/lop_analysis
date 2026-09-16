# unit_label_demand_0916 — ラベル切替の要求が各個体・P/B/Vへどう配られるか

起草: Codex_V10個体配分_0916 / 依頼: Issa「これお願いします」（各個体の旧ラベル支持結線と要求の分布を確かめる）
親: Obsidian `駆動源問題_0909` §12、V10作業リスト H1/A2、`spec_l1_push_split_0915.md`。

## 1. 閉じる穴と範囲

和の恒等式は個体の符号を決めない。固定した入力・重みで旧ラベル o を新ラベル n に替えたとき、u_new = d + e、d = J_o - J_n、e = u_old。今回測るのは **P/B/Vごとの旧ラベル支持 a*d の分布、実際の抑制 a*u_new の分布、十分条件の被覆**。恒等式の一致は実装検算であって機構の発見に数えない。

既存の150タスク走の群別変位・事後のearly/lateの符号は閲覧済み。今回の局所Jacobianと旧残差は既存保存物には無い。既存軌道の10タスクprefixを再現するため独立の確認走とは呼ばず、**既知軌道への新規計測（測定・集約方法は実行前固定）**とする。Wの因果効果・タスク正味のAdam変位・LoP・150タスク全体には結論を延長しない。

## 2. 学習と保存

- `src/relu_gelu_silu_rl_0914.py` の Engine を変更せず使用。演算形状も揃えるため `l1_push_split_0915.MODELS` の48モデルをそのまま学習し、主解析はRLの15モデル（LR/ELU1/R/GELU/SILU × seed 0,1,2）。主対象はGELU/SILU、残りは対照・報告のみ。
- 既存と同一の1200画像、784-100-100-10、float32、Adam lr=.001、batch16、6000更新/タスク、初期値・全乱数ストリームを継承。10タスクで停止。TF32なし、決定的計算、Adam状態をタスク間で保持。
- task 2–10の更新数 **0,25,100,300,1000,6000** でRL全モデルの全重み・bias・Adam一次/二次モーメント・時刻を保存。0は新ラベルに対する最初の更新前。6000でも比較する旧ラベルは直前タスクのラベルのまま。初期重みも別途保存。
- 元画像subsetと10タスク全ラベル、batch順序、モデル順序、全task末の既存評価量を保存。観測は学習用乱数を消費しない。主データの値はspecとコードのcommit後に生成する。

## 3. 計測（各保存点の1200画像、両隠れ層、全100個体）

保存float32パラメータをfloat64へ変換して、親の活性化式で再評価する。これは保存状態での高精度局所診断であり、float32実行の逐演算丸めとは区別する。

Jは最後の隠れ層では出力重みV、第1層では V diag(phi2') W2。各標本で p、a、z、phi'、J_o、J_n、d、e=sum(p J)-J_o、u=sum(p J)-J_n、L=max_c|J_c-J_o|、R=(1-p_o)L、gamma=sign(a)d-R を求める。全クラスmaxに旧クラス自身のゼロを含めても同じ。

- P: z>0、B: z_c<z<=0、V: z<=z_c。GELU z_c=-.7517915239、SiLU z_c=-1.2784645428。対照のz_c=-1は名目区分。各画像で群が変わるため、ユニット自体を固定のP/B/V所属と呼ばない。
- 主集計は **o!=n** の画像のみ。同ラベル画像は別に数を出しd=0を検算する。
- 群ごとの全標本・個体組数n、占有率、非ゼロ活性組数n_activeを記録。abs(a)>1e-12を非ゼロとする。勾配符号の判定余裕tol=1e-10*(1+max_c abs(J_c))。
- 非ゼロ活性組において、旧支持率 f_support=mean(sign(a)d>tol)、新支持率、符号未確定率、抑制率 f_suppress=mean(sign(a)u>tol)、増幅率、十分条件成立率 f_cert=mean(gamma>tol)。分母ゼロはNA。p_oは同じ母集団で平均する。
- 大きさを伴う主統計: C=sum(a*d)/sum(abs(a*d))、S=sum(a*u)/sum(abs(a*u))。ゼロ分母はNA。C>0は旧支持寄与の総量が新支持を上回る、S>0は抑制要求の総量が増幅を上回ること。これはlogit局所スケール方向の要求で実変位ではない。
- 被覆: cert_mass=sum(1_cert*abs(a*u))/sum(abs(a*u))、cert_force_mass=sum(1_cert*abs(phi'*u))/sum(abs(phi'*u))。多数を保証していても力を担う部分が少ない場合を検出する。
- 生のbias方向要求 F=-sum(phi'*u)/n_all、F_switch=-sum(phi'*d)/n_all、F_residual=-sum(phi'*e)/n_all を分ける。n_allは同じ保存点の全変更ラベル画像×100個体数で、各群の和が全体に閉じる。**Adam後の移動と呼ばない。**
- 同時点の一様新ラベル期待値に対する旧支持 a*(J_o-mean_c J_c) も副次記述。実ラベルdrawとの違いを表示し、期待Adam更新と解釈しない。
- 個体IDを保持した群別の加法十分統計を保存し、個体→タスク→seedの構成を追えるようにする。群別集計に加えて個体別全群のC/S/被覆・占有・z平均を提供する。

## 4. 集約と判定（事前固定）

主窓は **task2–10・step0**。各seed内で9切替の分子と分母をそれぞれ合計し比を取る（個体・画像組をプール）。最後に3seedの中央値と範囲を報告し、独立標本数を画像×個体数とした検定は行わない。占有・有効分母を必ず併記。task2単点も別表にする。

GELU/SILU × layer1/2 × P/B/Vの各行について:

- Q1: 3seedすべて C>1e-8 → OLD_SUPPORT_DOMINANT、すべて C< -1e-8 → NEW_SUPPORT_DOMINANT、残り MIXED_OR_NEUTRAL。NAが1seedでもあれば INSUFFICIENT_MASS。
- Q2: 同じ基準をSに適用し SUPPRESSION_DOMINANT / AMPLIFICATION_DOMINANT / MIXED_OR_NEUTRAL / INSUFFICIENT_MASS。
- Q3: 3seedすべて f_cert>.5 → CERT_COVERS_MAJORITY、すべて <=.5 → CERT_NOT_MAJORITY、残り MIXED、NAは INSUFFICIENT_MASS。併せてcert_massとcert_force_massを出す。閾値失敗を抑制の不在・反証としない。
- step25/100/300/1000/6000は同じ集約の記述表で時間変化を示す。step0条件を後続時刻へ据え置かず、その時点で再計算する。固定IDの追跡は保存するが、全期間の普遍性は判定しない。
- 学習前との比較: 初期重みで **同じtask1旧ラベル/task2新ラベル** を評価し、学習後task2-step0と比較。初期時点の群と学習後の群は構成が変わるため、群別差は固定集団の因果効果と呼ばない。全群を合算した個体・画像集合でのC/S比較を別途出す。

予測（Codex、Issaの予測ではない）: 切替時のPでは旧支持と抑制が総量で優勢。B/Vはaの負号と下流結線に依存し、位置だけから予測しない。十分条件の成立率が低い可能性を残す。

## 5. 検算・停止と保存

- G0（学習前）: 合成float64状態で両層のJ/u/dをautogradと比較。u=d+e、abs(e)<=R、条件成立時の符号を検算。旧/新反転、下流ゲート省略の変異で失敗すること。P/B/Vの分割と加法集約を検査する。観測の前後で親Engineのp/m/v/tがbit不変、次の更新も観測無しとbit一致。
- G1: 全48モデルのtask1–10で親 `l1_push_split_0915` のonline_acc/online_ce/train_ceとcnorm_l1を照合。完全一致しなければTRAJECTORY_DIFFERSを全結果に付記し、独立に軌道差の原因を調べる。近似一致へ基準を変更しない。評価統計の丸め差は別途報告。
- G2: 本解析の各保存点で恒等式・残差上界・群分割・分子分母閉包を検査。float64最大絶対誤差1e-9*(1+最大量)を許容。非有限または閉包違反時は該当結果の結論を出さず原因を調べ、変更は別commitで記録する。
- 学習コード/spec/hash/データSHA/環境/乱数subset/hashをprovenanceへ。解析コードを別ファイルにし、実行時のgit hashをanalysis_provenanceへ。
- 生のcheckpoint、入力・ラベル・順序、個体加法統計は `~/Projects/obsidian-research-data/unit_label_demand_0916/` に退避、SHA256 manifestをcommit。summary/verdict/群表/個体要約/検算はrepoへ。最後にmainへmerge/push、worktree・実験branchを削除。
