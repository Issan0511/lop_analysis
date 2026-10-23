# spec: 装填成分の押しが後期に半々になる経路（chargpt_loadsign_1023）

状態: 登録 / 2026-10-23 / 実装 Codex 01a1676a / 親 chargpt_latepush_1023

## 1. 問いと範囲

Issa: 「装填成分の押しが後期に半々になるのは何が変わったからなのか、保存状態だけで調べて」。
前回 C3-2 L1 の s_load は早期 0.719–0.798、後期 0.460–0.519。
新しい学習、optimizer step、API ワーカー、サブエージェントは使わない。
上流パラメータが更新されて位置を動かす経路は対象外。保存状態での入力表現は係数として使う。
主は C3-2 の L1、ReLUcap-2 の L1 を対照にする。seed 0–9、課題末 1–10,25,39–50,89–100（35 点）。
窓は早期 1–9、中期 39–50、後期 89–100。各 seed 内で中央値、seed 間の範囲を報告する。

## 2. 保存状態での恒等分解

load10.py の probe と vL=κF、および前回の固定 softmax 誤差の定義を継承する。
現課題 probe の FFN 入力平均を ybar、生きた unit を z>0 が一度でもある unit とする。
勾配の probe は load10.uniform(seed)。その各 token n で、最後の FFN について

    y = n2(r), z = W y, a = phi(z), t = r + U a
    c = t - mean(t), sigma = sqrt(mean(c²) + eps), q = c/sigma
    h = gamma * q, e = softmax(original_head(h)) - 1/V
    b = gamma * (e @ vL), K = y @ ybar
    D = (b - mean(b))/sigma
    R = -q * mean(q*b)/sigma
    S_i = mean_n K_n phi'(z_ni) [(D_n+R_n) @ U_i]

最後の LN の平均引きは D に含め、分散の微分による射影を R とする。eps を省略しない。
S = S_direct + S_radial は unit ごとに厳密。S>0 が SGD の負勾配で沈める側。
Adam の分母、実際の更新、上流の更新は入れない。
ReLUcap の phi と phi' は原実装を使う。原実装の cap は学習時の重み制約であり、forward の活性のクリップではない。今回の固定状態では両腕とも通常の ReLU とその微分になる。

## 3. 同じ状態での診断操作（学習しない）

3 因子の全 8 組合せを計算する。元の body、gate、ybar、vL、U、sigma は固定。

- K: original / abs(K)。符号だけを取り去る。
- LN: exact D+R / direct D。分散微分の射影だけを取り去る。forward の LN は変更しない。
- error: original e / load-only error eL=softmax(h @ vL.T)-1/V。後者は装填成分だけの logit が作る誤差で置換し、backward readout は同じ vL。

追加記録: K を token 間で一定の正値にした s、S_direct と S_radial の符号、相殺度 |S|/(|S_direct|+|S_radial|)、direct が正から exact が非正になる割合と逆方向、gate が両 probe で開く数、押しの絶対値。
K>0 の割合は現課題と一様入力の両方で記録する。前回の kn_positive は現課題だけだった点を明示する。
LN の幾何: 中心化した t、r、Ua の norm、FFN/残差の norm 比、両者の cosine、U_i の中心化方向に占める q への射影の大きさ（平均二乗 cosine）。
出力側: vc と W の norm 比、vL norm、元の誤差と load-only 誤差の cosine、e と装填 logit の内積、誤差 entropy を記録。
符号割合は元の生きた unit 集合を使う。0 は正に数えない。比の 0/0 は欠測とし件数を残す。

## 4. 走る前の判定

各単独操作と組合せについて、後期 s の窓中央値 >=0.75 かつ元より >=0.20 増える seed が 8/10 以上なら、その操作で後期の沈める偏りが回復したとする。
元の早期→後期低下量と操作後の低下量の差を seed ごとに出す。3 因子の相互作用を無視して単独原因と断定しない。
回復の基準を満たさなくても全値・連続差を報告。どの操作でも満たさなければ、未特定と結論する。
これらは保存状態における経路の十分性の診断であり、学習史上の原因の証明ではない。
予測は特定因子に固定しない（探索的な経路候補比較）。判定基準と比較範囲を測定前に固定する。

## 5. 検証と実行

checkpoint は chargpt_seed10_1022_bundle.tar（sha256 5729ec0279f25d884cfec0aef8c4234d2f9788d57208fed99a5c7eff4ad23e61）からメモリに読み、展開しない。
source は前回保存した原実装、probe/readout は前回 measure.py と load10.py から import。
CPU float64、1 process、最大 8 threads。前に pgrep と nvidia-smi で他の走を確認し、止めない。
本走前に seed0 の両腕、課題末 1,9,50,100 で上記 S を autograd と照合（最大絶対誤差 / 最大信号 <=1e-10）。
前回の unit 配列の S_load と全状態で照合し、s_load と n_alive は厳密一致、最大相対誤差 <=1e-10 を要求。κ も前回と照合。
gate 不一致なら停止する。恒等分解・8 組合せの baseline 一致も検査する。
spec を先に commit し、その後 code を commit して実行する。

## 6. 保存

analysis/chargpt_loadsign_1023/ に code、results/chargpt_loadsign_1023/ に states.csv、windows.csv、judge.json、summary.md、検証・provenance。
unit 配列は obsidian-research-data/chargpt_loadsign_1023/ に退避して manifest を commit。
vault は既存の 可塑性喪失/測定/ に結果ノートを MCP で作り、親の後期分解ノートへリンクする。
完了後 CLAUDE.md §4 に従って main へ merge/push、worktree と自分のブランチを片付ける。

## 7. 初回結果を見た後の追補（第二段階の測定前に登録）

第一段階では C3 の direct s が早期 0.986–0.996 から後期 0.493–0.749 へ低下し、単独診断の回復基準はすべて 0/10。従って最終 LN の分散微分だけを原因としない。ReLUcap は direct にすると後期 0.998–1.000（10/10 回復）。この結果を見てからの追加であると明記し、元の判定を変更しない。

direct の信号を同じ保存状態の「自 unit」「他 FFN unit」「残差入力」の寄与に厳密分解する。
これは upstream parameter update の寄与ではなく、固定 forward activation が出力誤差を作る寄与の分解。

softmax の centered logit f に対して alpha=logmeanexp(f), d_i=(p_i-1/V)/(f_i-alpha)>0 とし（0/0 の極限は 1/V）、

    H = diag(d) - d d^T / sum(d)
    p-u = H f

が成り立つ。H は半正定値。P は特徴次元の中心化、Gamma=diag(final LN gamma)。
QL = vL Gamma P U、QF = vc Gamma P U。残差 logit numerator は vc Gamma P r。
実際の誤差に対する direct 成分の自 unit 項は

    S_self_i = mean K gate_i a_i [QL_i^T H QF_i] / sigma²。

残差項は mean K gate_i [QL_i^T H (vc Gamma P r)]/sigma²。
他 unit 項は direct-self-residual とし、独立に全 FFN 出力を通した direct_ffn も計算して一致を確認。
load-only error でも同様に計算する（QF=QL、H も load logit から）。この場合 K を abs(K) にした self は非負であり、内部検算になる。

全700状態で各項の正の割合、絶対値中央値、相対大きさ self/(|self|+|other|+|residual|)、self が正で合計が非正になる割合を出す。
実誤差側 self の符号が早期→後期で崩れるか、他 unit または残差からの負の信号が増えるか、両者かを記録する。
load-only 側の direct の符号が崩れるなら、その原因はこの分解で cross 項（他 unit/残差）に局在する。ただし実誤差側へそのまま一般化しない。
前段 direct との恒等一致 <=1e-10、softmax 恒等式 <=1e-10、absK の load-only self >= -1e-12*最大絶対値を要求。
gate は seed0 両腕の task1,9,50,100、次に全700状態。第二段の解釈は探索的と明記する。

## 8. 補題との対応を調べる追補（第二段階結果後・測定前）

Issa「他unit由来って何でわかるの」「配置がおかしくなるから」「補題1はどこで狂うの」。第二段階はsoftmax誤差側の出力源分解であり、補題の自己形とは別の量である。この区別を明記する。

第三段階は装填F自体を、現課題probeのFFN出力源に分ける。T=Gamma P U、G=T^T T、B=(Y-u)^T(a/sigma_task)、F_r=(Y-u)^T(Gamma P r/sigma_task) とすれば F=B T^T+F_r が厳密。
unit i の最終LN分散微分前の読み出しは

    κF T_i = κ B_i G_ii + κ sum_{j≠i} B_j G_ji + κ F_r T_i。

この3項で同じ元のsoftmax誤差を逆伝播する。これは補題への接続を検査する装填側の分解で、第二段階の予測側の分解とは別である。
さらに最終LN分散微分も各F成分に対して分ける。unit i のF-selfの読み出しはκ B_i T_i^T、その分散微分の戻りは -κ(e·B_i)(q·U_i)(q·Gamma T_i)/(D sigma_uniform)。F-residualはκF_r、F-otherは全体との差とする。
これで装填の自己成分だけでも符号が揃わないか、他unitの装填が混ざることで崩れるかを識別する。両方なら両方とする。input/labelの非対角相関を直接測ったものとは呼ばない。
全700状態、同じ窓・alive。Fの再現・direct/exactの各合計を前段値と最大相対誤差1e-10以内で照合する。新しい学習なし。
これは追加の探索的解析。補題H1の旧ラベル平均、特徴・κ固定、π→y、単純な線形読み出しの条件を、今回のGPT固定状態へ無条件に移さない。κ>0の下界・自己形自体のReLU正値性・実際のS_loadの正値性を区別する。

## 9. 入力分布と正答極限の確認（第三段階後・測定前）

Issa「一致したら更新することがない」「実装はどうだった」「入力次元が悪いとか」。実装は2階Markov確率遷移から次文字をサンプルする箱Mで、固定入力への決定的ラベルの暗記ではない。元の勾配probeは一様文字列、Fは現課題文字列である。
同じ700状態で、勾配probeだけをFと同じ現課題文字列に揃える。新ラベル期待は引き続き一様（誤差p-u）。元と同じalive/ybarを使い、full/loadの押しを比較する。これは同じ入力への独立ラベル付替えの診断であり実際の次課題の系列ではない。
同じ現課題probeで、旧ラベルのCE誤差p-yを使う押しもfull/loadで記録する。旧課題の正答率、真ラベル確率、最大確率、entropy、CE、生成過程の条件付きentropy Hも出す。
後期s_loadが入力を揃えるだけで>=.75かつ元より>=.20となるseedが8/10以上なら入力分布変更で回復と記録。元の事前判定は変えない。旧課題CEの勾配と切替p-uの勾配を混同しない。
gate8状態で新ラベル期待full/loadと旧ラベルCE fullをautogradに照合、相対誤差1e-10以内。CPU倍精度、保存状態のみ。結果はprobe_control_states/windows.csvへ、unit配列は作らずコードから再現する。

## 10. 読み出しの整列・課題間回転とELR（第五段階、測定前登録）

2026-10-23夕、Fableからの依頼。問いは「後期に他unitから戻る項が逆向きになるのは、回転の凍結で読み出し方向が現課題の装填に追従しなくなったためか」。既存判定は変えない。新規学習・optimizer・APIワーカー・サブエージェントなし。

### 10.A 整列の量を型の合う形に定義する

§8と同じ700状態、L1、aliveは現課題probeでz>0が一度でもあるunit。
勾配probeは元の独立一様文字列、誤差は元のheadのp-u。H1監査のmatched-inputには変更しない。
T_iは128次元、B_jは65クラスのベクトルなので、Σ_{j≠i}T_j B_j^Tは行列であり、そのままT_iとのcosや正負は定義できない。
§8の他unitのdirect押しと符号を厳密に照合するため、クラス軸を既存の係数で縮約する：

    E_i = mean_n[(K_n gate_ni / sigma_n) e_n]             (65次元)
    R_i = Σ_{j≠i} T_j <B_j,E_i>                         (128次元)
    alpha_i = cos(T_i,R_i)
    S_other,direct,i = κ <T_i,R_i>。

R_iには自己unitと固定残差F_rを入れない。κ>0なのでalphaの符号とS_other,directの符号が一致する。
これは元の押しに対する整列の帳簿で、誤差・gate・K・sigmaを含む。headの幾何だけの独立指標とは呼ばない。
ノルム0のcosは欠測、件数を保存。主のcos>0割合は全aliveを分母とし、欠測は正に数えない。有限cosのみの割合も記録。
各状態のalive unitのcos中央値・q10/q25/q75/q90・cos>0割合を出す。
早期1..9・中期39..50・後期89..100について、状態内の統計→seed内窓中央値→10 seedの範囲の順で集約。
unit配列も保存するので分布を再集計可能にする。

### 10.A.2 回転の定義と相関

同一unit indexを対応させ、前課題末t-1と当課題末tで
rotT=1-cos(T_i(t-1),T_i(t))、rotW=1-cos(Wup_i(t-1),Wup_i(t))を計算。
T=Gamma P UなのでrotTには最終LN gammaの変化も入る。記録としてraw down列UのrotUも計算する。
これは課題間の端点変位であり、課題中の回転の累積量ではない。小ささだけで学習中の凍結とは断定しない。
1課題差の保存ペアのみ使う（2..10,40..50,90..100）。10→25、25→39、50→89は除く。
主相関は各seedの後期90..100で、現時点aliveかつ有限なペアをunitごとに集め、3ペア以上あるunitのalpha中央値・rot中央値のSpearman。
tiesは平均順位、定数列・3unit未満は欠測。n_units/n_unit_task_pairsを必ず報告。
副記録: unitごとのalpha<0の頻度とrot中央値、各課題内Spearmanの窓中央値、unit×課題をpoolした相関。いずれも独立サンプルの検定・p値として扱わない。
早期・中期も同じ集約で記録。前課題にもaliveのunitだけに限定した相関も感度分析として記録。

符号の注意: 「整列が負/低いほど回転が小さい」はrho(alpha,rot)>0を予測する。
依頼の「Spearmanが負」は、**不整列度m=-alpha**を用いたrho(m,rot)<0として採点し、rho(alpha,rot)も併記する。測定後に定義を反転しない。

### 10.B ELRの保存状態

測定前のファイル棚卸しで、既存worktree
~/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/row5/elr/elr/
のseed5..9にstate_task50.ptとstate_task100.ptを確認。seed0..4はこの場所でcheckpointが見つからなかった。
元のworktree・checkpointは変更しない。resultsとobsidian-research-dataのmanifest・bundleの一覧も検査し、存在/欠落の範囲を記録する。
後期主比較は手元で利用できるseed5..9の課題100末。課題50末は補助。C3/ReLUcapも同じseed・同じtaskの値に揃える。
ELRはL0のphi=ReLU(k*z)で、kはstate_dictに保存されないlambdaの閉包。
原コードrow5_elr.pyに従い、同じseedのC3 unit_stats.npzのfloat32行ノルム比から課題末kを復元する。
L1は原形のまま。L0の行ノルムを新たに射影する操作や学習はしない。
保存per_task.csvのk中央値・L0行ノルム、layer_stats.csvのL1 chain統計とforward復元を照合（数値rtol1e-4+atol1e-6、分率は1/512以内）。
再現しなければBを止め、Aのみ完了して理由を記録する。5seedしかないことを10seedへ一般化しない。
同じF分解でself/other/residualのdirect・exact、radial、合計s_loadを出す。n_aliveと大きさも併記する。
モデル復元後のL1 s_loadは元のfixed_error_gradientsを用いたautogradと照合する。

### 10.C 予測と採点（Fableの確率を保持）

- P1: 早期C3/ReLUcapともcos>0が.95以上（0.7）。運用上、各腕でseed窓中央値が閾値を満たす数を報告し、両腕とも8/10以上なら支持。
- P2: 後期C3のcos>0が.40..60（0.7）。該当8/10以上なら支持。
- P3: 後期ReLUcapのcos>0が.90以上（0.7）。該当8/10以上なら支持。
- P4: 後期C3で不整列度と回転のrho<0が8/10以上（0.5）。rotT主、rotW・rotUは別々に採点する。
- P5: ELR後期L1の他unit項は沈める割合.8以上（0.5）。同じdirect他unit成分で利用可能なseedの該当数/分母を報告。10seedの判定は保留し、利用可能な状態に限る採点を明示する。

P1〜P3の正負割合は§8の既測定s_other,directと恒等的に同じなので、独立な予測成功/再現とは数えない。
新しく得るのはcosの大きさ・分布、課題間回転との結び付き、ELRでの分解。
因果の断定はしない。「回転が小さいunitほど不整列」という共変動を測る検査。

### 10.D 検証と置き場

seed0両腕の35状態で、§8のraw_loadingと全成分の押し・aliveを照合。
全700状態でもR_i内積と既存other,direct、成分和を相対誤差1e-10以内で確認し、符号は厳密一致を要求。
R_iは行列積で作り、gateでは独立のj≠i明示和と数unitで照合する。回転・rank tiesも小型テンソルで検算。
sourceは既存measure.py/load10.py、checkpointは既存tarからメモリへ。CPU float64、8 threads。開始前pgrep/nvidia-smi、他の走は止めない。
code: analysis/chargpt_loadsign_1023/rotation_*.py。results/chargpt_loadsign_1023/rotation/ にstates/windows/correlations/ELR/summary/judge/gate/provenanceを置く。
生unit配列を同じ実験名のobsidian-research-dataへ退避し、既存backup_manifestを保持して追補分のmanifestを別途commit。
既存結果ノートへMCPで追記。mainへmerge/push後、自分のworktree/branchを削除する。
