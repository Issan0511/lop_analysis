# spec: L0の沈みを作る誤差信号と自己形（chargpt_l0force_1023）

登録: 2026-10-23 / Issaの依頼 / Codex / 保存状態のみ、新しい学習なし。

## 1. 問いと既知
端点帳簿で「行の回転が運ぶ」と分かっても、その向きの勾配を生む理由にはならない。Issaの主仮説は、自己出力を消す力とReLUの負側の値・微分の弱さによる自己形である。L0 carrierの既知の結果（39→100でW項10/10、回転9/10、後期の頭/胴の逆転）を見た後の新しい診断。以前の判定を変更しない。

## 2. 対象
C3/ReLUcap、seed0..9。課題末9,39,99,100、および課題100の更新100。計100保存状態。早期代表9、沈み始め代表39、後期は99/100末と課題100の頭を区別する。700状態の全窓を代表した推定とは呼ばない。
checkpointはseed10束、更新100はl0carrier退避済み入力を読む。主の勾配入力はその状態の現課題の固定chain probe。学習と同じく位置0を損失から除外する。測る位置・幅は固定uniform probe（全位置）のL0全512unit。状態別aliveはuniformで一度でもz>0、全unitも併記。

## 3. 誤差信号の厳密分解
真の条件付き分布qは既存ChainM、pは保存網のsoftmax、u=1/65。
switch=p-u、target=q-u、fit=p-q。switch=target+fit。
同じforwardとJacobianで別々に逆伝播し、期待CE fitを測る。実現ラベルによるsample=p-yも記録する。不可約entropy H(q)とKL(q||p)を分け、p=qでもH(q)>0だがfitは0と明記する。switchはこの固定入力で一様ラベルへ置換した帳簿であり、実際の次課題の入力まで平均したものではない。

## 4. 自己形の経路を閉じる
L0のactivationをa、down列をU0_i、最終LNの中心化をP、gammaをΓ、標準偏差をσ_n、クラス中心化headをVとする。B_i=VΓPU0_i、t_ni=B_i/σ_n。
元のcomponents.pyと同じ厳密PSD secantを使う。fはlogit、α=logmeanexp(f)、δ=f-α、d_c=expm1(δ_c)/(65δ_c)（0は連続極限）、H=diag(d)-dd^T/sum(d)。Hf=p-u。
最終LNの分散微分とL1のJacobianを通らない残差直通経路の自己出力項は

    b_self,ni = φ'(z_ni) φ(z_ni) t_ni^T H_n t_ni.

これは全信頼度で非負だが、位置係数Kやrの微分を掛けた押しの正値は自動ではない。directの残りを同じL0の他unit出力とその他の残差出力に分ける。最終LN分散微分を別項にし、全逆伝播との差をupper（L1注意/FFN Jacobianを通る経路）とする。

    g_switch = g_self + g_other0 + g_residual + g_LNrad + g_upper
    g_fit = 同じ5項 - g_target

全項はL0 up重みの勾配に直し、加法性を検算する。selfは残差直通経路の自己出力であり、全後段経路を含むunit自己信号とは同一視しない。secantによる非線形の配分は一意ではない。self>0の恒真を発見や十分な原因証明としない。

## 5. 位置への力
uniform入力をx、m=mean(x)、C=sample covariance(x)、mu_i=w_i·m、s_i²=w_i^T C w_i、r_i=mu_i/s_i。
S_mu,i=g_i·m（SGDのdmu=-ηS_mu）、S_r,i=g_i·[m/s_i-mu_i Cw_i/s_i³]（dr=-ηS_r）。正が沈める側。
S_muを行の半径/接線に分けるが、それ自体を原因名にしない。rの純半径項が0を確認する。
SGD計量を主とし、保存Adamの第二momentが利用できる状態では固定分母1/(sqrt(v/(1-beta2^step))+eps)でも同じ線形帳簿を出す。保存第一momentによる方向は別に記録し、期待grad/分母を実際のAdam一歩と呼ばない。更新100のmomentが無ければ欠測とする。

## 6. 集約と判定
状態/seed/arm/計量ごとに、aliveと全unitのS_mu・S_rの中央値、正割合、絶対値中央値。self/other0/residual/LNrad/upper/negative_targetの絶対値和に対するself分担をunitで計算して中央値。
実際の期待CEでS_r>0のunitに限り、selfのS_r>0の割合、およびselfを引くとS_r<=0となる割合を出す。該当unit数も記録。これは固定帳簿の除去で、再学習介入ではない。
J1（主）後期の課題100更新100と末を別々に判定: C3の10seed中8以上で、aliveのうち期待CEで沈めるunitが32以上、うちself正が80%以上、self除去で非正になるものが50%以上なら DIRECT_SELF_CARRIES、そのほかNOT_ESTABLISHED。この判定が外れても全後段を含む自己機構を否定しない。switchにも同じ基準を記録する。
J2: fitの下向き勾配で最大の絶対値項と符号を記録。相殺、上流を含まないこと、momentとの差を隠さない。閾値判定はしない。
J3: confidence/target置換はswitchとtargetの符号・大きさ比較。entropy単独で原因判定しない。
相2は独立サブエージェントが理論条件を点検する。単に自己項が正なら相2成立とはしない。GPTのCE/後段構造で残差化模型の条件が未確認なら、その旨を記録する。

## 7. 検算・限界・予測
実装前spec commit。最初にseed0両腕の末9/100と更新100で、uniform r/s/aliveを既存l0carrier（末9/100）または退避layer_stats（更新100）と照合（r,s相対1e-3、alive完全一致）。勾配加法性・secant恒等式・解析up勾配とautograd一致は相対1e-9。r微分は有限差分（パラメータ更新保存なし）と純半径0で点検。gate通過前に全seedを測らない。
新しい学習、optimizer生成、外部APIワーカーなし。サブエージェントはIssaが明示依頼した独立理論点検のみ。CPU使用前にpgrep/nvidia-smi、他走を止めない。
Issaの予測: 自己形が原因（確率指定なし）。実装者はJ1の採否を予測しない。既知端点から原因を後付けしない。
保存状態の勾配は未保存の全軌道を積分した原因寄与ではない。局所でselfが担うかをまず識別し、判定不能なら何が欠けるかを具体的に残す。

## 8. 保存
analysis/chargpt_l0force_1023/、results/chargpt_l0force_1023/（states.csv,summary.md,judge.json,gate.json,provenance.json）。unit配列はobsidian-research-dataに退避しmanifestをcommit。結果ノートを既存の可塑性喪失/測定/に作成。mainへpushし、このworktree/branchを片付ける。

## 9. 実装前の細部（測定前）
upperはFFN1枝とattention1枝にも分けて記録する。h0=L0後、h1=h0+attention1、h2=h1+FFN1とし、g1=∂L/∂h1、g0=∂L/∂h0、g2=∂L/∂h2ならFFN1枝=g1-g2、attention1枝=g0-g1。attention枝はその先のFFNとの複合経路を含む。J1の上位項upperはこの和のままで変更しない。自己形の存在と、平均位置/相対位置を沈める符号を分けるため、selfのS_muとS_rを必ず両方出す。

## 10. 全後段経路を含む自己項（第一段階を見た後、追加測定前）
第一段階の100状態で、直通自己項は沈める側だが後期には大きな打消しと上位経路がある。このため直通だけで自己機構を否定せず、全後段を通る局所自己出力を追加で測る。元のJ1を差し替えない。
対象は両腕10seed、課題末39/100（40状態）。uniformでaliveなunitからnumpy RNG seed=10230000+seed*1000+taskを用いて最大32unitを無作為抽出（両腕で母集団が違うのでunitの対応差とはしない）。選択は勾配を見ない。
L0出力前のactivation行列aと、その後の全計算f(a)を考え、J=∂f/∂aを保存点で固定する。unit iの全token列だけを残したa^(i)について、f_i=J a^(i)をJVPで求める。前段残差は固定、後段注意の時間混合・FFN・最終LNは全て微分する。
同じsecant Hで、unit i自身に戻る自己信号は [J^T H f_i]_{:,i}。これにReLU gateとL0入力を掛ける。局所線形化した出力源の全unit合計はJ a、残りはf-Jaとして、

    full_switch = full_self + full_other0 + full_residual
    full_fit = 上3項 - full_target

を選択unitごとに厳密に閉じる。J^T H JはPSDだがtoken間交差があり、各tokenの自己信号やS_rの正値は保証しない。a_iとの内積の非負性を確認する。これは全経路の局所自己出力の帳簿で、有限のunit除去や軌道上の因果介入ではない。
追加判定Q1: 各seedの選択unitで期待CEでrを沈めるunit≥8、その80%以上で自己項も沈める、その50%以上で自己項除去により非正、を8/10seed以上なら FULL_LOCAL_SELF_CARRIES。switchにも同じ判定を別に出す。SGD主、固定Adam分母記録。各状態別に示し、全期間の原因を同定したとはしない。
独立検算: 明示Hで自己係数を照合、JVPは有限差分、同一unitに対する自己項はJVP同士の双線形積でVJPと独立照合。非線形source配分の定義依存を明記する。実行前に追補commit。
