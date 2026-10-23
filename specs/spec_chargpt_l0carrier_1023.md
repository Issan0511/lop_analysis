# spec: L0の沈みの担い手を保存状態で分ける（chargpt_l0carrier_1023）

状態: 登録（測定前に量・集約・判定・予測を固定）/ 2026-10-23夜
起案: Claude（Fable、V13）/ 実装: Codex 01a1676a / Issaの依頼
親: chargpt_handover_1023、chargpt_loadsign_1023。既存の判定は変更しない。

## 1. 問いと登録前に知っていたこと

L0（block0 FFN）の相対位置rが後期に−0.35から−3.5へ深まるのを運ぶのは、行wと入力平均mの整列か、mそのものの動きか、biasか、入力の広がりか。

Fableが登録前にlayer_stats.csv（uniform, layer0, u=10000）で見た情報:
- C3 L0のr_medは課題10–30で約−0.35、その間sd_medは2.5→5.3。35–40からrが深まり100で約−3.5（8/10 seed）、sd約20。
- seed4/8はsdが13.7/12.0まで伸びてもrが−1.1/−0.4で死なない。ReLUcapはr約−0.8、sd約0.3–0.5で平ら。
- 課題1–50の切替100更新ではrは上がり（6/10 seedで正）、胴で沈む。
- K1照合値: C3 s0 t10 r−.4293, sd2.4617, mu−1.0724, dead0; t50 r−1.5437, sd8.4266, dead.0566; t100 r−3.4058, sd20.5484, mu−69.5471, dead.6191。C3 s4 t100 r−1.0957, sd13.7017, dead.0039。cap s0 t100 r−.7577, sd.3215, dead0。

登録前のコード確認: 両腕ともup層のbiasは無く、LNもgammaのみ。b≡0なのでBIAS_SINKSは構造上不可能。下のFableの元の確率は修正せず残す。まだ本解析のforward計測は行っていない。

## 2. 状態・入力・コホート

C3/ReLUcap × seed0..9 × 課題末1..10,25,39..50,89..100（35点）=700状態、L0全512unit。
checkpointはchargpt_seed10_1022_bundle.tarをメモリへ読み、学習しない。
probe定義はanalysis/chargpt_seed10_1022/load10.pyのuniform(seed), data(seed,task)をimportしてそのまま使う。
主はseedごとに固定されたuniform probe、記録は現課題chain probe。どちらも64×128 token。
x_tildeはFFN直前LNの出力（gammaを含む実際のforward入力）、d=128。

固定コホートは各arm/seedの課題10末、**uniform probeでz>0が一度でもあるunit**とする。chainも同じコホートを使い、入力分布間でunit集合を変えない。後で死んでも除外しない。全512unitの集計も併記。
コホート数、現時点alive数、死（全token z<=0）、p<.005割合を記録する。
chainの状態間差にはprobe（課題分布）の変更も入るため、学習だけによる入力移動とは呼ばない。

## 3. 量と恒等式

m=mean_n x_tilde_n。||m||, ||m||/sqrt(d)、token共分散Cのparticipation ratio tr(C)^2/tr(C^2)を記録（Cはddof=1、PRは一様な正規化倍率によらない）。

unit jごと:

    z_nj = w_j·x_tilde_n + b_j
    norm_j = ||w_j||, wh_j = w_j/norm_j
    c_j = cos(w_j,m)
    shat_j = sd_n(wh_j·x_tilde_n)
    mu_j = w_j·m+b_j
    s_j = norm_j*shat_j
    r_j = mu_j/s_j = ||m||*c_j/shat_j + b_j/s_j = A_j+B_j

sdは元のlayer_statsと同じ標本標準偏差（ddof=1）。恒等式にはepsilonを足さない。norm=0、sd=0、||m||=0は欠測数を記録し、判定への混入を止める。今回の有限性は各状態で検査。
状態集計の中央値はtorch.quantile(.5)/numpy.median（偶数個では中央2個平均）。K1用のみ、元実装と同じr_quantile(.5)、sd/muのtorch.median（中央下側）も保存する。
mean(z)=mu、sd(z)=s、r=A+B、実forward zとx_tilde W^T+bを最大絶対誤差1e-10以内で検査し、相対誤差も記録。

## 4. 端点の帳簿

保存課題末を昇順に隣り合うもの全て接続: 1→2→…→10→25→39→40→…→50→89→90→…→100（34対/arm/seed）。

    dmu_w = Δw·m_old
    dmu_input = w_old·Δm
    dmu_cross = Δw·Δm
    dmu_bias = Δb
    Δmu = dmu_w+dmu_input+dmu_cross+dmu_bias

    dmu_stretch = Δ||w|| * (wh_old·m_old)
    dmu_rotate = ||w_new|| * (Δwh·m_old)
    dmu_w = dmu_stretch+dmu_rotate

    dr_source = dmu_source/s_new  (source=w,input,cross,bias,stretch,rotate)
    dr_width = -mu_old*(s_new-s_old)/(s_old*s_new)
    Δr = dr_w+dr_input+dr_cross+dr_bias+dr_width

各unitの恒等式と、鎖のΔrがr_end−r_startに望遠鏡和で一致することを1e-10で検査。中央値の和は和の中央値と一般に一致しないため、unit別に和を取ってから中央値を取る。集計された成分中央値の合計をΔr中央値へ一致させない。

帳簿の窓はearly1→10、bridge10→25、bridge25→39、middle39→50、bridge50→89、late89→100、主のcumulative39→100。
各窓でunit別に各項を足す→固定コホート中央値→seed間範囲。対ごとの中央値と、対ごとの中央値を窓内で中央値にした補助統計もCSVに保存する。窓内の対数/課題距離を併記。長い橋を1課題分とは扱わない。

両端で固定probeを用いた代数的帳簿であり、途中の更新を積分した因果寄与率ではない。50→89を細分すると各項の分担は変わり得る。
b=0ならrは行ノルムに陽には依存しない。純粋な正の行スケール変更ではrが不変で、帳簿の伸び項と幅項が相殺する。大きい負の伸び項だけでrの沈みの因果を言わない。

## 5. 判定（主probe=uniform、主集合=固定コホート）

- **J1**: C3の39→100累積。unitごとに加算後、各項のコホート中央値を比較。inputが最も負ならINPUT_MEAN_MOVES、w（伸び+回転をunitごとに合算）が最も負ならW_TERM、biasが最も負ならBIAS_SINKS。7/10以上同ラベルならそのラベル、それ以外はMIXED。候補に負の項が無い場合、または1e-10以内の最小値同率はそのseedをUNASSIGNEDとし7票に数えない。幅・交差は必ず別に記録。W_TERMなら伸び/回転の中央値のどちらがより負かを併記。
- **J2**: C3各seedの25→100で||m||比>=2ならINPUT_CONCENTRATES、コホートc中央値の差<=−.2ならALIGNMENT_TURNS、shat中央値比<.7ならSPREAD_SHRINKS。複数可。各ラベルの該当seed数とseedを報告（依頼に全体の票数閾値は無いため勝手に追加しない）。ReLUcapも記録。
- **J3**: ReLUcapの39→100累積drについて、コホートmedian(|Σ_pair Δr_j|)<=.3のseedが8/10以上ならCAP_FLAT、それ以外NOT_CAP_FLAT。ここで主は**累積後に絶対値**を取る。曖昧さを避け、|median(ΣΔr)|とmedian(Σ|Δr|)も別列で報告し、判定を差し替えない。
- **J4**: C3 seed4/8の主累積とJ2量を、残る8seedの範囲/中央値と並べて記述。新たな有意差閾値や事後の勝者判定は加えない。

状態推移は課題10/25/39/50/89/100で||m||・c・shat・||w||・b/s・r・sd・PRを両腕について表にする。

## 6. 保存された更新100の頭・胴（補助記録）

登録前のファイル棚卸しで、別worktree chargpt_cl_1019 のM2/{arm}/seed{seed}/ckpts/u100_task{t}.ptが両腕全10seedのt90..100で存在することを確認（220状態）。元bundleにはu100は無い。
これらは読み取りのみ。主の固定uniform probeで、t−1末→tのu100（頭）とu100→t末（胴）の同じ帳簿を計算し、固定課題10コホートで集約する。chainのu100は追加しない。
各t末とu100は元layer_statsのuniform/layer0/u=10000または100と照合する。元ファイルSHAをprovenanceに記録し、再現用の入力複製をobsidian-research-dataへ退避する。学習・optimizer作成なし。
主J1の保存課題末の鎖は変更しない。頭/胴を挿入すると帳簿項の分担が変わり得ることを記録。欠けた入力があればその対を省き件数を報告する。

## 7. 走る前の予測（Fable。Issa欄は空欄）

- J1: INPUT_MEAN_MOVES .40、BIAS_SINKS .30、W_TERM .25（回転 .15、伸び .10）、MIXED .05。biasなしと判明しても元の確率を変更しない。
- J2: INPUT_CONCENTRATES .50、ALIGNMENT_TURNS .40、SPREAD_SHRINKS .10。各seedで採点し該当数を示す。全体閾値が無いので単一の○×を後付けしない。
- J3: CAP_FLAT .80。
- Issa:

## 8. 検証・保存

測定前に本specをcommit。その後に実装。束SHAは既存manifest（5729ec0279f25d884cfec0aef8c4234d2f9788d57208fed99a5c7eff4ad23e61）へ照合。
最初のgateでK1の5状態、両probeの恒等式、少なくとも1対の帳簿、純粋な行スケール変更の相殺を検査。K1は元CSVへ相対1e-3（0は絶対1e-6）、上記丸め値はさらに丸め誤差5e-5を許容。
K1失敗なら全数計測を止めてprobe等の違いを報告する。勝手に元値を書き換えない。
全700状態でも元CSV統計と恒等式を確認。CPU float64/8threads、開始前python本数・nvidia-smiを確認し他の走を止めない。

code: analysis/chargpt_l0carrier_1023/。results: results/chargpt_l0carrier_1023/（states.csv, ledger.csv, windows.csv, summary.md, judge.json, provenance.json, gate/checks, unit配列）。
git外のunit配列と借用入力の複製は~/Projects/obsidian-research-data/chargpt_l0carrier_1023/へ退避しSHA付きbackup_manifestをcommit。元bundleと別worktreeには書き込まない。
ノート: 可塑性喪失/測定/文字GPTで継続学習_L0の沈みの担い手を保存状態で分ける_結果_1023.md（Obsidian MCP）。親: [[文字GPTで継続学習_担い手の交代を保存状態で確かめる_結果_1023]]。新しいvaultフォルダを作らない。
終了後origin/mainをmergeしてmainへpushし、自分のworktree/branchを削除する。新規学習・APIワーカー・サブエージェントは禁止。
