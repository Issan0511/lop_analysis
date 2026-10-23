# spec: 相2の方向対応が立ち上がる条件を保存状態で分ける

状態: 登録 / 2026-10-24 / Issaの「実測調べてよ」への追補 / Codex

## 1. 問い・既知・対象

統合ノート「Transformerの押しと相2_実測の統合_1024」§7の未測定条件を測る。既知: C3 L0の相2予測とCE切替のbalanced accuracyは課題1〜4では低く、課題6〜9で高い。pmaxやCE head残差の比だけではこの変化を説明していない。SiLU seed1000のL2は課題50以降に沈み、課題100末の全経路Adam固定分母の押しは相2と対応する。

主対象: C3/ReLUcap × seed0〜9 × 課題末1〜9（180状態）、L0自身のup行からrへ戻る押し。L1・保存Adam分母版は記録。前回と同じprobe・同じ全後段微分を使う。追加対象: SiLU/SiLUcap seed1000、課題50/100末の4状態、L2/L3の全パラメータ経路（SGD・Adam固定分母）。新しい学習、optimizer、API worker、subagentは起動しない。

## 2. 符号と固定した二乗模型

S=〈勾配, M∇r〉、Δr=−ηSなので正が沈む。rは固定一様probeの平均/不偏標準偏差。勾配の鎖probeはload10.pyと同じ64系列、先頭位置を除くN=8128。Mは恒等または保存Adam分母の逆、運動量は含まない。Hは最終LN出力N×d、C=65、Vはクラス中心化headの転置d×C、G=HᵀH。固有値閾値1e−10 λmaxでG⁺とΠを作る（1e−8/1e−12のrankも記録）。

P=softmax(HV)、U=1/C、Q=現課題の真の確率、Y=C(Q−U)、V*=G⁺HᵀY、Vr=ΠV。Δ=P−U−HV/Cは近似でなく定義する。一様点二次近似のLS解への距離であり、CE最適解への距離ではない。

bodyへ戻す信号を厳密に分ける:

- actual = (P−U)Vᵀ/N
- sq_full = HVVᵀ/(CN)
- sq_range = HVrVrᵀ/(CN)
- ls_task = HV*V*ᵀ/(CN)
- fit_gap = sq_range−ls_task
- null = sq_full−sq_range
- ce = ΔVᵀ/N

actual = ls_task + fit_gap + null + ce。全て同じ逆伝播・同じr微分で射影する。後段attention/FFN/LN・他unitの影響を残す。共有headの直接更新はSiLU全経路で direct=∂V CE を独立計上し、actual_full=actual_body+directを検算する。

## 3. 旧課題平均と個別課題の差

ChainMは各2文字文脈の確率をDirichlet(.01,…,.01)から独立生成する。Y=C(Q−U)についてβ=E‖Y_row‖²=C(C−1)/(C·.01+1)。文脈行列K_nm=同じ文脈なら1。

- iid_mean = β HG⁺/(CN)
- ctx_mean = β HG⁺HᵀKH G⁺/(CN)
- context_shift = ctx_mean−iid_mean
- task_departure = ls_task−ctx_mean

従って actual_body = iid_mean + context_shift + task_departure + fit_gap + null + ce。βは生成則から固定し、実測への係数fitはしない。H・probeが旧課題に依存して学習・生成されているため、この「平均」はH/probeを固定してラベルを引き直す条件付き模型である。task_departureを独立な標本ノイズだと断定しない。

fit診断はB=G^{1/2}Vr、B*=G^{1/2}V*、e=‖B−B*‖F/‖B*‖F。Vr−V*=C G⁺{Hᵀ(P−Q)−HᵀΔ}の恒等式も検算。CE head勾配が小さいだけでLS fit成立とはしない。

## 4. 平均方向の安定性Z（主の早期180状態）

D_i=J_H[M∇r_i]、A_i=sym(G^{+1/2} HᵀD_i G^{+1/2})。iid相2の無係数の押しはtr A_i。k=64の独立Gaussian出力模型でZ_i=√(k/2)|tr A_i|/‖A_i‖F。これは実Markov課題の確率保証ではない。

全512unitについて固定Gaussian64本（torch seed1024024、特徴128次元、状態間共通）を使う。w_b=G^{+1/2}z_b、信号Hw_bw_bᵀを逆伝播してs_bi=z_bᵀA_i z_bを得る。正確なtr A_iを中心にvhat_i=mean_b(s_bi−tr A_i)²、Zhat_i=√k|tr A_i|/√vhat_i。これはMonte Carlo推定であって上界の保証ではない。前半/後半32本の推定とその差も出す。

独立検算: C3 seed0の課題1/6/9、固定unit 0/127/255/511についてD_iをJVPで直接作り、A_iの固有値、正確なZ、fit上界、CE上界を記録。tr A_iと逆伝播、各Gaussian二次形式と射影の一致を検査する。固定unitがゼロならそのまま記録する。測定後にunitやサンプル数を選び直さない。

## 5. 集約・事前に固定する記述判定

主はC3 L0 SGD。早期窓=1〜4、対応成立後=6〜9、5は別記録。各状態は生きたunit、各窓はseedごとの状態中央値、seed間は範囲を報告する。SiLUはresponsive集合（既存dphi閾値）を使う。数値ゼロは最大絶対値の1e−10、方向比較は同じactual非ゼロ集合を使い予測ゼロを失敗として数える。実測の両符号が10unit以上でBA、Spearmanも併記。以前の共通非ゼロ集計とは集合の違いを明記する。

- J1 FIT_CLOSER: eの窓中央値が20%以上低下するseedが8/10以上。
- J2 GEOMETRY_MORE_STABLE: Zhatのunit中央値の窓中央値が25%以上上昇するseedが8/10以上。split-half両方でも同条件が成り立つか併記。Gaussian診断の安定性の記述であり実課題の保証ではない。
- J3 CE_RELATIVELY_SMALLER: ‖S_ce‖2/‖S_sq_full‖2の窓中央値が20%以上低下するseedが8/10以上。
- J4 TASK_DEPARTURE_SMALLER: ‖S_task_departure‖2/‖S_ctx_mean‖2の窓中央値が20%以上低下するseedが8/10以上。

上記は並立可。どれも満たさなければ今回の四条件の単調な改善では説明できないと書く。全項のL2比（分母は実測または対応模型）、符号・BAを記録。各項を帳簿から引いたときのBA変化も出すが、再学習・長期因果の介入とは呼ばない。とくにfitとCEの大きな相殺があれば単項の大きさを寄与率とはしない。SiLUの4状態は記録で、10seed判定へ混ぜない。

事前予測（Codex、探索的）: J1 .35、J2 .65、J3 .35、J4 .50。既存の方向一致を知った上での予測。以前の判定を変更しない。

## 6. gate・保存・完了

束SHA確認。C3 seed0の課題1/6/9、ReLUcap seed0課題9、およびSiLU seed1000課題100で旧生配列のP2・actual・rを再現（相対1e−8以下）。各信号・勾配・r帳簿、head残差恒等式を相対1e−8以下で検査。不一致なら本走前に停止する。小行列で文脈平均の基底列和を独立検算する。JVPで作るAの検査も相対1e−8以下。小さな項への相対誤差は同じ合計または比較両辺の最大値で正規化し、尺度を保存する。

走る直前にpython数とGPUを確認。他の走を止めない。CPU float64・8thread。コードはanalysis/chargpt_phase2conditions_1024、結果はresults/chargpt_phase2conditions_1024（states.csv、components.csv、comparisons.csv、windows.csv、judge.json、summary.md、gate.json、provenance.json）。生配列とログはgit外に置き、CLAUDE.md §4でobsidian-research-dataへ退避してmanifestをcommitする。統合ノートへMCPで実測を追記し、mainへpush、今回のworktreeとbranchを片付ける。
