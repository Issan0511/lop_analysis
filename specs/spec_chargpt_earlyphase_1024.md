# spec: 課題1〜9で相1型・相2型の押しを同じ合計へ照合する

状態: 登録 / 2026-10-24 / Issaの依頼・Codex / 親 chargpt_l0phase2_1023

## 1. 問いと既知

「Transformerでは相1→相2が早いのか、最初から相1の条件が合いにくいのか」。既知はC3 L0の課題9末で相2head模型とCE切替のSpearmanが0.829〜0.896であること。課題1〜8のこの比較は未測定。MLPで付けた相の名前をGPTの同定済み機構として移さない。

対象はC3/ReLUcap、seed0〜9、課題末1〜9の180状態。L0主、同一計算で得るL1は記録。主指標は一様probeで定めたrの、自unitのup行を通る一次の押し。勾配probeはload10.pyと同じ現課題鎖64系列×128位置、先頭位置を損失から除く。新課題はラベルだけ独立一様平均であり、次課題入力の実再生ではない。新しい学習・optimizer作成・API worker・subagentなし。

## 2. 相1/相2を同じ場所・同じ経路で比較する

Hは最終LN後・最終head直前のN×128特徴、J_iはHのL0（記録L1）up行への微分。両模型でHとJを旧ラベル平均の間固定する。正の共通係数は省き、実測への係数fitをしない。

- **相1型 head（P1）**: クラス中心化旧ラベルYに対しV=κYᵀH+V0、V0の旧ラベル相関0、κ>0、予測p−u→YというH1を同じHへ置く。E[Y V] ∝ Hなので、予測勾配はJᵀH = ∂(‖H‖²/2)。単純なFFN自己項だけを残す比較ではなく、後段注意・FFN・LNの微分を全て通す。
- **相2型 head（P2）**: G=HᵀH、最小ノルム共同LS readoutの旧ラベル共分散G⁺。予測勾配Jᵀ(HG⁺)。既存headcov.pyのiidと同じ。有効rank閾値λ>1e−10 λ_max。
- **文脈対照**: C_nm=同じ直前2文字の文脈なら1、それ以外0。P1ctx=Jᵀ(CH)、P2ctx=Jᵀ(HG⁺HᵀCHG⁺)。Cを密行列にせず文脈別和で計算。両者とも同じラベル共分散に替える。これは特徴固定の条件付き共分散であり、実学習の旧ラベル条件付き分布を同定しない。
- **従来の局所自己形（local_self、記録）**: 各FFN unitの∇w_i Σφ(z_i)²/2。P1/P2と同じ鎖probe・同じr微分に射影。後段を通さないのでhead同士の主比較とは分ける。

H1の特徴をHとした模型と、FFN出力を特徴とした素の自己形は同じ量ではない。特に最終LNでHのノルムが固定に近いとP1は弱くなり得る。P1の勾配normと最終LN γの変動を併記し、P1の不一致だけで相1が存在しないとしない。

## 3. 実測と条件の診断

- 実際のswitch勾配: ∇W CE(p,u)、すなわち出力誤差p−u。fit勾配: p−q_current。両者とも他unit・後段の全微分を含む。
- r=μ/s、μ=w·m（up biasなし）、Covは固定一様probeの不偏共分散。∇w r=m/s−μ Cov w/s³。押しS=〈∇wL,∇wr〉、Δr=−ηS。SGD主、保存Adam分母で∇wLを割る版も記録（運動量・実更新ではない）。
- P1/P2/local_self/文脈対照/実測は同一関数でこの射影を行う。μへの押しも記録してrの幅微分との違いを残す。
- headの有限状態診断: F=QᵀH（Q=q−u）、κ_q=〈Vc,F〉/‖F‖²、cos²(Vc,F)。固定誤差の装填信号(p−u)κ_qF、正答分布に置換した信号Qκ_qF、残り(p−u)(Vc−κ_qF)を逆伝播。装填+残り=実測の恒等式を確認する。Qはone-hot極限ではなく確率分布の記録でありH1そのものとは呼ばない。
- 旧課題のp_max、q_max、H(q)、KL(q||p)、〈p−u,q−u〉のcos、head fit残差比 ‖(p−q)ᵀH‖/‖(p−u)ᵀH‖。後者が小さいことだけでH2全条件成立としない。

## 4. 集約と固定した記述判定

生きた集合は一様probeでz>0が1位置以上。非有限値は失敗とし、都合のよいunit抽出をしない。押しの0は別に数える。主のP1/P2比較は生きたunitのうち実測・P1・P2の三者が全て非ゼロの**共通集合**。その数と除外数を報告。各実測符号が10unit未満ならbalanced accuracyを未判定とする。理論間の零点だけを合格に数えない。

主はC3 L0 switch SGD。指標はbalanced accuracy B（実測で沈む群・浮く群の的中率の平均）。Spearmanは記録で、量の順位が合わないことを方向定理の反証にはしない。各seedでD=B(P2)−B(P1)、窓1〜3・4〜6・7〜9の中央値を取る。P2優位=P2のB≥0.65かつD≥0.05。P1優位=P1のB≥0.65かつD≤−0.05。これは**有限状態での予測器比較の閾値**で、相の真偽を決める定理ではない。

- J1 FROM_FIRST_SAVED: 課題1末でP2優位かつ窓7〜9でもP2優位のseedが8/10以上。これは最初の保存点からの優位で、課題1の10000更新中に移行が無かったとは言わない。
- J2 EARLY_CROSSOVER: 窓1〜3でP1優位、窓7〜9でP2優位の同じseedが8/10以上。各課題のDも全点図示し、交代時点を後付けで選ばない。
- J3 otherwise: 両者のB≥0.65かつ|D|<0.05が両窓で8/10以上なら BOTH_MATCH、それ以外は MIXED_OR_UNRESOLVED。J1/J2を満たしても「相1が無い」「相2が原因」とは書かない。
- ReLUcap、L1、Adam、文脈対照、local_selfは同じ表を記録。主判定をこれらから選び直さない。

予測（実装前、Codex）: J1成立0.55、J2成立0.15、残り0.30。既知の課題9末の対応から置いた探索的比較の予測で、完全に新規の仮説検証ではない。

## 5. 走る前の検査

1. 束SHAを照合。tarからメモリに読む。checkpointを展開・変更しない。
2. C3/ReLUcap seed0 課題9末で、既存headcovのP2 SGD/Adam、scalarのswitch/fitとalive/r/μ/sを再現（配列の最大相対誤差≤1e−9、符号一致）。不一致なら本走を止める。
3. P1を‖H‖²/2の独立autogradに照合、local_selfを活性の二乗損失に照合。P2は半log擬似detの有限差分（rank維持、相対≤1e−4）。rの行微分も有限差分≤1e−6で確認。小行列でP1のラベル平均、P2の共同LS期待、文脈共分散を基底列和で独立検算。
4. rank閾値1e−8/1e−12のrankを全状態で記録。rankが変わる状態のみP2の押しも計算して符号感度を報告する。主の閾値は変更しない。
5. 実行直前python本数とGPU状態を記録。他の実行は停止しない。

## 6. 保存

code: analysis/chargpt_earlyphase_1024/。results: results/chargpt_earlyphase_1024/（states.csv、comparisons.csv、windows.csv、judge.json、summary.md、図、gate.json、provenance.json）。生unit配列・ログはgit外に置き、CLAUDE.md §4に従いobsidian-research-dataへ退避しmanifestをcommitする。結果ノートは既存の相2結果ノートへ追補。origin/mainへ統合・push後、今回のworktreeとbranchを削除する。
