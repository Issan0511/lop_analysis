# spec: L0 の相殺後の押しを相2と照合する（chargpt_l0phase2_1023）

状態: 登録 / 2026-10-23 / Codex。新しい学習・optimizer・API worker は使わない。

## 1. 問いと既知の情報

Issa の問いは「自己項が正なのになぜ他項に打ち消されず合計がその向きになるのか」。正の自己項を見つけたり、それを合計から差し引いて符号が消えたりするだけでは答えにならない。chargpt_l0force_1023 のその解釈は c5f8d196 で撤回済み。既知: 同解析の現在課題の平均勾配は全 alive unit のおよそ半分以下しか沈めず、自己項と他項の大きな相殺がある。L0carrier の後期の頭と胴は逆向きになる。今回はこの既知の合計値を隠さず、別に計算する相2の予測が追うかを調べる。

参照: vault 駆動源問題「他のunitがいても自己形の図は壊れない_3歩の説明_1006」「ReLUとくねくねへの拡張_押しの格と死の吸収と谷の止め役_1008」、V13付録 C.5。Claude memory の force-field-self-rest-decomposition-caution / drift-measuring-instead-of-proving / gradient-channel-split-pitfall を確認済み。

## 2. どこまでを実測するか

固定特徴・二乗損失・独立等分散の旧ラベル・共同最小二乗 fit の相2では、他 unit と定数を除く射影 P_i を使い、全読み出しの相関を含む期待の押しは

    Q_i = <u_i, P_i a_i> / ||P_i a_i||^2

（正のラベル分散を除く）になる。u_i は位置を下げる勾配方向に対応する特徴微分。この式は実 GPT の CE・非線形後段・Markov ラベル・学習済み特徴への無条件の定理ではない。次の二段の saved-state diagnostic を実施し、その適用条件と実際の合計との一致を別々に判定する。

### A: L0 特徴上の帳簿

腕 C3/ReLUcap、seed 0..9、課題9/39/99末・課題100の更新100/末（100状態）。load10.py の固定一様 probe と現課題 chain probe（各64系列×128位置）。r の定義は一様 probe、sd は不偏標準偏差、bias 無し。

    k_mu = mean(x), k_r = mean(x)/s_i - mu_i Cov(x) w_i/s_i^3
    a_i=ReLU(z_i), u_i=gate(z_i) (x k_i)

P は (a) 定数のみ、(b) 定数と z_i、(c) 定数と全ての他 unit の a_j。主は (c)、mu は記録、r は主。SGD と保存 Adam 分母を固定した k_r/den_i を分ける。損失から位置0を除く。alive は一様 probe における z>0 が1点以上。

全列を標準化し、Gram の固有値を最大値の1e-10で切る（特異値比1e-5）。rank不足で自分の列が他列で表せるときは identifiable=False とし Q を0に置換しない。nullspaceへの座標射影が1e-7以上の列を非識別とする。感度検査は Gram 閾値1e-8/1e-12。残存エネルギー比・非識別数も報告する。正負は相対 roundoff 許容1e-10付きで計算しゼロを正扱いしない。

### B: 後段を通した相殺（40状態）

課題39/100末、両腕×10 seed。同じ probe の先頭8系列（事前固定。64系列の全勾配も別に照合）と、前回と同じ uniform alive から事前乱数抽出の32 unit。抽出は seed=10230000+1000*seed+task、結果の符号で選ばない。

L0 activation から logits までの Jacobian J を、後段の注意・FFN・LN・時間混合を全て含めて取り、f_i=J[e_i a_i], t_i=J[e_i u_i] を作る。全512 unitの f_j を使う。class共通成分は中心化。固定残差 f_bg=f-logit_mean-sum_j f_j も1列の nuisance として含む。

    Qout_i = <t_i, Pout_i f_i> / ||Pout_i f_i||^2

Pout は全ての他 f_j と f_bg を除く。これは保存された出力方向の「スカラー振幅を共同 fit する線形局所モデル」の相2期待値であり、自由な128次元の down 行を再fitしたものではない。この限界を必ず報告する。比較用に他 f_j だけを除く値も保存する。Euclidean centered-logit metric を固定し、結果を見て metric を替えない。

同じ8系列で full current-fit (p-q) と independent-label switch (p-1/V) の r 押しを JVP 内積で直接計算する。full は自己・他unit・固定残差・全後段を全て含む。q は実課題の遷移確率。ラベル確率 noise を足さない。

共同fitの必要条件も記録: 実際の振幅方向の CE 勾配 F' (p-q) が F'(p-1/V) に対して十分小さいか（norm 比、中央値相対残差）。小さいだけで二乗損失・独立ラベル等の仮定が成立したとは言わない。

## 3. 実際の合計と保存点間の動き

A の各状態は既存 raw の full fit/switch の own-W 投影と比較。B は同じ8系列の直接計算値と比較し、64系列の既存値との sign agreement と Spearman を併記。Bで全64系列の実 full 勾配も再計算し、L0入力を変える上流パラメータを含む r の全微分を JVP で記録する（SGD）。own-W の結果を全パラメータの結果と呼ばない。

保存点39→100、99末→100更新100、100更新100→100末の unit 別 -Delta r と、区間開始点の Q を比較する。入力変化を含む実 -Delta r と、同じ保存された W_new を x_old に当てる own-W の有限差を分ける。途中の力の積分や因果寄与率とは呼ばない。全 unit の平均で帳尻を合わせず、同じ unit の順位相関と符号を見る。

## 4. 判定（走る前に固定）

- J1 GEOMETRY: C3 課題100末で、識別可能なaliveの Qout_r>0 が0.75以上のseedが8/10以上なら幾何予測あり。ただしこれ単独で原因支持にしない。各seed16unit未満しか識別できなければ主判定不足。
- J2 TOTAL_TRACKING: C3 の課題39/100末をseed内で結合し、Qout_r と実 full own-W r push の Spearman>0.5、かつ符号の balanced accuracy>0.65 が8/10seed以上。fit と switch を別判定し、どちらかをもう片方の代用にしない。actualが単符号のseedはbalanced accuracy判定不能。全パラメータSGDも記録。
- J3 MOTION: A の chain Q_r と保存点間 -Delta r の Spearman>0.5 のseedが8/10以上かを区間別・腕別・own-W/実変化別に記録。長い区間の相関は必要条件でも因果証明でもない。
- joint-fit 必要条件の記録: Bの norm(F'(p-q))/norm(F'(p-u)) <=0.1 のseed数。ゼロ除算は未定義。満たさない場合、局所振幅モデルの共同fit前提をその保存状態で支持しない。
- J1だけ○でJ2が外れた場合は「相殺後の幾何は沈める向きだが、観測された合計の説明は未成立」。J2○でも frozen-state 診断を長期因果と呼ばない。

## 5. 予測（Codex、結果を見る前）

J1○0.65。J2 switch○0.35、fit○0.15。J3は各区間○0.25。必要条件の残差比<=0.1が8/10seedで成立する確率0.30。自己項正値と実fit勾配の大きな相殺を既に見たので、fitの一致には低い予測。Issaの予測は補わない。

## 6. 実装検査と置き場

specを先にcommit。BUNDLE sha256、既存 raw sha256、source code hashを記録。独立な小行列LSとleave-one-out射影、rank退化、nonGaussian ReLU符号反例(1,2,-1,-1,-1)、JVP有限差分、実際の全勾配との内積一致、1seedの旧full値一致を検査してから全seedを走らせる。失敗なら修正の理由を記録して検査からやり直す。学習更新は0。

code=analysis/chargpt_l0phase2_1023/、results=results/chargpt_l0phase2_1023/（states/units/comparisons/judge/provenance/summary）。大きな配列・ログはobsidian-research-dataへ退避しmanifestをcommit。結果note=可塑性喪失/測定/文字GPTで継続学習_L0の相殺後の押しを相2と照合する_結果_1023.md。時間相関を含むnoise feedback項は疎な保存点から実測できないので、この走では符号だけの再計算を原因確認としない。

## 7. probe縮小の追加検査（主結果を見た後、追加計算の前）

主計算40状態は完了。Qoutは識別可能unitで全て正だが、実fitの8系列/64系列の符号一致は最小0.5だった。近似probeの影響を確認するため、あらかじめ照合に使ったC3 seed0の課題39/100末で、全64系列のF'FとU'Fをchunkごとに足してQoutを出す。unit抽出と閾値はそのまま。8系列と64系列の符号・Spearman・識別数、64系列の実合計との関係を記録する。主判定J1〜J3を置き換えない。これはprobeの数値検査であり、原因を支持する別の判定を追加しない。
