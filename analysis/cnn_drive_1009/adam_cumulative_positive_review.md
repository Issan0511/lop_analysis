# Moving Adam cumulative bridge：独立最終監査

2026-10-09。対象 /tmp/cnn_adam_cumulative_positive_1009.md を全読了。親の指示に従い数値実験は再実行せず、理論部分を監査した。

**判定：PASS。blocking な式の誤り・仮定の欠落は見つからない。**

## 1. 累積恒等式と残差

- §2.1 は g_t=b_t+[β/(1-β)](b_t-b_{t-1}) の有限 summation by parts。初期端点、終端、weight variation の符号と添字は正しい。
- predictable vector を用いる分解と、scalar proxy を先に分離する §2.2 の分解はいずれも exact。現在の denominator に依存する項を martingale にしていない。
- §3 の Cauchy–Schwarz による denominator/momentum norm bounds は正しい。momentum の望ましい符号を仮定していない。
- §4 の debiased second-moment recursion、reciprocal-root ratio の恒等式、nonincreasing proxy に対する有限 momentum-boundary bound は正しい。
- §2.3 の unweighted momentum の Cesàro cancellation と bias-correction の可算誤差も正しい。これを Adam denominator の消滅と混同していない。

## 2. 長期確率評価

§5.1 の二乗可積分 martingale series と Kronecker による正規化極限は、提示された deterministic increasing normalization と分散可算条件の下で正しい。

§5.2 の Doob maximal probability bound は、stopped martingale の総分散が記述どおり上から抑えられるなら正しい。finite total signal の場合に martingale 誤差が自動的に相対消滅する、としていない。

geometry event の持続または stopping を別途要求し、SGD 軌道の結論を actual Adam 軌道へ無断で代入しないという制限も明記されている。

## 3. actual coordinate Adam の不変構造

u_0,...,u_L が全て uniform unit vector であるため、Conv 内の raw gradient entries は g_a/√(d_l d_{l-1}) で同一。zero moment initialization から同一更新が維持され、全 raw parameter を更新する coordinate Adam が channel symmetry を保つ。

head の各 spatial/class についても channel copies の更新は同一。B の class centering は coordinate Adam では一般に保たれないが、本文はこれを要求せず、class-common 成分を差し引いた CE signal を使っている。

振幅更新

  Δa_l=-η√D_l bhat_a/(√vhat_a+ε√D_l)

の scaling は正しい。u_0 の uniform 条件も明記されているため、first-layer input 座標で Adam の denominator が outer-product 方向を壊す問題はない。

## 4. bias-corrected bound と positive-amplitude cap

debiased EMA weights の Cauchy–Schwarz による比率 bound は正しい。K_β の式に (1-β_1) が見えないのは

  (1-β_1)/(1-β_1^t)≤1

を用いた保守的上界であり、誤った省略ではない。β_1²<β_2 が幾何級数の収束を保証する。

η_t≤min_l a_l/[2√D_l K_β] は過去 state だけで定まり、どの current gradient 実現でも各 a_l を少なくとも半分残す。従って各有限時点で positivity と strict MaxPool routing が維持される。

この cap は a_l の正の一様下界や、正の極限値まで保証するものではない。本文の Adam 結論はそのような下界を使っていない。

## 5. current-state CE と literal capacity self

任意の正 a_l、任意の current B に対する radial CE signal は convexity または pairwise identity から非負。logit-bound による quantitative lower bound も、softmax covariance の centered quadratic-form lower boundを積分することで得られる。

代表 first-Conv channel の mean projection と amplitude gradient の比例係数 μ_X/√d_1 は正しい。μ_X は original input field の平均であり、pooled spatial feature の平均と取り違えていない。

L=2、unbalanced a_1,a_2 での

  K'_full=2a_1k[bb^T+a_2²Q],
  K'_self=2a_1k[bb^T+a_2²u_{1,c}²Q]

は全 raw Conv/head Jacobian の block 展開と一致する。λ>0 に対する両 logdet derivative は strict positive。CE は R=(k/a_1)f により同じ沈降側だが、class-constant logits で停止する。本文は strict sign agreement と nonreversal を区別している。

## 6. balanced-label actual-Adam の長期例

各画像に全 C class の copy を入れる batch は exact uniform-label objective を計算する。この設計では、実現する各 Conv amplitude gradient が非負であることが network geometry から導かれる。従って nonnegative moment は結論であり、一般 bridge の仮定として導入していない。

positive cap と組み合わせると全振幅は非増加で、極限を持つ。最初の update batch に nonclassconstant output の画像が一つでもあれば first-step gradient は strict positive、記載された δ_1 の式は bias-corrected first Adam step と一致する。後の全時点でこの strict net decrease が残る。

これが証明するのは balanced label enumeration の moving CNN。iid label noise がある場合の denominator/momentum residual budget を証明したことにはならない。この境界は明瞭である。

数値確認で actual scalar denominator を predictable proxy とできるのは、固定 full-image batch と balanced labels により、その gradient が現在 state の deterministic function だからである。本文の設定では正当。

## 7. 非 blocking な明確化

厳密な記述として、冒頭に ε>0 と明示するとよい（1/ε の proxy を使うため）。また §7 の first-step strictness は「最初の update batch に含まれる画像」での nonclassconstant output と書くと、ランダム画像抽出の読者にも曖昧さがない。

今回の文脈では ε=1e-8 と fixed full-image batch の実例が明示されており、どちらも計算や結論を変える問題ではない。

**最終的な射程：actual moving Adam の exact 累積恒等式、条件付き残差定理、channel-symmetric positive geometry の維持、balanced-label actual Adam の strict permanent mean decrease は成立する。通常 iid labels の残差支配、標準 RL-CIFAR、ReLU death/可塑性喪失までは証明していない。**
