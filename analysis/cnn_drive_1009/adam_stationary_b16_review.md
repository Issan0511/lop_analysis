# B=16 stationary Adam 反転 certificate の独立監査

2026-10-09。監査者 existing_drive_proof。repo は編集しない。

最終判定：PASS。数学、主 certificate の外向き sqrt 区間実装、独立有理 moment 再帰との全 moment 完全一致を確認した。主ファイル /tmp/cnn_adam_stationary_b16_1009.py と .md を読了し、certificate() を実行した。独立に sqrt 関数を一切使わず、有理数の二乗比較だけでも厳密な負の上界を証明した。MC は補助診断として区別する。

## 1. 主張の正確な範囲

K_t iid Binomial(16,9/10)、g_t=900001/1000000-K_t/16 とする。E g_t=1e-6>0。

β1=a=9/10、β2=b=999/1000、ε=1e-8 の Adam moments にこの **固定された iid scalar gradient driver** を入力する。定常の normalized update

  A_t=m_t/(sqrt(v_t)+ε)

について E A_t<0、ほぼ確実な長期時間平均も負である。したがって「Adam の符号変化は一歩だけで、長期平均すれば必ず E g の向きに戻る」という一般命題への反例になる。

この結果だけで actual CNN/CE のパラメータ軌道の長期ドリフトが反転したとは言わない。実パラメータを更新すると一般に logits、p、gradient 分布が変わる。固定 iid driver と actual moving-state trajectory は区別が必要。

なお、この driver は uniform 10-class labels と矛盾しない。9 classes を一つの集合 S とすると、P(y∈S)=9/10。S の全 logits を同じ scalar θ で増加させ、ある固定状態で p(S)=0.900001 とすれば、その θ 方向の batch CE gradient は g=0.900001-K/16 である。ただし θ 更新後まで同じ p(S) とは限らない。

## 2. 定常表現と support bound

二側 iid 列上の定常解は

  m_t=(1-a)Σ_{j≥0}a^j g_{t-j},
  v_t=(1-b)Σ_{j≥0}b^j g_{t-j}².

絶対収束する。r=E g²=0.005625000001、Δ=v-r とする。

  M=max|g|=0.900001,
  z_min=min g²=(0.025001)²=0.000625050001>0.

確率 1 で |m|≤M、z_min≤v≤M²。support 全 17 点から min を取っており、K=14 の g=0.025001 が最小絶対値。K=15 は -0.037499 なので、g=0 に近い未考慮の atom はない。

## 3. 有理 moment 再帰の独立導出

ξ=g²-r、U_pq=E[m^p Δ^q]、Z_pq=E[g^p ξ^q] とする。過去の (m,Δ) と現在の g は独立だから

  m'=a m+(1-a)g,
  Δ'=b Δ+(1-b)ξ

を二項展開して

  U_pq =
    Σ_{0≤i≤p,0≤j≤q,(i,j)≠(p,q)}
      binom(p,i)binom(q,j)
      a^i(1-a)^(p-i)b^j(1-b)^(q-j)
      U_ij Z_(p-i,q-j)
    / (1-a^p b^q).                                   (1)

U_00=1。(p,q)≠(0,0) の分母は正。右辺は総次数の小さい moment だけなので一意に再帰計算できる。全 atom、確率、a,b,r が有理数なので、この部分に丸め誤差はない。g と ξ の同時 moment を用いることが重要で、m と v を独立とする操作はない。

独立実装 /tmp/cnn_adam_stationary_b16_independent_1009.py は Fraction で (1) を再導出・実行した。E m=E g、EΔ=0、E m²=μ²+(r-μ²)(1-a)/(1+a) も assert で確認した。

主要値：

|moment|独立計算値|
|---|---:|
|E m²|0.000296052632578947368421052631578947...|
|E[mΔ]|2.78752477700693756194e-7|
|E[mΔ²]|1.23832398263450911099e-11|
|EΔ⁶|7.78359326621998504348e-22|
|EΔ¹²|3.72730505231070355268e-41|

独立実装の最後の Decimal sqrt は照合用高精度評価であり、それ自体を outward-rounding certificate と称していない。主実装の整数 sqrt 区間に加え、独立実装にも sqrt 不使用の有理比較証明を追加した（§7.1）。

主実装は Δ の cumulant 加法と moment 変換、mixed moments の履歴一致式を使う。これは本監査の二項再帰とは異なる経路である。両 module を読み込んで、7 個の support/joint quantities と Δ の 0〜12 次全 13 moment を Fraction として直接比較する assert を実行し、全て完全一致した。主上界が独立有理上界より小さいことも assert で確認した。

## 4. Taylor 中心項と good event

f(v)=v^(-1/2) を r の周りで二次まで展開する：

  P2(Δ)=r^(-1/2)-(1/2)r^(-3/2)Δ+(3/8)r^(-5/2)Δ².

全事象にわたる期待の中心項は

  T2=E[mP2(Δ)]
    =μ/sqrt(r)-E[mΔ]/(2r sqrt(r))
      +3E[mΔ²]/(8r²sqrt(r)).

独立評価 T2=-0.00031508311583357736884334726591...。

good event A={v≥0.7r} 上では r と v を結ぶ全区間が 0.7r 以上。f'''(x)=-(15/8)x^(-7/2) なので Lagrange remainder と Cauchy-Schwarz により

  |E[1_A m(f(v)-P2(Δ))]|
    ≤(5/16)(0.7r)^(-7/2) E|m||Δ|³
    ≤(5/16)sqrt(E m² EΔ⁶/(0.7r)^7).                  (2)

符号に依存しない上界であり、good indicator を落とすのは正しい。独立評価は 0.0000391608913306860474710821953482...。

## 5. bad event：中心項の bad 部分も含む

B={v<0.7r} では Δ<-0.3r。偶数 moment Markov bound は

  P(B)≤EΔ¹²/(0.3r)^12
      =6.98987740953236822703e-8....                   (3)

ここで bad event の exact integrand だけを小さくするのでは不十分で、既に全期待で計算した P2 の bad 部分も差し引く必要がある。採用されている bound は両方を含んでいて正しい。

x=Δ/r∈(-1,-0.3) なので 1-x/2+3x²/8≤15/8。従って

  |E[1_B m(f(v)-P2(Δ))]|
    ≤M[1/sqrt(z_min)+15/(8sqrt(r))]P(B).             (4)

独立評価は 0.00000408898217753643142159557555413...。P2 の 15/8 は bad event が下側だけであることに依存する。上側まで含めた event にそのまま使うことはできないが、ここでは下側 event なので問題ない。

## 6. ε の効果

任意の符号の m に対して

  |m/(sqrt(v)+ε)-m/sqrt(v)|
    =ε|m|/[sqrt(v)(sqrt(v)+ε)]
    ≤ε|m|/z_min.

従って絶対期待差は

  ε sqrt(E m²)/z_min
    =0.000000275276858520498725809503499... .         (5)

これは m の符号を固定せずに扱っており、ε を単に無視していない。

## 7. 合成上界

(2),(4),(5) を T2 に加えれば

  E[m/(sqrt(v)+ε)]
    ≤ -0.000271557965466834391224859991509... <0.      (6)

の数値候補を得る。負 margin は約 2.716e-4 であり、残差が負中心を打ち消さない。good/bad/Taylor/ε を含む全寄与が列挙されている。

E g=1e-6>0 に対して E Adam update<0 である。m の平均自体は μ>0 であり、反転は m と v の相関、および非線形正規化による。m/v 独立近似ではこの結論を落とす。

### 7.1 sqrt 実装にも依存しない独立の有理 certificate

A=μ-E[mΔ]/(2r)+3E[mΔ²]/(8r²)<0 とすると T2=A/sqrt(r)。以下をすべて Fraction の不等式として実行・確認した。

1. A<0 かつ A²>(0.0003150831)² r。従って T2<-0.0003150831。
2. (25/256)E m² EΔ⁶/(0.7r)^7<(0.000039161)²。従って good error<0.000039161。
3. r>(3/40)²、sqrt(z_min)=25001/1000000 なので bad error≤M P(B)[1000000/25001+25]<0.0000040891。
4. ε²E m²/z_min²<(0.0000002753)²。従って epsilon error<0.0000002753。

したがって厳密に

  E A < -0.0003150831+0.000039161+0.0000040891+0.0000002753
      = -2715577/10000000000 = -0.0002715577 < 0.

各平方比較は符号が分かった非負量同士であり、不等号の向きを失っていない。Decimal の丸め、整数 sqrt の実装、MC のいずれにも依存しない独立 certificate である。

### 7.2 主実装の sqrt interval と丸め方向

主実装の n=isqrt(floor(x·10^160)) は floor(sqrt(x)·10^80) と等しいので、[n/10^80,(n+1)/10^80] は sqrt(x) を含む。両端を二乗する assert も確認した。

- T2 の係数 A<0 なので、sqrt(r) の上端で割ると上界、下端で割ると下界。実装は正しい。
- good error と epsilon error は正の sqrt の上端を使う。実装は正しい。
- bad error の 1/sqrt(r) は sqrt の下端、sqrt(z_min) は正確な rational min|g| を使う。実装は正しい。
- 最終 upper<0 は float に変換する前の Fraction 比較。小数は表示専用。

主実装が出す期待値区間と別再帰での moment がすべて整合しており、符号 certificate に未処理の誤差項は見つからなかった。

## 8. 定常期待から長期平均へ

定常 A_t は二側 iid 列の可測な shift function であり、support bound から有界。iid shift の ergodicity より

  (1/T)Σ_{t=1}^T A_t → E A_0 <0   almost surely.     (7)

外部 ergodic 定理に依存しない同等の証明も可能。長さ J の補正済み有限 moving averages による A_t^(J) を考えると、有界で J-dependent である。時刻を J 個の residue class に分ければ各列は iid なので strong law が適用でき、時間平均は E A_0^(J) に収束する。無限 moving average との差は以下と同型の O(a^J+b^J) の一様上界を持つため J→∞ で (7) を得る。

### zero initialization と通常の bias correction

m_0=v_0=0 から開始した通常の moments を m_t^0,v_t^0、補正値を

  mhat_t=m_t^0/(1-a^t),   vhat_t=v_t^0/(1-b^t)

とする。共通の future iid 列で定常解に couple すれば

  m_t^*=(1-a^t)mhat_t+a^t m_0^*,
  v_t^*=(1-b^t)vhat_t+b^t v_0^*.

有限補正平均も凸結合だから |mhat|≤M、z_min≤vhat≤M²。よって

  |mhat_t-m_t^*|≤2M a^t,
  |vhat_t-v_t^*|≤M² b^t.

h(m,v)=m/(sqrt(v)+ε) のこの support 上の偏微分を評価すると

  |h(mhat_t,vhat_t)-h(m_t^*,v_t^*)|
    ≤2M a^t/sqrt(z_min)+M³ b^t/(2z_min^(3/2)).        (8)

従って bias-corrected Adam も、その Cesàro 平均は同じ負の定常期待に収束する。初期の bias correction はこの反転を除去しない。通常の uncorrected Adam も有限初期誤差が幾何減衰するため同様。

固定 η>0 の driver-driven 更新 θ_{t+1}=θ_t-η A_t なら

  (θ_T-θ_0)/T → -η E A_0 >0

となる。ただしこの結論の前提は **g_t の法則が θ_t の変化と無関係に固定**されていることである。actual CE/CNN でこの前提を維持する追加議論は本証明にはない。減衰学習率についても、(7) だけから任意の重み付き平均の結論を無条件に主張しない。

## 9. 最終監査の留意点

この反例は「長期平均なら Adam でも符号保証が自動的に戻る」という一般化を排除するのには十分である。一方で、実 RLCIFAR の長期沈降を否定・証明するものではない。

厳密な符号定理を実訓練へ接続する際は、新ラベル CE gradient の期待だけでなく、Adam の moments と current-state gradient の joint law、または current-state 条件付きの optimizer projection を含める必要がある。

補助 MC の文章には一点だけ軽微な明確化を提案した。「厳密な期待値区間との差が約 1.6 SE」という表現より、「Taylor 中心との差が約 1.6 SE、厳密期待値区間までの最短距離は約 1.1 SE」が正確。真の期待値は区間のどこかであり、中心そのものとは証明していない。符号証明には影響しない。
