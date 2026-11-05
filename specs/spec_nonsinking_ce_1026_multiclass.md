# 実行前追補: 10 クラスの通常 CE（2026-10-26）

主 spec の実行前に追加。二値 CE だけの特殊性を調べる。主 spec は書き換えない。
C=10、N=10、H=10、入力 `x_n=sqrt(2C)(e_n-1/C)`（R^C の simplex）。
W_i=0.1 x_i、b_i=-0.25、読み出し V_ik=0.5 if i=k else 0、出力 bias=0。
全 W,b,V,c を通常の mean softmax cross entropy と full-batch SGD lr=0.05 で 20,000 steps 更新する。
各 unit が自分のクラスだけで開くのは初期条件であり、ゲートを固定しない。

解析的予測: 拡張 Gram は対角 2C-1、非対角 -1。active 値 A と inactive 値 B、読み出しの自クラスと他クラスとの差 d に対し
`q=(C-1)/(exp(A*d)+C-1)`、
`A'=A+eta*(2C-1)*q*d/C`, `B'=B-eta*q*d/C`, `d'=d+eta*A*q/(C-1)`。
A,d は増え CE→0。平均 μ は増え、r=μ/σ は 1/(2 sqrt(C-1))=1/6 へ下から近づく。
μ/r、CE、accuracy、正側率、実ゲート変化、上記再帰の照合誤差を記録する。
独立な NumPy softmax CE 全パラメータ勾配を autograd と照合する。
全 step の単調性・有限走と永続性の区別・退避方法は主 spec と同じ。
