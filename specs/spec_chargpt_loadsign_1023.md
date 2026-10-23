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
