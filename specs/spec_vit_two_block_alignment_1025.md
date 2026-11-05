# 2blockの整列介入と通常学習の対応（1025）

T5の初回1条件（seed0、課題10、100更新、第1block、入力固定・固定計量）を既に閲覧した探索的追試。そこでゲート共分散項は下向きだが小さかった。これを全条件へ一般化せず、以下の整列測定の前にspecとコードをcommitする。

## 対象

基準走seed0–4、課題10/60、更新100/500の計20状態。両blockで、同じ学習画像250枚上のCLS μ・σ・rを測る。これはその起点から独立一様な次ラベルへ切り替える仮想実験で、実際に保存された課題開始0→20更新の変位と区別する。

切替の入力は (a)同じ学習画像、(b)次課題11/61の画像。bでも整列は旧課題の画像・ラベルから定義し、位置を測るprobeは同じ旧画像250枚に固定する。

## 同一状態のhead介入

最終分類headのclass-centered行列Vc、最終特徴H、旧ラベルQに対しF=(Q−1/C)^T H、L=κF、κ=<Vc,F>/||F||²、R=Vc−L。

- base: Vc
- half: R+0.5L
- zero: R
- norm_unload: RをVcと同じFrobenius normへ拡大
- perm0–3: 旧ラベルを画像間で置換して作る方向をRとLに直交化し、Lと同じnormへ揃えてRに加える

biasとheadのclass平均は維持する。置換のseedは`stream_seed('two_block_alignment', seed, task, start, j)`。各headで予測pを再計算し、独立一様ラベル期待の損失`logsumexp(f)−mean(f)`を逆伝播する。headのnorm・直交制約・整列係数・CE・pmaxを記録する。これは最終headの一つの整列定義に対する操作であり、各中間FFNのfc2の全ての整列を除いた実験ではない。

元headについてだけ、確率pを固定した`load=(p−u)L/N`と`remainder=(p−u)R/N`の分解を併記する。確率を再計算する上の介入とは別。通常学習は`fit=(p−Q)Vc/N`。

## 位置への押しと実Adam

SGDと起点Adam分母固定・momentumなしの二つで、全bodyを通る一次のΔμ、Δσ、Δrを測る。学習率1e−4を掛け、負が沈降。自fc1だけの寄与も解析式で測り、全体−自fc1を上流の寄与とする。正確な一次の加法分解であり、有限更新の因果割合とは呼ばない。

各介入で元Adamの全状態を復元し、全画像の期待勾配に対する1更新を実torch.optim.Adamで行う。通常学習の1更新も測る。これはminibatch Adam更新の期待値とは異なる。実Δμ・Δσ・Δrを同じprobeで測る。checkpointを変更しない。

旧6blockと同じ記述判定: 課題・起点・block・入力条件別に、固定計量でbaseのunit平均Δr<0、かつ4permutation対照平均よりbaseが下向き、が4/5seed以上なら局所的な整列効果あり。finite Adamでも対応するseed数を別に示す。half/zero/norm_unloadの方向も示し、単なるhead縮小と区別する。整列が全沈降の必要十分原因とは言わない。

unitの対応は、同じ旧画像上のfitとloadの全経路・自fc1のΔrについて、共通valid unitで2×2の符号割合、反対向きの割合、周辺割合から独立に組み合わせた場合との差、cosine・順位相関を示す。数値ゼロはmax(1e−12, 1e−6×最大絶対値)以下とし、除外数を記録する。T5との比較を追加する場合はμの共分散源をμの整列源と比較し、T5のμをrの寄与と読み替えない。

gateはsynthetic checkpoint（seed97、課題2、25更新）。全head制約、body forwardの一致、load+remainder=baseの位置、局所解析式とJVP、直接の期待CE逆伝播とbody信号逆伝播の一致、測定が元のparameterを変えないことを確かめる。gateの数値を本結果へ混ぜない。元6blockで検算済みのbody/probe/JVP関数はそのまま再使用し、source SHAを記録する。

恒久データは`obsidian-research-data/vit_two_block_1025/alignment/`。保存状態ができた順に実行しても20状態全てを対象とし、結果による選別をしない。
