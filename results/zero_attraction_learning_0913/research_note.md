# 零点復元と重み収縮：新規学習結果 0913

状態: 完了。19条件×10seed×500task（各5M更新）の学習、76 checkpointの機序・代数対照、7条件×2境界の自然更新再生、集計・バックアップを完了。

## 結論

原点で傾き2となるSnake `phi(z)=z+sin(2z)/2` は、通常Snakeよりlate task451–500の自由重み成長倍率が約9.4%低い。seed対応bootstrapの成長比95%CIは約0.894–0.917。登録判定はDIRECTIONAL_PHASE_SUPPRESSIONであり、20%以上の強い抑制判定には届かない。

しかし自由重みRMSは初期比1.656倍で、集団全体が初期より縮んだとは言えない。一方、個別unitの70.8%は縮み、最終task500のseed内unit成長倍率中央値の平均は0.0416倍。多数の縮小unitと、少数の増大unitに分かれる。

near-rootは「各unitの全32入力の前活性平均が零点±0.1」と「全32入力が零点±0.1」を区別する。lateでpeakの前者69.4%、後者62.2%。平均だけが0でも分布幅が小さいとは限らない。

![[可塑性喪失/測定/補助データ/零点復元_新規学習_0913/synthesis/unit_distributions.png]]

## 全条件を揃えて分かった範囲

late task451–500、全unit・10seedの平均。自由重み成長は初期比RMS、収縮率は各unitのlate平均二乗ノルムが初期未満の割合。

| 条件                | 自由重み成長 |  全W成長 | 収縮unit割合 | 平均が零点±0.1 |
| ----------------- | -----: | ----: | -------: | --------: |
| 通常Snake q0        |  1.827 | 1.148 |    27.2% |      7.1% |
| 原点傾き2 Snake q0    |  1.656 | 1.013 |    70.8% |     69.4% |
| 原点傾き0 Snake q0    |  1.825 | 1.137 |    30.1% |     16.8% |
| Leaky a=.1, q0    |  5.259 | 2.979 |     0.0% |      0.3% |
| Leaky a=.1, q=-.5 |  4.527 | 2.581 |     0.0% |      0.4% |
| Leaky a=.1, q=+.5 |  2.944 | 2.047 |     2.8% |     14.5% |
| Leaky a=.3, q0    |  3.401 | 1.963 |     0.1% |      2.1% |
| Leaky a=.3, q=-.5 |  2.883 | 1.685 |    11.2% |     13.3% |
| Leaky a=.3, q=+.5 |  3.875 | 2.230 |    31.5% |     11.5% |
| Leaky a=.7, q0    |  1.906 | 1.055 |    44.2% |     47.3% |
| Leaky a=.7, q=-.5 |  1.679 | 0.942 |    82.5% |     84.6% |
| Leaky a=.7, q=+.5 |  1.851 | 1.038 |    78.9% |     81.4% |
| 線形                |  0.594 | 0.378 |    93.5% |     81.3% |

全19条件の表は [[可塑性喪失/測定/補助データ/零点復元_新規学習_0913/summary.md]]。

Leakyでは定数の効果が傾きに依存する。a=.3ではq=+.5がq0より約14.0%増大（探索比較、10/10seedで増大）。q=-.5は3つの傾き全てで対照より小さいが、a=.1ではlateの収縮unitは0%。この条件では抑制と初期からの収縮が違う。

a=.7では両offsetで着座・多数unitの縮小が起きる。ただしlate MSEは約0.46–0.48で、a=.3,q0の約0.00017より高い。線形対照は集団全体も縮むがMSE約0.77を残す。小さいWを良い可塑性と同一視しない。

Snake q0 3条件のlate MSEは通常4.1e-11、peak6.4e-9、valley2.9e-11。全て非常に小さく、今回の課題だけではpeakによる可塑性維持の優位を示さない。

![[可塑性喪失/測定/補助データ/零点復元_新規学習_0913/learning_trajectories.png]]

## 自己項がcに届くまで

MSE `L=E[delta^2]`、`o_i=v_i phi(z_i)` の1隠れ層モデル。今回の自由入力方向をuとする。

`g_self=2 v_i^2 E[phi phi' x_free]`

`g_rest=2 v_i E[(delta-o_i) phi' x_free]`

誤差を変えてもモデル状態を固定すればg_selfは変わらず、g_rest側が変わる。自己項が縮める向きでも総更新で相殺されうる。

task区間について `I=-sum_s <u_s,Delta u_s>` をself/rest/roundへ分けると、

`c N D = I_self + I_rest + I_round + K`

`G = ||u_end||^2-||u_start||^2 = Q-2(I_self+I_rest+I_round)`

`D^2=Q+2K`、`Q=sum_s ||Delta u_s||^2`、`K=sum_{r<s}<Delta u_r,Delta u_s>`。

task21と101の自然更新再生は、7条件・各10seedで最終uとvが元ログとbyte一致。自己項の大きい正値はほぼrestの負値と相殺される。角度に残るのはその小さな差とKで、自己項だけからcの符号を決められない。cのunit単純平均とcNDの平均も別の量。

task21 peakは200更新時G=-0.00856だが、10000更新終端ではG=+0.02054。初期の収縮をtask全体の収縮と呼ばない。

task101・10000更新の全unit/seed平均では、通常Snakeは `I_self=4.228353, I_rest=-4.229546, K=0.0009328` から `cND=-0.0002603`、peakは `I_self=4.805010, I_rest=-4.801484, K=0.0001281` から `cND=+0.0036539`。丸め項も別保存し、最大誤差約1.35e-11でnorm収支が一致した。

peakのtask101では平均cNDが正でも、`D²=Q+2K=0.0077570` が `2cND=0.0073078` より大きく、平均Gは+0.0004492である。一般に `G=D²-2cND` なので、侵食側の角度だけで収縮を断定しない。

![[可塑性喪失/測定/補助データ/零点復元_新規学習_0913/synthesis/transition_ledger.png]]

## 誤差の大きさに関する代数対照

全19条件×4 checkpointの固定モデルで次を検算した。仮想targetを作る代数対照であり、新たな自然学習条件ではない。

- 同じ誤差RMSで、残差と `v phi'(z)(u・x_free)` の整列を反転すると、自由重み方向の勾配は符号反転。モデル状態を固定しているので自己項は同じ。
- 全活性化へq=.5を加え、現在の出力biasを `c_out-q sum(v)` で補償すると総W勾配は不変（最大誤差1.94e-15）。自己項とrestの配分は変わる。定数は通常deltaへ直接届くため、補償なしの一般論として「v経由しか効かない」とは言えない。
- v=0ではそのunitの入射W,b勾配が0。出射v自体の勾配が0という意味ではない。

したがって「誤差が大きいほど自己項が強くなる」は、状態を固定した式としては成立しない。誤差から変わるのは総勾配・相殺残差・更新幅であり、それを通してcとnorm収支が変わる。学習経過でvや前活性分布まで動けば、自己項自体も間接的に変わる。

## 解釈の範囲

- この研究でいう自由重みは入力末尾5bitへの重み。前活性分散は厳密に `Var(z)=||u||^2/4`。全20次元のWノルムと区別する。
- 局所的に `phi(z)=s(z-z0)` で、平均mと中心化入力xiの共分散Sigmaを使うと、自己項は `partial_m L_self=2v^2s^2(m-z0)`、`grad_u L_self=2v^2s^2 Sigma u`。傾きだけでなくv²が効く。
- 微小変位プローブの正の復元係数は、他unitとvを固定した学習済み状態への局所復元。原点への復元や自然な同時学習の因果効果と同一視しない。
- 初期W,b,v、教師、入力列は一致するが初期予測・損失は異なる。位相変更は傾きだけの介入ではない。
- lab PCの旧平行移動Snakeは実装を回収できず、この明示式による新実験を旧実験の忠実再現とは呼ばない。
- scalar MSE・plain SGD・CPU。CE/Adam・多層への一般化、自己項除去介入、同一初期出力比較は未実施。
- `<w_i,grad_w L>=E[delta o_i]` は一般のSnakeの入射重みにはそのまま使えない。今回のLなら一般式は `2 E[delta v_i phi'(z_i)(z_i-b_i)]`。出射vの式や正の1次同次の場合と分ける。

## 成果物と来歴

主spec: [[零点復元と重み収縮_学習実験_spec_0913]]
追加解析spec: [[自己項から侵食角cまで_自然更新追跡_spec_0913]]
旧ログの再解析: [[零点着座と重み収縮_事後再解析_0913]]
包括的な次段階設計: [[零点への復元はW増大を抑えるか_検証設計_0913]]

- 学習コード: `/home/issan/Projects/claude/zero_attraction_0913` branch `codex/zero-attraction-0913`、本走HEAD `0ba13e8`、主spec `1689251`。
- 解析コード: `/home/issan/Projects/claude/zero_attraction_analysis_0913` branch `codex/zero-attraction-analysis-0913`。
- 生ログ: 学習worktree `results/zero_attraction_learning_0913/logs/*.npz`。
- checkpoint: 同 `ckpts/*_step{0,200000,1000000,5000000}.pt`。
- 最終結果: 解析worktree `results/zero_attraction_learning_0913/`。途中集計は同 `partial_analysis/` に保存。
- 自然更新分解: 同 `natural_trace/`、`seed_ledger.csv` と `verification.json`。
- 個別unitプローブ: 同 `mechanism/`、`seed_summary.csv` と `verification.json`。代数対照とfixed15/bの大きさ: 同 `controls/`。
- Vaultの小さい集計・図: [[可塑性喪失/測定/補助データ/零点復元_新規学習_0913/summary.md]]。
- ログ列・checkpoint照合、分散恒等式、norm収支は検算済み。入力SHAは `analysis_verification.json`。
- Git commitはローカルのみ。rawはGitに含まれない。
- rawバックアップ: `/home/issan/Projects/obsidian-research-data/zero_attraction_0913/`。`training/` に190 NPZ・76 PTと実行ログ（全体約3.95 GB）、`analysis/` に個別unit分解と最終集計。
- SHA256 manifest: `training_backup_manifest.json`、`analysis_backup_manifest.json`。学習コードが本走HEAD/hashと一致することを終了後に確認。
- 解析コードHEAD: `422f9dc`。自然再生コードの変更履歴にはoffsetの共有参照復元修正がある。修正後に全14軌道の一致を確認し、不一致だった試行は結果に使用していない。
- 総合検算: `completion_verification.json`。主学習前に62テスト、全条件30kの有限性と再現性検査がPASS。

第一段階として登録した学習・checkpoint測定は完了。自己項を除く長期介入、同じ初期出力を固定した学習、CE/Adamは後続段階として保留する。
