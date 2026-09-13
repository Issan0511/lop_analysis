# ELUの沈下・W増大・学習余力：3実験の結果（2026-09-13）

3本を事前登録して実行し、元データからの再集計と独立監査を完了した。すべて ELU(alpha=1) と leaky ReLU(slope=.1)、3 seed。**「ELUなら沈下は無害」という一般化はできない。ただし、深くしたときの精度水準の低下と、タスクを重ねる進行的なLoPは区別する。**

| 問い | 今回の結果 |
|---|---|
| 中心化Wノルムが同じでもELUを深くすると害があるか | 制御した平均前活性−1→−4で到達精度が約1.8–1.9ポイント低下。自然な沈下が進行的LoPを媒介する割合は未同定。 |
| W増大の害はELUで小さいか | 到達精度では深さにより比較の向きが変わり、登録した実質差判定は不確定。副次の適応中テストCE面積では両深度ともELUで悪化が小さい。 |
| ゼロ近傍に残る小さいWの集団が支えるか | 小さいWとの共通部分を凍結した主比較はseed間で混在。ゼロ近傍25個全体の更新を止めると、両活性化で学習曲線が悪化。 |
| 環境依存か | 同一画像・更新予算でもPermutedとRandom Labelで大きく異なる。RLのELUは早期に崩れ、後からのW1制限では機構を判別しにくい。 |

![3実験の主比較](three_studies.png)

Aの点はseed、B/Cの誤差棒は3 seedのt95%区間。Cの小さいELU応答は無害性の証拠ではない。PMの天井とRLの崩壊後の床を下の全軌道と合わせて読む。

## B：Wと平均深さを別々に固定

既存のCPU Permuted MNIST軌道task20から分岐しtask120まで実行。各ユニットの中心化W1行ノルムを5/10、実際の訓練入力に対する平均前活性を−1/−4へ、毎タスク冒頭と各Adam更新後に固定した。行平均はtask20に固定し、biasで前活性の平均を調整する。評価は各task625更新後、late=tasks101–120、seed別固定test 2048例。

| 固定したWノルム | ELU 平均−1の精度 | ELU 平均−4の精度 | 深くしたことによる低下［95% CI］ |
|---|---:|---:|---:|
| 5 | 92.495% | 90.693% | 1.802 pp［1.245, 2.359］ |
| 10 | 91.471% | 89.554% | 1.917 pp［1.459, 2.376］ |

同じノルムで平均を負へ動かす介入にも害があった。ただし全ユニットの平均を一様に固定するため、自然な個体差や旧dclampとは違う条件である。「自然に生じる沈下を防げばLoPの何割が消えるか」の答えではない。

進行的劣化Lはearly21–40とlate101–120の同予算精度の差。固定4条件のELUでは−0.317〜+0.151 ppと小さいのに対し、未介入は+1.587 pp（leaky未介入+1.646 pp）。**深さによる水準低下があることと、固定した深さで進行的劣化が続くことは同じではない。**

W5→W10の害を比較すると：

| 指標：ELUのW効果−leakyのW効果 | 平均−1 | 平均−4 |
|---|---:|---:|
| 到達精度低下の差（pp） | −0.478［−0.993, +0.038］ | +0.720［+0.058, +1.382］ |
| 適応中テストCE面積増加の差 | −0.04456［−0.08002, −0.00910］ | −0.03871［−0.06575, −0.01167］ |

負はELUのW増大の害が小さいことを表す。精度の主判定は登録した±0.5 ppの実質差基準で両方INCONCLUSIVE。深さによる差の変化Jも+1.198 pp［0.453, 1.943］で同判定。副次指標のCE面積では、仮説に沿った方向が両深度で出た。

CE面積はtasks101–120の各タスクでstep0,20,100,300,625を台形積分し625で割った値。**初期CEを揃えたり差し引いた指標ではないため、純粋な学習速度とは呼ばない。** また平均深さを固定しても分布の裾・飽和は動くので、このW効果を「沈下・飽和を一切介さない成分」とは呼べない。活性化間では各自のtask20状態から開始しており、W方向・読み出し・Adam履歴まで同一ではない。

出所：[B summary](../elu_depth_width_0913/summary.md)、[verdict.csv](../elu_depth_width_0913/verdict.csv) の late_test_acc / late_adapt_ce_auc。独立再計算：[audit.json](../elu_depth_width_0913/audit.json)、[auc_audit.json](../elu_depth_width_0913/auc_audit.json)。

## A：ゼロ近傍の集団の更新だけ止める

8固定置換で測った実際の P(|z|≤1) 上位25個と、中心化Wノルム下位25個の共通部分を候補にした。旧band_omegaのゲート基準の帯とは異なり、ELU/leaky共通の座標。既存task20/task100の同一関数・Adam・RNGから6分岐、10新タスク各625更新。W1行とbiasの実更新を止め、他のパラメータは学習する。対照は個数と現在の出力寄与・W2列ノルムを近づけた別集団。

| ELU task100 | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| 小さいW×ゼロ近傍の候補数 | 4 | 9 | 5 |
| 候補凍結−対応対照のCE面積 | +0.004910 | −0.002854 | +0.004420 |
| ゼロ近傍25個凍結−対応対照のCE面積 | +0.025773 | +0.013409 | +0.016694 |

小さいWとの共通部分の主判定はMIXED。task20では候補1–2個で登録した最小数3に届かず、存在不足として扱った。閾値を広げていない。

副比較のゼロ近傍25個はELU/leakyとも3 seedで追加のCE面積悪化があり、開始関数が全く同じ最初の新タスクでも同方向。ただしELUは625更新後の精度でほぼ追いついた。task100から10新タスク平均の到達精度低下は−0.0011、−0.0590、+0.0011 pp。**「浅い集団が適応中の性能を支える」所見であり、「小さいWの予備集団がELU特有の優位を作る」とはまだ言えない。**

対応対照は近似である。現在の寄与 |ΔCE| の群／対照の比はELUの広い浅い群で0.87–1.06、leaky seed0は1.65で十分に揃っていない。このため活性化間の効果量を厳密な同条件比較として読まない。

出所：[A summary](../elu_reserve_0913/summary.md)、[verdict.csv](../elu_reserve_0913/verdict.csv)、[contrasts.csv](../elu_reserve_0913/contrasts.csv)（checkpoint100、target_minus_matched / shallow_minus_matched）。到達精度と最初のタスクは[all_seed_summary.csv](../elu_reserve_0913/all_seed_summary.csv)（checkpoint100。到達精度低下=100×[SHALLOW_MATCHED−SHALLOW]のheldout_acc、最初のタスクCE面積=SHALLOW−SHALLOW_MATCHEDのtask1_auc_ce）。対応診断は[matching_balance.csv](../elu_reserve_0913/matching_balance.csv)。後二者は登録指標の補足表示であり新しい主判定ではない。

## C：同じ画像・同じ計算量で環境を変える

同じMNIST 1200枚、784-100-100-10 MLP、Adam、50タスク各6000更新（80 epoch）。Permutedは毎タスク画像置換＋真のラベル、Random Labelは画像固定＋毎タスク新しいランダムラベル。ELU/leaky×無介入/W1幅制限×2環境×3 seed=24モデル。W1中心化ノルム制限はtask10終端の値をtask11から維持。PMの旧10000枚1周、RLの旧400 epochとは異なる有限予算比較である。

![環境ごとの精度水準と第2層ゲート](../elu_environment_0913/environment_levels.png)

登録したLoP量Lはonline精度のearly11–20−late41–50。D=L(無介入)−L(W1制限)、正なら制限で減衰が小さくなったことを表す。

| 環境 | D：ELU | D：leaky | leaky−ELU |
|---|---:|---:|---:|
| Permuted | +0.291 pp | +0.502 pp | +0.211 pp［0.145, 0.277］ |
| Random Label | −0.197 pp | +4.806 pp | +5.003 pp［3.464, 6.542］ |

活性化間差の環境差J（PM−RL）は−4.792 pp［−6.277, −3.307］、登録判定ENV_DEPENDENT。ただしこのラベルだけで機構を結論しない。

- PMはonline約99%、タスク終端の訓練精度100%という天井に近い。
- RLのELUは介入前に崩れていた。無介入のタスク終端訓練精度はtask1 98.83%、task5 100%、task10 31.61%、task50 11.11%。単調な低下ではない。登録したearly11–20ですでにonline11.71%、late41–50は10.22%。
- RLではleakyもearly到達精度84.66%で予算不足フラグ。全seedを残して集計した。
- RLのELU第2層では、低ゲート占有率 P(phi'<.05) がtask10 98.70%、task50 99.997%へ増え、同じ窓の平均前活性は−43.25、−154.13。入力・ユニット・seed平均の記述量であり、沈下が原因と証明する介入ではない。今回の制限対象はW1のみ。
- D_ELUが小さい／負でも「W増大が無害」ではない。制限は崩壊後に入り、床で改善余地が乏しい。Dが負なのは早期と後期の差の差で、後期の精度自体を悪くした意味でもない。

**環境依存の疑いは具体化した。特に「MNISTなら一律にELUに有利」ではなく、今回の反復・更新予算では、タスク構成により挙動が大きく変わる。** この実験は「ELUの第2層沈下を早期から防ぐと回復するか」を実施していない。

出所：[C summary](../elu_environment_0913/summary.md)、[verdict.csv](../elu_environment_0913/verdict.csv)、[levels.csv](../elu_environment_0913/levels.csv)（env/act/iv、early/late）。全軌道：[rows.csv](../elu_environment_0913/rows.csv)（train_acc、task1/5/10/50）、層別：[selected_task_diagnostics.csv](../elu_environment_0913/selected_task_diagnostics.csv)（RL/ELU1/ref、task10/50、layer2低ゲート・前活性）。単点の選択は登録済み測定の事後の記述表示で、主判定を置き換えない。[解釈上の限界](../elu_environment_0913/interpretation.md)。

## 実行・検証・保存

- A：72分岐、3600曲線測定行。task20/task100の元走に保存された5種の個体別配列（cnorm_i, m_i, bias_i, zcur_i, sdcur_i）との照合は全件厳密一致。同一snapshotからの分岐状態も完全一致。候補ID・対照・probe分離・凍結後の実更新・CE面積を検証。
- B：30軌道、18120曲線測定行。元の参照120タスク×6 workerの720ノルム配列が厳密一致。中心化ノルム相対誤差最大2.49e−8、実際の訓練入力平均誤差9.40e−7。主効果と副次CE面積の再集計も一致。
- C：24モデル×50タスク、1200タスク行、7200曲線行、700ユニット配列。各モデル300000更新。介入前の対応腕はパラメータ・Adamとも厳密一致。ノルム相対誤差最大2.38e−7。
- GPUの初期数値検証で失敗した試行も保存。歴史的初期化の演算順を完全に合わせ、元の許容誤差のままランダムラベルの検証を通した後に本走した。許容誤差を緩めていない。BではSSHが一度切れたがremote workerは完走し、再実行はしていない。
- 3 seedのpilot。seedを独立反復とし、タスクを独立sampleに水増ししない。今回の登録内容・主判定は結果を見て変更していない。

[統合検証](verification.json) / A [独立検証](../elu_reserve_0913/independent_consistency_audit.json) / B [独立検証](../elu_depth_width_0913/audit.json) / C [provenance](../elu_environment_0913/provenance.json)。

実験repo: https://github.com/Issan0511/lop_analysis （branch codex/elu-reserve-0913）
SSH worktree: /home/issan/Projects/claude/elu_reserve_0913
宿主base: 46b2b71、データ: /home/issan/Projects/claude/proj_004_drift/data/mnist

| 走 | 正式spec | 本走前の事前登録commit |
|---|---|---|
| A | specs/spec_elu_reserve_0913.md | b5a64757db2a790502b1947c76c4a304e28d1d8e |
| B | specs/spec_elu_depth_width_0913.md | efe777b065792184555a8a361267a326ede51272 |
| C | specs/spec_elu_environment_0913.md | 661651a94231af0c0c80551b8046a219f087a243 |

本走前の実装commitはA c04b11e、B 7c1d1d4、C a3a041c。Cの技術QA追補もa3a041cに登録。実行コードのSHA、データSHA、RNGは各provenanceに保存。集計・図は src/elu_three_studies_report_0913.py。

再解析用の .pt/.npz は元worktreeに加え、同一hostの別ディレクトリ /home/issan/Projects/obsidian-research-data/elu_three_studies_0913/files/ に相対構造を保って複製し、全件SHA256一致を確認した。Gitには表・コード・図・監査・[raw_manifest.json](raw_manifest.json)を保存し、cloneにrawバイナリが含まれないことを明示する。manifestの各repo_pathへ戻せば、集計と監査を再実行できる。同一host内の保管であり、別hostへの災害対策バックアップではない。

今回閉じていないのは、自然な沈下の媒介効果、小さいW集団のELU特有の保護効果、RLで崩壊前から第2層を制御したときの回復である。これらを今回の結果から既成事実にしない。
