# よく開いているが勾配の小さい個体: 保存状態の事後解析

状態: 事後解析・新規学習なし。元の20seed ReDo/noneの保存状態を用いる。問いはIssa「めっちゃ開いていて勾配が小さいやつっていますか？」。

第3課題10/50/780更新、第10/20/29課題末を解析する。開口率p>=.95と>=.99を記述上の高開口条件とし、現在課題の全画像でのpを使う。独立な検定や仮説の事前登録とは呼ばない。

保存した現在課題の固定256画像（モデルとラベルを保持）でCEの次を計算する。

- 各入力の逆伝播信号 δ_ni=1[z_ni>0] v_i^T(prob_n-y_n)。RMS、平均絶対値、平均を分ける。256枚平均lossによる1/256因子をかける前のper-example勾配であり、ReLUの局所微分そのものではない。
- incomingパラメータ勾配: gW_i=mean_n δ_ni h1_n、gb_i=mean_n δ_ni。両者の結合L2norm。
- outgoing勾配: gv_i=mean_n a_ni(prob_n-y_n) のL2norm。
- class-centered norm(v_i)、誤差RMS、δRMS/(norm(v_centered)*errorRMS)を併記し、vの小ささ・当てはまり・誤差方向との弱い揃いを区別する。
- 小さい勾配の記述閾値δRMS<1e-3と<1e-4、incoming/outgoing両norm<1e-3を報告する。閾値は可塑性喪失の定義ではない。ゼロ出力重みのreset直後は主要結論から除く。
- 保存された実ミニバッチのbias勾配と、本人W+biasによる位置の実更新も別表で記述する。Adamでは小勾配から小更新や凍結を推定しない。位置の更新量はパラメータ更新全体のnormではない。

図と集約値、具体例のID、ソースhash、個体配列を保存する。ソースは /home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019/raw/ で、元配列は変更しない。コードと結果はmainへ保存し、Obsidian測定ノートに反映する。
