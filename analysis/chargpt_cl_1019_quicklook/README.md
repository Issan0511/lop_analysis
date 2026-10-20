# chargpt_cl_1019 の下見（seed 1000、未登録）の解析スクリプト

1020 午後、Issa の問い「幅が MLP より爆発的に伸びる理由」「深い層の μ/σ」「なぜ 38 課題まで耐えるか」に答えるために書いた。
対象は `results/chargpt_cl_1019/quicklook/M/{KKT1,KKT1b3}/seed1000` と `crosscheck_white-san/M/C3/seed0`。全部 CPU で動く（走っている GPU を触らない）。
worktree の root で `python analysis/chargpt_cl_1019_quicklook/<script>.py`。

- kkt1_ql.py — 腕ごとの要約（verdict の seed_stats）・層別・up のノルム・5 課題ごとの超過損失
- width_decomp.py — σ_i = ‖w_i‖ × √(trΣ_y/d) × 揃い の分解、γ、入力の共通成分、壁 c・マハラノビス、残差 rms
- carrier_source.py — 残差の共通成分 x̄ を源（emb・attn・ffn）に分ける、Snake のオフセット (h̄−μ)/σ
- scale_drive.py — σ の伸びの指数、checkpoint での尺度微分 ∂L/∂log s とその共通成分の分
- replay_gamma.py — 課題 25 末＋Adam 状態から 1,000 更新を再生（切替/継続/α 凍結）、γ・W_up の動径の歩
- scale_scan.py — 切替の瞬間と収束状態で FFN 出力（共通/変動）を s 倍したときの損失
- mu_attrib.py — 課題 25→50 の Δμ を 自分の重み/上流 ȳ/交差 に割る、課題ごとの沈みの表
- b3_source.py — KKT1b3 の共通の書き込みの行き先（W_down h̄ と b_down の相殺）、x̄ の源
