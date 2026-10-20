# raw/（kappa_reload_cnn_1020）

保存契約: `units_*.npz` は unit 配列のみ。状態のスカラー（κ・‖F‖・‖v^c‖・A・B・L・D・J・射影割合・K_n の統計・検査の値など）は `../rows.json`、採点に使った数値床は `../floors.json` にある。監査や再集計は「raw の npz ＋ rows.json ＋ floors.json」で行う。

- `units_<腕>_s<seed>.npz`: 軸 0 は `t`（0 = 初期値、以下は保存境界）。m・sd・pplus・npos・alive・m_patch・G_full・G_own・G_up・G_own_jvp・G_blocks（c1 c2 f1 f2 head の (W,b) ブロック）・alpha は 232 unit（c1 16・c2 16・f1 100・f2 100 の順）。S_exact・S_self・S_oh・S_proj・kappa_i は f2 の 100 unit、S3 は f1 の 100 unit。符号集計の mask は保存しておらず、alive と floors.json の床から作り直す。
- `replay_<腕>_t<tt>.npz`: 再生の記録。steps × seed × 232 の m・sd・pplus・alpha、floor_dm（同じ状態の再評価差）、区間ラベル、hash と bit 照合（meta・fidelity）。`_b10` は SNA を 10 束で回した参考の走で、本表には使わない。
- `kappaR_<腕>_s<seed>.npz`: A1.5 の own 方向の感度の整列（kappa_R・eps・rho・e_pi・c_R・margin と検算の値）。
- `diag_*.npz`: Snake t1 の最初の 1 歩の分解、LR t1 で m = 0 にした複製の 1 歩。
- `returners_g1.json`: T7 で戻った seed-unit の一覧（p⁺ 基準）。
