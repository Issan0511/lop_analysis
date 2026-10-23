あなたは実装と計算の担当です。日本語で報告してください。作業ディレクトリは git worktree `/home/issan/Projects/claude/wt/chargpt_l0_carrier_1021`（ブランチ claude/chargpt_l0_carrier_1021）。commit はしないでください（親の Claude が commit します）。

# 依頼
`specs/spec_chargpt_l0_carrier_1021.md`（commit 済み ac28a29f、予測は書き換えない）の **(A) L0 の沈みの経路** と **(B) L1 の担い手の出所** を実装・計算し、予測 P-A1〜P-A5・P-B1・P-B2 を採点してください。(C) は別セッションが回すので触らないでください。

# 書いてよい場所（ここ以外に書かない）
- `analysis/chargpt_l0_carrier_1021/`（コード）
- `results/chargpt_l0_carrier_1021/`（checks.json・表の CSV・summary.md。unit ごとの配列は `units/` に npz で、小さく）

# 読むだけの場所
- データ: `/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/M2/{C3,ReLUcap}/seed{0..9}/ckpts/`（init.pt、task<k>.pt = 課題 k 末、task<k>_u100.pt = 課題 k の更新 100、*_adam.pt）と同じ階層の per_task.csv・unit_stats.npz。HEAD は a970f12c。絶対に書かない。
- 分解の手本（流用してよいが写さず import する）: `/home/issan/Projects/claude/wt/chargpt_cl_1019/analysis/chargpt_cl_1019_quicklook/decomp_dmu.py`（窓の定義、probe、Δμ = Δw·ȳ + w·Δȳ + Δw·Δȳ の割り方、σ_a で割る、沈む unit の限定）。L1 の担い手の出所の手本は同じ quicklook の解析（残差の各項の token 平均を LN2₁ の後に射影する。ファイルは `analysis/chargpt_cl_1019_quicklook/` を探す）。モデルは `src/chargpt_cl_1019.py`・`src/chargpt_cl_1019_kune.py`（cap の腕は `K.C`。2 ブロックは環境変数 CHARGPT_NLAYER=2 で作る。cap の腕の読み込み方は `chargpt_cl_1019_kune.py` の cap の実装と、quicklook の cap_summary.py を見る）。
- 定義は spec §2 を正とする。probe は課題の鎖の probe（`probes_M(chain, seed, task)`）。両端とも同じ probe。
- Python: `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`、CPU（`CUDA_VISIBLE_DEVICES=""`、`torch.set_num_threads(8)`）、`PYTHONDONTWRITEBYTECODE=1`。

# 計算するもの（spec §2）
(A) 両腕 × seed 0〜9 × 層 L0・L1:
- 窓 (i) 切替直後 100 更新（課題 t−1 末 → 課題 t 更新 100）と (ii) 課題の残り（更新 100 → 課題 t 末）、t ∈ 1〜10・40〜50・90〜100。粗い窓 10 末→25 末→39 末も（報告のみ）。
- 各窓で Δμ_i を 自分 Δw·ȳ_a ／ 上流 w_a·Δȳ ／ 交差 に割り、σ_a で割る。unit の平均と、沈む unit（Δμ < 0）に限った平均。負の量（負の成分の絶対量）の上流の割合も。
- 窓ごとの |w_a·Δȳ| の中央値、‖w_a‖ の中央値、‖Δȳ‖、揃い |w·Δȳ|/(‖w‖‖Δȳ‖)。
- 課題ごとの回転 cos(w_i^{(t)}, w_i^{(t−1)}) の中央値（課題末どうし、連続する課題末があるところ）と lr/‖w_i‖ の中央値。
- データそのものの差: 同じ重み（課題 t 末）で課題 t の probe と課題 t+1 の probe の m_i の差（Δμ/σ）。
(B) L1 の担い手の出所（両腕 × seed 0〜9、課題 1〜10・40〜50・90〜100 の課題末）: ȳ₁ に占める 埋め込み・注意 0・FFN0・注意 1 の射影の割合。

# 検査（先に通す。通らなければ止めて理由を書く）
- 恒等式の残差 ≤ 1e−10（float64）。
- seed 1000 の C3-2 の L1（quicklook/M2/C3/seed1000）を同じコードで計算し、上流の割合 0.958・集中 0.937（課題 2〜10 の切替窓）を再現する（許容は丸め）。
- cap の腕で L0・L1 の幅 s_i が課題 1 末の値を超えないこと（cap の検査の再掲、1 seed でよい）。
- 変異対照: Δw を 0 にした窓で分解の上流の項だけが残ること、probe を入れ替えたときデータの差の項だけが変わること。

# 出力
- `results/chargpt_l0_carrier_1021/checks.json`、`table_decomp.csv`（腕 × seed × 層 × 窓）、`table_scale.csv`（|w·Δȳ|・‖w‖・‖Δȳ‖・揃い・回転・lr/‖w‖）、`table_data_shift.csv`、`table_carrier_source.csv`、`summary.md`（日本語。定義・検査・表（seed の中央値と 8/10 の数え方）・予測 P-A1〜P-A5・P-B1・P-B2 の採点（○・✗・判定不能と理由）・言えること／言えないこと。固定状態の帳簿を因果で書かない。数は表へ、一文一主張）。
- 最終回答には summary.md の要点（表と採点）を書いてください。
