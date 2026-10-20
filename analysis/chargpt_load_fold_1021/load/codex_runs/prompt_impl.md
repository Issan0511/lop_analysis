あなたは実装と計算の担当です。日本語で報告してください。作業ディレクトリは git worktree `/home/issan/Projects/claude/wt/chargpt_load_fold_1021`（ブランチ claude/chargpt_load_fold_1021）です。commit はしないでください（親の Claude が commit します）。

# 依頼
`specs/spec_chargpt_load_fold_1021.md` の §2（(A) 装填）と §4 の (A) の検査を実装し、計算し、予測 P0-1〜P0-7 を採点してください。spec は commit 済み（8122b8b1）で、予測は書き換えないでください。§3（(B)）は親がやるので触らないでください。

# 書いてよい場所（ここ以外に書かない）
- `analysis/chargpt_load_fold_1021/load/`（コード）
- `results/chargpt_load_fold_1021/load/`（結果: checks.json、表の CSV、summary_load.md）

# 読むだけの場所
- 定義の正本（同じ worktree の中、origin/main 由来）: `src/kappa_reload_cnn_1020.py`、`results/kappa_reload_cnn_1020/summary.md`、`specs/spec_kappa_reload_cnn_1020.md`、`analysis/kappa_reload_cnn_1020/codex_derivation.md`。κ・cos²・Q・帰無（クラス置換）・補題（A = ⟨v^c, F⟩ の恒等式、κ_lb）・G_own・G_full（JVP）・「生きた unit」の定義は、ここに揃えてください。文字 GPT に合わせて変えた所は summary_load.md に式で書く。
- 文字 GPT のコードとデータ（別セッションの worktree。絶対に書かない）: `/home/issan/Projects/claude/wt/chargpt_cl_1019`（HEAD f71be184）。
  - モデル: `src/chargpt_cl_1019.py`（埋め込みと head は重みを共有: `s.wte.weight = s.head.weight`。bias なし。LN は γ だけ）。活性化と cap の腕: `src/chargpt_cl_1019_kune.py`（`K.C` が本体のモジュール）。
  - checkpoint の読み込み・probe・鎖の作り方の手本: `analysis/chargpt_cl_1019_quicklook/decomp_dmu.py`（`probes_M(chain, seed, task)`、`uniform_probe_M(seed, device)`、`task_data_M` など）。unit の記録の意味は `src/chargpt_cl_1019.py` 425〜451 行（q は z < 0 の割合）。
  - checkpoint: `results/chargpt_cl_1019/quicklook/M/{C3,SiLU,GELU}/seed1000/ckpts/`（init.pt、task1〜10・25・39〜50 の課題末 task<k>.pt と更新 100 の task<k>_u100.pt があれば、Adam は *_adam.pt）、`M/{ReLUcap,SiLUcap,GELUcap}/seed1000/ckpts/`（init、task1・10・25・50）、`M2/{C3,SiLU}/seed1000/ckpts/`（2 ブロック、init と task1〜10・25・39〜50・89〜100）。`per_task.csv` に課題ごとの鎖のエントロピー H と記録。
- Python: `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`。CPU だけで回す（`CUDA_VISIBLE_DEVICES=""`、`torch.set_num_threads(8)` 程度）。GPU は別セッションの走が使う。`PYTHONDONTWRITEBYTECODE=1` を付ける（他の worktree に __pycache__ を作らない）。chargpt のモジュールは `sys.path` に `/home/issan/Projects/claude/wt/chargpt_cl_1019/src` を足して import する（ファイルを写さない）。provenance に chargpt の HEAD f71be184 を書く。

# 計算するもの（spec §2 のとおり。要点）
1. head の再整列: 各課題末 k で、課題 k の鎖の probe の token ごとに h_n = 最終 LN の出力、y_n = 次の文字の one-hot。F_k = Σ_n (y_n − 1/V) h_nᵀ。κ_k、κ_k/κ₁、cos²(v^c, F_k)、Q = κ‖F‖、帰無の分位点、補題の恒等式の誤差、κ_lb = N(log V − L)/‖F‖²（bias なしで B = 0）、破れの数、κ/κ_lb。init の状態でも同じ量。
2. 読み出しに隣接する FFN 層（4 ブロックは L3 = h.3、2 ブロックは L1 = h.1）の 512 unit: m_j、生きた unit、G_own,j（後ろ向き 1 回）、G_full,j（m(θ) の JVP 1 回、方向は ∇_θ L_u）、自己形 d_j の向きと G_own の符号の一致、沈める側（G > 0）の割合（生きた unit の中で）。L_u は一様の目標を一様 probe に当てた損失 mean_n[logsumexp f_n − mean_c f_nc]。
3. 初期化の押し: init での |G_own| の中央値 ÷ 課題 1 末での |G_own| の中央値。
4. P0-7: 課題末 k の状態で、課題 k+1 の鎖の probe と実際の次の文字（1 実現）で計算した G_own の沈める側の割合と、2. の値の差。
5. 参考に他の層（L0〜L2）の G_own の沈める側の割合も（4 ブロックの C3・SiLU・GELU だけでよい）。

# 検査（spec §4 (A)。全部通ってから本計算。通らなければ止めて理由を書く）
- 補題の恒等式の相対誤差 ≤ 1e−10（float64 で計算）。
- G_own（後ろ向き）と有限差分、G_full（JVP）と有限差分の一致（相対 1e−4、数 unit を抜き取り、float64）。
- m_j = w_j·ȳ の閉じ（bias なし）。
- 読み込んだ網の課題末の損失（課題 k の鎖で）が per_task.csv の記録と近いこと（評価の一致。許容は記録の量の定義を読んで決め、根拠を書く）。
- 変異対照: head の符号を反転した網で κ が負になる、一様目標の代わりに 1 クラスに全部置いた損失で G の符号の分布が変わる（空の検査を避ける）。
- 埋め込みと head の共有が読み込み後も保たれていること（同じテンソル）。

# 出力
- `results/chargpt_load_fold_1021/load/checks.json`、`table_head.csv`（腕 × 課題末）、`table_units.csv`（腕 × 課題末 × 層）、`summary_load.md`（日本語。定義・検査・表・予測 P0-1〜P0-7 の採点（○・✗・判定不能と理由）・言えること／言えないこと）。数は表へ、一文一主張。固定状態の帳簿を因果で書かない。
- 最終回答には、summary_load.md の要点（表 2 つと採点）を書いてください。
