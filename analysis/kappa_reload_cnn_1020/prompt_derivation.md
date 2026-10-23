あなたは Codex gpt-6-astra。役割は「独立の導出ノート」。この worktree（/home/issan/Projects/claude/wt/kappa_reload_cnn_1020）は git の作業ツリー。spec は `specs/spec_kappa_reload_cnn_1020.md`、MLP 側の定義は `analysis/kappa_reload_1020/measure.py` と `analysis/reload_dial_1020/analyze.py`（`kappa_parts`）。読むだけで、コードやデータは変更しない。答えは **日本語** で、`analysis/kappa_reload_cnn_1020/codex_derivation.md` に書く（このファイルだけ作成してよい）。数式は Markdown のインライン TeX でよい。

## 記号
N 枚の入力、K クラス、最終隠れ層の出力 φ_n ∈ R^H、線形 head ℓ_n = V φ_n + b（V ∈ R^{K×H}）、π_n = softmax(ℓ_n)、旧ラベル y_n（one-hot）、CE の平均 L = −(1/N)Σ_n log π_{n,y_n}。クラスで中心化した V^c = V − mean_c V（列ごと）、b^c = b − mean b。F = Σ_n (y_n − 1/K) ⊗ φ_n ∈ R^{K×H}。κ = ⟨V^c, F⟩/‖F‖²（Frobenius）。A = ⟨V^c, F⟩。B = Σ_c (n_c − N/K) b^c_c（n_c = クラス c の枚数）。

## 頼むこと
1. **κ の下界が構造によらないことの証明**: 恒等式 A = Σ_n[(ℓ_{n,y_n} − ℓ̄_n) − b^c_{y_n}]（ℓ̄_n = mean_c ℓ_nc）と、Jensen（logsumexp ℓ_n ≥ ℓ̄_n + log K）から A ≥ N(log K − L) − B、ゆえに κ ≥ [N(log K − L) − B]/‖F‖² を、φ_n がどんな網（CNN・Transformer・何でも）の出力でも成り立つ形で証明する。使った仮定を全部列挙する（線形 head・CE・one-hot ラベル・それ以外に何が要るか／要らないか: 最適化法、収束、ラベルの分布、重み減衰の有無）。等号条件も書く。
2. **「κ > 0」が H1（課題の末に読み出しと今のラベルの対応が正）の補題になる条件**: L < log K − B/N なら κ > 0。B/N の大きさの見積もり（head の bias の中心化した成分 × クラス数の偏り）と、それが小さい理由（一様ラベルで n_c − N/K = O(√N)）。
3. **conv 層・中間 fc 層では κ が定義できないこと、何に置き換わるか**: チャネル j の平均前活性 m_j は下流を非線形に通って logit に届く。画像 n の logit の感度 R_n = D f_n[u_j] ∈ R^K（u_j = ∇_θ m_j）を使うと、一様ラベルの期待の押しは G_j = (1/N)Σ_n ⟨π_n − 1/K, R_n⟩（既知の恒等式、`results/cnn_drive_verify_1009/summary.md`）。この形で
   (a) 最終隠れ層では R_n が入力によらない（R_n = g_in v_i の形）ので κ に戻ることを示す（S_exact_i = Σ_n K_n g_in ⟨v^c_i, π_n⟩ との対応。K_n がどこから出るか）。
   (b) 中間層では R_n が入力に依るので、「読み出しの装填」の代わりに何を測るべきか（例: 旧ラベルへの R_n の整列 κ_R = Σ_n⟨R_n^c, y_n − 1/K⟩ / Σ_n‖y_n − 1/K‖² のような量、またはその符号条件）を提案し、それが G_j の符号について何を言えるか（言えないか）を導く。
   (c) 下界 (1) に当たるものが中間層で作れるか。作れないなら、何が障害か（非線形・入力依存・上流項）を一段ずつ。
4. **上流項 G_up の符号について先験的に言えること**: 最終隠れ層の m_i = W4[i]·x̄ + b4[i] で、x̄ が上流の一歩で動く分 G_up_i = ⟨∂m_i/∂θ_up, ∂L_u/∂θ_up⟩。これが G_own と同符号になる条件・ならない例（c1_direction_1010 の定数モードと同じ構造か）。短くてよい。
5. **Keep（押しの大きさの比）に対する理論の予想**: κ_k ‖F_k‖ と |S_exact| の中央値の関係。κ が縮んでも押しが縮まない／縮む条件。

出力の形: 定理・補題は「主張 → 仮定 → 証明 → 等号条件 → この箱で満たされているか（spec の情報から言える範囲）」。推測は推測と書く。一文一主張。
