# spec: 後期の切替の押しを装填成分と残りに分ける（chargpt_latepush_1023）

状態: 登録（走る前に判定・予測を固定）/ 2026-10-23 / 起案 Claude（Fable、V13）・実装 Codex 01a1676a / 親: chargpt_seed10_1022

## 1. 問い
C3-2（2 ブロック、ReLU、LN は γ のみ）の L1 への切替の期待の押しは、課題末 1〜9 で 83〜89% が沈める側、課題末 89〜100 で 46〜67%（chargpt_seed10_1022 A2、10 seed）。後期に向きが薄まるのは、(a) 装填 κF の押しそのものが弱るからか、(b) head に残る旧課題の成分 v⊥ が unit ごとに乱数の向きを足して薄めるからか。

## 2. 分解（恒等式）
各課題末の状態で、A と同じ定義の中心化読み出し v^c、現課題の遷移確率からの F、κ = ⟨v^c, F⟩/‖F‖² を取り、v^c = κF + v⊥（⟨v⊥, F⟩ = 0）と分ける。自分の重みの経路の期待の押し S_i(v) は v に線形なので、S_i(v^c) = S_i(κF) + S_i(v⊥) が unit ごとに成り立つ。生きた unit の集合、probe、新しい独立課題への期待の取り方、Adam の分母の扱いは load10.py と同一にする。

## 3. 量（arm = C3-2 主、ReLUcap-2 記録。layer = L1 主、L0 記録。状態 = 課題末 1〜10・25・39〜50・89〜100、seed 0〜9）
- s_full・s_load・s_res: 沈める側（S_i > 0）の生きた unit の割合。それぞれ v^c・κF・v⊥ で計算。
- share_load: 生きた unit の中央値 |S_i(κF)| / (|S_i(κF)| + |S_i(v⊥)|)。
- cos²(v^c, F)、κ、生きた unit 数。
- 窓: 早期 = 課題末 1〜9、中期 = 39〜50、後期 = 89〜100。seed ごとに窓の中央値を取り、seed 間の範囲で報告する。
- 相関: seed ごとに、全課題末（36 点）の (s_full − 0.5) と cos² の Spearman。

## 4. 判定（走る前に固定。主は C3-2 の L1）
- J1 LOADING_SINKS_LATE: s_load の後期の中央値 ≥ 0.75 の seed が 8/10 以上。満たさなければ NOT_LOADING（後期は装填の押しそのものが弱い）。
- J2 RESIDUAL_DILUTES: s_res の後期の中央値が 0.40〜0.60 に入る seed が 8/10 以上、かつ share_load が早期 → 後期で下がる seed が 8/10 以上。片方だけなら PARTIAL。
- J3 TRACKS_COS2: Spearman > 0.5 の seed が 8/10 以上。
- 記録（判定外）: L0 の s_load は早期から 0.5 未満か（A の記録 0.29〜0.34 の再現）。ReLUcap-2 の L1 で J1〜J3 が同じか。J1 が外れた場合は、K_n（位置の係数、load10.py の定義）の符号の割合を課題末ごとに出す。

## 5. 予測（走る前。Fable、確率）
J1 ○ 0.65。J2 ○ 0.60。J3 ○ 0.55。L0 の s_load は早期から 0.5 未満 0.70。Issa の予測はこの下に Issa が書く。

## 6. 言わないこと
固定状態の一歩の期待の帳簿で、因果の寄与率ではない。実 100 更新の沈みの分担ではない。後期の L1 は大半が死んでいる時期があり、生きた unit が少ない状態は n_alive を併記して読む。

## 7. 置き場
spec: specs/spec_chargpt_latepush_1023.md。code: analysis/chargpt_latepush_1023/。results: results/chargpt_latepush_1023/（states.csv: seed, arm, task, layer, n_alive, kappa, cos2, s_full, s_load, s_res, share_load。summary.md、judge.md、judge.json）。checkpoint は chargpt_seed10_1022 の束（obsidian-research-data）から展開し、git に入れない。結果ノートは vault 可塑性喪失/測定/。

## 8. 実装上の確定事項（実装・計算前、Codex 2026-10-23）

原文 §1〜7 を保持したうえで、元コードとの対応と曖昧な点をここに固定する。

1. **点数**: `load10.TASKS` は列挙どおり課題末 35 点（10+1+12+12）、`STATES` は init を含む 36 点である。J3 は「全課題末」を優先して 35 点で判定する。init を含む 36 点の Spearman も補助列として全 seed・腕・層で記録する。init は窓には含めない。
2. **F の定義**: §2 の「遷移確率からの F」は略記。数値は `load10.head` と同じ現課題の固定 probe の実現ラベルで `F = (one_hot(y)-1/V)^T h`（和、平均でない）。h は最終 LN 後の body 出力。κ・cos² は `load10.head` を直接使用し、F の複製計算と一致確認する。
3. **線形性**: `load10.gradient(m, uniform(seed))` は一様目標 CE の勾配であり、head を丸ごと交換して softmax まで再計算すると v の線形関数ではない。分解するのは**元の状態の softmax 誤差 e = p_original−1/V を固定した逆伝播**。同じ body・入力 embedding のまま、最終 body 出力への勾配 `e @ v` の v だけを v^c・κF・v⊥ に替える。これにより full は元の CE の up.weight 勾配に一致し、unit ごとの加法性を検査できる。head/embedding の tied weight は書き換えない。これは同じ状態の勾配の帳簿であり、head 交換の介入ではない。
4. **押し・probe・Adam**: `load10.uniform`（勾配の入力）、`load10.data`（位置・生存判定の現課題 probe）、`load10.moments`、`load10.sink` をそのまま使用する。各層の `S_i = <g_up[i], mean(y_probe)>`。alive は現課題 probe のいずれかで z_i>0。`load10` の A は生の勾配で Adam の分母を使っていないため、本解析も Adam の分母を使わない（*_adam は読まない）。S>0 は降下方向で位置が下がる印。
5. **原コードの所在**: main には元 GPT の src が未統合のため、`033679e0f1a3d313e1e3110e22c8f1cac3c36628:src/chargpt_cl_1019{,_kune}.py` を git blob から自分の worktree 内に復元して import する。他の worktree は参照・編集しない。復元 blob と SHA256 を provenance に残す。
6. **数値検査の gate**: CPU float64。まず seed 0、両腕、init＋全課題末、両層について full の s・n_alive と元 `results/chargpt_seed10_1022/load/states.csv` を照合する。s は絶対誤差 <=1e-12、n_alive は完全一致、κ・cos² は相対誤差 <=1e-8。さらに full の unit ごとの S と元 `load10.gradient`＋`load10.units` の S が相対最大誤差 <=1e-10 で一致すること。失敗したら 10 seed 計算へ進まず報告する。
7. **全状態の sanity**: `<v⊥,F>` の正規化誤差 <=1e-12、`v^c−κF−v⊥` の相対最大誤差 <=1e-12、unit ごとの `S_full−S_load−S_res` の相対最大誤差 <=1e-10。ゼロ分母の share は NaN（数を記録）、n_alive=0 の割合も NaN。窓で非有限値があればその seed の該当判定は不成立、相関の欠測・定数も J3 不成立として記録する。中央値は numpy の通常の中央値（偶数のとき中央 2 値の平均）。Spearman は同順位に平均順位を与える。
8. **J2 の残りのラベル**: 両条件を満たすと RESIDUAL_DILUTES、片方なら PARTIAL、両方外れたら NOT_RESIDUAL_DILUTES。J3 の不成立は NOT_TRACKING_COS2。窓・閾値は変更しない。K_n=y_n・mean(y_probe) の正・負・ゼロの割合は常に補助列として計算しておき、J1 不成立なら summary に課題末ごとの表を出す。
9. **解釈の範囲**: v⊥ は現課題の F に直交する残差であり、この分解だけで「旧課題由来」「乱数」を同定したとは言わない。share_load は押しの絶対値の分担で、head のノルム分担 cos² と区別する。
