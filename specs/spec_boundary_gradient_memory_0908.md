# 切替直後の勾配と実更新 追補：切替前のAdam記憶
親: [[切替直後の勾配と実更新_spec_0908]] / 状態: 検証中 / 更新: 2026-09-08
初回計装走の実行開始後、結果集計前に固定した追補。主specは変更しない。
主specのhistory = beta1*m_previousには、切替前の記憶と切替後に既に学習した勾配の両方が入る。「historyが大きい」から「前タスクの慣性が原因」とは言えない。
保存m_boundaryを用い、step tのnumerator中の切替前成分beta1^t*m_boundaryを、実更新のdenominatorに通した量をpre_boundary_historyとして別の補助再生で保存。主計装のhistoryとの差をwithin_task_historyとする。分母には切替前の二次モーメントが残るため、これもAdamリセットの介入効果ではない。
範囲は主specと同じ全4腕3seed19境界100step。学習系列は変更しない。旧CPU z再現誤差2e-5以下を確認。1..20/50/100累積と毎stepを図・CSVに追加する。主判定を変更しない。事後追補として結果に明記する。
