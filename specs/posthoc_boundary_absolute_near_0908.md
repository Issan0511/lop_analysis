# 境界群検証の事後感度分析：絶対的な0近傍
2026-09-08 / Codex / boundary_groups_0908
主spec 1ed2efdはNを|jump|が小さい25unitと定義した。先行完了したLRの2seedを見ると、この相対的なNは|zbar|~2まで含む。Issaの「0周り」と同一視できるかが不明なため、追加学習なしで保存状態の分担計算だけを行う。
これは一部の主結果を見た後に設計した事後感度分析であり、主定義・主結果を置き換えない。
Dは主specのまま。N_abs=|jump|<=.5（人数上限なし）。D_only/N_only/both/restを再構成し、同じ5ブロックShapley CE分配を計算。4腕3seed、tasks101..119、625stepのみ。N_abs人数0もそのまま残し、良い結果だけ選ばない。
各seedのN_only人数、D人数、N_only/Dtotalの絶対寄与と1unitあたりの寄与を出す。N_only人数0の場合1unit当たりはNA。大小だけで因果主張はしない。
