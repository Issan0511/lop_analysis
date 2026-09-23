# sgd_postfit_cifar_0923 — Claude の予測（本走の前、Codex の予測を見る前）

書いた: Claude (Opus 5.5), 2026-09-23 13:0x。事象の定義は `PREDICTION_EVENTS.md`、判定は spec v2.2。
見た情報: 試走（pilot.json）、検査走（checks_pre.json、C8）、課題 1 の検査走の軌跡（Adam の暴発、SGD 0.03 の Δ‖W1‖² +5、adam_ce x=1 の凍結点）。

Issa の立場（引用、確率は付けない）: 「どっちかいうと adam のせいな気がしてきた」。

| # | 予測 |
|---|---|
| E0 | p = 0.95 |
| E1 | SGD_RESCUE 0.13 / PARTIAL 0.03 / NO_RESCUE 0.84 |
| E2 | SGD_RESCUE 0.88 / PARTIAL 0.09 / NO_RESCUE 0.03 |
| E2b | SGD_RESCUE 0.76 / PARTIAL 0.16 / NO_RESCUE 0.08 |
| E3 | SGD_RESCUE_W 0.13 / PARTIAL_W 0.02 / NO_RESCUE_W 0.85 |
| E4 | RESET_RESCUE 0.05 / PARTIAL 0.13 / NO_RESCUE 0.82 |
| E5 | RESET_RESCUE_W 0.04 / PARTIAL_W 0.12 / NO_RESCUE_W 0.84 |
| E6 | MATCHED_CE_HARMLESS 0.42 / MATCHED_CE_HARMFUL 0.03 / ADAM_WORSE_AT_MATCHED_CE 0.11 / SGD_WORSE_AT_MATCHED_CE 0.02 / MIXED 0.42 |
| E7 | p = 0.95 |
| E7b | p = 0.85 |
| E7c | p = 0.20 |
| E8 | p = 0.12 |
| E9 | p = 0.50 |
| E10 | p = 0.07 |
| E11 | p = 0.25 |
| E12 | p = 0.35 |
| E13 | p = 0.97 |
| E14 | p = 0.80 |
| E15 | 中央値 0.92、80% 区間 [0.60, 1.15] |
| E16 | 中央値 22,500、80% 区間 [17,000, 30,000] |

## 根拠

- **E1/E3/E7/E7b（S_hi は発散で負ける）**: 安定の上限は重みの大きさとともに下がる（課題 1–2 の ‖W1‖² ≈ 700 で 0.03〜0.1 の間、課題 49 の 76k で 0.001〜0.003）。上限 ∝ ‖W1‖²^−0.7 程度と読むと、0.03 が上限を超えるのは ‖W1‖² ≈ 1.5k、つまり課題 2〜5 あたり。F 並みに伸びるだけで、多くの seed が途中で NaN になる。5 本以上死ぬと、失敗込みの中央値は 16k 前後になり NO_RESCUE。
- **E2/E2b（安定な SGD は救う）**: SGD はどの η でも収束後の Δ‖W1‖² が +1〜+7 で、主に W3 を動かす（W3 は毎課題書き直される）。したがって S_lo と S_mid は F とほぼ同じ連鎖になり、F は停止チェーンと同じく遅れない。S_mid は後半に落ち込み（暴発）がありうるので、S_lo より少し不確か。
- **E4/E5（状態を戻しても救わない）**: 新しい課題の最初の勾配は、s_sw の v より 1〜2 桁大きい。最初の一歩はどちらの状態でも ≈ 3.16·lr·sign(g) に飽和し、s_sw の v は最初の数歩を 5% ほど小さくするだけ。課題の中の漂流は A と同じなので、AR ≈ A。
- **E6（CE を揃えた Adam）**: 参照腕は S_mid になる見込み（S_hi は 8 seed 残らない）。X は約 2 e-fold。静かな漂流は e-fold あたりほぼ一定の費用で、深さ 12 e-fold のうち 2 だけなので、A_ce の W の余りは A の 2 割弱で、ρ_Ace は 0.75 前後になる。HARMLESS と MIXED が五分。
- **E11/E12（交差 fork）**: 状態の効果は上の理由で小さい。重みの効果は、1 課題ぶんの漂流（t48 で +2,900）を連鎖の比例（55k で +1,900 更新）に当てると +100 更新ほどで、50 組では境目。
- **E9**: e-fold は S_mid > S_lo でほぼ確実。G は Adam の当てはめ由来の約 21k が大半を占め、SGD の差（課題あたり数十）は seed のばらつきに埋もれるので五分。
