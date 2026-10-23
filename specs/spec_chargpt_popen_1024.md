# spec: 開いた割合と相2平均との差の向き（chargpt_popen_1024）

2026-10-24。Issa の仮説を既存保存配列で検査する。新しい学習・forward・API worker・subagent は起動しない。

## 既知と問い
chargpt_phase2conditions_1024 で、L0 の unit ごとの押しを並べたベクトルについて、個別課題成分 task_departure と相2平均 P2 の cos が課題末1–4から6–9で負から正へ変わったことを見ている。この結果を受けた事後解析であり、その発見自体の事前登録ではない。開いた割合の値を見る前に以下を固定する。

## 対象と定義
C3/ReLUcap × seed0–9 × 課題末1–9 の180状態。主は L0 SGD（自分のup行経路の r=mu/s の押し）、Adam固定分母を記録する。生きたunit集合を親解析と揃える。
chargpt_earlyphase_1024/raw の p_i=mean_n 1[z_ni>0] を chargpt_phase2conditions_1024/raw の押しに結合する。p は同じ固定一様probe上の unit ごとの入力に対する開き率。各入力で開くunitの割合とは異なる。r・alive・P2・actual の一致を全180状態で確認し、相対誤差1e-8超なら停止。
M_i=相2平均 iid_mean（P2の正の定数倍）、D_i=ls_task−ctx_mean、実際の切替の押し=actual。正の押しはrを下げる。

## 集計
窓は課題末1–4、6–9。各状態のunit中央値や割合→各seed内の窓中央値→10seed範囲。
- p の中央値と p>1/2 のunit割合。
- M_i D_i>0（補強）の割合、M_i>0、D_i>0、実際の押し>0 の割合。
- unit別 Spearman(p,M)、Spearman(p,D)、Spearman(p,MD)。同じunitの窓中央値での p の変化と補強への反転を追い、p=1/2を横断したかを記録。
- 固定p帯 [0,.1,.2,.3,.4,.5,.6,.8,1] ごとに反対/補強を数える。窓ごとにstate×unitをpoolしseed内で集計（独立反復とは扱わない）。同じp帯でも時期差が残るか記録。後から良く当たる閾値を選ばない。
- 向きの変化が M の符号変化、D の符号変化、両者のどれによるかを固定unitで分ける。全体のcosは大きさも重みになるので、unit割合と混同しない。

## 解釈
1/2はユーザーが指定した閾値ではなく、既存の簡略理論との照合用。1/2を横断しなくても連続的なpの効果は否定しない。pとの相関や条件付き時期差は記述であり、pを変える介入による因果検査ではない。理論の残差化や分布の仮定を外してTransformer全経路へp=1/2の符号法則を流用しない。以前の判定を変更しない。

## 保存
analysis/chargpt_popen_1024/、results/chargpt_popen_1024/。元npzを複製せずパス・sha256をprovenanceに記録。統合ノート Transformerの押しと相2_実測の統合_1024 に追記する。
