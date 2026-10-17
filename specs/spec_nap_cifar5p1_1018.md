## 5. 5+1 の NaP 比較

**run id:** `nap_cifar5p1_1018`  
**推奨ファイル:** `specs/spec_nap_cifar5p1_1018.md`  
**継承:** `cifar5p1_mlp_0920`、`baselines_cifar5p1_1008` の評価/較正規則。  
**原典:** Lyle et al., *Normalization and effective learning rates in reinforcement learning*, arXiv:2407.01800v1、Algorithm 1、§3.3、付録A.4/D。

### 5.1 問いと比較条件

F51/std/ReLU/lr1e−4で、NaPをSNA/KKT1と同じseed対応・後期窓に並べる。原典の万能性、他のlrでの順位、RLでの有効性はこの走から主張しない。

評価seed0–9、較正seed100–109。R/SNA/KKT1はG0と再利用資格を満たした既存記録を使う。NaPはtask1の最初から全30課題、他のハイパーパラメータはF51のまま。

### 5.2 NaP の固定実装

各隠れ層を **linear(no bias) → LayerNorm(affineあり) → native ReLU** とする。LNは各画像の100 unit軸で正規化、平均・分散はcorrection=0、eps_LN=1e−5、γ=1、β=0で初期化。ここでβはLNのoffsetで、AdamのβやEMAと別物。BN/EMAは入れない。最終はlinear100→100、softmax前の追加LNは入れない。

元の初期化streamでW1,b1,W2,b2,W3,b3を一度生成してWの対応を保ち、**NaPでは全linearのbiasを使用しない**。出力biasも無しと固定する。この選択を参照表に記し、暗黙にRと同じbias数と扱わない。LNのoffsetは学習する。

毎Adam更新の直後に、各seed・各層の **行列全体のFrobeniusノルム** を初期値ρ_lへ戻す:

`W_l ← ρ_l·W_l/||W_l||_F`、l=1,2,3。

これは unit別上限cap12と違い、**各層の球面への射影**。小さいノルムも初期半径へ戻す。半径はt0で一回保存、task1末へ取り替えない。W3も射影することを明記する。出力層はLNで正規化されないため、その射影はlogit尺度を変え得る。

ReLUに対応するLNのscale/offsetについて、Algorithm 1のとおり各層の連結ベクトルを射影:

`a=sqrt(100/(||γ_l||²+||β_l||²))`、`γ_l←aγ_l`、`β_l←aβ_l`。

aは両方の更新前候補から一度計算する。β単独を初期ノルム0へ射影しない。Adam momentや時刻は射影しない。元のelementwise Adam式を使い、lrは全パラメータ1e−4で一定。Wノルム0/連結ノルム0は `PROJECTION_UNDEFINED` とし、勝手なreinitや小定数で救わない。

**原典からの位置づけ。** 正規化と重み射影、scale/offsetの別処理は原典に従う。毎更新射影、F51の幅・データ・100次元出力、LNのepsilon、固定lrは本比較で固定した具体化。NaPとRの分岐時出力を一致させる介入実験ではない。

### 5.3 較正・候補の選び方

**候補数は1。** 上記の制約（lr固定・毎更新射影・初期半径）では探索すべきWD係数はない。少数候補という規則を満たすために無関係な格子を作らない。seed100–109で全30課題を一回走り、有限性・射影・速度・後期窓を記録し、その設定と全出力/hashを固定してから評価seedを開く。

較正が不調ならそれを開示して同じ設定を評価するか、実装不良ならS検査へ戻す。性能を見てLN eps・射影頻度・γβ制約を変える場合は別候補・新版として開示し、同じ評価seedの成績を見て選ばない。候補数1であることをベースライン表の較正条件にも載せる。

### 5.4 主指標と統計

各seedでW=hard t21,23,25,27,29 online平均。主比較は **NaP−KKT1、NaP−SNAの2本**。両側の正確な対応符号検定、差0は除き、**2検定のHolm補正**を固定。効果は対応差の中央値と平均、全10差を出す。

- `NAP_WINS` ⇔ Holm補正p<.05 かつ対応中央値>0。
- `COMPARATOR_WINS` ⇔ Holm補正p<.05 かつ対応中央値<0。
- `EQUIVALENT_WITHIN_0.005` ⇔ 有意差なし、かつ差の順序統計区間[d_(2),d_(9)]が全て[−0.005,+0.005]内。
- その他 `UNRESOLVED`。

順序統計区間の被覆はn=10で97.8515625%/本、2本の同時被覆はunion boundで少なくとも95.703125%。これは対応差の**母中央値**の区間で、平均のt区間と混ぜない。未決を同点としない。NaP−R、早期窓、最終memo、eff_rank、死亡率、m/s/r、LN前後の分布、射影量は報告のみ。

freshは2種類を区別する。共通の手法なしR-fresh(t29)は元baselineと共通の参照値、**NaP自身のfresh(t29)**はNaPの初期W/LN/Adamから同じt29バッチを780更新する。`fresh_self−continual_NaP` をNaP内のgapとして出す。異なるfreshを持つ手法のgapだけを使って性能順位を決めない。

### 5.5 S 検査

| S | 本物で要求 | 必ず落とす変異 |
|---|---|---|
| S-host | NaP無効化の専用経路がR宿主と一致、SNA/KKT1再利用資格を確認 | CIFAR-10定数を使う／outputを5次元へ |
| S-LN | 独立float64参照で画像別100軸LNと勾配が算術許容内、層間でγβ別。biasなしの配線を検査 | バッチ軸LN／ReLU後LN／biasを残す |
| S-Wproject | W1/W2/W3の層全体ノルムがt0半径、方向保存、上にも下にも射影。moment不変 | 行別射影／capだけ／t1半径／W3だけ除外 |
| S-affine | 非零offsetを含むfixtureでγβ連結norm=√100、同じaで同時scale | βだけ0へ／γ更新後normからβ係数を再計算 |
| S-grad | native ReLUの0点・負側とLNを含むautograd照合、LNの連鎖勾配あり | 手書きReLU微分だけでLNを迂回 |
| S-fresh | NaP-self freshのW/LN/Adam/半径がt0に戻り、t29のバッチが継続網と同一 | LN状態を持越す／freshだけprojection無し |
| S-stats | 正確符号検定・Holm・順序統計区間を人工データで独立照合 | 片側p／補正なし／seedを並び順で対応 |

### 5.6 Issa の事前予測欄

| 記入項目 | Issa 記入欄 |
|---|---|
| 対KKT1 / 対SNAのラベルと確率 | 未記入 ／ 未記入 |
| NaP後期mean online、2対応差の中央値と範囲 | 未記入 |
| 勝つ/負けると予想する機構的な理由 | 未記入 |
| NaPが両者に勝った場合に方法の主張をどう直すか | 未記入 |
| 記名・時刻・署名対象 spec hash | Issa ／ 未記入 ／ 未記入 |

費用: 1候補×10較正seed＋10評価seed、全30課題とfresh。LNと射影により既存Rの20秒をそのまま保証しない。LNのみ・射影のみのablationをこの比較の必須腕に追加しない。

---


## 共通規約と実行予測

`spec_six_runs_common_1018.md`を本仕様の一部として継承する。特に同ファイル§8のIssaの実行指示・論文由来の採用予測を正本とする。
