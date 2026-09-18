"""Human-readable result tables and retained-state figure from saved summaries."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import report as r

def main():
 p=r.OUT
 def read(f):return list(csv.DictReader((p/f).open()))
 primary=read('primary.csv');secondary=read('secondary.csv');ret=read('retained_secondary.csv');adj=read('retained_followup_adjusted.csv')
 def row(rows,arm,age,metric,coord=None):return next(x for x in rows if x['arm']==arm and int(x['age'])==age and x['metric']==metric and (coord is None or x['coordinate']==coord))
 def m(rows,arm,age,metric,coord=None):return float(row(rows,arm,age,metric,coord)['mean'])
 def ci(x):return f"{float(x['mean']):+.4f} [{float(x['lo']):+.4f}, {float(x['hi']):+.4f}] {x['label']}"
 cells=[(a,g) for a in r.ARMS for g in r.AGES];labels=[f"a={'.1' if '0p1' in a else '.7'} / {g//10000}" for a,g in cells]
 fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');yy=np.arange(6)
 for metric,label,color,shift in [('null_B_field_RMS','Previously A-invisible field','#007c91',-.14),('total_input_jump_RMS','Full input-switch change','#b8762c',.14)]:
  v=[m(ret,a,g,metric) for a,g in cells];axes[0].barh(yy+shift,v,height=.26,label=label,color=color)
 axes[0].set_yticks(yy,labels);axes[0].invert_yaxis();axes[0].set_title('B exposure before learning');axes[0].set_xlabel('RMS over units / inputs / 15 switches');axes[0].legend(fontsize=8)
 for metric,label,color,shift in [('B_Cn_qpos','Positive inherited field','#007c91',-.12),('B_Cn_qneg','Negative inherited field','#b8762c',.12)]:
  rr=[row(ret,a,g,metric) for a,g in cells];v=np.array([float(x['mean']) for x in rr]);lo=np.array([float(x['lo']) for x in rr]);hi=np.array([float(x['hi']) for x in rr]);axes[1].errorbar(v,yy+shift,xerr=[v-lo,hi-v],fmt='o',capsize=2,color=color,label=label)
 axes[1].set_yticks(yy,['']*6);axes[1].invert_yaxis();axes[1].axvline(0,color='grey',lw=1);axes[1].set_title('Counteraction in the A-null component');axes[1].set_xlabel('Coefficient (positive = opposing the field)');axes[1].legend(fontsize=8)
 for metric,label,color,shift in [('B_mean_transport_null','Change in A-null weight','#007c91',-.14),('B_mean_transport_span','Change in A-visible weight','#b8762c',.14)]:
  v=[m(ret,a,g,metric) for a,g in cells];axes[2].barh(yy+shift,v,height=.26,label=label,color=color)
 axes[2].set_yticks(yy,['']*6);axes[2].invert_yaxis();axes[2].axvline(0,color='grey',lw=1);axes[2].set_title('Mean B response change from W');axes[2].set_xlabel('Signed preactivation change');axes[2].legend(fontsize=8)
 for ax in axes:ax.grid(axis='x',alpha=.15)
 fig.suptitle('Retained A-invisible weight is exposed at B; selective erosion is conditional\nSecondary follow-up, 10 seeds; middle intervals are descriptive 95%',fontsize=13)
 fig.savefig(p/'retained.png',dpi=180);plt.close(fig)
 text=r'''# 直交成分の堆積と次タスクでの顕現を分けて検証 — CondA 0918

## 結論

**「Aで見えなかった重みがBで現れる」ことは確認できた。ただし今回のA中に新しく残った、A開始時wに直交する増分とは別の成分だった。正側の選択的侵食までを一般的な一連の機構として支持する結果ではない。**

新規増分uの生の前活性への寄与は、全6条件でAよりBの方が平均4〜5%小さかった。中心化重みでも同方向。Aの入力から完全に見えない実重みnは存在し、Bで切替直後の前活性変化と同程度のRMSで現れたが、nはA開始前から存在し、A中はほぼ保存されていた。これは「以前の重みの保持・顕現」を支持する観測であり、「Aの学習で新しい不可視成分が堆積」を意味しない。

## 対象・事前登録

CondA、20–100–1、leaky a=.1/.7、通常SGD lr=.005、unhalved MSE、各10 seed。既存学習20/100/500タスク時点からのswitch_force_0918の切替後10000更新をTask Aとする。今回はその終点からさらにTask Bに切り替える。Bの候補15種類×全32入力を学習前に列挙し、別途1種類のBを10000更新、A継続対照と同じ5ビット乱数列で比較した。全W/b/v/cを通常更新。主解析float64、元のfloat32でも主判定を照合。

ラベル「20」はA前までの学習タスク数であり、Aはその次、Bはさらに次のタスク。Aの既存軌跡自体は以前に解析済みだが、今回の主指標は事前登録後に計算した。

- 本体spec: `specs/spec_orth_reveal_0918.md` / 46a0664。
- 本体runner: dede695、独立集計・再実行:38463d0。
- 保持成分の追加解析spec: `specs/spec_orth_reveal_retained_followup_0918.md` / 59dc999。本体結果を見た後に登録した別の二次解析。コード2fcd953。
- 本体の18比較と追加の12比較は別family。追加解析を本体の不支持の代用品にしない。

## 1. Aで新しく残ったw直交成分

各unitでA開始時w0、A終了時w1を使う。

$$D=w_1-w_0,\quad \alpha=\frac{\langle w_0,D\rangle}{\|w_0\|^2},\quad p=\alpha w_0,\quad u=D-p.$$

u⊥w0であり、w1=(1+α)w0+u。これは1タスクの正味増分の分解であり、回転するw_tに対する各stepの直交成分の単純累積とは異なる。

q_A=u·x_A、q_B=u·x_B。seed内でunit・入力・B候補の二乗をプールしてからRMS比を取る。G=log(RMS_B/RMS_A)を主指標とした。「Aで平均が0」と「全入力から見えない」は区別した。

| a | A前の学習タスク数 | B/A RMS比 | Gの18比較補正区間・判定 |
|---|---:|---:|---|
'''
 for a,g in cells:text+=f"| {'.1' if '0p1' in a else '.7'} | {g//10000} | {m(secondary,a,g,'RMS_B_over_A','whole'):.4f} | {ci(row(primary,a,g,'G'))} |\n"
 text+=r'''
全6条件でG<0。活性φ(z)−φ(z−q)への有限アブレーション、出力重みvを掛けた寄与、unitを合算した出力差でも、B/A比の平均は1を下回った。全体としての顕現増大を、平均前活性だけの議論やゲートの非線形性に置き換えて支持することもできなかった。

一方、「どのunitも見えやすくならない」ではない。主結果を見た後の記述的点検では、unit×切替候補の12〜25%でRMS_B>RMS_Aだった。ただし、その組が占めるBの寄与二乗量は約3〜10%。一部の顕現を排除する結論ではない。

## 2. u方向はその後削られるか

B開始時のq_Bの正負を固定した。C±は、それぞれの側で後のW更新による応答変化がq_Bに逆らう係数。正値なら対抗・抑制。q_Bの符号と実際のz_Bの符号は別であり、4セルと新規活性化セルも保存した。

K=−Σ〈u,w_t−w1〉/Σ||u||²は、固定u方向の係数低下。開始時の係数は1なので終了時は1−K。Sは(C+−C−)のB切替−A継続差、Kdiffも同様の対照差。

| a | タスク数 | C+ B | C− B | K B | K 対照 | S判定 | Kdiff判定 |
|---|---:|---:|---:|---:|---:|---|---|
'''
 for a,g in cells:
  vals=[m(secondary,a,g,k,'whole') for k in ['Cplus_B','Cminus_B','K_B','K_control']]
  text+=f"| {'.1' if '0p1' in a else '.7'} | {g//10000} | "+' | '.join(f'{v:+.4f}' for v in vals)+f" | {row(primary,a,g,'S')['label']} | {row(primary,a,g,'Kdiff')['label']} |\n"
 text+=r'''
正側の抑制は平均として見えるが、「負側より選択的に、切替によって追加で削る」という主指標Sは6条件とも補正区間が0を跨いだ。これは効果0の証明ではない。a=.7では固定u方向の係数低下がA継続より大きいことは3時点とも確認された。ただしuはAでも既に見えており、この結果だけで「隠れていたuの顕現」を支持しない。

正側/負側の**更新元**からの〈u,Δw〉も毎step実更新で集計した。固定した**受け手**q_Bの符号とは混同しない。正側の更新がuを削る平均傾向と、負側更新が戻す平均傾向は複数条件にあるが、これも顕現と歴史的蓄積を結ぶ因果証明ではない。

Aでのノルム二乗の平均収支（平行項＋u二乗）は、a=.1で+2.035+3.069、+1.602+5.926、−2.991+4.460、a=.7で−3.158+4.292、−4.111+5.035、−3.412+4.803。最初の2条件は平均の平行項自体が正であり、「w方向は常に侵食」でもない。これらは記述的なseed平均で、多くの平行項の95%区間は0を跨ぐ。

## 3. 見えないuが新規蓄積したように見える分解上の落とし穴

Aの全入力が張る部分空間をS_A、その直交補空間への射影をN_Aとする。今回は入力空間20次元、S_Aのrankは6。固定したAの通常SGDでは

$$D=-\eta\sum_t h_t x_t\in S_A,\qquad N_A D\simeq0.$$

従って、

$$N_Au=-N_Ap$$

であり、uだけにA不可視成分があっても、pの逆向き成分がそれを相殺する。実更新DにA不可視な重みが新規に加わったことにはならない。

実測の||N_A D||/||D||はfloat64で最大6.45e−14、native float32で最大4.80e−5（全/中心化座標）。uの不可視成分の二乗比はwholeで平均0.46〜3.68%。その小成分もpと相殺した。中心化重みでも対応する制約を検算した。Adamの座標別前処理にはこの入力spanの制約をそのまま移せない。

## 4. 実際の重みに残っていたA不可視成分 — 追加解析

ここは本体結果を見た後に別specで登録した。新しいuではなく

$$n=N_A w_1\simeq N_Aw_0$$

を追った。n·x_A≈0なのにn·x_B≠0となる。〈n,w0〉≈||n||²なので、nは一般にw0には直交しない。まさにユーザーが区別を求めた二条件である。

| a | タスク数 | nの重み二乗比 | Bで現れるnのRMS | 切替による全前活性変化のRMS | nの正側係数Cn+ | nの負側係数Cn− | 選択性Sn判定 |
|---|---:|---:|---:|---:|---:|---:|---|
'''
 for a,g in cells:
  vals=[m(ret,a,g,k) for k in ['null_weight_energy_fraction','null_B_field_RMS','total_input_jump_RMS','B_Cn_qpos','B_Cn_qneg']]
  text+=f"| {'.1' if '0p1' in a else '.7'} | {g//10000} | "+' | '.join(f'{v:.4f}' for v in vals)+f" | {row(adj,a,g,'Sn')['label']} |\n"
 text+=r'''
ここでCn±は、**nが属するA不可視部分そのものの変化**だけで測った抑制係数。全Wの応答変化による抑制とは分けた。Aでのnの保存誤差は最大2.29e−14、Aでの前活性寄与は最大2.82e−14。

nの顕現は切替直後の全前活性変化と同程度のRMSを持つ。しかし二つの成分には交差項があるので「nが変化の何%を説明した」と二乗比をそのまま読めない。正負両方に現れるが符号分布は完全対称ではなく、a=.1の履歴が長い状態では負側の二乗量が大きかった。無方向な顕現なら対称、という仮定を実データへ無条件には置かない。

追加の12比較補正では、正側選択性Snはa=.1・100タスク状態だけ正、他は未確定。n方向の係数低下対照差はa=.7の3時点で正。ただし全n方向の係数低下平均は約0.29〜0.33%であり、u方向の14〜21%という数字とは異なる分母・対象である。

Bで下げ直す更新は、不可視成分を削る以外にも起こる。例えばa=.1・20タスク状態ではW由来の平均前活性変化−.04378のうち、A不可視部分の変化が−.00418、A可視部分の変化が−.03960。biasは別に−.00462。顕現した成分そのものの大幅な消去と、ほかの方向からの応答の打消しを同一視できない。

## 支持される範囲と未検証の接続

1. Aで新しく残ったw0直交増分が、Aで隠れBで全体として強く現れる予測：今回の主指標は逆方向。
2. A入力から見えなかった実重みがBで現れること：確認。A中の新規蓄積でなく以前からの保持。
3. その正側が選択的に削られること：追加解析の一条件で支持、条件共通の機構としては未確定。
4. 長期のw増大が保持成分nをどう作ったか、回転するw_t直交更新の履歴がnを作ったか：この二タスク実験では未検証。既存checkpointのnは観測したが、その歴史的生成源を特定していない。

## 検算・ファイル

本体の恒等式・実更新由来の集計、独立NumPy再計算、autogradが全てPASS。元SCREnv乱数列と計測を外した通常SGDで全12本を再実行し、保存時点のW/b/v/cの624比較が全てbitwise一致。本体主判定18個はnative float32でも一致。追加解析はfloat64。事前の負例は平均ゼロと不可視の混同、p-u相殺の省略、符号群の反転、固定群と移動ゲートの混同を検出した。

- コード: `analysis/orth_reveal_0918/`
- 小さい結果・全区間・seed別値: `results/orth_reveal_0918/`
- 生配列/ログ/スクリプト退避: `~/Projects/obsidian-research-data/orth_reveal_0918/`
- 元checkpoint・A軌跡のSHA256: `provenance.json`
- 退避データのSHA256: `backup_manifest.json`
- 図: `primary.png`, `retained.png`

関連: [[タスク切替で平均入力の反対向きの力は再生するか_結果_0918]]、[[AとQの恒等式をCondAで実更新照合_結果_0917]]、[[W増大メカニズム_0909]]。
'''
 (p/'RESULTS.md').write_text(text)
if __name__=='__main__':main()
