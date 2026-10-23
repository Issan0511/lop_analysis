# 零点への着座と重み収縮：既存ログ再解析 0913

親: W増大メカニズム_0909 / 状態: 検証中 / run_id: zero_attraction_reanalysis_0913
格: **既存データの事後・記述的再解析**。新しい学習実験ではない。
既存の結果ノート（着座・重みの集計）を読んだ後の計画であり、未知結果への事前予測とは呼ばない。
本手順を解析実装・新集計の実行より先にローカルcommitする。元の登録判定は変更しない。

## 問いと対象

Issaの仮説: 活性化の零点付近への復元と、そのユニットの入力重みの収縮が結びつき、W増大を抑える。
閉じる穴: 平均位置の着座と分布幅、全Wノルム、タスク内の動径収支を区別した同一個体の記述が足りない。

対象は act_offset_0906 の完走したLeaky 6条件・各10seed・各100unit。
A: LRoff0_1216, LRoffm0p5_1216, LRoffp0p5_1216; a=.1, lr=.01, 5M, task500。
B: LRoff0_lr0p00125_1216, LRoffm2_lr0p00125_1216, LRoffp2_lr0p00125_1216; a=.1, lr=.00125, 40M, task4000。
A/Bの水準差を用量効果にしない。発散したAの±2は解析対象に存在しないため除外理由を表示する。

データ:
- /home/issan/Projects/obsidian-research-data/act_offset_review_0908/full/logs/
- 同 tail/ckpts/ のstep0・final checkpoint。
- 先行取得の archive SHA は同ディレクトリREADMEと旧ノート参照。
- 今回読むファイルごとのSHA256を新出力sources.jsonに記録する。
コード基点: b50f127768ec1c56a8ca8da3c4e9ede31fade958。

## 座標と窓

活性化定数はqと書く（侵食角cと区別）。phi_q(z)=Leaky(z;.1)+q。
零点z0はq>0なら-q/.1、q<0なら-q、q=0なら0。折れ目0とは別。
free重みu=W[:,15:20]はタスク内に変動する5bitを担い、既存MNISTの中心化Wとは異なる射影。
固定タスク内の32点支持で Var_x(z)=||u||^2/4、半幅=sum|u|/2。
全Wは20次元で、固定15bitへの重みはそのタスク内では平均位置を変える。bも別に記録。
この恒等式の一致は検算であり、独立な機序証拠に数えない。

step0は初期値。task終端は step=10000*t。
窓: early task2–20; middle task21–100; A late451–500; B late3951–4000。
task1には先行切替がないので、タスク切替の収支に含めない。endpoint初期比にはstep0を使う。
初期過渡と後期ドリフトを分け、最終状態の小ささを「初期から縮んだ」と読み替えない。

## 記述量

全unitを含め、ALIVEや|v|による足切りはしない。
unit・時間をseed内で等重み集約し、独立単位はseed（10）。群の割合と全体の分布を併記。
1. 全WのRMSノルム、free重みのRMSノルム、入力内sigma、zbar-z0のRMS、|v|のRMS。
2. free重みのlate/初期比、全Wのlate/初期比（seed内で二乗平均の平方根どうしの比）。
3. unit別に窓平均free norm < 初期normとなる割合。ゼロ初期normは比から除外して件数表示。
4. 最終taskで |zbar-z0|<=.1 の群を「零点近傍」として記述。±.05, ±.2の感度分析も記録。
   これは結果による群分けであり、近傍化の因果効果を推定しない。
   群内のfree比、全W比、|v|、分散、人数を出し、縮んだunitだけの抜粋を避ける。
   「平均が近い」ことと「全入力が近い」ことを分け、max_x|z-z0|<=.1も別集計。
5. 初期とfinal checkpointの全W/b/vを直接確認し、全W^2=free^2+fixed15^2を検算。
6. タスク終端系列から各unit・各task:
   du=u_t-u_(t-1), Q=||du||^2, R=2<u_(t-1),du>, G=||u_t||^2-||u_(t-1)||^2。
   G=Q+Rを検算。N=||u_(t-1)||, D=||du||, c=-<u_(t-1),du>/(ND)。
   cはN,D>1e-12時だけ定義し未定義数を記録。単純平均cからGを再構成しない。
   窓のQ,R,Gをunit平均後にtask総和しseed別表示。c_eff=-sum(dot)/sum(ND)も補助表示。
   これはtask内の全更新を観測した帳簿ではない。自己項/Adam/経路持続の分解には用いない。

## 比較と図

主にA内±.5対0、B内±2対0で、free growth比のseed対応差を表示する。
95% percentile paired bootstrap 5000回、rng20260913。小標本の記述的CIであり検定による一般化・多重比較後の確証とは呼ばない。
腕別時系列は各seedのunit二乗平均→平方根、seed中央値とIQR。
図はA/Bを別列にし、free norm、全W norm、零点からの距離、|v|を並べる。
近傍群は事後選択であることを表と図にも明記。

## 検証

- 各腕10seed、全500/4000task+step0、重みstepと他の記録stepを照合。
- 32点支持再構成のzmin/zmax、半幅を記録値と照合（float32蓄積を踏まえatol5e-4/rtol1e-4）。
- 初期/finalのfree重みがcheckpointに一致（atol1e-6/rtol1e-5）。
- 全Wノルムがcheckpointと一致（atol5e-5/rtol1e-5）。
- 32点の分散とfree^2/4、およびG=Q+Rはfloat64でrtol1e-10/atol1e-10。
- sourceにない値、欠落、NaNは補完せず記録して該当判定を保留。
- 比較表は独立にcheckpointの初期/finalから再集計して時系列終点と照合する。
- 既存のsummaryのALIVE中央値を今回のALL RMSと数値一致させる検査はしない（集約が違う）。

## 成果物と限界

results/zero_attraction_reanalysis_0913/ に summary.md, seed_summary.csv, group_summary.csv, paired_comparisons.csv, task_ledger.csv, verification.json, sources.json, trajectories.npz, trajectories.png/pdf。
既存ログは書き換えない。解析コード・手順・小出力は専用worktree、rawの重複コピーは不要。
この再解析が支持できるのは「着座群にfree重み収縮が同居する」「条件の変更でfree幅が変わる」という記述。
自己項がその収縮を生成した、傾きの強さが原因、Adam/MNISTでも普遍、可塑性維持に有効、は後続介入が必要。
