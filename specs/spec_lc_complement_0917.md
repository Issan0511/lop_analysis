# Lillo & Cheney 補完 C1–C4 — 実装仕様・登録

状態: 実装・検査段階。本ファイルの最初の commit が登録。短い smoke は本走に含めない。
作成: 2026-09-17 / Codex / 依頼: Issa「設計案を実装しましょう」
親: Obsidian `可塑性喪失/spec/LilloCheneyを補完する_押しの持続と判定の混同_設計案_0917.md`
親の取得時 SHA256: `458acc63df3bd2b5783ea34cb154aafcaac43fa58eb2f35ad865b4463ae01823`
V10 H4（定義と押しの持続）、H5（活性化の条件）、A8（先行研究との接続）。
実装: `src/lc_complement_0917.py` と `analysis/lc_complement_0917/`。
本依頼の成果物は実行可能な実装・検査・本走手順。本走の完了や予測の当否を意味しない。

## 1. 設計案から具体化した点

- C1 は (入力, ユニット) の分類。ユニットが全入力で死んだ割合とは区別する。
  LC の σ は先頭から固定順16枚ずつのバッチ内、層全体を平坦化した出力の標本標準偏差に
  1e-8 を足す。最終端の短いバッチも同様。主データは1200枚なので75バッチ。
- LC は設計案に書かれた数式を明示的に適用する。上流コードの module dispatch は再現しない。
  公開 `utils/dead_units.py` は経路により ModuleList をそのまま判定する箇所や
  負側床の幅を max(0.02,0.05σ) とする別の RL 経路があるため、
  「彼らの全実装と同一の dead unit 指標」とは呼ばない。
  照合: https://github.com/lute47lillo/activations_plasticity/blob/iclr2026/utils/dead_units.py
- C2 の K_rev は最初の継続タスク内の **第10,20,…,80 epoch**（1始まり）の全75更新だけ。
  γ は 1.5,.5,.25,2 の順を2巡。これは設計案の10 epoch間隔の翻案であり、
  論文側の別スケジュールや CNN 全体の再現ではない。
- C2 の区間は seed 内差の平均に対する両側95% Student t 区間（df=9）。
  標本分散0なら点区間を返し、degenerate を記録。seed が足りなければ主判定しない。
  0を含む区間は同等性の証拠ではない。PERSISTENCE_MATTERS はこの登録条件のラベルに限る。
- C3 は別作業 `claude/lc_elu_lr_0917` の α×lr とは独立の傾き×漸近値。
  「彼らのプロトコル」を明確化し、先頭1200枚、固定順、native ELU/CELU を採用。
  当方の seed 別無作為1200枚・毎epochシャッフル・expm1微分とは条件が違う。
  この違いを隠して resp_ee と bit 一致を要求しない。公開実装を複製せず当方コードで実装。
- C4 の「回復時間の上界」という呼び方を訂正する。一歩の上限が与えるのは
  **回復時間の下限（必要条件）**。有限時間回復を保証せず「時間で律速されない」とも言えない。
  また一般の勾配履歴で 3.16η は一様上界でない。§5 の Cauchy–Schwarz 上界を使う。

## 2. C1 — 判定の混同

resp_ee の t2,5,7,10,15,20 × seed0–9 の保存状態を使用。学習の再実行なし。
全26腕について既存 build_shift/make_forward を使い、分岐時の出力と実際の訓練微分を読む。
LC: |a|<.05σ、ELU/CELU は a<−α+.02 との和集合。TR: |g|<1e−3、別列で g=0。
GPUの付随対照ではReLUは|a|<1e−5、leaky(.1)は|a|<max(.05σ,10×.1σ)。対照は報告のみ。
A=LC∩{|g|≥.1}/LC、B=TR∩not LC/TR。ゼロ分母は未定義（JSON null / CSV空欄）で、0にしない。
n_eff = (Σ|g|)²/(NΣg²)、全入力0なら0。ユニット平均を層の値とする。
GELU/SiLU の負微分は絶対値で扱い、符号の相殺を死亡とみなさない。

E は既存 `arms.csv` の k=1 の online_acc。
26腕の seed 平均の順位相関を主報告、seed ごとの26腕の順位相関も併記。
入力や腕を独立標本としてCIを出さない。両層を報告し第2層の比較を主とする。
R2_10 と N10 の LC 要素数一致を全seed・両層で検査する。

GPUの自然軌道は neff_pred のB・Lエンジンを全モデル同じ並びで再実行する。
seed0–2、RL/PM、150タスク。記録済み online と各層の終端 z 平均の全行完全一致が必須。
開始場の E は同じタスクの online、終了場には E を割り当てない（測定時刻を混同しない）。
RL GELU t50 の A は task50 phase=end の layer2 で読む。
この部分はチェックポイントが無いため再生コストがあり、「走ゼロ」はCPU分岐再解析に限る。

## 3. C2 — 押しの持続

resp_ee の箱と更新則をそのまま呼ぶ。float32、CPU1スレッド、flush_denormal=True。
保存状態の出所のtorch・CPU capability・入力データ・ソース・checkpoint hashを照合。
宿主mucap_el_runは後から別のcap処理を追加済みなので、記録commitのblob hashを確認した上で
resp_eeが使用するforward2のAST一致を要求する。他の依存ソースはファイル全体のhash一致。
各自然腕は元のprefixのstate hashとonline_accに完全一致しなければ停止。
各腕でparams、m、v、時刻、ラベル/バッチ生成器を独立cloneし、Adamはリセットしない。
分岐t2,t5、seed0–9、6000更新×2タスク。

| 腕 | 最初の継続タスクの操作 |
|---|---|
| N | γ=1 |
| K_rev | §1の周期衝撃 |
| K_mid2 | 更新index [3000,3075) にγ=2 |
| K_sw2 | 更新index [0,75) にγ=2 |
| K_hold2 | 全6000更新γ=2 |
| K_hold2_L2 | 第2層のみ全6000更新γ=2、報告のみ |

**2タスク目は全腕・全層でγ=1**。非線形直前のzに掛け、逆伝播もγ倍を含む。
オンライン精度/CEは更新前。終端は衝撃有効時と解除時の両方でmemo、LC/TR/厳密0/n_effを読む。
全更新の精度とCEをnpzに保存して衝撃中・解除後の経過を検査できるようにする。
主判定はt5,k1。t2およびk2は報告。下記ラベルは排他的でなく併記できる。

- PERSISTENCE_MATTERS: CI(K_rev−N)が0を含み、CI(K_hold2−K_rev)の上端<0。
- TIMING_MATTERS: CI(K_sw2−K_mid2)の上端<0。
- SHOCK_HARMS: CI(K_rev−N)の上端<0。
- いずれも満たさなければ INCONCLUSIVE。多重比較の補正なし（設計案を踏襲）。
- 欠損・重複・非有限・出所不一致があれば主判定を出さずエラーにする。

## 4. C3 — 傾き×漸近値

784–100–100–10、先頭1200枚/255、batch16、各epoch同じ順、400epoch×50タスク。
seed0–2。大域torch seedから各タスクの訓練1200ラベル→検証500ラベルを50回引いた後、
nn.Linear3層を初期化。検証ラベルは生成順再現のため消費し訓練しない。
torch.optim.Adam既定、momentはタスクを跨ぐ。CPU1スレッド、flush_denormal=False。
native F.elu/F.celuの実際のautograd微分を測る。C2の−16.64の床は持ち込まない。

| 腕 | 負側の式 | c | 傾きck | lr |
|---|---|---|---|---|
| E1 | ELU(1) |1|1|1e−4|
| S36 | exp(3.6z)−1 |1|3.6|1e−4|
| C36 | CELU(3.6) |3.6|1|1e−4|
| E36 | ELU(3.6) |3.6|3.6|1e−4|
| E1_lr1e3 | ELU(1) |1|1|1e−3|

毎task: online、floor=最多ラベル率、F=(online−floor)/(1−floor)、memo、zbar、||μ₂||、
LC/TR/0/n_effとユニット別n_eff。Fはclipしない。T_halfは最初のF<.5のtask、
一度も下回らなければnullと右打切りflag。後半t41–50のF、生涯onlineをseedごとに出す。
傾き効果=(S36+E36−E1−C36)/2、漸近値効果=(C36+E36−E1−S36)/2、差も報告。
3seedのt区間は記述的な補助。後半平均F≥.5を「生存」の予測閾値とし、機構の因果証明にはしない。
本走の既存結果をこの箱へ混ぜない。長走の再開機能は無く、各task終端にcheckpointを保存する。

## 5. C4 — 正しい必要時間の下限

m,vが0初期値、β₁²<β₂、ε≥0として、Cauchy–Schwarzから

    |m_hat_t|/sqrt(v_hat_t) ≤ C_t
    C_t = (1−β₁)/sqrt((1−β₂)(1−β₁²/β₂))
          × sqrt((1−(β₁²/β₂)^t)(1−β₂^t))/(1−β₁^t).

(1−r^t)(1−β₂^t)≤(1−β₁^t)²（r=β₁²/β₂、AM–GM）なので係数の一様上限は
(1−β₁)/sqrt((1−β₂)(1−β₁²/β₂)) ≈7.27。任意の勾配列に対して有効。
3.16は孤立した勾配等の特別な履歴の値で一般の一様上界として使わない。
μ固定、外部γ変化なしなら |Δzbar|≤ηC(||μ||₁+1)。深さdから戻るには少なくとも
ceil(d/[ηC(||μ||₁+1)]) 更新が必要。ただしその回数で戻れるという意味ではない。
第2層はΔμ項もあり本式だけでは上界にならない。動く入力に適用するなら
Δzbar=Δw·μ+w·Δμ+Δw·Δμ+Δbの残りを別に抑える必要がある。

## 6. 予測の扱い

親設計案のClaude予測を継承: C1 RL GELU L2 t50 A≥.3(60%)、RL EE L2 t10 A<.05(85%)、
|ρ(E,LC)|<|ρ(E,n_eff)|(75%)、R2_10/N10のLC一致(構成上)。
C2 t5 PERSISTENCE_MATTERS(65%)、TIMING_MATTERS(45%)、K_hold2≤.2(60%)、
t2全腕がNとの差.05以内(55%)、k2にK_sw2の傷が残る(40%)。
C3 E1生涯≤.40(70%)、E36≥.75(55%)、C36≤.45(65%)、S36後半F≥.5(50%)、
傾き主効果>漸近値主効果(55%)。C3の具体化した実装条件は§4を優先する。
Issa予測は未記入。本実装依頼を予測への同意とは解釈しない。

## 7. 検査・実行・保存

`python -m unittest analysis.lc_complement_0917.checks -v`:
境界γ、k2解除、2層への作用とγの連鎖律、自然継続のstate一致、clone独立性、
LCゼロ付近/負側床、TRとn_effのゼロ/負微分、C3関数と標準kernelの一致、
一般Adam上界と3.16反例、C1–C3集計のラベル/欠損/重複の拒否、CLI煙試験。
実データcheckpointのhashと前向き一致も検査する（効果の検定はしない）。
全C2のsmokeとC3の5腕smokeは別出力・provenance.smoke=true。
本走コマンドは `analysis/lc_complement_0917/README.md`。既存出力の上書きを拒否。
実行開始時にgit/spec/source/dataの出所を記録。COMPLETE以前やsmokeのshardは主集計を拒否。
生npz/pt/logは `~/Projects/obsidian-research-data/lc_complement_0917/`、小さいCSV/JSON/説明はgit。
完了時はCLAUDE.mdに従いmanifest、mainへのmerge/push、worktree片付けを行う。
