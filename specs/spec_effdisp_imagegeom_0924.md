# CIFAR画像の次元・分散・最適化器と実効変位 — 0924

状態: 本走前の事前登録。起草2026-09-24 JST、Codexの予測（Issaの予測ではない）。
Issaの明示依頼: SCR Adamの確認に加え、64×64への画素複製、grayscale32、16×16縮小を実行する。
V10 H1・実効変位の入力次元依存を調べる。既存の20→40bit走とは別実験。

## 1. 固定した設計

ローカルGPU、seed300–304の5本、leaky .1、d→100→100→10 MLP、CE。
各seed固定CIFAR10 train画像1200枚、各task新しいuniform10class乱数ラベル、50tasks。
各task400epochs×75batch×batch16 = 30000更新。no WD/reset、optimizer状態を持ち越す。
Adam lr=.001, beta=(.9,.999), eps=1e-8 / 純SGD lr=.01, momentumなし。
画像bank・label・epoch batch順は全geometry/optimizer間でseedごと共有、独立stream。
基準画像はuint8を/255したRGB（raw01）。channel標準化はしない。
したがって過去のstd入力のAdam/SGD倍率やSGD走の独立再現とは呼ばない。

| geometry | 変換（CHW） | d | 比較の意味 |
|---|---|---:|---|
| rgb32 | raw01 | 3072 | 基準 |
| dup64 | 各画素を縦横2回ずつ複製するnearest | 12288 | rank固定、非零共分散固有値4倍 |
| dup64_half | dup64/2 | 12288 | rank・非零固有値とも基準と同じ |
| gray32 | .299R+.587G+.114B | 1024 | 情報も変わる |
| avg16 | 非重複2×2ブロック平均 | 768 | 高周波情報も変わる |

5geometry×2optimizer×5seed=50系列。既存CIFARと同じ更新数を省略しない。
各変換はtask間で固定。Σもµも固定なので入力変化の帳簿項G=0。
raw01/標準化やlrの最適値を結果を見て選ばない。非有限値は当該系列DIVERGEDとして残す。

## 2. 初期化と最適化器の座標依存

基準は既存hostのLinear初期化（fan-in一様）。gray32/avg16は同じ規則のnative fan-inで初期化。
dupは基準第一層の重みを空間的に2×2へ複製し、dup64は/4、dup64_halfは/2。
全腕の第二層以降とbiasはseedごと共有。dupの初期関数は基準と一致する。
初期optimizer momentsは0。同じlrを全層で使う（geometry別のlr補正はしない）。
灰色化/縮小の初期関数は揃わない。比較には情報量・難度・初期関数の差が残る。

q=4、複製作用素AはAᵀA=qI、x'=sAx、W'=WAᵀ/(qs)。
非零固有値はqs²倍。初回第一層更新の関数変化はSGDでqs²倍、Adamで概ねqs倍（epsilon無視時）。
よってraw dupは両者4倍、half dupはSGD1倍/Adam2倍。
これは初回/対応する同一状態での式で、50task後のDや固定点の倍率を保証しない。
純SGDのhalf dupは実数精度・対応状態のもと全軌道が同じ。丸め誤差と長期の軌道分岐を別記する。
lr補正による同値性は小さな独立算術検査だけ行う（追加の長時間腕にはしない）:
第一層SGD η'=η/(qs²)、Adam η'=η/(qs), ε'=sε。他層/biasは変更不要。

## 3. 保存と計量

初期状態と全task末の全パラメータ、optimizer moments/step、正確な画像bank/indices、task labels、乱数定義とsource hashを保存。
各task train/start/online accuracy、CE、層別mean|phi'|とactive unit fractionを保存。
第一層のみを主計量とし、population共分散を画像bankで定義する。
V=tr(WΣWᵀ), Q=tr(DΣDᵀ), X=tr(WprevΣDᵀ), D=Wnext−Wprev。
ΔV=Q+2X, R=sqrt(V), d_eff=sqrt(Q), c=X/sqrt(Vprev Q), rho=sqrt(Q/Vprev)。
mean channel ||Dµ+db||²、raw ||W||²、||D||²も併記。量はfloat64。
大きな12288²共分散を作らず、中心化画像への射影で正確に計算してよい。
rankはsingular value > 最大singular value×1e-10のfloat64数値rank（dupの理論rank一致も報告）。
effective rankはparticipation ratio tr(Σ)²/tr(Σ²)。上位10PCのQ割合も報告する。
この1200枚bankの中心化rankは最大1199。ambient dimensionだけで実効rankを代用しない。

## 4. 判定・予測（結果を見て変更しない）

既存 `spec_effdisp_validation_0924.md` §4 のP1/P2/P3/P4を閾値ごと継承。
late=task41–50、P3calibration=task6–25/test26–50。群PASSは5seed中4seed以上。
STOPPED/ゼロQ/無効係数を隠さない。leakyでも学習停止をactive plateauに数えない。
各geometry/optimizerのR,d_eff,c,rho,train_accをearly6–15/late41–50のseed内中央値で集約。
geometry/rgb32のpaired比、Adam/SGDのpaired比をすべて掲載。95%CIはseed bootstrap5000回、rng924。
bootstrapは同じseedの対を保つ。cは符号を保つ差も併記し、0近傍の比は無効とする。

- E1（90%）: half dup/SGDのlate Rとd_effはrgb32/SGDの各.8–1.2倍に入り、5seed中4seed以上で成立。
- E2（70%）: half dup/Adamのearly d_eff比は>1.2、5seed中4seed以上。初回の2倍を全taskの2倍へ外挿しない。
- E3（60%）: raw dupのearly d_effは両optimizerで基準より大きい（各>1、4/5seed）。
- E4（未指定・report）: gray/avg16のR・D・精度の方向は決めない。rank低下だけでは削り/供給と学習難度の競争が決まらない。
- E5（30%）: 全geometryでcが同じ、という強い仮説。各optimizerごと各geometryのlate c中央値と基準との差の絶対値≤.05が全4比較で4/5seed以上なら支持。固定点Rが次元だけで決まるとは予測しない。
- P2が不成立なら「幅の固定点の高さ」という表現を避け、観測時間窓の幅と書く。
- P3は独立のheldout予測。恒等式の一致や同じ窓のD/(2|c|)を予測の成功に数えない。

## 5. 実装・資源

簡単な実装/解析はgpt-6-solに委譲し、rootが仕様/式/主要出力を検証。
2task以下の縮小smokeは/tmpに保存し、本走と混ぜない。速度・finite・同値算術だけを見る。
入力分布factorial `effdisp_inputscope_0924` のGPU走が終わってから本走を開始。
4つの入力次元ごとにベクトル化する予定。既存の所要時間から追加10–20時間程度と推定（未bench）。
途中の結果でtasks/seeds/armsを縮めない。コマンド確認は10分間隔。ローカルのみ。
生データは `obsidian-research-data/effdisp_imagegeom_0924`、SHA256manifestを残してmainへ統合。
