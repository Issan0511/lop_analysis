# erosion_race_0919: 沈降と新ラベル学習の競争

登録者: Codex / 三理論の統合・侵食と回復。2026-09-19 JST。
実行許可: Issa「この仮説面白そうなので検証して」。本 spec を commit する前に本走を開始しない。

V10作業リストへの対応: H1/A2（上流の駆動源とAdam実変位の接続）の初期第1層部分を検証し、H4（応答と学習）の機構候補を時間介入で調べる。W増大全体・H8崩壊時刻・長期生存条件まで閉じる実験ではない。

## 1. 問いと既知情報

仮説は、誤った予測の修正中に特徴の応答が失われる速さと、新しい正解への適合が成立する速さの競争が、活性化間の差を生む、である。平均入力だけでは更新の符号も Leak/Snake の生存も説明できない。

既知の同箱実験 rlcifar_mlp_battle_0918 (seed 0–9) では raw の ReLU/GELU は早期に低性能、ELU は課題1では学習でき後に低下、LR/SNA は比較的維持される。この既知結果を見て設計した検証であり、仮説の完全な盲検発見ではない。今回の seed 1001–1005 はこの親実験の 0–9 と分離する。50課題後の帰結を2課題の走から証明しない。

検証を分ける。(A) 実更新の下向き量と上向き量、(B) 応答とラベル適合の時間順、(C) 初期沈降に一時介入した後の学習。A/B の相関のみで C の因果を主張しない。CE の confidence 項を定義だけで「侵食」、label 項を「回復」と呼ばない。

## 2. 固定実験箱

親: src/rlcifar_mlp_battle_0918.py、base main 81fb7f2。raw CIFAR10 から seed ごと固定1200画像、ランダム10分類ラベルを課題ごと再抽選、3072–100–100–10 MLP、batch16、Adam lr=.001, beta=(.9,.999), eps=1e-8、400 epoch = 30000更新/課題、2課題。初期化・画像・ラベル・順序の独立 RNG を親から再利用。optimizer は課題をまたいで継続。

seed=1001,1002,1003,1004,1005。全走を同じ R=5 の stacked baddbmm で実行。同一 seed の全腕で入力・初期パラメータ・ラベル・batch列を揃える。親の旧 R=20 との bit 一致は要求せず、今回の R=5 の親エンジンと短走で一致を検査する。

9走 (=45 seed 条件): R/base, GELU/base, ELU/base, LR/base (a=.1), LK001/base (a=.01), SNA/base, GELU/early, GELU/late, ELU/early。SNA は親の adaptive Snake (c=.6,beta=.01,alpha clip [.005,3])。SiLU、固定Snake、入力中心化は今回は加えない。

pilot/check は seed 9995–9999 の短走のみ。本走の結果を見て閾値・介入窓・seed・腕・終了時点を変えない。実装不具合は修正履歴と無効走を明示する。

## 3. 実軌道上の CE/Adam 分解

pi=(.1,...,.1), logits=f とし、L_conf=mean_x[logsumexp(f)-pi^T f], L_label=-mean_x[(y-pi)^T f]。g=g_conf+g_label。通常の CE autograd で求めた g による親の Adam 更新を実更新の正本とする。conf 勾配を別 backward で求めても実際の g を置き換えない。

第1層の一次モーメントだけ、各課題開始時の m_start を別に保存する。課題内 s 更新後、m_hist=.9^s m_start、m_conf は課題頭で0から conf 勾配を EMA、m_label=m_actual-m_hist-m_conf。3成分すべてに実軌道の v_actual と全履歴 t の bias correction を共有する。各 u_k=-lr*m_k/(1-beta1^t)/(sqrt(v_actual/(1-beta2^t))+eps)。過去課題の履歴を現在ラベルの学習と混同しない。

これは実軌道の加法的寄与帰属であり、各成分だけで独立に走らせた Adam の反実仮想ではない。浮動小数点の実パラメータ差と式上更新との差も残差として記録する。

## 4. 一時介入

第1層の拡張入力平均 h=(mu_x,1)、各unitの conf 由来更新 u_conf=(delta_w,delta_b) に対して、

    correction_i = -min(h dot u_conf_i,0) * h / ||h||^2

を通常 Adam 更新に加える。conf 成分の負向き平均変位だけを除去する。optimizer の m/v は書き換えない。正方向の conf 変位、label/hist 成分はそのまま。補正は実際の総変位にも別寄与として入れる。

early: 課題1の更新1–5000だけ。late: 課題1の5001–10000だけ。課題2は全腕介入なし。解除後の25000更新 (early) または20000更新 (late) と、その後の新課題で効果を調べる。

この介入は入力を中心化せず、微分の床を直接付与しない。ただし h 方向の重み変更は画像間差にも作用し得るため「平均だけを変え、中心化特徴を厳密に保存する操作」とは主張しない。late は同長の時期対照であって、全経路が同一の sham ではない。機構の一意な媒介証明にはならない。

## 5. 計測

各課題の更新0,1,10,30,75,150,300,750,1500,3000,5000,7500,10000,15000,22500,30000で全1200画像を評価。0は新ラベルに切り替えた後・更新前。

機能: 全集合の CE, accuracy, A=mean_x[f_y-mean_c(f_c)]。per_task の online accuracy は親と同じ更新前batch正解率の全30000更新平均。これは固定ランダムラベルへの適合であり、未見画像への汎化ではない。

応答: 各層/unitの z の平均・標準偏差、phi(z) の標準偏差、mean(phi'^2)、ppos、dead=max_x|phi'|<1e-6。中央値等と unit 配列を保存。実活性化の入力に微分を適用する (LN腕は使用しない)。初期と課題末に effective rank を補助計測できる。

更新収支: 各 seed の固定最初64画像×第1層100unitから、初期 z>0 の対だけを一度選び、2課題を通して同じmaskを使う。固定初期 unit sigma_z を max(sigma,1e-6) として、毎更新の実パラメータ差による delta_z/sigma を計測。E=mean_mask max(-delta_z/sigma,0)、R=mean_mask max(delta_z/sigma,0)。全更新を加算し診断間窓ごとに出力。正側から消えた対も追跡するため、現在の正側集合への流入出で母集団が変わらない。これは初期正側だった対の輸送であり、「常に正側で起きた量」とは呼ばない。

同じ mask 上の符号付き conf,label,hist,correction 寄与と closure residual を保存。max は実総変位に適用し、成分を個別に rectify して足さない。64画像は学習用 subset からの固定機構probeで、1200画像全体の厳密な flux ではない。観測は RNG/パラメータ/optimizer/SNA状態を変えない。

## 6. 登録判定

主たる因果判定 C1: GELU early-base の課題1末 accuracy の seed 対応差。中央値>=+.10 かつ5 seed中4以上が正なら EARLY_HELPS、それ以外 NO_CLEAR_RESCUE。全対応差を併記する。n=5の記述的登録基準であり、有意差検定・個体を独立標本とする判定ではない。

C2 (時期・副判定): GELU early-late の課題1末accuracy差が中央値>=+.10かつ4/5正なら EARLY_ADVANTAGE。それ以外 NO_EARLY_ADVANTAGE。C3 (持続・副判定): GELU early-base の課題2 online accuracy差が同じ基準を満たすか。C4: ELU early-base の課題1末accuracy差を同基準。ELU は天井効果の可能性を明記する。

M1 (起源): 課題1更新1–750の conf,label,hist の初期正側probe輸送 C,L,H を積算。seedごとに C<0 かつ max(-C,0)>0.5*(max(-C,0)+max(-L,0)+max(-H,0)) を満たすか。R/base と GELU/base のそれぞれ4/5以上なら CONF_NEGATIVE_DOMINANT、それ以外 NOT_ESTABLISHED。成分の大きな相殺がある場合は実 net と併記する。

M2 (下向きと回復): 課題1更新1–750について、GELU対LR、GELU対SNAの各seedで N=E-R を比較。DeltaN=N_G-N_other=(E_G-E_other)+(R_other-R_G)。対応差5本と、その算術平均による厳密な二項分解を示す。平均DeltaN>0のとき第1項がDeltaNの2/3以上なら LESS_DOWNWARD、第2項が2/3以上なら MORE_UPWARD、どちらでもなければ MIXED。両項の一方が負でも符号を隠さない。平均DeltaN<=0なら NO_LOWER_NET_EROSION。この記述判定は活性化の全寿命を代表しない。

M3 (時間順、第1層が主・第2層は副): V(t)=median_unit[sd(phi(z_t))/max(sd(phi(z_init)),1e-6)]。初期sd<=1e-6のunitは比率から除く。V<=.1が連続2診断で成立する最初の診断を応答低下到達、accuracy>=.5が連続2診断で成立する最初の診断を適合到達とする。各seed・各課題の区間打切り到達時点を列挙し、未到達を無限時間と偽装せず censored とする。task2のVも初期基準。deadは副指標であり leaky の微分床をそのまま生存証明にしない。

総合: C1成立かつM1成立なら、初期conf由来負方向輸送を弱めると適合が改善する版を支持。C1不成立なら「5000更新の猶予で十分」は支持せず、他の競争版まで一括棄却しない。M1不成立ならconfidence項が侵食の主因という説明を修正する。M2がMORE_UPWARDなら「Leak/Snakeは侵食自体が少ない」という説明を採用しない。M3は順序情報であって単独の因果判定にしない。

## 7. 事前予測 (Codex署名、2026-09-19、本走前)

| key | 予測 | 確率 |
|---|---|---:|
| P1 | C1=EARLY_HELPS | .60 |
| P2 | C2=EARLY_ADVANTAGE | .65 |
| P3 | C3 は持続改善の基準を満たさない | .65 |
| P4 | M1=CONF_NEGATIVE_DOMINANT | .65 |
| P5 | GELU対LRのM2はLESS_DOWNWARD | .55 |
| P6 | GELU対SNAのM2はLESS_DOWNWARD | .55 |
| P7 | ELU/base の課題1適合到達は4/5以上、GELU/baseは1/5以下 | .80 |

不一致も全て採点。閾値を結果後に最適化しない。探索的追加集計はそう表示する。今回は2課題で打ち切り、成功/失敗を理由に自動で50課題や新腕へ拡張しない。

## 8. 実装検査・成果物

CE勾配恒等式、課題切替を含むAdam3項closure、更新寄与closure、介入の符号/射影、matched positive branch、ELU平行移動の差分縮小、leaky分散床を検査。親baselineとの同R短走、観測on/off、CUDA graph/eager、SNA状態とRNG不変を検査してから本走。

spec、engine、checks、launcher、report、results/erosion_race_0919/{summary.md,verdict.json,figure.png,各走CSV/provenance} を残す。生unit配列・checkpoint・ログ・pilotはobsidian-research-dataへsha256/bytes付きで退避。本走git_hashは実行前commitを指す。最終報告で限定された支持/不支持と未検証範囲を明記し、repo mainとvaultへ統合する。
