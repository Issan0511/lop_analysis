# 劣化後のCH′に自然なBを追加する直接回復対照

登録状態: REGISTERED
実験ID: `cnn_chb_delayed_1021`
作成日: 2026-10-21 / 依頼: Issa（元会話 01a14da9-6ed1-7d12-9eb3-ed79510a9c72）

## 問いと範囲

既に劣化したCH′の課題30末からBを後付けすると、次課題の学び直しが改善するか。学習開始からのCH′＋B予防とは別の直接回復実験。全層人工応答交換 `cnn_response_chb_1021` と幅capの結果を保持する。人工応答、初期補正、κ探索、seed選別、窓探索、追加追跡実験は行わない。予測は精度とCEの改善だが、両者は別に判定する。未確定・悪化も登録どおり報告する。

## 源と固定条件

- 最新origin/main `1ab160647149d36ff30582c3ca3fba0b5755deb5` から専用worktree/branch。
- 唯一の親: `cnn_chprime_bias_1018` の同精度CHf task30完全状態。
- source: `/home/issan/Projects/obsidian-research-data/cnn_chprime_bias_1018/host_gpu_recovery_1018/results/cnn_chprime_bias_1018/CHf/state.pt`
- SHA256: `936b3e285a979f97097234dd76c816fa24417e773f5ce2efbf19c6f6c8d0c4c3`。
- 元の12591要素list型backup_manifestとsource state、全10 checkpoint、provenance、per_task、stream_t30の計14ファイルをSHA/bytes照合。compact source_manifestを保存。古いTF32=trueのP183は使わない。
- seed10–19を順に全て保持。1200枚/seed、batch16、400epoch×75更新=30000更新。次の課題31。lr=.001、Adam β1=.9/β2=.999/eps=1e−8。tc=900000から継続。
- ReLUは親のtorch.clamp(min=0)（ゼロ微分1）。入力中心化、特徴ごとのEMA中心化、raw z幅EMA、全W/b、head、Adam m/v、EMA/V、label/batch RNGを継承。task切替時は親と同じ観測用hupdカウンタだけをゼロ化。初期学習状態を補正しない。
- 数値条件: torch2.13.0+cu130、RTX5060Ti、float32自然forward/Adam、B幅計算float64、matmul/cudnn TF32=false、deterministic=true、benchmark=false。CUDA graphのstale capture stream overrideをプロセス内設定・記録。

## 2腕と実装

1. `CH_continue`: CH′をそのまま継続、Bなし。
2. `CH_addB`: 同じCH′状態からBを有効化。

両腕とも同じ新labelと各epochのbatch順、同一コードの自然forward。自然CF.forwardを直接呼び、人工Fieldは構築しない。旧CH_self/CH_swapの曲線は必要時に参考比較のみとし、主対照に流用しない。

Bは親 `src/cnn_chprime_bias_1018.py` の既存関数をそのまま呼ぶ。各更新の順序は自然forward/CE/勾配→Adam→そのforwardの活動からEMA更新→同じbatchを再forwardしてbiasなしqの幅を計算→c1,c2,f1,f2のbiasを順次±.2sへ射影。下流には修正biasと受け手EMAを反映した自然活動を渡す。conv幅は各空間位置の画像軸分散を空間平均してsqrt、FC幅は画像軸分散のsqrt。W/head/Adam状態は射影が直接変更しない。

**初回のBは最初のAdam/EMA更新後**。初回online精度/CEと初回勾配、Adam更新、EMA更新、W/headは2腕一致する。初回射影後のbiasから自然な軌道が分かれる。Bによるbias移動をAdamのbias更新量に混ぜず別に保存する。

## 評価と事前判定

onlineは各batchの更新前予測。各seed内で算術平均し、seedを単位として対応比較。精度の主窓は更新1–3000。CEも同窓を別の確認的評価項目にする。

- 精度便益 = CH_addB accuracy − CH_continue accuracy（報告は百分率点）。
- CE便益 = CH_continue CE − CH_addB CE（小さい損失を正の便益）。CEは生の平均、log変換・clipなし。
- 確認的検定族は上記2項目。各exact two-sided sign test、厳密なゼロ差はtieとして検定母数から除き、2項目をHolm補正、α=.05。
- 項目ごとに、Holm p≤.05かつ平均便益>0かつ正差seed数>負差seed数なら `IMPROVES`。負方向が同様なら `WORSENS`。有意だが平均と多数符号が一致しなければ `HETEROGENEOUS`、その他は `INCONCLUSIVE`。精度とCEを別々に報告し、片方だけ改善を両方の回復とは呼ばない。
- 対応平均差の95% percentile seed bootstrapは20000回、numpy.default_rng(1021)、同じ10seed再標本indexを両項目に使用。検定判定にCIを追加条件としない。
- 副窓は更新1001–3000、全1–30000、末尾27001–30000。精度・CEとも全seedの水準・差と平均を示す。副窓に新しい成功判定や検定を付けない。
- 失敗seedは除外せず、非有限loss/勾配/学習状態検知以降accuracy=0、CE=+infとし継続記録。CE比較は片側+infを悪化、両側+infをsign tieと扱う。非有限CEが一つでもあれば数値平均便益/CIを未定義(null)、CE判定 `INCONCLUSIVE_NONFINITE`。全seed水準・失敗更新を別記。データ欠落や実装資格不合格は完走/科学的判定としない。

改善は劣化したCH継続に対する回復を意味する。fresh/予防CHB水準への完全回復や同等性は試験していない。bias/forward/応答/尺度/Adamの軌道が同時に動くため、応答だけの媒介率・責任層は同定しない。

## 診断と保存

更新0,1,100,750,3000,15000,30000に全1200枚/seedの診断。各層の位置m、画像軸幅s、相対位置r、positive率(z>0)、native平均微分(z≥0)/分散、pool winner微分、bias/幅、Wノルム、活動effective rank、EMA/V、Adam分母。ゼロ幅の比は非有限を隠さず保存。全seed/層/位置の配列とunit集約を残す。

全30000更新の精度/CE、5層の相対W更新、Adam bias更新RMS/相対norm、B込みbias移動、勾配/Adam分母量、B発火率を保存。Bは全epochの順序と各unitの幅・射影前後bias・hitを保存する。epoch境界完全再開状態、診断点完全checkpoint、label/batch stream hash、source/code/runtime/provenance、失敗・ログを残す。

## 実行資格（結果を見て緩和しない）

1. CPU合成: 初期一致と初期射影なし、初回Adam/EMA/W/head/online一致、B無効と元CF経路一致、独立4層逐次射影、ゼロ幅/境界保持、保存再開、統計の両方向/非有限分岐。
2. spec/code/checksをcommit/pushしたreleaseだけが実データに進める。ソースSHA/設定/全checkpoint/RNG/画像subsetを照合。既存task30表を元chunk20で再現。
3. GPU実データ資格: 全1200枚の自然初期全特徴/入力/pool winner/logits/CEはbit一致。開始勾配はatol1e−7,rtol1e−6。各腕1/75stepの自然親経路との全状態はbit一致。初回Bの独立射影はatol1e−7,rtol1e−6、幅はmax(abs差/max(1,abs参照))<1e−4。
4. graph/eager75step全状態はmax(abs差/max(1,abs参照))<1e−4。capture巻戻し・同一実行反復・ファイル保存再開・診断前後state/RNGはbit一致。unused ReLU fixAの元来のNaNだけを許可し、bad=trueは不合格。元表memo差は≤1/1200、他有限量は上記relative<1e−4。
5. 全資格成功とrelease/source/runtime contract一致後のみ本走。検査による更新は毎回同一源へ巻戻し、本走へ持ち越さない。閾値調整や条件探索をしない。不合格時は原因修正を別commitで記録し、結果を確認する本走には進まない。

GPUは `/tmp/lop_analysis_gpu.lock` を共用し逐次実行。監視間隔5分以上（完了間近・即時対応が必要な時を除く）。完了後、保存曲線から別実装で主判定と全窓を再計算し、stream/diagnostics/射影/終端SHAを監査。Vault結果正本・分離済み部品表/段落地図へMCP経由で反映。生データSHA/bytes退避→main merge/push→専用worktree/branch整理→別checkout＋退避データでreport JSON全体再生成一致→元会話へ報告。
