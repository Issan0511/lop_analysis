# spec: SiLU最終FFNの一つ前の沈降を相2の合計と照合する

状態: 登録 / 2026-10-24 / Issa「しらべて」 / 実装Codex

## 1. 問い・既知・対象

4ブロックSiLUの最終FFN(L3)が正側へ出て、その一つ前L2が沈む現象を、相2の共同fitの式が説明するか。既知はseed1000でL3正側化、L2への誤差信号のFFN3経路による増幅約15倍、後期L2沈降。これらは符号の説明でも相2の証明でもない。

ローカルで読める同じ軌道の保存点を使う。SiLU seed1000の課題末1,5,10,25,50,100、および課題100更新100。対照SiLUcap seed1000の課題末10,25,50,100。全11状態。対象L2主・L3記録、各512unit。10seedのSiLU保存物はmanifest上byakkoにあり、今回ローカル束には含まれない。到達可能な状態の範囲を明記し、seed1000を典型としない。

学習なし、optimizer作成なし、API worker/subagentなし。他worktreeは読取のみ。自分のworktree以外のcheckpointを変更・削除しない。

## 2. 予測と実測（測る前に固定）

最終LN後の全128特徴Hを同じ現課題鎖probeで得る。親chargpt_earlyphase_1024と同じ64系列×128位置、先頭位置を損失から除く。

- 相2head予測P2=J_HᵀH(HᵀH)⁺。特徴を旧ラベル平均の間固定、自由なheadを共同最小二乗fitした独立・等分散旧ラベル模型。正の全体係数省略。
- 文脈対照P2ctx: C_nm=同じ直前2文字なら1。J_HᵀHG⁺HᵀCHG⁺。iidを主から差し替えない。
- 実測switch_full: 実網のCE(p,u)、fit_full: CE(p,q_current)、全パラメータの勾配。実網のhead/embedding共有を保つ。
- switch_body/fit_body: 実誤差(p−u)/(p−q)を固定した現在のheadでHへ戻し、body経路だけ微分。共有embeddingのhead直接勾配を含むfullとの違いを記録。自由head模型P2にはこの直接更新の同一性はない。

全て全unit・後段注意・FFN・LNの微分を含む。押しはS_i=〈∇θr_i,gθ〉、Δr_i=−ηS_i。r_i=μ_i/s_iは固定一様probeの前活性平均/不偏標準偏差。JVPで全パラメータ経路の押しを出す。自unit up行だけの押しも同じ勾配から記録し、full−ownを上流経路として恒等的に確認。主はfull。μの押しも記録する。

SGDと保存Adamの分母固定(補正済み2次モーメント、eps込み)を分ける。Adamの運動量・実更新は再現しない。主の符号照合はSGD、Adamも同じ表を報告。

## 3. 集約と判定

SiLUの負側を死と決めない。全512unitを記録し、主比較の応答あり集合は一様probeでmax|φ′|>1e−6。p(z>0)、平均|φ′|、応答なし数、r中央値を併記。非ゼロ判定は各配列のmaxabsの1e−10以下を数値ゼロとし分離する（1e−8/1e−12で感度記録）。

予測と実測の共通非ゼロ集合で符号一致、balanced accuracy(実測沈む群と浮く群の的中率の平均、各10unit以上)、Spearman、双方の沈む割合を出す。さらに全unitのSの平均・中央値を残す。単に相関があることと、合計として沈むことを区別する。

記述判定（単一seedの適用テストで、定理の真偽ではない）:
- SWITCH_MATCH: SiLU L2の課題50/100末でP2full対switch_fullのBA≥.65が両方。片方ならPARTIAL、どちらも満たさなければNOT_MATCHED。
- FIT_MATCH: 同じ条件をfit_fullへ適用。切替の成功を課題内の沈降の説明へ持ち越さない。
- LATE_SINK: 同じ2状態でP2fullとfit_full双方の全unit S平均が正、rが50→100で下がるか。Adam版も別に判定する。満たしても軌道積分・因果同定とは呼ばない。
- L3上昇: 課題10→25のr中央値とP2/実測の向きは記録のみ。相2に必ず上昇する性質があるとはしない。

予測（Codex、走る前）: SWITCH_MATCH .50、FIT_MATCH .20、LATE_SINK(SGD) .20。独立二値Brierを報告。

## 4. 検査・数値感度

checkpointはSHA256記録、model/source/probeの由来を保存。layer_statsのr中央値・sd中央値を同じ一様probeで相対1e−3以内(丸め絶対1e−5も許容)に照合。不一致は本走を止める。課題10と100でbody戻し＋head直接の勾配=全CEを相対1e−9で確認。

P2はGram固有値閾値1e−10×最大。1e−8/1e−12のrankを全状態記録し、rankが変わる場合にはその予測も出す。主閾値不変更。課題10と100、相2・実switchの全方向をnorm1にし、rのJVPをパラメータ摂動±1e−4/1e−5/1e−6で独立確認（最終刻みのmax誤差/max|予測|≤1e−4）。P2自体も半log擬似detの有限差分で確認、rank不変を要求する。失敗時は保存し報告、閾値を緩めない。

実行前にpython本数・nvidia-smiを保存。他の実行を止めない。CPU float64。

## 5. 保存

code analysis/chargpt_siluphase2_1024/。results同名（states.csv、checks/provenance、summary/judge、全unit配列）。raw/logsはobsidian-research-dataへ検証付きで退避してmanifestをcommit。specを先にcommit。完了後mainへmerge/push、worktree・branch削除。Obsidianの既存「文字GPTで継続学習_L0の相殺後の押しを相2と照合する_結果_1023」にSiLUの別対象として追補。
