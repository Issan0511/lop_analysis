# sna_grow_1017 — 幅の伸びだけを追う適応 α は、CNN の SNA の劣化を止めるか

状態: **登録（この commit）。実装・検査・本走はこの commit より後。**
作成: 2026-10-17 / 設計・予測・判定: Claude（親）/ 実装・走・集計: Claude（Opus、workflow のサブエージェント）/ 起点 origin/main = 063b7c60。
起案: Issa「幅が縮むのに合わせに行くのは変。縮んでも飽和 LoP は起きない」（1017）。理論（V12 §4）が LoP の駆動として置くのは幅の伸び（H5）で、縮みではない。適応 α を入れた理由は「幅が伸びても波の形を保つ」ことなので、縮みを追う理由は理論に無い。

## 0. 既知（このノートの前に見たもの。予測の成功に数えない）

- **CNN（RL-CIFAR、Kumar の CNN、sna_cnn_cause_1009、seed 10–19）**: 窓 t21–30 の online の seed 平均は SNA 0.9576、SNAc3（全層 c = 3）0.9649、SNAfrz1（課題 1 末で α 凍結）0.9694。対応差の seed 間 SD は SNAc3−SNA 0.0036、SNAfrz1−SNA 0.0039。窓 t11–20 は SNA 0.9635、SNAc3 0.9660、SNAfrz1 0.9727。99% 到達 epoch（中央値）は SNA 21 → 33 → 49 → 70（t1/t10/t30/t50）、SNAc3 19 → 41 → 35 → 28。課題末の精度は全腕・全課題で 0.97 以上（SNA は最悪でも 0.9975）。
- **切替直後の縮み（再走 t20 → t21、seed 10–12、scratch `dist_anim/`）**: SNA の f2 の幅は 1 epoch で 15.3 → 2.2 に縮み、平均も 0 へ寄る（z̄ −13 → −3.3 at 10 epoch）。α は 10 epoch で 5.8 倍に上がる。同じ軌道で α を切替時の値に据え置くと、f2 の傾きの平均は 3/5/10 epoch で 0.49/0.65/0.74（実際は 0.16/0.30/0.38）、f1 は 0.44/0.59/0.72（実際 0.32/0.42/0.50）。c2 は逆で、据置 0.21〜0.38 に対し実際 0.52〜0.62。軌道の変化（feedback）は含まない。
- **5+1 CIFAR × MLP（controls_cifar5p1_1008、seed 0–9、後期窓 hard 21–29）**: SNA 0.6648、SNA_fixT1（課題 1 末で α 凍結）0.6717（SNA − fixT1 = −0.0069、95% [−0.0128, −0.0010]）、SNA_fix0（α = 0.6 固定）0.6034。**MLP で適応が固定 α に勝つ分は、課題 1 で一度合わせることで取れている**（課題 1 の中で幅が約 7 倍になるため）。課題 2 以降も追い続ける利点は見えていない。

## 1. 問い

**Q1（CNN、主）** 課題 1 は SNA のまま幅に合わせ、課題 2 以降は α を「これまでの最大の幅」に合わせる（伸びだけ追い、縮みは追わない）と、SNA の劣化は止まるか。
- 止まれば: 劣化の原因は「縮みを追うこと」（切替直後の縮みに α が追従して fc を谷に留める）と読める。
- 止まらなければ: 原因は「課題の終わりに深い層が 2αW = 1.2 の鋭さのまま谷に座ること」のほう（この規則でも課題末は幅がほぼ最大なので 2αW ≈ 1.2 に戻る）。SNAfrz1 が効いたのは伸びまで追わなかった（ゲートが洗われた）から、と読む。

**Q2（MLP、副）** 同じ規則は、5+1 CIFAR × MLP で SNA の成績を落とさないか。

## 2. 腕の定義（両箱共通の規則 `grow`）

- 課題 1 は宿主の SNA と同一（α = clip(c/√V, lo, hi)、V はバッチ分散の EMA、β = 0.01、c = 0.6、lo 0.005、hi 3.0）。
- 課題 1 の終わりに、適応している全サイトで V_max ← V（unit／チャネルごと）。以後は毎更新、V の EMA 更新の直後に V_max ← max(V_max, V)。α = clip(c/√V_max, lo, hi)。V の EMA そのものは宿主のまま続ける。
- c・β・clip・初期化・学習率・データ・ラベル列・バッチ列は宿主のまま。全サイト（CNN は c1・c2・f1・f2、MLP は第 1・第 2 層）に同じ規則を当てる。

### 2.1 CNN（`src/sna_grow_1017_cnn.py`）

- エンジンは `src/sna_cnn_cause_1009.py` を無改変で import し、活性化の束 `BundleSnake` だけを部分クラスで置き換える（V_max と、run ごと・サイトごとの grow のフラグ。どちらも静的なテンソルで CUDA graph に乗せる）。腕名は `SNA+grow1`（宿主の `frz` の仕組みに乗せ、課題 1 末に grow を入れる）。状態の保存・再開・run ごとの切り出しに V_max とフラグを含める。
- 束: slots = {SNA, SNA+grow1} × seed 10–19（R = 20）、30 課題 × 400 epoch、チェックポイントは宿主の CKPT_TASKS。出力 `results/sna_grow_1017/cnn/`（per_task.csv・curves.npy・provenance・ckpt は git 外）。

### 2.2 MLP（`src/sna_grow_1017_mlp.py`）

- `src/controls_cifar5p1_1008.py` の `SnakeEngine`（宿主 `cifar5p1_mlp_0920.run()` の手順）に、モード `grow` を足す（課題 1 末に V_max ← V とフラグを立て、捕捉し直す。EMA は止めない）。腕 `SNA_grow`、seed 0–9、std、Adam 1e−4、30 課題、fresh 対照つき（宿主どおり）。参照（SNA・SNA_fix0・SNA_fixT1）は `results/controls_cifar5p1_1008/c2/` の記録を使う（同じエンジン・決定的。§3 の K-mlp-rep で bit 再現を確かめる）。出力 `results/sna_grow_1017/mlp/`。

## 3. 検査（本走の前に全部通す）

- **K-cnn-off**: 束 {SNA, SNA+grow1} × seed 10 で課題 1 を回すと、両 slot の重み・m・v・V が課題 1 末まで bit で一致する（課題 1 は SNA と同一のはず）。さらに SNA slot の課題 1 末の重みが、宿主エンジンの同じ構成（`E.Bundle` の素の SNA、同じ slots）と bit 一致する。
- **K-cnn-grow**: 合成の V 列（伸びと縮みを含む）で、grow 後の α が単調非増加で、各時点で clip(c/√max V) に一致する（相対 1e−7）。課題 1 末に V_max が V に初期化される。
- **K-cnn-graph**: grow が入った状態から 2 epoch、CUDA graph と eager で重み・V・V_max が bit 一致。
- **K-cnn-resume**: 課題 2 の途中で state を保存・読み直して続けた結果が、通しで回したものと bit 一致（V_max とフラグが state に入っていること）。
- **K-mlp-rep**: モード `host` の SNA を新しいコードで 2 課題回すと、記録 `results/controls_cifar5p1_1008/c2/` の SNA の per_task 行が bit で一致する。
- **K-mlp-grow / K-mlp-graph**: CNN と同じ内容を MLP で。
- 閾値は bit 一致か、相対 1e−7（演算順の違いが入りうるものだけ）。

## 4. 量と判定（帯は §0 の雑音から）

### 4.1 Q1（CNN）

- 主量 D = W21(SNA+grow1) − W21(SNA)、W21 は課題 21–30 の online の平均。対応は同じ束・同じ seed。尺度は既知の差 SNAc3 − SNA = 0.0073（束 A、固定値として登録）。
- 雑音: 対応差の seed 間 SD 約 0.0037 → 10 seed 平均の SE 約 0.0012。0.75 × 0.0073 = 0.0055 は 0 から約 4.6 SE、0.25 × 0.0073 = 0.0018 は 0.0073 から約 4.5 SE。
- **GROW_RESCUES**: D ≥ 0.0055 かつ正の seed ≥ 8/10。
- **GROW_NOT**: D ≤ 0.0018。
- **GROW_PARTIAL**: それ以外。
- 副（報告のみ）: 窓 t11–20 の同じ差、99% 到達 epoch の推移、課題末の 2αW（c2・f1・f2）と f2 の座席・傾きの平均（mob）、c1 の幅、logit の大きさ、切替直後の損失、暴走の有無（f2 の幅 > 24 を伴う切替直後の損失 > 100）。
- 副（機構、報告のみ）: SNA+grow1 の課題 20 末のチェックポイントから課題 21 を再走し、f1・f2 の傾きの平均を 3/5/10 epoch で記録する（SNA は既知の 0.16/0.30/0.38 と 0.32/0.42/0.50 を同じ手順で取り直して並べる）。

### 4.2 Q2（MLP）

- 主量 M = E_late(SNA_grow) − E_late(SNA)、後期窓 hard 21–29 の online、seed 0–9 の対応差。
- **NO_LOSS**: M ≥ −0.005 かつ 95% 区間の下端 ≥ −0.010。**LOSS**: M ≤ −0.010。**UNRESOLVED**: それ以外。
- 副（報告のみ）: SNA_grow − SNA_fixT1、SNA_grow − SNA_fix0、層ごとの α・幅・2αW（t29）。

## 5. 予測（Claude、本走の前）

- **Q1**: GROW_RESCUES 0.40・GROW_PARTIAL 0.35・GROW_NOT 0.25。点予測 D = +0.004（区間 −0.002〜+0.010）。理由: 据え置きの反実仮想では fc のゲートが覚え直しの最中に倍近く開くが、c2 は逆に閉じる。課題末には fc の 2αW が 1.2 に戻るので、谷に座ること自体は残る。
- 副の機構: SNA+grow1 の f2 の傾きの平均は 3/5/10 epoch で 0.45〜0.75（SNA 0.16〜0.38 より高い）、確率 0.7。
- 副: SNA+grow1 の課題末の f2 の 2αW の中央値は 1.0〜1.2（課題末の幅が最大より少し小さいため）、確率 0.7。暴走は起きない（入力層は SNA と同じく上限 3 に張り付くか、幅の最大に合わせた α で固定に近い）、確率 0.8。
- **Q2**: NO_LOSS 0.8。点予測 M = +0.003（SNA と SNA_fixT1 の間）。

## 6. 手順

1. 実装と検査（CNN と MLP を並行）→ 独立のコード審査（反証役）→ 直し → 本走（CNN R = 20 で約 1 時間、MLP は数分）→ `analysis/sna_grow_1017/verdict.py` が §4 の帯を当てて `results/sna_grow_1017/verdict.json`・`summary.md` → 機構の再走 → 監査（言い過ぎと転記）→ 親が読みを書く。
2. GPU は共有（ollama が載ることがある）。評価は chunk し、OOM を 2 回出したら止めて報告する。
3. 終わったら CLAUDE.md §4（git 外の生データを obsidian-research-data/sna_grow_1017/ へ、manifest、main 統合、片付け）。

## 7. 結果

（本走の後に書く）
