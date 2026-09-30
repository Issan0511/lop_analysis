# sink_roots_r11b_0930 — spec_sink_roots_0930 §6.2 R11b（5+1 CIFAR）の実施（走る前に記録）

書いた: Claude（fork2）、2026-09-30。R11b は `specs/spec_sink_roots_0930.md`（登録 dd8e3ee）§6.2 で登録済みで、予測は同じ spec の §9.1。CIFAR-100 がこのマシンに無く保留していたが、0930 15 時に Issa が取得を許可した（「やりましょう」）。sink_roots_0930 の branch は main に統合して片付けたので（661f83c）、CLAUDE.md §2 のとおり origin/main から新しい worktree `wt/sink_roots_r11b_0930`（branch `claude/sink_roots_r11b_0930`）を切った。登録の中身（箱・量・判定・予測）は変えない。この文書は、登録の文面で決まっていなかった細部を走る前に固定するためのもの。

## 1. データ

- `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` を `proj_004_drift/data/cifar100/cifar-100-python.tar.gz` に置く（worktree からは `data` の symlink で見える。cifar5p1_mlp_0920 の `DATA_DIR`）。
- sha256 が cifar5p1_mlp_0920 の provenance の値 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` と一致しなければ走らせない。

## 2. 走

- エンジン `src/sink_roots_c5p1_0930.py`（cifar5p1_mlp_0920 の R = 1 の eager の 1 歩に probe を足したもの、CPU・1 スレッド、1 seed 1 プロセス）。
- 腕: ELU・GELU・SILU・LR・R・KKT1（spec §6.2 の 6 活性化。名前は cifar5p1_mlp_0920 の ARM_ORDER）× seed 0〜2 × 30 課題、std、lr 1e−4（0920 と同じ）。18 走。
- 置き場: `obsidian-research-data/sink_roots_r11b_0930/<腕>_s<seed>/`（arrays.npz・rows.json・provenance.json）。

## 3. 検査（本走の前）

- **C1 1 歩の一致**: 同じ腕・seed の 2 課題を、cifar5p1_mlp_0920.run()（seeds=[seed]、CPU、graph なし、fresh なし）とこのエンジンで回し、課題ごとの online_acc が一致すること（ELU と KKT1、seed 0）。probe が学習の計算に触れていないことの確認。変異対照: lr を 1.0001e−4 にすると一致しないこと。
- C1 の比較量の追加（本走の前、最初の C1 の後）: 最初の C1 では online_acc が 2 課題とも一致したが、変異対照（lr 1.0001e−4）でも online_acc が変わらず、online_acc だけでは違いを見分けられなかった。0920 の evaluate() が課題末にその課題の画像で出す連続量（zbar・zbar_min、両層）を、このエンジンの記録から再計算して比べる量に足した（相対 1e−8、0920 の印字は 10 桁）。この量で本体の一致（相対 ≤ 4e−10）と変異対照の不一致（相対 2e−5〜6e−5）の両方を確かめた（`results/sink_roots_r11b_0930/C1_step_check.json`）。
- **C2 記述**: 0920 の本走（`results/cifar5p1_mlp_0920/<腕>_std_lr0.0001/per_task.csv`、別の並び・機材）との課題ごとの online_acc の差（ビット一致は求めない）。

## 4. 読み（spec §6.2 の Q11b と §9.1 の予測の、登録の文面で決まっていなかった細部）

- 課題 2〜30 の切替を使う（課題 1 には切替が無い）。
- 押し = その課題の訓練画像での m（前活性の画像平均）の、格子点 78 と 0 の差。対象 = 切替時（格子点 0）にその課題の訓練画像で上端（z の最大）> 0 の unit。課題ごとに対象の unit の中央値を取る。
- **Q11b**: 押しの中央値が負の課題の割合（層 1・2 × 活性化 × seed）。
- 予測「第 2 層の押しは 6 活性化とも過半の課題で負」（0.6）: 活性化ごとに 3 seed の課題（3 × 29）をまとめた層 2 の割合が、6 活性化の全てで 0.5 を超えれば当たり。seed ごとの割合も並べる。
- 予測「新クラス部が打ち消し部より大きい」（0.55）: 課題ごとに、層 2 の切替時に開いた unit で |S2_new| の中央値 > |S2_cancel| の中央値となるかを数え、活性化ごとに 3 seed をまとめて過半の課題で成り立つことが 6 活性化の全てで起きれば当たり。
- 記述: 入力の蹴り（同じ重みで新課題と旧課題の画像の m の差、格子点 0）、固定の検査画像 2,000 枚での m の変位、S2_cancel・S2_new の符号、easy 課題と hard 課題の別。
