# vit_long_1024 — LN を外した Joudaki ViT を、課題数と課題あたりの更新数の両方で延ばす（喪失は到達時間で測り、課題末の重みを残す）

状態: **登録前の草案。** 腕・予算・判定・閾値は Issa の確認の前で、まだ動かしてよい。走らせるのは Issa の go の後で、走る前に「登録」の commit で固定する。データ（Tiny ImageNet）の再取得も Issa の go の後に行う（§10.2）。
作成: 2026-10-24 / 起案: Claude（Opus 5.5、Fable の依頼）/ 決定: Issa（1024。課題数と課題あたりの更新数を両方増やす、指標を到達時間に替える、課題末の重みを保存する）。
親 spec: `specs/spec_joudaki_vit_battle_0919.md`（箱・学習の設定・追補 4 の高速化）、`specs/spec_randlabel_joudaki_vit_0919.md`（乱数ラベルの課題、§2.1 の枚数の較正）、`specs/spec_factorial_joudaki_vit_0919.md`（§5 の smoke、§6 の N セル）、`specs/spec_chargpt_loadsign_1023.md`（J5 の恒等分解）、`specs/spec_kappa_cnn5p1_1022.md`（κ・cos²・Q の定義）。
コード（読むだけ）: `analysis/joudaki_vit_battle_0919/`（model.py・data.py・run.py・launch.py・report.py。sha256 で凍結、upstream/ に原典を固定）、`analysis/randlabel_control_joudaki_vit_0919.py`、`analysis/factorial_joudaki_vit_0919.py`、`analysis/fresh_control_joudaki_vit_0919.py`。
ブランチ: `claude/vit_long_1024`（worktree `wt/vit_long_1024`、origin/main 95b989d4 から切った）。

## 0. 一行

原典の ViT から LN を外すと喪失は出たが（factorial の N セル）、40 課題 × 500 更新では 10 seed 中 4 seed が崩れ切らず、課題平均の online 正解率は天井と床の間で差が縮んだ。この spec は同じ N セル（腕 A: LN なし・ReLU・乱数ラベル 250 枚）を課題数と課題あたりの更新数の両方で延ばし（推奨は 100 課題 × 1,500 更新・seed 0〜4）、喪失を「訓練集合の暗記率が 0.9 を超えるまでの更新数 T90」で測る。LN ありの同じ設定（腕 B）を盾の対照に置き、課題末の重みと Adam を保存して、走った後で段①（装填の押し）の帳簿を当てる。

## 1. 問い

（次の commit で書く）

## 2. 既知（登録前に見たもの。独立な予測の成功として数えない）

（次の commit で書く）

## 3. 予測

（次の commit で書く）

## 4. 腕と箱

（次の commit で書く）

## 5. 却下した代案

（次の commit で書く）

## 6. 測る量

（次の commit で書く）

## 7. 判定（走る前に固定）

（次の commit で書く）

## 8. 検査（走る前に通す）

（次の commit で書く）

## 9. 言わないこと

（次の commit で書く）

## 10. 置き場・手順・計算量

（次の commit で書く）

## 11. 本文への反映（走る前に決める）

（次の commit で書く）
