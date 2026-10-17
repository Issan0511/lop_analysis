# 新しい走1: RC/raw/ReLU ReDo × CHB の引継ぎ

Issaの2026-10-18 02:36:16 JSTの改訂指示。新runは `redo_chb_rlcifar_1018`。既存のGPU controllerとは別のworktree/queueで扱う。

## 旧走の扱い

旧 `redo_position_2x2_1018` は02:10 JSTに完走し、`abd5220ba3670c13afacb31cfe9974edfdf3f9cb` にCOMPLETE/MIXEDが保存されている。元予測POSITION_CARRIESは支持されなかった。旧spec・予測・全結果を維持する。新走の成績で旧予測不成立を置き換えない。

旧runの本走CLIを再実行しない。完了後でもfreshやmetadataを再計算する。STOPも完了manifestを変えるので、supersession用途には使わない。新worktreeの `results/redo_chb_rlcifar_1018/campaign_supersession.json` で役割だけを変更した。

既存 `six-gpu-1018b` は旧走1を除いた5走を明示選択しており、finish queueはCHB/CNNだけ。旧走1を再実行する予定はない。新runは現在のhardcoded controllerに存在しない。既存controller・state・service・実行中HEADを編集して割り込ませない。

## 登録条件

正本 `specs/spec_redo_chb_rlcifar_1018.md`。初回登録de8a437f、世代定義の追補09d175aa。基点main094489c3。新しいRCのReDo×CHB結果は設計時に未観測だが、CHBの既知結果と旧5+1のMIXEDを知って作った設計である。

- 固定1200枚の乱数ラベルCIFAR-10/raw、ReLU、3072–100–100–10、seed10–19、CUDA R10、Adam1e−3、200課題×30,000更新、TF32 off。
- none/redoはnative ref（C/Hなし）、CHB/bothは既存のC+H+Bパッケージ。CHBのb1ゼロ化は初回Adam後から、b2 WDλ=.1713、EMAβ=.01、重みWDなし。
- ReDoはτ=.1、絶対更新1560周期、直近4batch16の64画像。native Adam→B→EMAの後、graph外で一度介入。CHBでは実際に渡るcentered hidden featureでscoreを計算する。bias moment・EMA・global Adam時刻は保持。
- 3846イベント/腕。再初期化直後のstrict max z>0で復活を定義し、後の観測再死亡と置換打切りとevent-associated deathを分ける。
- 主窓181–200、7対応同時区間。予測CHB_POLICY_CARRIES＋REDO_SHORTFALL。主ラベルは方策比較で、純粋なdrift mediationを同定したという意味ではない。

5+1の本文用比較は `f51_context.md` とJSONに監査した。共通10seedの対応差中央値比ではReDo42.247%、H41.511%。残差の全てをrankの因果寄与とは記さない。Hは早期から約.54・後期約.55で、水準と経時劣化を区別する。

## 実行の前提と入口

cwdは `/home/issan/Projects/claude/wt/redo_chb_rlcifar_1018`。Pythonは `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`。この新worktreeだけのdata symlinkから既存cloneの正本archiveを読み、実bytes/subset/stream/hashを照合する。

実行可能状態の正本は各source-bound証跡と `README.md`。CUDA検査・独立監査・本走の実施前に、それらを通過したと記録しない。ここには後から最終確認結果を追記する。

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.redo_chb_rlcifar_1018.checks
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.redo_chb_rlcifar_1018.gpu_checks
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.redo_chb_rlcifar_1018.launch preflight
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.redo_chb_rlcifar_1018.launch run --arm all
```

各正式CLIのhelp/READMEを確認する。CPU/GPU/独立reviewの証跡は現在のimport closure、spec、archive、cohortに一致する必要がある。GPUチェックも本走も共有 `/tmp/lop_analysis_gpu.lock` の単一ownerが取得する。launcherの外側に同じlockのflockを重ねない。GPUがbusyならその状態を保持し、他のjobを止めず、空きの確認は5分以上の間隔にする。

既存の長期queueが完了してから新runの独立queueを起動する。Work環境は物理GPUの存在を確認できてもdevice nodeへアクセスできず、本走はホスト上のGPUが使えるセッションで行う。現在の環境をCUDAが使えると偽装しない。

本走はnone→redo→CHB→bothを全て新規に実行し、4×10×200行と全診断を要求する。途中の正当な保存状態だけをresumeし、source/HEAD・データ・generator・probe buffer・世代・時刻の不一致を拒否する。実行中にsource/HEADを変更しない。判定前に保存物のhash・対応を検証する。

完了後は本runのrawデータを別途退避してmanifestを保存する。既存controllerのretentionはこの新runを自動処理しない。他のworktreeを消さず、CLAUDE.mdに従って統合・片付けを進める。追加GitHub公開は本引継ぎの許可範囲に含めず、従前の自動承認拒否の制約を保持する。

## Work側の最終確認

実装source commit `5e44f9a9bc7143e4ba567da81c5aea0754e45188`。CPU全8群・指定負例35件がPASS。別担当2名がソースと実行検査を独立確認し、実ソースへのextra-B（診断前/後）・世代死亡理由の誤りも検出した。19ソースのreviewed hashが上記commitのblobと一致することを `source_commit_verification.json` に保存。

GPU検査CLIと `launch run --arm all` を実際に起動したが、Workではいずれもexit2/BLOCKED_CUDA。本走は未開始、科学labelはnull、gpu_release.jsonはall_pass=falseで全CUDA検査NOT_RUNのまま。ホスト上で本物のGPU releaseを通してから実行する。証跡は `results/redo_chb_rlcifar_1018/`、旧走1の完走記録や既存controllerは変更していない。
