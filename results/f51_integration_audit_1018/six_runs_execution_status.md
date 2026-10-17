# 追加6走の実装・実行記録

Issaの「サブエージェントで全部実装実行」「予測は論文の主張で」の指示に基づく記録。6担当が別worktreeで実装し、別担当が相互監査した。方向予測は介入結果を観測する前の共通spec §8に固定した。確率・精度の数値予測は代筆していない。

## 固定した実装と方向予測

| 走 | run id | 実装の最終commit | 独立監査 | 方向予測 |
|---|---|---|---|---|
| 1 | `redo_position_2x2_1018` | `585b80fb633b6e67eb8795a7ca35aa77c67515e2` | 走3担当 | `POSITION_CARRIES` |
| 2 | `cap_act_box_2x2_1018` | `1352d2e67548bc5ac8f9f06fd25ed5a96b00b100` | 走5担当 | `ACTIVATION_SPLITS` |
| 3 | `chb_decompose_200_1018` | `ea66eb792564517f6bad00daae0c371006377510` | 走2担当 | `BIAS_STOP_SUFFICES` |
| 4 | `resp_step_controls_1018` | `e01cebd07a428e508c6a489a9fdaaecab865de73` | 走6担当＋root | `NOT_LR_ALL_THREE` |
| 5 | `nap_cifar5p1_1018` | `54effd4585992892e996729a0e1f8440a2e3164a` | 走4担当 | Rより改善する方向。対SNA/KKT1は`OPEN_COMPARISON` |
| 6 | `cnn_chprime_bias_1018` | `bba6291cccf7e8ed7a5f4a4b0da397e633360994` | 走1担当 | `RESCUED` |

各worktreeは `/home/issan/Projects/claude/wt/<run id>`、ブランチは `codex/<run id>`。継承元はmain上の `063b7c6085340475549981494e713a7ceea6dbbb`。後から増えていたremote mainの5commitは別実験の変更で、継承ソースとの衝突がないことを読み取り確認した。remote mainは更新していない。

## 検証の範囲

| 対象 | 実行したCPU検査 | 残る本走・CUDA検査 |
|---|---|---|
| G0 | 元F51のpilot全6群がPASS、定数・main祖先・347ソースhash照合 | CUDA main/iv/doorsと実測再現は未完了 |
| 走1 | 7群、指定変異14件の検出 | CUDA、G0 |
| 走2 | 10群、指定変異12件の検出 | CUDA、G0、較正・本走 |
| 走3 | 15群 | CUDA、200課題本走 |
| 走4 | native RM経路、並列対直列の一致、指定変異20件の検出 | RC/F51はCUDA待ち。RMの実行状態は後段に記録 |
| 走5 | 8群、指定変異24件の検出 | CUDA、G0、較正・本走 |
| 走6 | 10群、指定変異23件の検出 | CUDA graph/eager・巻き戻し検査、CNN本走 |

これらは実装検査であり、論文の予測を支持する実験結果には数えない。独立監査の承認範囲もソース・CPU検査までで、CUDA検証の代用にはしていない。

全runnerは登録したseed・課題数・演算経路を保つ。CUDA本走を実際に起動した結果、`torch.cuda.is_available() == False` により `BLOCKED_CUDA` を保存して停止した。CPUへの条件変更、seed削減、精度の代入は行っていない。F51のG0も `all_pass=false` のまま保存した。

## 重要な実装上の固定

- 走1はt28末で一回介入、t29を採点する。閾値0.1のReDoは完全死亡以外も選ぶので、「死亡unitだけ」の効果とは主張しない。
- 走2は既存2セルも現環境で再走する構成。`lr_only`は独立した較正seed100–109の実変位RMSから決め、倍率を1でclipしない。
- 走3はCH/CHBの再利用を無条件には認めず、対照2腕も再走する。初期bias凍結、重みWD、bias WDを分ける。
- 走4は勾配比と実際のfloat32パラメータ変位比を分ける。較正後に倍率を固定し、評価seedからは倍率を作らない。ゼロ分母は未定義として停止する。
- 走5はNaP原典2407.01800に照合。隠れ層のaffine LayerNorm、全3重み行列の初期ノルムへの射影、gamma/betaの連結射影を行う。対SNA/KKT1の順位予測は置かない。
- 走6は各チャネルの現在幅sを測り、biasを `[-0.2s,+0.2s]` に毎更新後射影する。convの幅は位置ごとの画像方向分散を位置平均して平方根を取る。狭いチャネルほど小さい上限になる。過去P183とのTF32条件差があるため、CHf対照も新しく走らせる。
- 全走で欠測・発散・検査違反を科学ラベルへ変換しない。保存済み完了証跡・ソース・データ・seed/腕対応・出力hashを照合してから判定する。

## RM本走

登録通り、較正100–109、評価0–9、各seed CPU1thread、6000更新/課題。独立seedを4プロセスで実行する。ソースとHEADは `e01cebd07a428e508c6a489a9fdaaecab865de73` で固定した。

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/resp_step_controls_1018/run_rm.py --out results/resp_step_controls_1018/RM --workers 4
```

完走、exit code 0。較正10seed×2腕×2課題=40行、評価10seed×4腕×2課題=80行を保存。接頭部を含め登録通り216万更新を実行した。保存データ・ソース・出力hash・seed/腕/課題の完全性と対応を検証し、t分布の独立した閉形式計算でも全5対応差の区間と判定が一致した。結果は `rm_execution_status.json` と `rm_independent_result_check.json` に保存した。

| t11の腕 | 10seed平均online精度 |
|---|---|
| 無介入N | 13.6725% |
| 場の交換R | 89.6584% |
| 勾配比lr対照LR_G | 10.0369% |
| 実変位比lr対照LR_U | 11.3429% |

結果保存commit: `d3d235d7e0dd3d3f34db75f23273d539c047728a`。独立した走6担当も、320ファイルのhash・自然N経路一致・実lr・全対応区間を追加照合し承認した。約2.34GB/5568ファイルの生データはworktree内に保持している。

**RMの登録判定は`NOT_LR`。** R−Nは+75.9859pt、m=15の同時区間は[73.9762,77.9957]pt。LR_G−Rは−79.6216pt（[−81.3140,−77.9291]）、LR_U−Rは−78.3155pt（[−79.3910,−77.2400]）。否定できたのは登録した二つの一定層別倍率対照までで、任意のlr調整全般ではない。RC/F51は未実行なので、3箱全体の科学判定は`INCOMPLETE`、全体labelはnullのままである。

## 再開用の入口

実験ごとの `analysis/<run id>/README.md` に本物の検査・較正・本走・集計CLIを記載した。CUDAを接続したら、G0の元F51検査を再実行して実測証跡を作り、各差分のCUDA検査を満たしてから本走へ進む。G0の偽のPASSフラグを作らない。

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/f51_integration_audit_1018/audit.py --repo /home/issan/Projects/claude/proj_004_drift --out /tmp/issa_six_specs_1018/g0_independent --cuda-validation
```

監査結果の正本は `integration_audit.json`。現在の共有先は `/tmp/issa_six_specs_1018/integration_audit.json`、保存版はこの結果フォルダにある。本走を止めた理由も各runの結果フォルダに保存した。利用可能なGPU実行先はこの環境・既存設定からは見つからなかった。

## 公開と保存

全6specの事前登録remote commitは `remote_registrations.json` に保存した。追加実装は走3だけremote commit `b068d4ea4a7c93863d79676550f09217eae66238` に到達した。その後、走6のGitHub commit作成が自動承認レビューに拒否された。理由は「実装・実行の指示はあるが、この外部へのコード・結果・履歴の送信は明示的に許可されていない」。以降の公開操作・再試行・迂回はしていない。

実際のlocal/remote対応と未公開状態は `remote_implementations.json` に記録した。ローカルcommitとworktreeは保持している。CUDA本走は未完了であり、main統合やworktree削除は行っていない。生ログを持つworktreeは退避・完了前に消さない。
