# WorkからローカルCodexへのGPU本走引継ぎ

更新: 2026-10-18 JST。依頼者Issaは全6走の実装・実行を指示済み。方向予測も論文の主張から固定済み。既存実装と実験条件を引き継いで残りを実行する。

**2026-10-18 10:00 JST追加: 走2のcap-matched fresh切り分け。** Issa「Capのfreshが悪かったわけではないん？」「切り分けて」を受け、元のfreshはcapなしだった点を確認し、同じ走2worktreeに事後追試を追加した。原spec・訓練ソース・原verdict・原rawは不変。追試spec `spec_cap_act_box_fresh_followup_1018.md` を新しいfresh-cap結果取得前の `7893b589` で固定、実装・原入力manifest・CPU検証・幅の事後集計は `af7bb74f`。F51/ELU・ReLU、seed 0–9、元の後期hard窓21/23/25/27/29を各々P0/Adam初期状態から学習。capには元のtask1終端半径を使い、最初のAdam更新後から適用する。fresh cap差とcontinual cap差の差を10seedで比較する。既存結果を知った後の追試であり独立な事前登録再現とは数えない。

`cap-fresh-followup-1018.service` が共有GPU lock待ちで、現CNN腕の終了後にCUDA graph/eager検証→追試20条件（各10seed）→原task29 fresh完全一致検査→集計→hash照合backupを行う。実装入口は走2worktreeの `analysis/cap_act_box_fresh_followup_1018/run.py`、進行は `results/cap_act_box_2x2_1018/posthoc_fresh/status.json`、ログは同じ場所の `run.log`。部分出力の上書き・自動再走はしない。新raw backup先は `obsidian-research-data/cap_act_box_2x2_1018/posthoc_fresh_1018/`。走3のpauseは維持。CNNとは同時学習しない。

Issaの追加指摘「5+1のELUは幅が大幅に増えていないのでは」に対し、原unit診断のhashを照合して `posthoc_fresh/width_summary.md` と全課題・seed別CSVに保存した。capなしELUの前活性幅の平均はF51 t1→29で第1層5.74→10.25、第2層3.20→5.46（t21→29はほぼ横ばい）。RC t1→50では16.98→239.04、12.91→501.45。第2層の平均unit相対位置はF51 −0.43、RC −1.96。F51は幅増加も負側への沈みも小さいという記述と整合する。箱間の学習率・更新数・課題分布が違うため、幅だけが因果的理由とは断定しない。

**2026-10-18 朝の走6回復:** 走2は較正・本走・analyzeを完了。その後06:11開始の走6GPU検査は、検査器の保存先引数 `out` をforward出力tupleで上書きしてJSON保存時にTypeErrorとなり、本走に進まなかった。09:35以降に調査し、局所変数の2か所だけを `activations` へ変更した（走6 commit `258012d6`）。訓練コード・閾値・条件は同一。元の独立レビューと失敗前証跡を `results/cnn_chprime_bias_1018/host_save_repair_1018/` に保持し、全他ファイルのhash一致と2識別子だけの差分を検証。元レビューを継承する範囲とrootによる保存処理差分監査を明記し、新しい独立担当がレビューしたとは記録していない。

CPU全10群と実CUDA全5項目（R10宿主byte一致、capture巻戻し、graph/eager byte一致、追加forward誤差、レビュー証跡）を再実行してPASS、結果は走6 commit `fc44083b`。`six-gpu-cnn-recovery-1018.service` が検査hashを確認して **CHf→CHf_B→verdictだけ** を実行中。`host_continuation/state_cnn_recovery.json`、ログ `logs/cnn_chprime_bias_1018_*` が現行。CHfのprovenanceがRUNNINGに進んだことを確認した。走3の停止は維持し、他の完了走を再訓練しない。

以前のretainerは全走をローカル保存済み（走3のpause snapshot含む）。新しい `six-gpu-retain-cnn-1018.service` はCNN回復controllerのlock解放を待ち、走6だけを `obsidian-research-data/cnn_chprime_bias_1018/host_gpu_recovery_1018/` にhash照合付き保存してcompact結果をlocal commitする。旧 `host_gpu_1018/` の停止時backupは上書きしない。進行は `retention_cnn_recovery.json`。GitHubへの追加公開はしていない。

**最新指示（2026-10-18 02:46 JST）: Issa「一旦走３は飛ばしましょう」。走3は一時停止、明示の再開指示まで自動再開しない。** CH task20完了checkpointで正式STOPし、元controllerも `STOPPED_AT_CHECKPOINT` で終了。checkpoint・source・HEAD・manifestの整合をCPUで確認（`host_continuation/run3_pause_checkpoint.json`）。CHB/追加3腕は未実行。走3の `results/chb_decompose_200_1018/STOP` とG0の `host_continuation/run_policy.json` を保持する。finish queueは走2→（走3をskip）→走6。走3は未完了のままで科学判定は出さない。

再開用HEAD `8f82fc9fff3e3d521d4253792ed8a58228b5a835` を保持するため、retainerは走3のrawを `obsidian-research-data/chb_decompose_200_1018/paused_20261018T0246JST/` にhash照合付きコピーするだけで、そのworktreeでは結果commitも行わない。元データも削除しない。再開が指示された場合は、pause policyとSTOPの解除、同じHEAD/checks/manifestでの `run --resume` が必要（CPU checks再生成も避ける）。以下の「走3を自動resume」の記述はこの指示以前の履歴。

## ホスト再開の追記（2026-10-18 02:10 JST以降）

この追記以降の現行進捗は `results/f51_integration_audit_1018/host_continuation/` の `state_after_repairs.json`（走4→3→6）、`state_nap_readout.json`（走5の比較対照・判定）、`state_cap_retry.json`（走2）が実行記録。初回サービスの履歴は `state.json` に保持。GPUは全サービス共通のflockで一つずつ利用する。以下の元の引継ぎ本文は停止時点の履歴として保持する。

- ホストのRTX 5060 Tiとtorch 2.13.0+cu130でCUDA利用を実測した。
- 初回G0は `gpu_host/` に保持。旧cu126結果との `eff_rank_l2` 111件だけが一致せず停止した。精度など他の全列とfreshは一致。既存 `spec_resp_cifar5p1_1007.md` §1.2/§7にも同じ環境差の規定がある。
- `0eff0529` の旧実装をクリーンな一時worktreeで現環境再実行し、無改変の旧S-nochangeにその独立参照を渡した。現環境同士の全列byte一致（eff_rankを含む）と変異検出を要求したまま、main/iv/doorsすべてPASS。許容値を広げていない。一時worktreeは削除済みで、再実行結果とソースhashは保存済み。
- 現行G0正本: `results/f51_integration_audit_1018/gpu_host_same_runtime/integration_audit.json`。共有 `/tmp/issa_six_specs_1018/integration_audit.json` に同じbytesを配置。
- `analysis/f51_integration_audit_1018/continue_gpu.py` が正式CLIを直列実行する。user service `six-gpu-1018.service` で開始。段階ごとのログと失敗は `host_continuation/` へ保存。一走の失敗はその依存段階を停止し、独立した後続走は継続する。部分出力の自動上書き・自動resumeはしない。
- 走1完走、登録判定MIXED。独立したdf9の閉形式による7同時区間の再計算も一致。結果local commit `abd5220b`。55ファイル、563,928,622 bytesを `~/Projects/obsidian-research-data/redo_position_2x2_1018/` に追加バックアップし、hashを照合した。worktreeの原データも保持。
- 走3のGPU競合検知はNVIDIAが列挙するデスクトップC+Gプロセスを訓練と誤認していた。既存宿主と同じPython訓練プロセスの検知へ揃えた。共有flockは維持。自己PID/デスクトップ/別Pythonの検査とCPU全15群を通し、local commit `8f82fc9f` に固定。
- 65件の以前の検査・停止証跡を `host_continuation/prior_evidence/` に退避してhashを保存。RMの既存本走は再実行しない。
- GitHubへの追加公開・実験worktree削除は行っていない。NaPの新規対照は既存資格検査がmain上の不変commitを要求するため、完走した対照のcommitだけローカルmainへfast-forwardする（02:23以降）。remote mainは更新しない。

02:18 JST追記: 初回サービスは走5の一時的な未固定ソース判定、走2のnautilus/ptyxis誤検知、走4の検査fixture不一致で各走を停止した。走5はソース無変更で清浄状態とCUDA検査を再確認。走2は自走の競合検知だけを修正（宿主・G0対象は無変更）、走4は比較する両側のラベル乱数を進める検査fixture修正を行い、CPU/CUDA全検査PASS。走3はCHのtask4完了checkpointを保存して停止し、同一HEAD/manifestの `run --resume` を後段に予約した。CPU checksを再生成するとtimestamp付きhashが変わるため、resume時は保存済みの検査証跡をそのまま検証させる。

現行サービスは **`six-gpu-1018b.service`**。走5→2→4→3(resume)→6。較正後NaPは実際にselected.jsonと較正証跡のlocal commitを完了してから評価へ移った。旧R/SNA/KKT1がソース差だけで再利用不適格なら、同じ全30課題/10seed条件の現環境対照を新規保存・commit・資格確認する。走4のRMは修正前の保存commit `d3d235d7` の無改変検証器でhash/admission/schedule/全結果を検証する `combined_resp_verdict.py` を用い、RC/F51は現行ソースで検証してから合算する。RM原データ・元provenanceを書き換えない。このRM独立経路は実行してPASS確認済み。

監視は通常5分以上の間隔。短い検査がその前に終わる見込みの場合と障害対応だけ例外とする。常駐サービスのログは `journalctl --user -u six-gpu-1018b.service`、段階はstate_after_repairs.jsonで確認する。停止が必要な場合は走のSTOP/checkpoint仕様を先に確認し、途中腕の上書き禁止を守る。controllerの `host_continuation/STOP` は次のコマンド開始を止める。走3のSTOPによるexit75ではcontrollerも後続を開始せず停止する。

02:24 JST追記: NaPの較正・評価は各30課題×10seed完走。比較対照は旧ソースとの差で資格が通らないため再走し、Rは新規対照commit `a00f342b` をローカルmainへfast-forwardして無改変の資格検査を通した。SNA/KKT1も同じ正式CLI・10seed/30課題で走らせる。`six-gpu-nap-readout-1018.service` は比較対照以降だけを実行し、NaPの較正・評価を再走しない。NaPの当初の未固定ソース判定と同時刻に、別セッションが `761530e3`（02:10:59 JST）でIssaのKKT1/SNA対NaPの予測追記を固定していた。追記は保持し、開始時のソース・specを証跡に残した。

走2はデスクトップのNUL区切りargvとChromiumが書き換えた単一文字列のprocess titleの両方を処理するよう再修正。実際の4形式をテストし、競合する訓練は拒否する。`six-gpu-cap-retry-1018.service` はCUDA検査から再実行する。過去2回はGPU競合検知で止まり、較正腕は作成されていない。明示された回復サービスは共有flockで待機し、走4などとGPUを同時使用しない。

`six-gpu-finish-1018.service` は元controllerと走2/5の回復queueが全部終わるまでlockで待機する。途中で競合によるnonblocking admission拒否があっても、走3は正当なcheckpointからresumeし、走6は未完了腕だけを正式CLIで続行する。完了済み走は再訓練せず正式reportを検証する。記録は `completion_state.json`。

`six-gpu-retain-all-1018.service` は上記finishも含む全controllerのlock解放を待ち、完走/停止が確認できてから原データを `obsidian-research-data/<run>/host_gpu_1018/` へコピーしてsha256照合、backup_manifest作成、10MiB未満のcompact結果をlocal commitする。元データとworktreeは保持。無確認のRUNNING状態・容量不足・異なる既存backup・他のstaged変更は停止して記録する。監視のpollは行わない。保存状況は `retention_status.json`。旧retain/retain-finalサービスは停止済みで、このretain-allが現行。

走3の初期CH実測は約40秒/課題で、同程度なら5腕×200課題はおよそ11時間（実測に基づく概算。CNN等の時間は別）。長時間実行が続くため、サービスが稼働している間は同じCLIを重ねて起動しない。

02:42 JST以降の追記（GPUデータ常駐の確認）: 走3は既に各seedの固定1200枚をCUDAへ一度転送し、10seed分の `X[10,1200,3072]`（float32、147,456,000 bytes）を全課題で保持する。各更新の画像抽出もGPU内で行い、10seedは束ねて計算、CUDA Graphも有効。CPUからのラベル転送は課題ごと、バッチ順序の転送はepochごと。CH task5–12の訓練部分は約1.35ms/更新、3万更新/課題。5腕×200課題で訓練だけ約11.25時間となるが、他腕の速度は未計測で境界診断・保存は別。GPU常駐をこれから追加する余地ではない。コード変更・追加GPUベンチマークは行わず、`host_continuation/gpu_residency_review.json` に根拠を保存した。

同時の進行確認で走4のRC/F51と合算verdict、走5の全比較対照とverdictはCLI成功を確認。走3は同一HEADから再開してCH task12まで進行していた。走2の回復queueは較正開始前のGPU入場で停止: 先行RCプロセスが共有lockを解放した後、プロセス終了前のCUDA contextを検出したため（PID915840）。科学データは未作成、ソース同一を確認。GPU競合拒否を緩めず、finish queueに走2の正式CLI全段階を追加し、先行プロセスが終了してから実行する。待機中だったretain→finishの順にサービスだけを止め、更新後finish→retainの順で再起動。動いている走3と元controllerは止めない。失敗と回復根拠は `cap_process_exit_recovery.json`、旧待機状態は `.before_cap_exit_recovery` に保存。finishとretainのサービス名・進行ファイル名は同じ。

## いまの状態

全6走を別サブエージェントが実装し、別担当が相互監査した。CPU検査と指定変異の検出は合格。CUDA本走の起動は試みたが、Workの実行環境からデバイスを使えず停止した。

**物理GPUは存在する。** ホスト名white-san、ASRock B760、PCI 0000:01:00.0のNVIDIA GPUにnvidiaドライバがbind済み、カーネルモジュール580.178.04も確認した。一方、WorkではPID1がcodexで、/dev/nvidia0・/dev/nvidiactlが見えず、nvidia-smiはdriver通信エラー、torch.cuda.is_available()はFalseだった。過去の「GPU不在」はこの隔離実行環境からのアクセス不能を指す。物理GPUがないという意味ではない。

作業は実際に `/home/issan/Projects/claude` 以下へ保存済み。唯一のclone `proj_004_drift` はmainのまま。既存worktreeを使う。最初にこの上位ディレクトリのCLAUDE.mdと適用されるAGENTS.mdを読む。

| 走 | worktree末尾（共通prefix: `/home/issan/Projects/claude/wt/`） | 保存HEAD | 事前予測 |
|---|---|---|---|
| G0 | f51_integration_audit_1018 | 本書を含むcodex/f51_integration_audit_1018 | 監査、科学予測なし |
| 1 | redo_position_2x2_1018 | 585b80fb633b6e67eb8795a7ca35aa77c67515e2 | POSITION_CARRIES |
| 2 | cap_act_box_2x2_1018 | 1352d2e67548bc5ac8f9f06fd25ed5a96b00b100 | ACTIVATION_SPLITS |
| 3 | chb_decompose_200_1018 | ea66eb792564517f6bad00daae0c371006377510 | BIAS_STOP_SUFFICES |
| 4 | resp_step_controls_1018 | d3d235d7e0dd3d3f34db75f23273d539c047728a | NOT_LR_ALL_THREE |
| 5 | nap_cifar5p1_1018 | 761530e3eb6fd00f0a2c85f6f31aed2f3a0d5818（実装54effd45と同一） | Rより改善する方向。追補でKKT1 > NaP、SNA > NaP |
| 6 | cnn_chprime_bias_1018 | bba6291cccf7e8ed7a5f4a4b0da397e633360994 | RESCUED |

各走のspecはそのworktreeの `specs/spec_<run id>.md` と `specs/spec_six_runs_common_1018.md`。CLIと中断時の扱いは `analysis/<run id>/README.md` を正本にする。

## 最初にホスト上でGPUを確認する

ユーザーが通常のホスト端末から開始したセッションで確認する。下記は読み取り確認のみ。

```bash
nvidia-smi
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

GPUが使えることを実測してから進む。使用Pythonは `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`（torch2.13.0+cu130）。環境差はprovenanceに残す。既存のGPU競合確認と共有lockを守り、**CUDA検査も本走も直列**にする。特に走2の検査自体はlockを取得しない。

## G0を実測PASSにする

F51を使う走1/2/4/5には独立G0が必要。過去のBLOCKED証跡は保持し、新しい保存先へ本物の監査を実行する。cwdはG0 worktree。

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/f51_integration_audit_1018/audit.py --repo /home/issan/Projects/claude/proj_004_drift --out results/f51_integration_audit_1018/gpu_host --cuda-validation
```

これは元F51のCPU pilotとCUDA main/iv/doorsを実行する。exit0かつall_pass/host_reproduction_pass/mutation_checks_passが本物の実測でtrueであることを確認する。現在の保存済みG0はCUDA未実施なのでfalse。フラグを手で書き換えない。

新しい監査JSONの絶対パスは `/home/issan/Projects/claude/wt/f51_integration_audit_1018/results/f51_integration_audit_1018/gpu_host/integration_audit.json`。引数対応runnerにはこのパスを渡す。PASS確認後に、既存runnerの既定参照先 `/tmp/issa_six_specs_1018/integration_audit.json` にも同じ正本のbytesを配置する。G0作成者と別の担当に監査証跡を確認させる。

## 残りの実行

推奨順は走1→5→2→4のRC/F51→3→6。各cwdでREADMEの正式CLIを使い、必要なGPU検査を先に通す。seed数・課題数・lr・演算形状を短縮しない。CPUへの本走移植は登録条件ではない。

- **走1**: `python -m src.redo_position_2x2_1018 run --integration-audit <上記G0絶対パス>`。共通接頭部と4分岐、freshを実行する。
- **走5**: CUDA S検査→phase calib→select→selected.jsonと較正証跡をcommit→phase eval→R/SNA/KKT1の再利用資格確認→verdict。NaP順位予測は02:07:26/02:08:13JSTのIssa指示に基づく追補を採用する（KKT1 > NaP、SNA > NaP）。`specs/amendments/spec_nap_cifar5p1_1018_prediction_20261018T020813JST.md` とJSONを読み、原OPEN履歴と追補commit/hashを最終報告に併記する。
- **走2**: `python -m analysis.cap_act_box_2x2_1018.checks` → `python -m src.cap_act_box_2x2_1018 preflight --integration-audit <G0>` → `calibrate --integration-audit <G0>` → `run --integration-audit <G0>` → `analyze`。既存セルも再走する構成。部分腕は自動resumeできず、上書き拒否を守って中断証跡を保存する。
- **走4**: `analysis/resp_step_controls_1018/checks.py --cuda` の後、RC/F51それぞれcalibrate全10seed→固定schedule→evaluate全10seed。F51にはG0を渡す。完了したRMを合わせてverdict。RMS勾配比と物理的なAdam実変位比の両対照を実行する。
- **走3**: `python -m analysis.chb_decompose_200_1018.launch check` → `plan` → `run`。3新腕に加えCH/CHBも現環境で再走。seed10–19、200課題。resumeはこのrunの正当な保存状態だけから行う。
- **走6**: CPU checks→gpu_checks.py→新CHf対照→CHf_B→verdict。チャネル別の現在幅sを用いるbias射影±0.2sを固定。旧P183はTF32差があるため対照再利用を不許可としている。原データの容量確認を守る。

ここでのpythonはすべて冒頭の既存venvの絶対パス。細部は各READMEを読む。GPU検査が実装不具合を見つけたら修正し、別担当による監査と更新検査を通してから再開する。仮説が外れた結果をCHECK_FAILEDへ変更しない。

## 完了済みRMは保持する

走4のRMはソースe01cebd07a428e508c6a489a9fdaaecab865de73で較正100–109・評価0–9を完走し、結果保存commitはd3d235d7。登録通り216万更新。RM単独判定NOT_LR、3箱全体はINCOMPLETE。

| t11腕 | 10seed平均online |
|---|---:|
| N | 13.6725% |
| R | 89.6584% |
| LR_G | 10.0369% |
| LR_U | 11.3429% |

rootと走6担当が独立に検証済み。約2.34GB/5568ファイルを走4worktreeに保持し、`results/resp_step_controls_1018/rm_raw_retention_manifest.json` にhashを保存した。厳密なresume identityは実行当時のHEADにも依存するので、結果保存後のHEADでRM本走CLIを再実行しない。完成済み保存物のreadoutは現HEADでも通ることを確認済み。

全体の実行記録: `results/f51_integration_audit_1018/six_runs_execution_status.md`。原データを退避し、統合済みであることを確認するまではworktreeを削除しない。

## GitHubの状態

6specのremote事前登録は保存済み。走3の追加実装だけremote b068d4ea4a7c93863d79676550f09217eae66238へ公開された。その後の走6commit作成は自動承認レビューが「実装・実行の指示には、この外部へのコード・結果・履歴の送信の明示許可がない」と拒否した。以降の公開は停止している。今回の引継ぎも実験続行であり、追加公開の許可として扱わない。

local/remote対応は `results/f51_integration_audit_1018/remote_implementations.json`。remote mainにはローカルmainより5commit新しい別実験の変更がある。既存の履歴と他セッションの作業を保持する。実験完了後のmain統合・退避・片付けはCLAUDE.mdと実際の公開許可に従って進める。
