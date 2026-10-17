# WorkからローカルCodexへのGPU本走引継ぎ

更新: 2026-10-18 JST。依頼者Issaは全6走の実装・実行を指示済み。方向予測も論文の主張から固定済み。既存実装と実験条件を引き継いで残りを実行する。

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
| 5 | nap_cifar5p1_1018 | 54effd4585992892e996729a0e1f8440a2e3164a | Rより改善する方向。対SNA/KKT1はOPEN_COMPARISON |
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
- **走5**: CUDA S検査→phase calib→select→selected.jsonと較正証跡をcommit→phase eval→R/SNA/KKT1の再利用資格確認→verdict。NaPの順位予測は新しく作らない。
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
