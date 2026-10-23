# Joudaki ViT / Tiny ImageNet 13活性化実験 — 別マシンへの引き継ぎ

更新: 2026-09-19 JST。ユーザー依頼により元マシンでは停止済み。**新マシンで全13活性化・seed0–9を最初から実行する。旧checkpointからは再開しない。** 旧版の途中再開手順は本版で置き換えた。元マシンは停止を維持する。

## 1. 最初に読む要約

- GitHub: https://github.com/Issan0511/lop_analysis
- ブランチ: `codex/joudaki_vit_battle_0919`。このMDを含む最新commitを取得する。
- 凍結学習実装のcommit: `677fc85`。並列測定追加: `89fedb1`。以後の引き継ぎ追加でも学習Pythonソースは変更していない。
- 旧試行は18/130 run・751/5200タスクまで保存済み（KKA23 / seed1 / task31終了）。**これは保管用の途中データ。新しい本走はSNA / seed0 / task1から始め、130 runすべて新マシンで取り直す。**
- seed0は13腕完了。seed1はSNA、KKA、R、LR、KKT1が完了し、KKA23が途中。その他のseedは未実施。
- 元サービス `joudaki-vit-battle-0919.service` は正常終了。rawルートの`STOP`を残してあり、自動再開しない。
- コード・仕様・高速化を再利用する。**旧raw約49GiBは元マシンに保管し、新本走のために転送する必要はない。** 移すのはデータセットのみ。旧run記録を新しいraw/runsへコピーしない。
- 本走の勝敗・順位は未集計。新マシンで完了した130 runだけで集計し、旧途中データは混ぜない。GPU/ソフトウェア環境を全条件で揃えるためのやり直しであり、途中の精度による条件選別は行わない。

仕様: [spec_joudaki_vit_battle_0919.md](spec_joudaki_vit_battle_0919.md)。実装説明: [README](../analysis/joudaki_vit_battle_0919/README.md)。停止状態・環境・転送manifest・run記録: [handoff_0919](../results/joudaki_vit_battle_0919/handoff_0919/)。

## 2. 固定条件と変更してはいけないもの

Tiny ImageNet、40個の非重複5クラス課題、各500更新、batch128、seed0–9。ViT patch8 / dim384 / depth6 / heads6 / FFN1536。Adam lr1e-4、weight decayなし。float32、TF32無効、`torch.compile(fullgraph=True)` + fused Adam。

13腕: SNA, KKA, R, LR, KKT1, KKA23, SL, RSL, LK001, LK03, ELU, SILU, GELU。SNA系のEMA更新はoptimizerの後。dropoutはATen乱数列、RSLは専用乱数列。各taskでheadをresetし、optimizer momentsは継続。

`analysis/joudaki_vit_battle_0919/` 以下の**全Pythonファイル**と `src/rlcifar_mlp_battle_0918.py`, `src/pmnist_0905.py`, `src/pmnist_rlcifar_0907.py` はSHA256で凍結している。ここにPythonファイルを追加するだけでも再開検査が失敗する。移行のためのパス修正や機能追加は不要。補助スクリプトが必要なら凍結ディレクトリの外に置く。

- `REPO`はソース位置から自動決定する。
- `RAW`は移行先の `$HOME/Projects/obsidian-research-data/joudaki_vit_battle_0919`。
- `DATA`は移行先の `$HOME/Projects/obsidian-research-data/datasets/tiny-imagenet-200`。
- `done.json`があるrunはlauncherが飛ばし、`checkpoint.pt`があれば途中再開する。そのため**新マシンのraw/runsは空で開始する**。元マシンのcheckpointは保管し、削除しない。
- 別GPU/別CUDA/PyTorchをまたぐbit一致は保証していないため、旧試行と新本走は分離する。新本走の全活性化・全seedで同一GPU/ソフトウェア環境を維持し、新環境のprovenanceを新規作成させる。

## 3. データセットだけ転送する

元マシンの旧rawは `/home/issan/Projects/obsidian-research-data/joudaki_vit_battle_0919/`（約49GiB）。STOPを残したまま保管する。新本走には使わない。Git内の`handoff_0919/run_records`と`transfer_manifest.json`も旧試行の保管記録であり、新本走の入力ではない。

転送先はまだ指定されておらず、転送未実施。元マシンで`DEST`を実際のSSH接続先に置き換えて実行:

```bash
DEST='user@destination-host'
ssh "$DEST" 'mkdir -p "$HOME/Projects/obsidian-research-data/datasets"'
rsync -a --partial --info=progress2 /home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200/ "$DEST:Projects/obsidian-research-data/datasets/tiny-imagenet-200/"
rsync -a --partial --info=progress2 /home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200.zip "$DEST:Projects/obsidian-research-data/datasets/"
```

データは展開済み約481MiB、zip約237MiB。`.venv`とコンパイルキャッシュは移さず再構築する。Python開発ヘッダも新マシンで用意する。旧rawの`runtime/`にあるUbuntu x86_64 Python3.12ヘッダは同環境の場合だけ参考になるが、新マシンではOSの開発パッケージを優先する。

新本走は全snapshotを最初から保存するため、空き400GiB程度が目安。コードは25GiB未満で停止する。

## 4. 移行先のコードと環境

Linux x86_64、NVIDIA CUDA GPU、Python 3.12、C/C++コンパイラとPython開発ヘッダを想定する。元環境はPython3.12.3、PyTorch2.13.0+cu130、CUDA13.0、Triton3.7.1、NumPy2.5.1、Pillow12.3.0、Matplotlib3.11.1、RTX5060Ti16GB、driver580.173.02。全pip packageは`handoff_0919/requirements.freeze.txt`を参照。torchvisionは不要。

移行先でも既存cloneを優先。未作成の場合のみcanonical cloneを作り、実験worktreeを作る。以下は移行先に同名worktree/ブランチがない場合:

```bash
mkdir -p "$HOME/Projects/claude/wt"
# cloneが存在しない場合だけ:
git clone https://github.com/Issan0511/lop_analysis.git "$HOME/Projects/claude/proj_004_drift"
git -C "$HOME/Projects/claude/proj_004_drift" fetch origin
git -C "$HOME/Projects/claude/proj_004_drift" worktree add "$HOME/Projects/claude/wt/joudaki_vit_battle_0919" -b codex/joudaki_vit_battle_0919 origin/codex/joudaki_vit_battle_0919
cd "$HOME/Projects/claude/wt/joudaki_vit_battle_0919"
python3.12 -m venv "$HOME/Projects/claude/proj_004_drift/.venv"
PY="$HOME/Projects/claude/proj_004_drift/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install 'triton==3.7.1' 'numpy==2.5.1' 'pillow==12.3.0' 'matplotlib==3.11.1'
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

移行先でのwheel入手・driver互換性は未検証。上記versionが入らない場合は、無断で別versionや半精度に切り替えず、環境差を明示して検査する。Pythonヘッダ不足なら移行先OSのPython3.12-dev/build-essential相当を用意する。複数GPU機でもこの実験にはまず1枚だけ見せ、全条件で固定する。

## 5. データ検証と新本走の出力先を用意（GPU学習なし）

以下は**新マシンでのみ**、リポジトリrootから実行する。既存rawがあれば上書きせず停止する。既に旧rawを転送していた場合は、新マシン上で別の保管名へ退避してから行う。元マシンの旧rawを移動・削除しない。

```bash
"$PY" - <<'PY'
from pathlib import Path
import hashlib, json, datetime
from analysis.joudaki_vit_battle_0919.run import RAW, DATA, source_hashes
assert not RAW.exists(), f'Existing output must be archived separately first: {RAW}'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()
assert sha(DATA.parent/'tiny-imagenet-200.zip') == '6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821'
assert len(list((DATA/'train').glob('*/images/*.JPEG'))) == 100000
assert len(list((DATA/'val/images').glob('*.JPEG'))) == 10000
RAW.mkdir(parents=True)
(RAW/'STOP').touch()
(RAW/'fresh_run_plan.json').write_text(json.dumps({
    'created':datetime.datetime.now().astimezone().isoformat(),
    'mode':'fresh_all_130_runs', 'old_data_excluded':True,
    'start':'SNA seed0 task1', 'source_hashes':source_hashes()
},indent=2)+'\n')
print('Fresh output prepared; production has not started:', RAW)
PY
```

同じコードがマシンごとのHOME配下に保存するため、コード内のRAWパス変更は不要。旧・新データの絶対パス表記が同じでも別マシンの別ディスクであり、結果を混合しない。

## 6. 新環境で検査する（本走はまだ開始しない）

初回compileに時間がかかる。新環境で速度実装の数値検査・KKA/RSLの実画像再開一致・report検査を実行し、結果と環境を新rawの`migration_checks/`に残す。検査seedは100で、本走seed0–9と分離される。再開検査は今後同じ新マシンで中断・継続するときのために実施する。旧checkpointは使わない。

**注意: `checks`自体も`STOP`に従う。検査の間だけ外し、finallyで戻す。** launcherとの共通lockを取得するため、本走との同時実行を防ぐ。検査スクリプトはGit管理のJSONを上書きするので、以下で必ず元に戻す。`source_hashes_at_validation`を付けずに検査JSONをlauncher用に上書きすると、launcherは起動を拒否する。

```bash
"$PY" - <<'PY'
from pathlib import Path
import datetime, fcntl, json, os, platform, subprocess, sys, torch
from analysis.joudaki_vit_battle_0919.run import RAW, REPO, source_hashes
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save=RAW/'migration_checks'/stamp
save.mkdir(parents=True)
lock=(RAW/'launcher.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
stop=RAW/'STOP'
assert stop.exists(), 'Expected preflight STOP'
assert json.loads((RAW/'fresh_run_plan.json').read_text())['mode']=='fresh_all_130_runs'
assert not (RAW/'runs').exists() or not any((RAW/'runs').iterdir()), 'Expected empty production runs'
stop_bytes=stop.read_bytes()
files=[REPO/'results/joudaki_vit_battle_0919'/name for name in
       ('speed_checks.json','checks_compile.json','report_checks.json')]
original={p:p.read_bytes() for p in files}
meta={'git_hash':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),
      'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
      'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
      'source_hashes':source_hashes(),'mode':'fresh_all_130_runs','old_data_excluded':True,
      'status':'started'}
(save/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
    (RAW/'fresh_preflight_pass.json').unlink(missing_ok=True)
    stop.unlink()
    for module,args in [('check_speed',[]),('checks',['--engine','compile']),('check_report',[])]:
        with (save/f'{module}.log').open('w') as log:
            subprocess.run([sys.executable,'-m',f'analysis.joudaki_vit_battle_0919.{module}',*args],
                           cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    for p in files:
        result=json.loads(p.read_text())
        assert result.get('all_pass'), str(p)
        result['source_hashes_at_validation']=source_hashes()
        (save/p.name).write_text(json.dumps(result,indent=2)+'\n')
    meta['status']='passed'
    (RAW/'fresh_preflight_pass.json').write_text(json.dumps({
        'checks_directory':str(save),'source_hashes':source_hashes(),
        'mode':'fresh_all_130_runs','all_pass':True
    },indent=2)+'\n')
finally:
    stop.write_bytes(stop_bytes)
    for p,data in original.items(): p.write_bytes(data)
    if meta['status']!='passed': meta['status']='failed'
    (save/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
print('New-host checks passed:',save)
PY
```

失敗時は本走を開始しない。元の検査結果を成功扱いに書き換えず、新環境のlogを確認する。新環境検査PASSは「その環境内での再開一致」であり、旧GPUと新GPUの長期軌道の一致を意味しない。

## 7. 新マシンで全130 runを最初から開始

§5と§6が通った後、新環境の`migration_checks/<timestamp>/`を結果記録へコピーしてcommit/pushすると環境履歴をGitでも保持できる。元マシンのSTOPはそのまま。初回起動前に旧runが混入していないことと、新環境の検査PASSを機械的に確認する。

```bash
set -e
cd "$HOME/Projects/claude/wt/joudaki_vit_battle_0919"
PY="$HOME/Projects/claude/proj_004_drift/.venv/bin/python"
RAW="$HOME/Projects/obsidian-research-data/joudaki_vit_battle_0919"
git status --short
# src / analysis / specs に未commit変更がないことを確認。
"$PY" - <<'PY'
import json
from analysis.joudaki_vit_battle_0919.run import RAW, source_hashes
assert not (RAW/'runs').exists() or not any((RAW/'runs').iterdir()), 'Old or existing runs found; do not fresh-start here'
passed=json.loads((RAW/'fresh_preflight_pass.json').read_text())
assert passed['all_pass'] and passed['mode']=='fresh_all_130_runs'
assert passed['source_hashes']==source_hashes()
(RAW/'STOP').unlink()
PY
# 直前の検査が成功した場合だけ次を実行する。
systemd-run --user --unit=joudaki-vit-battle-0919 \
  --description='Joudaki ViT Tiny ImageNet 13 activation battle' \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$RAW/launcher.log" \
  --property=StandardError=inherit \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  "$PY" -m analysis.joudaki_vit_battle_0919.launch
```

systemd user sessionがない環境ではtmux内で`"$PY" -m analysis.joudaki_vit_battle_0919.launch >> "$RAW/launcher.log" 2>&1`。ジョブがlogoutで止まらない実行基盤を使う。最初に**SNA seed0 task1**が始まることを確認し、その後の監視は5分以上あける。停止は`touch "$RAW/STOP"`でtask境界保存を待つ。強制killより優先する。

新マシンで取り直した全130 runがterminalになればlauncherが自動でreportを生成する。旧マシンの18完了runや途中runをこの集計rootに持ち込まない。完了後は図表・Obsidian結果ノート・生データmanifestを整備し、共有CLAUDE.mdの手順でmain統合・worktree整理する。**現在は旧試行停止・新本走の準備待ちであり、実験完了としてmain統合やworktree削除をしない。**

## 8. 速度と既知の制約

元GPUでKKAはcompileにより約139→63ms/更新（2.22倍）。ただし総計260万更新に評価・診断・保存が加わる。旧試行の残り時間見積もりは、新本走には適用しない。新マシンでは全260万更新を実行する。

同一GPUの通常1/2/4プロセスでKKA合成入力を測定し、総更新/秒14.30 / 13.16 / 13.92だったため、本走は1本。MPSは未測定。別マシンでも勝手に13本を同時起動しない。launcherは1GPU逐次、lockは同一ホスト/同一raw内だけであり、2台間の排他は保証しない。

CPUやGPUを占有させたくない場合、開始時刻や実行GPUを新マシンで決める。今回の元マシンには自動再開予定を作っていない。
