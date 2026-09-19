# Joudaki ViT / Tiny ImageNet 13活性化実験 — 別マシンへの引き継ぎ

更新: 2026-09-19 JST。ユーザー依頼により元マシンでは停止済み。**元マシンを再開せず、移行先だけで再開する。**

## 1. 最初に読む要約

- GitHub: https://github.com/Issan0511/lop_analysis
- ブランチ: `codex/joudaki_vit_battle_0919`。このMDを含む最新commitを取得する。
- 凍結学習実装のcommit: `677fc85`。並列測定追加: `89fedb1`。以後の引き継ぎ追加でも学習Pythonソースは変更していない。
- **18/130 run完了、751/5200タスク完了。KKA23 / seed1 / task31終了でcheckpoint保存済み。次はtask32。**
- seed0は13腕完了。seed1はSNA、KKA、R、LR、KKT1が完了し、KKA23が途中。その他のseedは未実施。
- 元サービス `joudaki-vit-battle-0919.service` は正常終了。rawルートの`STOP`を残してあり、自動再開しない。
- コード・仕様・検査結果・ここまでの小さな実行記録はGitに保存。**重み・前活性・checkpoint等のraw約49GiBとデータセットはGitに含まれない。§3で転送する。Git cloneだけでは途中再開できない。**
- 本走の勝敗・順位は未集計。完了後に全130 runで集計する。

仕様: [spec_joudaki_vit_battle_0919.md](spec_joudaki_vit_battle_0919.md)。実装説明: [README](../analysis/joudaki_vit_battle_0919/README.md)。停止状態・環境・転送manifest・run記録: [handoff_0919](../results/joudaki_vit_battle_0919/handoff_0919/)。

## 2. 固定条件と変更してはいけないもの

Tiny ImageNet、40個の非重複5クラス課題、各500更新、batch128、seed0–9。ViT patch8 / dim384 / depth6 / heads6 / FFN1536。Adam lr1e-4、weight decayなし。float32、TF32無効、`torch.compile(fullgraph=True)` + fused Adam。

13腕: SNA, KKA, R, LR, KKT1, KKA23, SL, RSL, LK001, LK03, ELU, SILU, GELU。SNA系のEMA更新はoptimizerの後。dropoutはATen乱数列、RSLは専用乱数列。各taskでheadをresetし、optimizer momentsは継続。

`analysis/joudaki_vit_battle_0919/` 以下の**全Pythonファイル**と `src/rlcifar_mlp_battle_0918.py`, `src/pmnist_0905.py`, `src/pmnist_rlcifar_0907.py` はSHA256で凍結している。ここにPythonファイルを追加するだけでも再開検査が失敗する。移行のためのパス修正や機能追加は不要。補助スクリプトが必要なら凍結ディレクトリの外に置く。

- `REPO`はソース位置から自動決定する。
- `RAW`は移行先の `$HOME/Projects/obsidian-research-data/joudaki_vit_battle_0919`。
- `DATA`は移行先の `$HOME/Projects/obsidian-research-data/datasets/tiny-imagenet-200`。
- `done.json`があるrunはlauncherが飛ばす。途中runは`checkpoint.pt`からモデル・Adam・EMA・RNGを復元する。KKA23 seed1のcheckpointを消さない。
- 同一環境のtask境界再開はbit一致検査済み。**別GPU/別CUDA/PyTorchをまたぐbit一致は保証していない。** 移行環境と切替位置を記録する。seed1は途中でハードウェアが切り替わるため、その事実を最終結果に明記する。最終reportはハードウェア混在を自動拒否しない。

## 3. 元マシンからrawとデータを転送

元マシンのパス:

```text
/home/issan/Projects/claude/wt/joudaki_vit_battle_0919
/home/issan/Projects/obsidian-research-data/joudaki_vit_battle_0919/        約49GiB
/home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200/  約481MiB
/home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200.zip 約237MiB
```

転送先はまだ指定されておらず、**この引き継ぎ時点では転送未実施**。元マシンで、`DEST`を実際のSSH接続先に置き換えて実行:

```bash
DEST='user@destination-host'
ssh "$DEST" 'mkdir -p "$HOME/Projects/obsidian-research-data/joudaki_vit_battle_0919" "$HOME/Projects/obsidian-research-data/datasets"'
rsync -a --partial --info=progress2 /home/issan/Projects/obsidian-research-data/joudaki_vit_battle_0919/ "$DEST:Projects/obsidian-research-data/joudaki_vit_battle_0919/"
rsync -a --partial --info=progress2 /home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200/ "$DEST:Projects/obsidian-research-data/datasets/tiny-imagenet-200/"
rsync -a --partial --info=progress2 /home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200.zip "$DEST:Projects/obsidian-research-data/datasets/"
```

`STOP`も転送する。`--delete`は使わない。`.venv`やTorchコンパイルキャッシュは転送せず再構築する。raw内の`runtime/`は現行のPython開発ヘッダを含む。学習時の各snapshot/preactivationのSHA256は既存manifestを利用し、その他のファイルは引き継ぎ時にハッシュを計算した。移行先で以下の全ハッシュ検査を行う。

転送済み49GiBに加え、残りのsnapshot等に数百GiBの空きが必要。元見積もりは最終保存量最大約350GB。開始時に空き400GiB程度を確保するのが目安。コードは25GiB未満で停止する。

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

移行先でのwheel入手・driver互換性は未検証。上記versionが入らない場合は、無断で別versionや半精度に切り替えず、環境差を明示して検査する。Pythonヘッダ不足なら移行先OSのPython3.12-dev/build-essential相当を用意する。コピーしたruntimeヘッダはUbuntu x86_64 Python3.12用。複数GPU機でもこの実験にはまず1枚だけ見せる（checkpointのCUDA RNGは1枚分）。

## 5. 転送・checkpointを検証（GPU学習なし）

リポジトリrootで実行。ハッシュ検査は全raw約49GiBを読み、時間がかかる。`STOP`を除く前に行う。

```bash
"$PY" - <<'PY'
from pathlib import Path
import hashlib, json, torch
from analysis.joudaki_vit_battle_0919.run import RAW, DATA, source_hashes
handoff = Path('results/joudaki_vit_battle_0919/handoff_0919')
assert (RAW/'STOP').exists()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()
for row in json.loads((handoff/'transfer_manifest.json').read_text())['files']:
    p=RAW/row['path']
    assert p.is_file() and p.stat().st_size == row['bytes'], str(p)
    assert sha(p) == row['sha256'], str(p)
assert sha(DATA.parent/'tiny-imagenet-200.zip') == '6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821'
assert len(list((DATA/'train').glob('*/images/*.JPEG'))) == 100000
assert len(list((DATA/'val/images').glob('*.JPEG'))) == 10000
state=torch.load(RAW/'runs/KKA23/seed1/checkpoint.pt', map_location='cpu', weights_only=False)
assert state['task'] == 31
assert state['source_hashes'] == source_hashes()
print('Transfer verified; next task: KKA23 seed1 task32')
PY
```

## 6. 新環境で検査する（本走はまだ開始しない）

初回compileに時間がかかる。以下は旧検査結果を保持し、新環境で速度実装の数値検査・KKA/RSLの実画像再開一致・report検査を実行して、結果と環境をrawの`migration_checks/`に残す。既存runの`provenance.json`は書き換えない（再開時も自動更新されないため、この移行記録が必要）。

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
assert stop.exists(), 'Expected paused experiment'
stop_bytes=stop.read_bytes()
files=[REPO/'results/joudaki_vit_battle_0919'/name for name in
       ('speed_checks.json','checks_compile.json','report_checks.json')]
original={p:p.read_bytes() for p in files}
meta={'git_hash':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),
      'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
      'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
      'source_hashes':source_hashes(),'resume_boundary':'KKA23 seed1 after task31',
      'status':'started'}
(save/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
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
finally:
    stop.write_bytes(stop_bytes)
    for p,data in original.items(): p.write_bytes(data)
    if meta['status']!='passed': meta['status']='failed'
    (save/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
print('New-host checks passed:',save)
PY
```

失敗時は本走を開始しない。元の検査結果を成功扱いに書き換えず、新環境のlogを確認する。新環境検査PASSは「その環境内での再開一致」であり、旧GPUと新GPUの長期軌道の一致を意味しない。

## 7. 移行先で再開

§5と§6が通った後、新環境の`migration_checks/<timestamp>/`を結果記録へコピーしてcommit/pushすると環境履歴をGitでも保持できる。元マシンのSTOPはそのまま。

```bash
cd "$HOME/Projects/claude/wt/joudaki_vit_battle_0919"
PY="$HOME/Projects/claude/proj_004_drift/.venv/bin/python"
RAW="$HOME/Projects/obsidian-research-data/joudaki_vit_battle_0919"
git status --short
# src / analysis / specs に未commit変更がないことを確認。
rm "$RAW/STOP"
systemd-run --user --unit=joudaki-vit-battle-0919 \
  --description='Joudaki ViT Tiny ImageNet 13 activation battle' \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$RAW/launcher.log" \
  --property=StandardError=inherit \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  "$PY" -m analysis.joudaki_vit_battle_0919.launch
```

systemd user sessionがない環境ではtmux内で`"$PY" -m analysis.joudaki_vit_battle_0919.launch >> "$RAW/launcher.log" 2>&1`。ジョブがlogoutで止まらない実行基盤を使う。最初にKKA23 seed1 task32が始まることを確認し、その後の監視は5分以上あける。停止は`touch "$RAW/STOP"`でtask境界保存を待つ。強制killより優先する。

全130 runがterminalになればlauncherが自動でreportを生成する。完了後は図表・Obsidian結果ノート・生データmanifestを整備し、共有CLAUDE.mdの手順でmain統合・worktree整理する。**現在は停止・移行待ちであり、実験完了としてmain統合やworktree削除をしない。**

## 8. 速度と既知の制約

元GPUでKKAはcompileにより約139→63ms/更新（2.22倍）。ただし総計260万更新に評価・診断・保存が加わる。直近の元GPU実測では残り約49時間だったが、これは移行先の予測ではない。

同一GPUの通常1/2/4プロセスでKKA合成入力を測定し、総更新/秒14.30 / 13.16 / 13.92だったため、本走は1本。MPSは未測定。別マシンでも勝手に13本を同時起動しない。launcherは1GPU逐次、lockは同一ホスト/同一raw内だけであり、2台間の排他は保証しない。

CPUやGPUを占有させたくない場合、再開時刻や実行GPUを移行先で決める。今回の元マシンには自動再開予定を作っていない。
