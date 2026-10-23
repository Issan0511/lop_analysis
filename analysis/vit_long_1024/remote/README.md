# vit_long_1024 を借りた GPU で回す

本走は Vast.ai の RTX 5060 Ti（white-san と同じ機種、Ubuntu のコンテナで root）で回す。spec は `specs/spec_vit_long_1024.md`（§10.3）。クラウドの API はどのスクリプトも呼ばない。

## 1. 送るもの（white-san で作る）

```bash
bash /home/issan/Projects/claude/wt/vit_long_1024/analysis/vit_long_1024/remote/pack_for_remote.sh /tmp/vit_long_1024_remote.tar.gz
rsync -avP -e 'ssh -p PORT' /tmp/vit_long_1024_remote.tar.gz root@HOST:/root/
```

- 中身は、このブランチの commit 済みのコード一式（`git archive`、使う分だけ）、`SOURCE_COMMIT`（commit の hash）、`SOURCE_HASHES.json`（ソースの sha256）、white-san の検査の記録（`results/vit_long_1024/checks/`）、ファイルごとの sha256 の一覧 `tiny-imagenet-200.manifest_1024.sha256` である。
- 送った写しには `.git` が無い。走は `SOURCE_COMMIT` を git の hash として記録し、ソースが `SOURCE_HASHES.json` と一致するときだけ本走を始める。
- データは送らない。借りた機械が Kaggle の認証なしの URL から取る。

## 2. 借りた機械で

```bash
mkdir -p /root/vit_long_1024 && tar -xzf /root/vit_long_1024_remote.tar.gz -C /root/vit_long_1024
bash /root/vit_long_1024/lop_analysis/analysis/vit_long_1024/remote/setup_remote.sh   # apt・uv の venv・データ（15 分ほど）
bash /root/vit_long_1024/lop_analysis/analysis/vit_long_1024/remote/run_remote.sh     # データの要る検査の後、launcher を背景で起動
```

- 置き場の既定は white-san と同じ相対の形で、`/root/Projects/obsidian-research-data/`（データは `datasets/tiny-imagenet-200/`、出力は `vit_long_1024/`）。`VIT_LONG_DATA`・`VIT_LONG_RAW`・`VIT_LONG_PY` で変えられる。
- `setup_remote.sh` は、uv で python 3.12.3 の venv を作り、`requirements.txt`（white-san の .venv の版をそのまま）を入れる。torch は https://download.pytorch.org/whl/cu130 から取る。Tiny ImageNet は入れ子の重複 `tiny-imagenet-200/tiny-imagenet-200/` を除いて外側の 1 組だけを展開し、件数と一覧で照合する。
- `run_remote.sh` は、データの要る検査（S-data・S-stream・S-arm1・S-smoke・S-speed、約 10 分）を流し、通ったら launcher を setsid と nohup で背景に起動する。S-speed の 1 更新の ms と、10 走と fresh の見込みの時間を表示する。
- 機械が途中で止まったら、`run_remote.sh` をもう一度流す。続きの課題の境から回る（課題ごとに再開用の checkpoint を書く）。
- 見る: `cat /root/Projects/obsidian-research-data/vit_long_1024/status.json`。止める: `touch /root/Projects/obsidian-research-data/vit_long_1024/STOP`（課題の境で止まる）。
- 全部終わると、`vit_long_1024/DONE` と `vit_long_1024/summary/judge.md`（J0・J1・J3 と予測の採点）が書かれる。

## 3. 戻すもの（white-san で受け取る）

```bash
# 出力の置き場（約 37 GB。再開用の checkpoint.pt は完了した走では要らないので外してよい）
rsync -avP -e 'ssh -p PORT' --exclude 'checkpoint.pt' \
  root@HOST:/root/Projects/obsidian-research-data/vit_long_1024/ ~/Projects/obsidian-research-data/vit_long_1024/
# 小さな記録と借りた機械の検査の記録（white-san の記録は checks_data_white-san.json に残る）
rsync -avP -e 'ssh -p PORT' \
  root@HOST:/root/vit_long_1024/lop_analysis/results/vit_long_1024/ ~/Projects/claude/wt/vit_long_1024/results/vit_long_1024/
rsync -avP -e 'ssh -p PORT' root@HOST:/root/vit_long_1024/pip_freeze_remote.txt ~/Projects/obsidian-research-data/vit_long_1024/
```

- 受け取った後、white-san で `python -m analysis.vit_long_1024.collect` と `python -m analysis.vit_long_1024.judge` を流し直してよい（同じ値になる）。
- 生データの退避と main への取り込みは `~/Projects/claude/CLAUDE.md` §4 に従う。
