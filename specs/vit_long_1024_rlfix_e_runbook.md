# vit_long_1024 追補 E の byakko の分を先生のマシンで回す手順

2026-10-26、Claude（Opus 5.5、white-san）。spec は `specs/spec_vit_long_1024.md` の追補 D と追補 E。判定・閾値・腕・seed・予測は spec が正で、この手順書は置き場とコマンドを決める。

## 0. 何をするか

- 2 ブロックの ViT（Tiny ImageNet、同じ 2,500 枚に課題ごとに乱数ラベルを引き直す、1 課題 5,000 更新 × 100 課題）を、仕事の一覧 `analysis/vit_long_1024/jobs_rlfix_e_byakko.txt` のとおり回す。
- 中身は、本走 10 本（SL・R0L・SRMS・R0RMS の組、seed 10252012〜10252014）と、新品 20 本（1 課題だけ）。white-san でも別の分を同時に回している。
- コードは書き換えない。新しく commit・push もしない。結果はギガファイル便で white-san に送る。

## 1. 置き場

このマシンには lop_analysis の clone がある（remote は github.com/Issan0511/lop_analysis）。

```bash
cd <lop_analysis の clone>                  # 無ければ git clone https://github.com/Issan0511/lop_analysis.git
git status --short                           # 変更が残っていたら止めて報告する
git fetch origin
git worktree add ../lop_vit_rlfix_e origin/claude/vit_long_1024 --detach
cd ../lop_vit_rlfix_e
git log --oneline -1                         # Issa がプロンプトに書いた commit であること
export E=$(pwd)
export PY=<lop_analysis の clone>/.venv/bin/python    # 文字 GPT で使った .venv
$PY -c "import sys,torch,numpy,PIL;print(sys.version.split()[0],torch.__version__,numpy.__version__,PIL.__version__,torch.cuda.get_device_name(0))"
```

- 版は python 3.12.x・torch 2.13.0+cu130・numpy 2.5.1 であること。Pillow が無ければ `uv pip install --python $PY "pillow==12.3.0"`（版は `analysis/vit_long_1024/remote/requirements.txt` の値）。torch の版が違ったら止めて報告する。
- 出力の置き場とデータの置き場を決めて、環境変数にする（ディスクは出力に約 8 GB、データに約 1 GB 要る）。

```bash
export VIT_LONG_RAW=<出力の置き場>/vit_long_1024_2blk_rlfix         # 例: /home/kubo/project/Nakatsuka/vit_long_1024_2blk_rlfix
export VIT_LONG_DATA=<データの置き場>/tiny-imagenet-200
```

## 2. Tiny ImageNet を取る

white-san と同じ Kaggle の版（akash2sharma/tiny-imagenet）を、認証なしの URL から取る。

```bash
mkdir -p $(dirname $VIT_LONG_DATA) && cd $(dirname $VIT_LONG_DATA)
curl -L --fail --retry 3 -o tiny-imagenet-200.kaggle_akash2sharma.zip \
  https://www.kaggle.com/api/v1/datasets/download/akash2sharma/tiny-imagenet
stat -c %s tiny-imagenet-200.kaggle_akash2sharma.zip        # 497536564 であること
sha256sum tiny-imagenet-200.kaggle_akash2sharma.zip          # 05fafc211a347868093af2133504de066329059df16c2d362d068a73859f466e であること
unzip -q tiny-imagenet-200.kaggle_akash2sharma.zip -x 'tiny-imagenet-200/tiny-imagenet-200/*'   # 入れ子の重複は展開しない
find tiny-imagenet-200/train -name '*.JPEG' | wc -l          # 100000
find tiny-imagenet-200/val/images -name '*.JPEG' | wc -l     # 10000
```

大きさか sha256 が違ったら止めて報告する。

## 3. 検査（全部通るまで本走しない）

```bash
cd $E
$PY -m analysis.vit_long_1024.checks_rlfix_e --skip-e3
```

- E1（RMSNorm の網の作り方）・E2（RMSNorm の腕の通し走）・E4（seed ごとの固定の画像とラベル）・E5（汚れの判定）が通ること。E3 は white-san の記録と比べる項目なので、このマシンでは飛ばす。
- 記録は `results/vit_long_1024_2blk_rlfix/checks_e_<このマシンの名前>.json`（commit しない。§5 の束に入れる）。
- E2 の `ms_per_update`（2 課題目の値）を報告に書く。white-san の RTX 5060 Ti では約 30 ms だった。

## 4. 本走

```bash
cd $E
setsid nohup bash analysis/vit_long_1024/run_queue.sh analysis/vit_long_1024/jobs_rlfix_e_byakko.txt \
  > $VIT_LONG_RAW/queue_byakko.out 2>&1 < /dev/null &
```

- 一覧の奇数行と偶数行を、2 本ずつ並べて回す（同じ組の 2 腕が同時に走る）。GPU には 2 本だけ載る。
- 進み具合: `$VIT_LONG_RAW/run_queue_jobs_rlfix_e_byakko.log`（start・end と終了コード）、各走の `$VIT_LONG_RAW/runs/<腕>/seed<seed>/status.json` と `per_task.csv`。
- 止めたいとき: `touch $VIT_LONG_RAW/STOP`（課題の境で止まる）。再開は `rm $VIT_LONG_RAW/STOP` の後に同じ命令を流す（終わった走は飛ばし、途中の走は課題の境から続ける）。
- **止める規則（spec 追補 E.6）**: 戻す腕（R0L・R0RMS）が課題 2〜5 で一度でも 0.9 に届かなかったら（`per_task.csv` の `T90` が 5025）、その組を止めて報告する。
- 1 本はおよそ 100 課題で数時間かかる（white-san の RTX 5060 Ti では 1 本だけで約 5.5 時間、2 本並べると 2 本で約 11 時間）。最初の組の課題 5 が終わったら、1 課題の秒数（`per_task.csv` の `seconds`）を報告する。

## 5. 結果を送る

全部が終わったら（ログの最後が `ALL_DONE`、本走 10 本の `done.json` の `status` が `complete`）:

```bash
cd $VIT_LONG_RAW
cp $E/results/vit_long_1024_2blk_rlfix/checks_e_*.json .
tar --exclude='checkpoint.pt' -cf vit_rlfix_e_byakko.tar runs run_queue_jobs_rlfix_e_byakko.log checks_e_*.json
sha256sum vit_rlfix_e_byakko.tar > vit_rlfix_e_byakko.tar.sha256
ls -l vit_rlfix_e_byakko.tar                                  # 約 7 GB
uv venv -q gfvenv --python 3.12 && uv pip install -q --python gfvenv/bin/python "gigafile==3.3.1"
gfvenv/bin/gfile upload vit_rlfix_e_byakko.tar                # 出た URL を報告する
```

- 再開用の `checkpoint.pt` は入れない（課題 100 末の重みと Adam の状態は `ckpt/` にある）。
- ギガファイル便の正式なドメインは `gigafile.jp`。
- 送った後も、このマシンの `$VIT_LONG_RAW` は消さない（white-san で受け取りを確かめるまで）。

## 6. 報告（日本語）

節目ごとに短く報告する（検査、最初の組の課題 5、半分、全部）。最後の報告に入れるもの:

- 検査の結果（E1・E2・E4・E5）と E2 の 1 更新の ms。
- 本走 10 本と新品 20 本の終了コード、かかった時間。
- 止める規則に当たった組があったか。
- 束の URL、`vit_rlfix_e_byakko.tar.sha256` の中身、束の大きさ。
- 手順から外れたこと（あれば）と、その理由。

## 7. してはいけないこと

- コード・spec・仕事の一覧・seed・腕を変えない。変える必要が出たら止めて報告する。
- このブランチにも他のブランチにも commit・push しない。
- 一覧にない走を足さない。GPU に 3 本以上載せない。
- Obsidian の vault には書かない。

## 8. 向こうの Claude Code に貼るプロンプト

```text
lop_analysis のブランチ claude/vit_long_1024（commit <commit>）にある
specs/vit_long_1024_rlfix_e_runbook.md（手順書）に従って、
2 ブロック ViT の追補 E のうち byakko の分（analysis/vit_long_1024/jobs_rlfix_e_byakko.txt）を、このマシンで回してください。
spec は specs/spec_vit_long_1024.md の追補 D と追補 E です。判定・腕・seed・予測は spec が正です。

- 出力の置き場: <置き場>
- データ（Tiny ImageNet）の置き場: <置き場>
- 手順書の §3 の検査が全部通るまで本走しないでください。
- 節目ごとに日本語で短く報告し、最後にギガファイル便の URL と sha256 を報告してください。
- コードは書き換えず、commit・push もしないでください。
```
