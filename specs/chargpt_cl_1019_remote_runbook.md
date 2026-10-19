# chargpt_cl_1019 を別マシンで回す手順

2026-10-20、Claude（Opus 5.5、white-san）。spec は `specs/spec_chargpt_cl_1019.md`（vault の v1.1 の写しと追補 0〜2）。

## 0. 何をするか

- 本走は箱 M だけである（α 0.01、N_M 10,000、50 課題、fresh は課題 10・25・50）。腕は C3・C4・B3・N4 の 4 本、seed は 0〜9。
- 走は seed 単位で各マシンに割り振る。**1 seed の 4 腕は同じマシンで回す**（seed で対応させる比較を同じ GPU の中で行うため）。
- **担当（2026-10-20）：別マシンが seed 0〜9 を全部回す。判定はこの 40 走で出す。** white-san は seed 0・1 を先に回して早く結果を見る（判定外。同じ seed をマシン違いで回した照合を兼ね、`results/chargpt_cl_1019/crosscheck_white-san/` に置く）。
- 1 走は RTX 5060 Ti で 41.8 分、1 seed の 4 腕で 2.8 時間。fp32（TF32 なし）が速い GPU ほど縮む。A100 は fp32 だと 5060 Ti とほぼ同じである。

## 1. 環境

- Linux、NVIDIA GPU 1 枚以上、CUDA が使えること。1 走あたり GPU メモリ約 1.2 GB、ホストの RSS 約 2 GB、ディスク約 35 MB（checkpoint を含む）。
- 基準機の版：python 3.12.3、torch 2.13.0+cu130、numpy 2.5.1。torch と numpy は**同じ版**にそろえる。乱数の列が版で変わると、照合（S-portable）で止まる。

```bash
git clone https://github.com/Issan0511/lop_analysis.git
cd lop_analysis
git checkout claude/chargpt_cl_1019
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python "numpy==2.5.1"
```

- Tiny Shakespeare は要らない（箱 W は v1 で回さない）。

## 2. 準備（照合・検査・速さ）

```bash
PY=.venv/bin/python bash src/chargpt_cl_1019_remote.sh prepare
```

順に次を行い、どこかで落ちたら止まる。最後に `PREPARE_OK` が出れば準備は完了である。

1. 環境を `results/chargpt_cl_1019/remote_<host>/env.txt` に書く。
2. **S-portable**：箱 M の課題 1〜3 の遷移表・batch 列・probe と 4 腕の初期値の sha256 を、基準機の `results/chargpt_cl_1019/expected_streams.json` と照らす。データ列はすべて CPU の乱数で作るので、機種によらず一致するはずである。
3. **検査（箱 M）**：spec §8 と追補 0 §G の検査を回す（2 分前後）。全部 PASS であること。
4. **速さ**：C3・seed 0・2 課題を 3 通り回し、1 走の時間を `remote_<host>/cost/cost.json` に書く。

## 3. 起動

担当の seed を渡す（別マシンは seed 0〜9）。

```bash
PY=.venv/bin/python bash src/chargpt_cl_1019_remote.sh run 0 1 2 3 4 5 6 7 8 9
PY=.venv/bin/python bash src/chargpt_cl_1019_remote.sh status      # 進み具合
```

- GPU が複数あるときは、担当を分けて GPU ごとに 1 本ずつ起動する（例：2 枚なら `CUDA_VISIBLE_DEVICES=0 ... run 0 1 2 3 4`、`CUDA_VISIBLE_DEVICES=1 ... run 5 6 7 8 9`。起動ログは同じ名前になるので、2 本目は手で `production_launch_<host>_gpu1.log` に分ける）。1 枚の GPU に 2 本載せても合計の速さは増えない。
- 完走した走は飛ばすので、止まったら同じコマンドで再開できる。途中で止まった走は最初からやり直す。

## 4. 結果を戻す

seed の 4 腕が終わるたびに：

```bash
bash src/chargpt_cl_1019_remote.sh push 0
```

- 完走した走（`provenance.json` があるもの）だけを commit し、`claude/chargpt_cl_1019` に push する。push の前に `git pull --no-rebase` で white-san 側の commit を取り込む。
- checkpoint（`ckpts/*.pt`）は gitignore されていて、向こうに残る。要るときはギガファイル便で送る。

## 5. してはいけないこと

- 箱 W を回さない。腕・seed・予算・判定・閾値を変えない（spec の追補 2 が正）。
- torch.compile や TF32 を入れない（追補 0 §A）。
- 1 seed の 4 腕を別々のマシンに分けない。
- 走っている起動スクリプト（`chargpt_cl_1019_production.sh`）を書き換えない。bash は走りながらファイルを読み進める。
- push 済みの commit を rebase・amend・force push しない（provenance の git hash が指す commit が消える）。

## 6. 向こうの Claude Code に貼るプロンプト

```text
lop_analysis のブランチ claude/chargpt_cl_1019 の specs/chargpt_cl_1019_remote_runbook.md に従って、
文字 GPT 継続学習 chargpt_cl_1019 の箱 M の本走を、このマシンで seed 0〜9 の全部について回してください。
手順は 1 環境 → 2 prepare（PREPARE_OK まで）→ 3 run → 4 seed ごとに push。
prepare で S-portable か検査が落ちたら、走らせずに結果（remote_<host>/ の中身）を報告してください。
spec（specs/spec_chargpt_cl_1019.md）の腕・予算・判定は変えないこと。報告は日本語で。
```
