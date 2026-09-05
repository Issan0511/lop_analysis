# 引き継ぎ: `edge_law_0905` の本走を別マシンで回す

宛先: 別マシン（`lop_analysis` の clone がある環境）で走らせる担当。
親: `specs/spec_edge_law_0905.md`（事前登録・**この spec の §4 以外で判定しない**）／`configs/edge_law_0905.yaml`（腕表の正本）。

> **やってほしいのは計算だけです。** 検査（S-null / S-mirror）と登録判定の解析は、委託参照ログ（`results/p3_extend_0902/logs`・2.6 GB・**git に入っていない**）を持っている元マシンで行います。そちらは 30 腕を回して**ログを送り返す**だけで完結します。

## 0. 前提

- `git pull` で `origin/main` を最新にする。必要なのは commit **（本走用の commit ハッシュはこのファイルの最後に追記します）** 以降。
- `python3 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"` が動くこと（開発は torch 2.13.0 / numpy 2.5.2）。venv があれば `.venv/bin/python`、無ければ system python3 で可（**判定に使う数値は元マシンで再計算するので、bit 一致は要求しません**）。
- ディスク: 生ログ約 **3.2 GB**。
- メモリ: 1 プロセス（10 seed）の peak RSS は 5M 腕で **0.9 GiB**、15M 腕で **1.4 GiB**。**並列数 = min(コア数, floor(空き GiB / (1.5 × 1.4)))** で決める（コア数だけで決めない）。
- CPU 時間: 合計 **約 620 分**。16 並列なら壁時計 60〜70 分（臨界パスは 15M 腕の約 52 分）。

## 1. 走らせ方

```bash
cd <repo>
OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.edge_law_0905 --launch-plan --parallel <N>
```

が 30 腕ぶんのコマンドと推奨並列数を出すので、その通りに流す。要点だけ再掲:

- **1 腕 = 1 プロセスで 10 seed 全部**を回す。`--seeds` で分割しないこと（**S-seed-split**: 並列 seed 数 `R` を変えると `env.step()` の入力列が 1 step 目からずれ、別の走になる）。
- `OMP_NUM_THREADS=1` を必ず付ける（付けないと遅く、かつ LAPACK の縮約順が変わる）。
- 出力は `results/edge_law_0905/`（`--outdir` 既定）。
- 長い腕から先に出す: `LRbm5_1216` / `Ebm4_1216`（15M）→ `LRlr0p005_1216` / `Elr0p005_1216`（10M）→ 5M 25 本 → `FBLR_1216`（500k）。
- 腕は独立なので途中で止めて再開してよい（`results/edge_law_0905/arm_status/<arm>_done.json` があるものは skip される）。

走り終わったら:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.edge_law_0905 --tail-extract
```

で `results/edge_law_0905/logs_tail/`（登録判定が読む窓だけの縮約 npz・全 30 腕で 100 MB 弱）ができる。

## 2. 送り返すもの

**A と B の 2 つだけ**。C は送らなくてよい。

| | 中身 | サイズ | 用途 |
|---|---|---|---|
| **A** | `results/edge_law_0905/logs_tail/` 全部 ＋ `arm_status/` ＋ 標準出力のログ | 約 100 MB | 登録判定（§4 の全部） |
| **B** | `results/edge_law_0905/logs/{LRnull_1216,Enull_1216,FLn_1216}_seed{0..9}.npz`（30 ファイル） | 約 300 MB | **S-null / S-null-E / S-mirror**（全記録の bit 比較が要るので縮約版では不可） |
| C | 残り 27 腕の生ログ | 約 2.8 GB | 送らなくてよい（そちらに残す。後で必要になったら別途） |

作り方:

```bash
cd <repo>/results
tar cf edge_law_0905_A.tar edge_law_0905/logs_tail edge_law_0905/arm_status
tar cf edge_law_0905_B.tar edge_law_0905/logs/LRnull_1216_seed*.npz \
                            edge_law_0905/logs/Enull_1216_seed*.npz \
                            edge_law_0905/logs/FLn_1216_seed*.npz
sha256sum edge_law_0905_A.tar edge_law_0905_B.tar
```

**ギガファイル便で送る**（Claude セッションからなら `/gigafile-uploader` スキル、手動なら https://gigafile.jp）。**URL と一緒に sha256 と、走らせた commit ハッシュ（`git rev-parse HEAD`）・python/torch/numpy のバージョン・`arm_status/*_done.json` の中身（発散腕の有無）を必ず添えること。**

> ダウンロード側の注意（元マシン向けメモ）: ギガファイル便は**ファイルごとに cookie jar を分ける**。先に `https://<n>.gigafile.jp/<id>` を踏んでから、同じ jar ＋ Referer で `download.php?file=<id>&dlkey=<key>` を叩く。jar を使い回すと 570 byte の HTML が返る。

## 3. そちらで見なくていいもの

- `--sanity`（事前検査）は元マシンで全 PASS 済み（`results/_preflight_edge_law_0905/sanity.json`）。**走らせても害はないが、S-null/S-mirror は参照ログが無いので落ちる**（`MISSING` になる）。
- `src.edge_law_analyze_0905`（登録判定）も参照ログが要るので元マシンで回す。
- **判定・解釈は書かないでください。** spec §4 の判定は元マシンで 1 回だけ回します。

## 4. 異常時

- **NaN で腕が落ちた**: そのまま報告する（`arm_status/<arm>_diverged.json` が出る）。spec §4.6 で「NaN seed が 2/10 を超えたら腕は `NOT_RUN`」と登録済みなので、こちらで処理します。**lr を下げる等の救済はしないこと**（登録外の介入になる）。
- **メモリが足りない**: 並列数を下げる。腕は独立なので分割して構いません。
- **`FBLR_1216`（full-batch・500k）だけ落ちる**: spec で `REPORT_ONLY` かつ「落ちたら `NOT_RUN`」と登録済み。他 29 腕が揃っていれば十分です。

---

## 走らせる commit

- 本走に使う commit: **（レビュー完了後にここへ記入）**
- 記入時点で `git log --oneline -1` と `git status --short src/` が clean であることを確認すること。
