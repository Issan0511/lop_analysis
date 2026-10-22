# chargpt_seed10_1022 を先生のマシン（byakko）で回す手順

2026-10-22、Claude（Opus 5.5、V13 のセッション、white-san）。spec は `specs/spec_chargpt_seed10_1022.md`（本体と追補 1）。判定・閾値・窓・開始点・予測は spec が正で、この手順書は置き場・実装・コマンドを決める。

## 0. 何をするか

- spec の **A（装填）・B（押しの出所）・C（L1 の生き返りの帰還）** を回す。**D（4 ブロック SiLU の新しい学習）は、Issa が貼るプロンプトで「D も回す」とあるときだけ**。
- A・B・C の入力は、white-san で回した 2 ブロック 10 seed の checkpoint（git に入っていない）。ギガファイル便の束で受け取る。
- 新しく書くコードと結果はブランチ `claude/chargpt_seed10_1022` にだけ commit・push する。

## 1. 置き場（2 つの checkout）

- **`$CG`** = ブランチ `claude/chargpt_cl_1019`（HEAD 033679e0）の checkout。道具（`src/`・`analysis/chargpt_cl_1019_quicklook/`）と、元の走の記録（`results/chargpt_cl_1019/M2/<腕>/seed<k>/` の online.npz・layer_stats.csv・per_task.csv など）がある。束の checkpoint もここに展開する。**このブランチには commit・push しない**（別のセッションのブランチ）。tracked のファイルも書き換えない。
- **`$S10`** = ブランチ `claude/chargpt_seed10_1022` の checkout（git worktree で作る）。spec・この手順書・新しい script・結果を置き、ここにだけ commit する。

このマシンには 4 ブロック本走で使った lop_analysis の clone がある（remote が github.com/Issan0511/lop_analysis）。見つからなければ新しく clone する。

```bash
cd <lop_analysis の clone>              # 無ければ git clone https://github.com/Issan0511/lop_analysis.git
git status --short                       # 変更が残っていたら止めて報告する
git fetch origin
git checkout claude/chargpt_cl_1019 && git pull --ff-only
git rev-parse HEAD                       # 033679e0f1a3… であること
export CG=$(pwd)
git worktree add ../lop_seed10 claude/chargpt_seed10_1022
export S10=$(cd ../lop_seed10 && pwd)
export PY=$CG/.venv/bin/python
$PY -c "import sys,torch,numpy;print(sys.version.split()[0],torch.__version__,numpy.__version__,torch.cuda.get_device_name(0))"
```

版は python 3.12.x・torch 2.13.0+cu130・numpy 2.5.1 であること（4 ブロック本走と同じ .venv）。違ったら止めて報告する。

## 2. 束を受け取って照合する

- URL は Issa がプロンプトに貼る。中身・各ファイルの sha256・束の sha256 は `$S10/results/chargpt_seed10_1022/transfer/bundle_manifest.json`。
- ギガファイル便は**ファイルごとに cookie jar を分ける**。先にページ（`https://<n>.gigafile.nu/<id>`）を同じ jar で開いてから、同じ jar と Referer で `https://<n>.gigafile.nu/download.php?file=<id>`（鍵があれば `&dlkey=<key>`）を取る。jar を使い回すと 570 byte の HTML が返る。

```bash
mkdir -p $S10/_incoming && cd $S10/_incoming          # _incoming は commit しない
curl -s -c jar.txt -o /dev/null "<ページの URL>"
curl -L -b jar.txt -e "<ページの URL>" -o chargpt_seed10_1022_bundle.tar "<download.php の URL>"
sha256sum chargpt_seed10_1022_bundle.tar              # manifest の tar_sha256 と一致すること
tar -xf chargpt_seed10_1022_bundle.tar -C $CG         # 中のパスは results/chargpt_cl_1019/... なので $CG の直下に展開する
```

展開後、manifest の全ファイルについて、`$CG/<path>` の bytes と sha256 が manifest と一致することを確かめる（1 つでも違えば止めて報告する）。束の中身:

| 中身 | 用途 |
| --- | --- |
| `results/chargpt_cl_1019/M2/{C3,ReLUcap}/seed{0..9}/ckpts/` の init.pt、課題末の task*.pt と task*_adam.pt（更新 100 の u100_* は入れていない） | A・B・C の入力 |
| `results/chargpt_cl_1019/quicklook/M2/C3/seed1000/ckpts/` の init.pt と task*.pt | A の照合（seed 1000 の保存値の出し直し） |
| `results/chargpt_cl_1019/quicklook/M/C3/seed1000/ckpts/task1.pt`・`task1_adam.pt` | B の照合（4 ブロック） |
| `results/chargpt_cl_1019/quicklook/rho/out/C3/t2_a_chain.json` | B の照合の保存値 |

## 3. 実装（`$S10/analysis/chargpt_seed10_1022/` に置く。`$CG` のファイルは書き換えない）

どの script も、道具は `$CG/src` を `sys.path` に入れて `import chargpt_cl_1019_kune as K; C = K.C` で使う（`CHARGPT_SRC` 環境変数で `$CG` を渡す）。元の script を写して変えるときは、写し元の path と git blob の hash を script の先頭に書き、変えた所だけ短く書く。

### 3.1 `load10.py`（A）

- 写し元: `$S10/analysis/chargpt_load_fold_1021/load/measure.py`（main 21989864）。定義（κ・cos²・Q・補題・帰無・G_own・G_full・自己形の符号）は変えない。
- 変える所: `SOURCE` を `$CHARGPT_SRC` に。`SEED` を引数に。置き場を `results/chargpt_cl_1019/M2/<腕>/seed<k>`（照合のときだけ `quicklook/M2/C3/seed1000`）に。腕は M2 の C3・ReLUcap。状態は init と課題末 1〜10・25・39〜50・89〜100。
- 足す所: 読み出し隣接の L1 に加えて L0 の G_own・G_full（記録）。生きた unit の数 n_live と |G_own| の中央値。実際の次の課題の 1 実現で出した s（`reanalyze_p07.py` の定義、記録・時間があれば）。
- 出力: `$S10/results/chargpt_seed10_1022/load/states.csv`（1 行 = 腕・seed・状態・層）と unit ごとの配列（大きければ git に入れない、§6）。
- `checks` の動作（spec §7 の A）: (i) seed 1000 の M2/C3 の 35 状態を出し直し、`$S10/results/chargpt_load_fold_1021/load/table_head.csv`・`table_units.csv` の M2/C3 の値と照合（追補 1 の許容）。(ii) 補題の恒等式・有限差分・変異対照（head の符号を反転すると κ < 0、一様目標を 1 クラスにすると G の符号が変わる）。(iii) 同じ seed の C3 と ReLUcap の init.pt と task1.pt がビットで同じ。

### 3.2 `rho10.py`（B）

- 写し元: `$CG/analysis/chargpt_cl_1019_quicklook/rho_split.py`（a6517244）。分割の定義・検査は変えない。
- 変える所: 定数 `SEED` を `--seed` 引数に（probe と鎖を作るのに使う）。`--nlayer`（2 か 4）を足して `C.L` を決める。`ROOT` を `$CHARGPT_SRC` に。
- 走らせ方: `--arm C3|ReLUcap --state <課題 k 末の .pt> --adam <課題 k 末の _adam.pt> --task k+1 --kind chain --nseq 32 --device cuda --jvp exact --seed <seed> --nlayer 2`。層は L0 と L1。k+1 は spec §4 の 10 課題。
- 照合（spec §7 の B）: 4 ブロック C3 seed 1000 の課題 1 末 → 課題 2（`--nlayer 4 --seed 1000`）を出し直し、`t2_a_chain.json` の各項と照合（追補 1 の許容）。
- 出力: `$S10/results/chargpt_seed10_1022/rho/<腕>/seed<k>/t<k+1>_a_chain.json`。

### 3.3 `revival10.py`（C）

- 写し元: `$CG/analysis/chargpt_cl_1019_quicklook/intervene_revival.py`（a970f12c）。腕の定義（control・lnex1・nocommon・gamma03f・gamma03）は変えない。
- 変える所: `ROOT` を `$CHARGPT_SRC` に。`--src` と `--seed` は既にある（`--src $CG/results/chargpt_cl_1019/M2/C3/seed<k>`）。
- 足す所: 各課題末に、固定の一様 probe（`C.uniform_probe_M(seed, ...)`、元の走の layer_stats.csv の `probe=uniform` と同じもの）でも `C.measure` を取り、L0・L1 の `frac_p_lt005` を記録する（今の script は開始課題の chain probe で測っている。両方残す）。control では、元の走の online.npz との一致（ビット一致の真偽、課題ごとの差の最大）と、元の layer_stats.csv（u=10000、probe=uniform）との L1 の死の差を記録する。
- 走らせ方（環境変数 `CHARGPT_NLAYER=2`）:
  - C1（C3-2 の 9 seed、開始 S_k と終わり E_k は spec §5 の表）: control・lnex1・nocommon。
  - C2（C3-2 の 10 seed、課題 25 末から 31 まで）: control・gamma03f・gamma03。
  - C3（記録・任意）: ReLUcap-2 の 10 seed（開始と R_k は spec §5 の C3）: control・lnex1。
- 出力: `$S10/results/chargpt_seed10_1022/revival/<C1|C2|C3>/seed<k>/<腕>/`（script の出力そのまま）。

### 3.4 `judge.py`

- spec の §3（A1・A2・A3）、§4（B）、§5 と追補 1（C1・C2）の規則を、そのまま実装する。ラベル・seed ごとの数・符号検定の p・記録の量を `$S10/results/chargpt_seed10_1022/judge.md` と `judge.json` に書く。
- 予測（spec の表）は採点だけする。予測の数を書き換えない。

### 3.5 `run_silu4.py`（D、指示があるときだけ）

- `$CG/src/chargpt_cl_1019_kune.py` を import して、保存の課題の集合（`C.CKPT_TASKS` と `K.DECOMP_TASKS`）に spec §6 の課題（課題末 11〜15・59〜70、更新 100 は 11〜15・60〜70）を足してから、その `main` を呼ぶ薄い wrapper。元のファイルは書き換えない。
- 走らせ方: `CHARGPT_NLAYER=4 CHARGPT_DECOMP_CKPT=1 $PY run_silu4.py run --box M --arm SiLU --seed <k> --tasks 100 --n_per_task 10000 --alpha 0.01 --fresh_tasks 10 25 50 100 --out $S10/results/chargpt_seed10_1022/silu4/SiLU/seed<k>`（seed 0〜9）。
- 走る前に、データ列が本走 4 ブロックと同じことを確かめる（`$CG/results/chargpt_cl_1019/M/C3/seed<k>/per_task.csv` の data_sha と、短い試走の per_task.csv の data_sha）。

## 4. 検査（全部通るまで本走しない）

spec §7 と追補 1 の検査を、`checks` として 1 つにまとめて回し、`$S10/results/chargpt_seed10_1022/checks/` に結果を書く。

- A: 3.1 の (i)〜(iii)。
- B: 3.2 の照合と、rho_split の検査（GW_vs_autograd・jvp_vs_backward・diag_direct・転置の変異）。
- C: 開始状態（S_k 末と課題 25 末）の一様 probe の L1 の死が、元の layer_stats.csv の値と一致すること。lnex1 の開始時の平均の分母が元と一致すること（定数の検査）。control の短い再生（1 課題）が動き、元の記録との差を出せること。
- どれかが落ちたら本走せず、落ちた検査と数を報告する。

## 5. 本走の順番

1. A（CPU、`torch.set_num_threads(8)` のまま）と C1・C2（GPU）を同時に始めてよい。GPU は同時 2 本まで（16 GB。1 枚に何本も載せても合計の速さはあまり増えない）。
2. C1・C2 が終わったら B（GPU）。
3. C3 は B のあと、時間があれば。
4. D は指示があるときだけ、最後に。

完走ごとに、かかった時間を `provenance.json` に残す。止まったら同じコマンドで再開できるように、完走した単位（seed × 腕）は飛ばす作りにする。

## 6. 結果の commit と push

- commit するのは `$S10` だけ。置き場は `results/chargpt_seed10_1022/{load,rho,revival,silu4,checks}/`、`judge.md`・`judge.json`、script は `analysis/chargpt_seed10_1022/`。
- 1 ファイル 20 MB を超えるもの（unit ごとの大きな配列、`*.pt`）は git に入れない。`results/chargpt_seed10_1022/raw_manifest.json`（path・bytes・sha256）に書いて、このマシンに残す。Issa が別に取りに来る。
- push の前に `git pull --no-rebase`。rebase・amend・force push はしない。`_incoming/` は commit しない。
- commit は節目ごと（検査、A、C、B、判定）に分ける。

## 7. 報告（日本語）

節目ごとに短く報告する。最後の報告に入れるもの:

- 判定のラベル（A1・A2・A3、B、C1・C2）と、seed ごとの数、符号検定の p。
- 予測の採点（spec の表の Opus 5.5 の欄。Issa の欄は空欄のまま）。
- 検査の結果。追補 1 の C の差（再生した control と元の記録の差）。
- spec から外れたこと（あれば）と、その理由。
- push した commit の hash と、git に入れなかった大きなファイルの置き場。

## 8. してはいけないこと

- spec の判定・閾値・腕・窓・開始点・予測を変えない。走る前に変える必要が分かったら、追補 2 として spec に書いて commit してから走らせ、報告する。走った後は変えずに報告する。
- `claude/chargpt_cl_1019` に commit・push しない。`$CG` の tracked のファイルを書き換えない。
- torch.compile・TF32 を入れない。
- D は指示がなければ回さない。
- Obsidian の vault には書かない（white-san 側で書く）。

## 9. 向こうの Claude Code に貼るプロンプト

```text
lop_analysis のブランチ claude/chargpt_seed10_1022 にある
specs/chargpt_seed10_1022_runbook.md（手順書）と specs/spec_chargpt_seed10_1022.md（spec、追補 1 まで）に従って、
文字 GPT の 1 seed の項目を 10 seed で確かめる解析 chargpt_seed10_1022 を、このマシンで回してください。

- 入力の checkpoint の束（ギガファイル便）: <ページの URL>
- 回す部分: A・B・C（D は回さない）
- 手順は 1 置き場 → 2 束の受け取りと照合 → 3 実装 → 4 検査（全部通るまで本走しない）→ 5 本走 → 6 commit・push → 7 報告。
- spec の判定・閾値・窓・開始点・予測は変えないこと。変える必要が出たら、走る前に追補 2 として spec に書いて commit し、報告すること。
- ブランチ claude/chargpt_cl_1019 には commit・push しないこと（別のセッションのブランチ）。
- 検査が落ちたら、走らせずに落ちた検査と数を報告すること。
- 報告は日本語で。節目（検査・A・C・B・判定）ごとに短く。
```
