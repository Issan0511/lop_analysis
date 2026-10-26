# ViT の走（追補 E、byakko の分）を手で回す手順

2026-10-26、中塚。先生の PC（byakko）で、2 ブロックの ViT の走を 30 本回していただく手順です。貼るコマンドは 5 行で、途中の判断は要りません。何か出たら、出た文をそのまま私に送ってください。

## 1. これは何か

- 可塑性の喪失の実験です。Tiny ImageNet の同じ 2,500 枚に、課題ごとに乱数ラベルを引き直して覚えさせます。1 課題 5,000 更新を 100 課題続けます。
- 腕は 4 つです。head を課題ごとに 0 に戻す／残す × 正規化が LayerNorm／RMSNorm。
- byakko の分は、本走 10 本（seed 3 つ分）と、新品の網で 1 課題だけ覚えさせる短い走 20 本です。白い機械（私の PC）で残りを同時に回しています。
- 時間は 2 日前後です。2 本ずつ並べて走り、GPU は 2 本だけ使います。

## 2. 要るもの

- 文字 GPT のときに使った `lop_analysis` の clone と、その中の `.venv`（python 3.12、torch 2.13.0+cu130）。
- 空きのあるフォルダが 2 つ。出力に 8 GB、データに 1 GB です。
- 下の `<clone>`・`<出力>`・`<データ>` を、実際のパスに置き換えてください。例:
  - `<clone>` = `/home/kubo/project/Nakatsuka/lop_analysis`
  - `<出力>` = `/home/kubo/project/Nakatsuka/vit_e_out`
  - `<データ>` = `/home/kubo/project/Nakatsuka/datasets`

## 3. 貼るコマンド（この順に）

### 3.1 取り出す（1 回だけ）

```bash
cd <clone> && git fetch origin && git worktree add ../lop_vit_rlfix_e origin/claude/vit_long_1024 --detach && cd ../lop_vit_rlfix_e
```

- `<clone>` の隣に `lop_vit_rlfix_e` というフォルダができ、その中に入ります。以後のコマンドはこの中で流します。
- 別の端末を開いたときは、`cd <clone>/../lop_vit_rlfix_e` で戻れます。

### 3.2 準備と検査（1 回だけ、10〜15 分）

```bash
bash analysis/vit_long_1024/remote/byakko.sh setup <clone> <出力> <データ>
```

- 版の確認、データの取得と照合、動作の検査をします。
- 最後に `SETUP_OK` と出たら次へ進んでください。
- `SETUP_FAILED: ……` と出たら、その行を私に送ってください。先へは進まないでください。

### 3.3 起動（1 回）

```bash
bash analysis/vit_long_1024/remote/byakko.sh start <clone> <出力> <データ>
```

- 本走が背景で始まります。端末を閉じて構いません。
- PC を再起動したときは、同じコマンドをもう一度流してください。終わった走は飛ばし、途中の走は課題の切れ目から続きます。

### 3.4 様子を見る（1 日 1 回でよいです）

```bash
bash analysis/vit_long_1024/remote/byakko.sh status <clone> <出力> <データ>
```

- 腕ごとに「終わった課題数 / 100」が出ます。
- `止める規則に当たった` と出たら、下の 3.6 の stop を流して、出た文を私に送ってください。
- `終了コードが 0 でない走がある` と出たら、その文を私に送ってください。
- `ALL_DONE` と出たら全部終わりです。3.5 へ。

### 3.5 送る（全部終わったら）

```bash
bash analysis/vit_long_1024/remote/byakko.sh send <clone> <出力> <データ>
```

- 結果を 1 つにまとめ（約 7 GB）、ギガファイル便に上げます。数十分かかります。
- 最後に URL と `sha256` の行が出ます。その 2 つを私に送ってください。
- 送ったあとも、`<出力>` の中身は私が受け取りを確かめるまで消さないでください。

### 3.6 止めたいとき

```bash
bash analysis/vit_long_1024/remote/byakko.sh stop <clone> <出力> <データ>
```

- 走っている課題が終わり次第止まります（数分）。再開は 3.3 の start です。

## 4. していただきたくないこと

- フォルダの中のファイルを書き換えること、`git commit`・`git push` をすること。
- 走を足したり、GPU に別の重い計算を同時に載せたりすること（2 本ずつで回っています）。
- 途中の出力のフォルダを消すこと。

## 5. 何か変なとき

- どの段階でも、画面に出た文をそのまま送っていただければ、こちらで読みます。
- `setup` で「python 3.12・torch 2.13.0+cu130 が要る」と出た場合は、`.venv` が別のものを指しています。`<clone>` のパスを確かめてください。
- `status` で「走っている数: 0」なのに `ALL_DONE` が出ていないときは、何かで止まっています。3.3 の start をもう一度流してください。
