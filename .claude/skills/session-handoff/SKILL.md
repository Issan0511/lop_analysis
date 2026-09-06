---
name: session-handoff
description: 飽和したセッションの文脈を、別マシンや新しいセッションの Claude Code に引き継ぐための手順。「引き継ぎ」「乗り換え」「別マシンで続き」「コンテキストが飽和」「handoff」「このセッションを閉じて続きは向こうで」「ギガファイルで結果を送って引き継ぐ」と言われたら、頼まれた形が曖昧でもこの skill を使う。送り側（引き継ぎノート・記憶の複製・データの束ね＋ギガファイル便）と受け側（受領・sha256 照合・展開・記憶の移植・受領確認）の両方をカバーする。
---

# セッション引き継ぎ（session-handoff）

## なぜこの形か

チャットの文脈は 4 層でできている。**どれか 1 層だけ渡しても続きは書けない**。

| 層 | 中身 | 運び方 |
|---|---|---|
| 1. 正本 | 研究の状態・判定・数値（vault の結果ノート・spec・`results/*/verdict.csv`） | vault（git）。引き継ぎノートから `[[wikilink]]` で指す。**転記しない**（数値の出所は正本のまま） |
| 2. セッション固有の状態 | 今どこまで来て・何が走っていて・次の一手は何で・何を踏んではいけないか | **引き継ぎノート**（vault の `引き継ぎ/` に 1 本・テンプレは `assets/handoff_template.md`） |
| 3. 記憶 | `~/.claude/projects/<proj>/memory/*.md`（ユーザーの流儀・落とし穴・マシン情報） | vault の引き継ぎフォルダに **ファイルごと複製**（受け側は自分の memory dir に移植） |
| 4. 生データ | `results/<run>/logs/*.npz`・ckpt・scratch のスクリプトと図・transcript | ギガファイル便（`scripts/pack_upload.py`）。manifest.json（sha256 付き）を vault に置く |

vault の [[現在地]] は「新しいチャットへの引き継ぎは 現在地 → 引用禁止 → 運用ルール の 3 本」と定めている。引き継ぎノートはそれを**置き換えない**——3 本の上に、この一連のセッションで積んだ層 2〜4 を足すもの。だから 現在地 には **1 行のリンクだけ**足す（[[vault-parallel-session-collision]]: 他セッションが同じファイルを編集中でも衝突しない）。

## 送り側の手順

1. **状態を機械的に拾う**: `bash scripts/collect_state.sh <repo> <vault>` の出力をノートの「機械的な状態」節に貼る（git の HEAD・未 commit・results の完走/発散・走っているプロセス・記憶の索引）。チャットの言い回しを出所にしない。
2. **repo と vault を push 済みにする**。commit されていないものは相手に届かない。他セッションの未 commit 変更（`git status` に出るが自分が触っていないもの）は **触らない・commit しない**。
3. **束ねて上げる**（背景で・長い）:
   ```bash
   nohup python3 scripts/pack_upload.py --out <scratch>/handoff_out --resume \
     "handoff_<date>_bundle=gz:<staging dir>" \
     "<run>=<repo>/results/<run>" ... > <scratch>/pack_upload.log 2>&1 &
   ```
   - 束（bundle）には transcript の gzip・`memory/*.md`・scratch のスクリプトと図を入れる。**小さい束を最初に**上げてパイプラインを確かめ、6 GB 級は最後。
   - データ tar はトップレベルが run 名になる（`-C parent name`）。受け側は `results/` に展開するだけ。
   - npz は既に deflate 済みなので無圧縮 tar（zstd で 2% しか縮まない）。
   - 1 ファイル 1 プロセスで上げる（gfile の cookie jar を分ける。同一プロセスで 2 本目が壊れた実績）。
   - `--resume` で manifest にある項目は飛ばす。失敗したら同じコマンドを再実行。
4. **引き継ぎノートを書く**（`assets/handoff_template.md`）。置き場所は `<vault>/可塑性喪失/引き継ぎ/<date>_<題>_<宛先>/`。同じフォルダに `memory/`（複製）と `manifest.json`（URL 付き）を置く。ノート名は vault 全体で一意に（wikilink はファイル名で解決する）。
5. **[[現在地]] に 1 行**（「直近のセッション引き継ぎ: [[ノート名]]」）。[[運用ルール]] のフォルダ表に `引き継ぎ/` の行が無ければ足す。vault を commit → push。
6. ユーザーに **受け側に貼るプロンプト**（`assets/bootstrap_prompt.md` を埋めたもの）を渡す。URL は manifest から。

## 受け側の手順（プロンプトに書いてあるが、skill としても持つ）

1. `git pull` で repo と vault を最新にする（vault が無ければ clone）。
2. 引き継ぎノートを読む → 「読む順」の vault ノートを読む（**全部**。要約で済ませない）。
3. `python3 scripts/receive.py --manifest <vault>/.../manifest.json --results-dir results [--only ...]` でデータを落とし、sha256 を照合して展開。束は `_handoff_dl/<bundle>/` に展開される。
4. `memory/*.md` を自分の memory dir（system prompt に書いてある `~/.claude/projects/<proj>/memory/`）へコピーし、`MEMORY.md` を統合する。**送り側のパス（/home/issan/…）はそのままでは使えない**ので、マシン固有の記憶は自分の環境で読み替える。
5. **受領確認**を書く: 自分の言葉で「今の絵」「次の一手」「踏まない線」を各 3 行以内に言い直し、最初の行動を 1 つ宣言してから始める。言い直せない箇所があれば、そこが引き継ぎの穴。

## ノートを書くときの原則

- **判定・数値は正本へのリンクで済ませる**（転記するなら run id と `verdict.csv` を併記し、事後の値には「事後・未登録」を付ける）。
- 「今の絵」は**結論だけでなく、棄却したものと外れた予測も書く**。次のセッションが同じ仮説をもう一度立てるのが最大の無駄。
- 「踏んではいけない線」は**このセッションで実際に踏んだ／踏みかけた**ものを優先する。一般論は [[引用禁止]]・[[運用ルール]] に任せる。
- 進行中の計算（他セッション・他マシン）は **pid・run id・触ってよいか**を書く。
- 相手の環境で **効かないもの**（bit 一致・絶対パス・GPU 有無・ドライバ）を明示する。

## 落とし穴

- transcript は生きているので gzip 時に「file size changed」と出る。最後の数ターンは入らない——ノートに書いたことが正。
- memory dir はセッションの cwd から決まる（`-home-issan-Projects-claude` のように）。repo の cwd が違うマシンでは別の dir になる。
- bit 一致検査（S-null / S-mirror）を持つ走は参照ログを作ったマシンでしか成立しない（GCP で実測）。同じ Raptor Lake（AVX2）同士なら通ることがある（lab の ISA probe は MATCH）。
- ギガファイル便は URL を知っていれば誰でも落とせる。保存 100 日。削除キーは manifest にある。
- vault のフォルダ表・現在地は他セッションと同時編集になりやすい。**編集前に `git log -3 -- <file>`**、追加は別ノート＋ 1 行リンク。
