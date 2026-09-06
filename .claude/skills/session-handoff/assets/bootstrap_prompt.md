（受け側の Claude Code に貼る。<> を埋める）

前のセッション（<送り側マシン>・チャット `<チャット名>`）の続きを、この環境で引き継いでください。

1. `git -C <repo> pull` と `git -C <vault> pull`（vault が無ければ `git clone <vault remote>`）。
2. `<vault>/可塑性喪失/引き継ぎ/<folder>/<ノート>.md` を読み、その §1「読む順」のノートを**全部**読む。
3. `python3 <repo>/.claude/skills/session-handoff/scripts/receive.py --manifest <vault>/可塑性喪失/引き継ぎ/<folder>/manifest.json --results-dir <repo>/results [--only <最初に要る項目>]` でデータを落として照合・展開（`pip install gigafile` が要る）。
4. 同フォルダの `memory/*.md` を自分の memory dir にコピーして `MEMORY.md` を統合する（パスはこの環境に読み替える）。
5. 受領確認（今の絵・次の一手・踏まない線を自分の言葉で各 3 行以内＋最初の行動 1 つ）を書いてから、§4 の一手に着手する。
