# worktree_cleanup_1023

2026-10-23 夜、Issa の依頼で、main に入った worktree 18 本とそのブランチ（手元と GitHub）を消した。

- 消す前に、各 worktree の git 外ファイル（__pycache__ を除く）を退避先と照合した。退避先に無かった 20 件は ~/Projects/obsidian-research-data/<worktree 名>/ へ移した（sha256 は移す前後で一致）。
- 一緒に消えたのは、__pycache__ と 0 バイトのロックファイル 6 個と、proj_004_drift/data を指すシンボリックリンクだけ。
- 消した worktree とブランチの先端の commit は backup_manifest.json にある。どれも main から辿れる。
- GitHub から消したブランチは 15 本。
- 生データの重複などは同じ夜に先に消した。経緯は ~/Projects/obsidian-research-data/_cleanup_1023/README.md。
