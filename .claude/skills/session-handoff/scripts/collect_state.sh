#!/usr/bin/env bash
# collect_state.sh <repo> <vault> — 引き継ぎノートに貼る「状態の要約」を Markdown で出す。
# 数値は git と results/*/arm_status から機械的に拾う（チャットの言い回しを出所にしない）。
set -u
repo="${1:?repo}"; vault="${2:?vault}"
echo "### 機械的な状態（$(date '+%Y-%m-%d %H:%M')・$(hostname)）"
echo
echo "**repo** \`$repo\` — $(git -C "$repo" rev-parse --abbrev-ref HEAD) @ $(git -C "$repo" rev-parse --short HEAD)（origin と$(git -C "$repo" status -sb | head -1 | grep -q 'ahead\|behind' && echo '差あり' || echo '同期')）"
n=$(git -C "$repo" status --short | wc -l); echo "- 未 commit: $n 件$( [ "$n" -gt 0 ] && echo ' → ' && git -C "$repo" status --short | head -5 | sed 's/^/`/;s/$/`/' | paste -sd' ' )"
echo "- 直近 commit:"; git -C "$repo" log -8 --format='  - `%h` %s' | cat
echo
echo "**vault** \`$vault\` — @ $(git -C "$vault" rev-parse --short HEAD)"
n=$(git -C "$vault" status --short | wc -l); echo "- 未 commit: $n 件（他セッションのものは触らない）"
git -C "$vault" status --short | head -5 | sed 's/^/  - `/;s/$/`/'
echo
echo "**results/**（arm_status から）"
echo "| run | 腕 | 完走 | 発散/失敗 | サイズ |"; echo "|---|---|---|---|---|"
for d in "$repo"/results/*/; do
  [ -d "$d/arm_status" ] || continue
  tot=$(ls "$d/arm_status" 2>/dev/null | wc -l)
  ok=$(grep -l '"status": *"COMPLETE"\|"status": *"COMPLETE_WITH_EXCLUSIONS"\|"status": *"RECOVERED"' "$d"/arm_status/*.json 2>/dev/null | wc -l)
  div=$(grep -L '"status": *"COMPLETE"\|"status": *"COMPLETE_WITH_EXCLUSIONS"\|"status": *"RECOVERED"' "$d"/arm_status/*.json 2>/dev/null | wc -l)
  echo "| \`$(basename "$d")\` | $tot | $ok | $div | $(du -sh "$d" | cut -f1) |"
done
echo
echo "**走っているプロセス**"
pgrep -af 'python.* -m src\.|python.* src/' | grep -v pgrep | sed 's/^/- `/;s/$/`/' | head -8
[ -z "$(pgrep -af 'python.* -m src\.|python.* src/' | grep -v pgrep)" ] && echo "- なし"
echo
echo "**メモリ索引**（\`~/.claude/projects/<proj>/memory/MEMORY.md\`）"
mem="$HOME/.claude/projects/$(echo "$(dirname "$repo")" | sed 's#/#-#g')/memory/MEMORY.md"
[ -f "$mem" ] && sed 's/^/> /' "$mem" || echo "- （見つからない: $mem）"
