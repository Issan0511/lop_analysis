# reload_dial_1020 — 課題 1 を当てはめた自信と切替の押し（登録走）

spec: `specs/spec_reload_dial_1020.md`（Claude の予測 d687f97e、Issa の予測 07628c01、どちらも段 switch の前）。
engine: `src/reload_dial_1020.py`（`push_direction_proof_0930/gen_states.py` と同じ箱と乱数列）。

| ファイル | 中身 |
|---|---|
| `src/reload_dial_1020.py --stage replay` | 課題 1 を再生し、精度 20・50・80・95% と初期値・課題末で θ と Adam 状態を保存。保存済みの mid[0]・end[0]・mid[1] と照合（`results/reload_dial_1020/checks.json`、全 10 走でビット一致） |
| `src/reload_dial_1020.py --stage switch` | 各 checkpoint から新ラベル 5 通りへ切り替えて 200 更新。1〜200 更新後の位置 m と開いた割合 p |
| `analyze.py` | checkpoint ごとの自信・κ・揃い A・κ の下界、期待の押し・自己形、draw ごとの GD 一歩・Adam 1 歩・200 更新 → `rows.json`。kappa_reload_1020 の 30 課題の課題末でも恒等式と下界を検査 → `bound_30tasks.json` |
| `summarize.py` | `summary.md`・`dial.png` |
| `extra.py` | 単調性、押し / 損失の下がり幅、下界が 0 以下の状態、Adam の沈みの時間変化 → `extra.json`・`extra.md` |

実行（CPU、合わせて 2 分ほど。段 replay と switch は 10 走を並列）:

```bash
OMP_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python src/reload_dial_1020.py --stage replay --act LR --seed 0
OMP_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python src/reload_dial_1020.py --stage switch --act LR --seed 0 --draws 5
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/reload_dial_1020/analyze.py
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/reload_dial_1020/summarize.py
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/reload_dial_1020/extra.py
```

生データ（checkpoint と switch の npz）は `obsidian-research-data/reload_dial_1020/`（`results/reload_dial_1020/backup_manifest.json`）。
