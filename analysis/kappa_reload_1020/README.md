# kappa_reload_1020 — κ の再装填（事後解析・未登録）

依頼: Fable（V13 の親セッション）経由の Issa「H1（学んだ読み出し κ が課題ごとに育ち直す）を実測に」（2026-10-20）。
入力: `obsidian-research-data/push_direction_proof_0930/states/{LR,R,ELU,SILU,GELU}_K10_s{0,1}.npz`（1 隠れ層 RL-MNIST、課題末と +200 更新の重み、30 課題。生成は同フォルダの `gen_states.py`）。新しい学習は回していない。

| ファイル | 中身 |
|---|---|
| `measure.py` | κ_k の当てはめと分解（偶然・特徴の変化・読み出しの変化）、切替の押し 3 種（自己形 S_self・旧ラベルの的 S_oh・新ラベル平均 S_exact）、実ラベルの全バッチ GD 一歩、Adam の最初の 200 更新の沈み、残り 3,800 更新の戻り、p = 1/2 と m = 0 を跨ぐ数。→ `results/kappa_reload_1020/rows.json`・`runs.json`・`raw/units_*.npz` |
| `check_closed_form.py` | GD 一歩と期待の一歩の閉じた式を autograd と照合（相対誤差 ≤ 2.4e-15）。変異対照（K_n の +1 を落とす）は 2.5〜4.3% の誤差で検出される |
| `summarize.py` | 帯ごとの表 `summary.md`、全行 `table_switch.csv`、図 `kappa_push_return.png` |

実行（CPU、約 40 秒）:

```bash
OMP_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/kappa_reload_1020/check_closed_form.py
OMP_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/kappa_reload_1020/measure.py
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/kappa_reload_1020/summarize.py
```

計算前の予測は side-session-notes の `kappa_reload_1020_predictions.md`（写しを `results/kappa_reload_1020/PREDICTIONS.md` に置いた）。Adam の状態は保存されていないので、Adam の最初の 1 歩は測れない。代わりに実ラベルの全バッチ GD 一歩を測った。
