#!/usr/bin/env python3
"""Read-only posthoc audit of archived stage1 S/P/H0 branches; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

RAW = Path('/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw')
REPO = Path('/home/issan/Projects/claude/proj_004_drift')
ACTS = ('R', 'ELU', 'SILU')
TASKS = (2, 11, 30)
ARMS = ('S', 'P', 'H0', 'fa')
GRID = np.arange(0, 4001, 100)
SNAPSHOTS = (0, 100, 200, 500, 1000, 2000, 3000, 4000)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ci(x):
    x = np.asarray(x, dtype=float)
    assert len(x) == 10 and np.isfinite(x).all()
    rng = np.random.default_rng(20261111)
    b = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(1)
    lo, hi = np.quantile(b, [.025, .975])
    return f'{x.mean():.6f} [{lo:.6f}, {hi:.6f}]'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('/tmp/p_residual_existing_audit.md'))
    p.add_argument('--json', action='store_true', help='also print all seed records and full mean curves as JSON to stdout')
    args = p.parse_args()
    data, sources, body_checks, unit_eligibility = {}, [], [], {}
    for act in ACTS:
        for seed in range(10):
            path = RAW / f'branch_adam_lr0.001_{act}_s{seed}.npz'
            meta_path = path.with_suffix('.json')
            meta = json.loads(meta_path.read_text())
            assert meta['steps'] == 4000 and meta['tasks'] == 30
            sources.append(dict(path=str(path), sha256=sha(path), metadata_sha256=sha(meta_path),
                                source_git_head=meta['git_head'], engine_sha256=meta['engine_sha256']))
            with np.load(path) as d:
                for t in TASKS:
                    su = d[f'S|t{t:02d}|units'][0]
                    unit_eligibility[act,t,seed] = int((su[2] > 0).sum())
                    for arm in ARMS:
                        key = f'{arm}|t{t:02d}|'
                        ev = d[key + 'eval']
                        assert ev.shape == (41, 2) and np.isfinite(ev).all()
                        assert d[key + 'online_acc'].shape == (40,)
                        assert d[key + 'online_loss'].shape == (40,)
                        assert d[key + 'loss_first'].shape == (200,)
                        assert np.array_equal(d[key + 'upts'], [0,1,10,50,200,1000,4000])
                        if arm in ('P', 'H0'):
                            same = np.array_equal(su, d[key + 'units'][0])
                            body_checks.append(same)
                            assert same, (act,t,seed,arm)
                        # Raw accuracy is float32. This is the exact float32
                        # representation of >= 1080 / 1200 correct inputs.
                        hits = np.flatnonzero(ev[:, 0] >= np.float32(.9))
                        t90 = int(GRID[hits[0]]) if len(hits) else None
                        data[act,t,seed,arm] = dict(activation=act, task=t, seed=seed, arm=arm,
                            source_npz=str(path), source_key=key+'eval',
                            accuracy_end=float(ev[-1,0]), ce_end=float(ev[-1,1]),
                            t90_grid=t90, censored=t90 is None, censor_horizon=4000,
                            reached_then_below=bool(len(hits) and ev[-1,0] < np.float32(.9)),
                            eval=ev.astype(float).tolist())
                    fresh = data[act,t,seed,'fa']['accuracy_end']
                    for arm in ARMS:
                        data[act,t,seed,arm]['deficit'] = fresh - data[act,t,seed,arm]['accuracy_end']

    def recs(act,t,arm):
        return [data[act,t,s,arm] for s in range(10)]
    def vals(act,t,arm,field):
        return np.array([r[field] for r in recs(act,t,arm)])
    def mean(act,t,arm,field):
        return vals(act,t,arm,field).mean()

    L = ['# POSTHOC EXISTING DATA — P residual versus H0 audit', '',
         '新規訓練なし。既存 branch の Adam lr=0.001、R/ELU/SiLU、seed 0–9、課題 2/11/30。Pall は読んでいない。', '',
         '## 要点', '',
         '- **課題11 ELUが、Pを残す場合と読み出しを外す場合の差を最も明瞭に示す。** P/H0の末accuracy=.7024/.9639、末不足=.2587/−.0028、CE=.8430/.2581。対応不足差P−H0=.2615 [95% CI .2449,.2785]。P/Sは全10seedでT90_grid未到達、H0は全10seed到達（平均3220、3000–3400）。P−H0の全seed平均T90差は未同定だが、保存gridに基づく差の平均下界は >780更新。',
         '- **課題2では読み出しを消す方がPより遅い。** T90差P−H0はR −1170、ELU −1610、SiLU −1480更新。したがって「Pで残る部分は常に読み出しを外せば消える」という一般化は成立しない。',
         '- 課題11 R/SiLUのP−H0末不足差は+.0468/+.1523だが、両腕とも全seedでT90未到達。課題30 ELUでも末不足差+.3733は大きいが、H0末不足.2418と90%未到達が残る。到達時間として何更新消えたかは4000更新窓内では分からない。',
         '- 課題30 ReLUは固定1200入力上で初期開口unitが全seedゼロ。P/S/H0の末accuracyはすべて.1156。これはこの状態・入力集合・予算の境界であり、永久的不能の主張ではない。',
         '- 課題30 SiLUはP→H0で末accuracyが.3615→.4204となる一方、CEは1.6761→1.6917。CE差P−H0の95%区間は[−.0497,.0219]。accuracy改善とCE改善を同一視しない。',
         '- 以上は単発branchの操作差。ViTの装填費用と同種か、またP学習後のbodyからheadを外して遅れが消えるかは、この既存監査だけでは未決着。新規loading_bridge recoveryは全Adamを消し、急性200更新後のbodyを使うため、本監査H0のT90と同じ推定対象ではない。', '',
         '## 1. 操作・時点・比較の意味', '',
         '- 課題番号 t はその課題内の更新。例えば t=11 は課題10末を共通出発点として課題11を学ぶ。loading_bridge の t_end=10 に対応する。',
         '- S/P/H0 は同一の body 重み・bias・Adam状態、同じ新ラベルと minibatch 順序から分岐。P/H0 の介入直後 units は S と180比較で完全一致。optimizer状態の同一性は snapshot/restore 実装に基づく。raw NPZはAdamそのものを保存していない。',
         '- P: hidden-unit 列ごとの独立クラス置換を head weight と head Adam m/v に施す。平均logitを保つbias補正。bias Adam・body Adam は保持。',
         '- H0: head weight/bias をゼロ化し、その両方のAdam stateを削除。body Adamは保持。bodyは凍結せず、残ったmomentumで初回から動き得る。H0A（全Adam消去）とは違う。',
         '- 同じ初期状態で更新を挟まない P→H0 は、P が変えた head/bias と対応optimizer状態を全消去するため直接 H0 と同一になる（ソース上の帰結）。しかし P の学習後に H0 を施す介入は、この既存分岐にはない。',
         '- P−H0 は「共通初期bodyから、Pを残す場合とhead+head Adamを消す場合の差」。重みだけの寄与、遅れの媒介率、ViTと同一機構であることはこの比較単独では分からない。', '',
         '## 2. 保存gridと打切り', '',
         '- eval=(41,2): 更新0,100,…,4000。列0=固定学習用1200入力のaccuracy、列1=平均CE。更新0は介入直後。held-out性能ではない。',
         '- online_acc/online_loss=(40,): 100 minibatchごとの更新前成績の平均。loss_first=(200,): 最初の200 minibatchの更新前CE。これらを全1200入力のevalと混ぜない。',
         '- unit統計: 0,1,10,50,200,1000,4000。push/conf0は更新0のみ。',
         '- T90_grid は eval 上で accuracy>=0.90 となる最初の記録更新数。全腕4000まで学習しているため到達後の終点も比較可能。到達後停止する新規loading_bridge recoveryとは違う。',
         '- 100更新gridの間に一時的に到達したかは不明。未到達は「保存grid上で4000まで未到達」であり、観測T90=4000へ代入しない。到達者だけの平均は全seed平均ではない。以下の区間はposthoc対応seed bootstrap 10000回の百分位95%区間（多重比較未補正）。',
         '- 末不足=同じseed/課題のfresh arm faの末accuracy−当該armの末accuracy。faは新規初期化した全体+optimizerで同じ課題を学ぶ基準。負の不足もそのまま保持。',
         '- ReLUの登録judgeは開口unit数に基づく課題打切りを使うが、本監査は依頼された固定課題を全seedで記述する。登録判定の再適用ではない。', '',
         '## 3. 終点（4000更新、10 seed平均）', '',
         '| act | task | arm | accuracy | deficit vs fa | CE |',
         '|---|---:|---|---:|---:|---:|']
    for act in ACTS:
        for t in TASKS:
            for arm in ARMS:
                L.append(f'| {act} | {t} | {arm} | {mean(act,t,arm,"accuracy_end"):.6f} | {mean(act,t,arm,"deficit"):.6f} | {mean(act,t,arm,"ce_end"):.6f} |')
    L += ['', '## 4. 対応差（正ならH0の成績が良い）', '',
          'Δdef=def(first)−def(H0)=acc(H0)−acc(first)。ΔCE=CE(first)−CE(H0)。不足の差はfresh基準が相殺される。', '',
          '| act | task | comparison | Δdef mean [95% CI] | ΔCE mean [95% CI] |',
          '|---|---:|---|---|---|']
    for act in ACTS:
        for t in TASKS:
            for arm in ('P','S'):
                a = vals(act,t,arm,'deficit')-vals(act,t,'H0','deficit')
                c = vals(act,t,arm,'ce_end')-vals(act,t,'H0','ce_end')
                L.append(f'| {act} | {t} | {arm}−H0 | {ci(a)} | {ci(c)} |')
    L += ['', '## 5. T90_grid 到達・打切り', '',
          '| act | task | arm | reached/10 | mean of reached only | min–max reached | end below90 after earlier reach |',
          '|---|---:|---|---:|---:|---|---:|']
    for act in ACTS:
        for t in TASKS:
            for arm in ARMS:
                rs=recs(act,t,arm)
                hits=[r['t90_grid'] for r in rs if r['t90_grid'] is not None]
                mn=f'{np.mean(hits):.1f}' if hits else '—'
                span=f'{min(hits)}–{max(hits)}' if hits else '—'
                L.append(f'| {act} | {t} | {arm} | {len(hits)}/10 | {mn} | {span} | {sum(r["reached_then_below"] for r in rs)} |')
    L += ['', '対応T90差は全10seedで両腕到達の場合だけ全seed平均を示す。そうでない場合は対応状態の件数を示す。片腕のみ打切りなら、打切り腕−到達腕の差には >4000−観測T90 という片側のgrid上限からの下界がある。両腕打切りでは順序も決められない。', '',
          '| act | task | comparison | both reached | first only censored | H0 only censored | both censored | all-seed T90 difference [95% CI] |',
          '|---|---:|---|---:|---:|---:|---:|---|']
    for act in ACTS:
        for t in TASKS:
            for arm in ('P','S'):
                x,y=recs(act,t,arm),recs(act,t,'H0')
                counts=[sum((a['t90_grid'] is None)==ac and (b['t90_grid'] is None)==bc for a,b in zip(x,y)) for ac,bc in ((False,False),(True,False),(False,True),(True,True))]
                diff=ci([a['t90_grid']-b['t90_grid'] for a,b in zip(x,y)]) if counts[0]==10 else 'not estimable as all-seed mean'
                L.append(f'| {act} | {t} | {arm}−H0 | '+ ' | '.join(map(str,counts))+f' | {diff} |')
    L += ['', 'seed順のT90_grid（0→9、`>`=grid上で4000まで未到達）:', '',
          '| act | task | S | P | H0 | fa | initial open units (S) |',
          '|---|---:|---|---|---|---|---|']
    for act in ACTS:
        for t in TASKS:
            vv=[]
            for arm in ARMS:
                vv.append(','.join(str(r['t90_grid']) if r['t90_grid'] is not None else '>' for r in recs(act,t,arm)))
            al=','.join(str(unit_eligibility[act,t,s]) for s in range(10))
            L.append(f'| {act} | {t} | '+' | '.join(vv)+f' | {al} |')
    L += ['', '## 6. 学習曲線の代表保存点（10seed平均）', '',
          '表は eval の代表点のみ。スクリプトの --json は全41時点の平均曲線とseed別全evalをstdoutへ出す。', '',
          '| act | task | update | acc S | acc P | acc H0 | acc fa | CE S | CE P | CE H0 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    full_curves=[]
    for act in ACTS:
        for t in TASKS:
            curves={arm:np.array([r['eval'] for r in recs(act,t,arm)]).mean(0) for arm in ARMS}
            for arm in ARMS:
                full_curves.append(dict(activation=act,task=t,arm=arm,updates=GRID.tolist(),mean_eval=curves[arm].tolist()))
            for step in SNAPSHOTS:
                ix=step//100
                aa=' | '.join(f'{curves[arm][ix,0]:.4f}' for arm in ARMS)
                cc=' | '.join(f'{curves[arm][ix,1]:.4f}' for arm in ('S','P','H0'))
                L.append(f'| {act} | {t} | {step} | {aa} | {cc} |')
    L += ['', '## 7. 出典・再現', '',
          f'- 閲覧したrepo HEAD: `{subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()}`。',
          '- 原raw生成commitは全30ファイルで `5773944b1a0484cc0c3797937413afe305a1dcb9`（sidecar JSON）。',
          f'- raw root: `{RAW}`。各数値は `branch_adam_lr0.001_{{R,ELU,SILU}}_s{{0..9}}.npz` の `{{S,P,H0,fa}}|t{{02,11,30}}|eval`。末値は行40、T90は行0–40の列0。CEは列1。',
          f'- source: `{REPO}/src/stage1_ctrl_1111.py`:44(grid),151(snapshot/restore),212(evaluate),220(run_task),265(bias compensation),290(P),309(H0),356(branch),385(S)。',
          '- P/H0の初期unit一致180比較、eval360配列のshape/finite、online/loss/uptsのgrid検査を通過。',
          '- 再現: `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python /tmp/p_residual_existing_audit.py`。出力は本報告のみ。`--json`追加で全計算結果をstdoutへ出せる。',
          f'- 集計script SHA256: `{sha(Path(__file__))}`。', '',
          '### 元ファイルSHA256', '', '| file | NPZ SHA256 | JSON SHA256 |', '|---|---|---|']
    for s in sources:
        L.append(f'| {Path(s["path"]).name} | {s["sha256"]} | {s["metadata_sha256"]} |')
    args.out.write_text('\n'.join(L)+'\n')
    result=dict(report=str(args.out),records=len(data),initial_body_checks=len(body_checks),sources=sources)
    if args.json:
        result.update(records=list(data.values()),mean_curves=full_curves)
    print(json.dumps(result if args.json else {k:v for k,v in result.items() if k!='sources'}, ensure_ascii=False))


if __name__=='__main__':
    main()
