"""追補 F の判定（spec F.5）と予測の採点（F.6）の表を、本走の csv と posthoc_f の json から作る（Fable、1104）。

読むもの: runs/<S|SC|R0>/seed*/per_task.csv・layer_stats.csv、runs/<fresh|freshC>/seed*/task*/per_task.csv、
posthoc_f/j5_S_seed*.json（J F1）、r1_S_seed*_t*_{base,fix_ffn0}.json（J F2・J F3）、r2r3_S_seed*_t*_{base,center,scale}.json（J F5 の副）、
r4_R0_seed*_t60_{base,br_body,br_newhead1}.json と posthoc_r0_late/fable_R0_t60_hl001_*.json（seed 10252010、J FR0）。
書くもの: results/vit_long_1024_2blk_rlfix/addendum_f/judge.md（判定と採点）・judge.json。欠けている json は — で埋め、ある分だけで判定する。
規則は spec F.5 のまま。F.11 で SN を外したので J F5 は SC と R2・R3 で読む（本文の「読み方」に明記）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json python -m analysis.vit_long_1024.posthoc_2blk.fable_f_tables
"""
import csv
import json
import statistics as St
from pathlib import Path

import numpy as np

RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix'
RUNS, PF = RAW / 'runs', RAW / 'posthoc_f'
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vit_long_1024_2blk_rlfix/addendum_f'
SEEDS = [10252010 + i for i in range(5)]
UNR = 5025
WIN = dict(early=range(2, 12), mid=range(21, 41), late=range(41, 61))
TH = dict(s_load=.56, sink=.78, input_share=.74, r1_reduction=.4, path_dead=.2, path_zbar=.2, width_ratio=5., spearman=.8,
          r_depth=-1., q_loss=1.5, q_noloss=1.3, r2_change=.10, r3_change=.40, r0_dead=.6, r0_unreached=10, freeze_tail=.9,
          newhead_t90=500)


def fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if isinstance(v, bool):
        return '○' if v else '×'
    if isinstance(v, int):
        return f'{v:,}'
    return f'{100 * v:.0f}%' if pct else f'{v:,.{nd}f}'


def per_task(arm, seed, task=None):
    p = RUNS / arm / f'seed{seed}' / (f'task{task}' if task else '') / 'per_task.csv'
    rows = list(csv.DictReader(open(p)))
    return {int(r['task']): (float(r['T90']) if str(r['reached90']).strip().lower() in ('1', 'true') else UNR) for r in rows}


def layer2(arm, seed):
    out = {}
    for r in csv.DictReader(open(RUNS / arm / f'seed{seed}' / 'layer_stats.csv')):
        if r['state'] == 'end' and r['layer'] == '2':
            out[int(r['task'])] = {k: float(v) for k, v in r.items() if k != 'state'}
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def count(flags):
    return sum(1 for f in flags if f is True)


def label(flags, need, of=5):
    return f'{count(flags)}/{of}' + ('（成立）' if count(flags) >= need else '（不成立）')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    J, lines = {}, []
    md = lines.append

    # ---------- T90 の窓と Q ----------
    md('# 追補 F の判定（spec F.5）と予測の採点（F.6）')
    md('')
    md('規則は spec F.5 を走る前に固定したまま。T90 の未到達は 5,025。窓は早期 2〜11・中期 21〜40・後期 41〜60。Q = 後期の T90 の中央値 ÷ 同 seed の新品の課題 50 の T90。')
    md('F.11 で SN を止めたので、J F5 は SC と R2・R3 で読む（下の「読み方」）。欠けている走は — で、判定はある分だけで出す。')
    md('')
    md('## 0. 学び直し T90 の窓（中央値、括弧は未到達の数）と Q')
    md('')
    md('| 腕 | seed | 早期 2〜11 | 中期 21〜40 | 後期 41〜60 | 81〜100 | 新品 T90 課題 25 / 50 / 75 / 100 | Q |')
    md('|---|---|---|---|---|---|---|---|')
    Q = {}
    for arm, fresh in (('S', 'fresh'), ('SC', 'freshC'), ('R0', 'fresh')):
        for seed in SEEDS:
            t = per_task(arm, seed)
            fr = {k: per_task(fresh, seed, k)[k] for k in (25, 50, 75, 100)}
            cells = []
            for rg in (WIN['early'], WIN['mid'], WIN['late'], range(81, 101)):
                v = [t[k] for k in rg]
                cells.append(f'{St.median(v):,.0f} ({sum(x >= UNR for x in v)})')
            q = St.median([t[k] for k in WIN['late']]) / fr[50]
            Q[(arm, seed)] = dict(q=q, late_unreached=sum(t[k] >= UNR for k in WIN['late']), tail_unreached=sum(t[k] >= UNR for k in range(81, 101)))
            md(f"| {arm} | {seed} | " + ' | '.join(cells) + f" | {fr[25]:,.0f} / {fr[50]:,.0f} / {fr[75]:,.0f} / {fr[100]:,.0f} | {q:.2f} |")
    md('')

    # ---------- J F1 ----------
    md('## J F1 `LOADED`（S の課題末 t10・t30: κ > 0 かつ補題の下界が成立かつ s_load ≥ 0.56。3/5）')
    md('')
    md('κ は ⟨W^c, F⟩/‖F‖²（LN なしでは ‖F‖ が大きく κ の値は小さい。符号と補題が判定）。cos² は head 行と F の向きの一致。「1 歩」は Adam のある課題末から課題 t+1 の最初の batch で実際に 1 歩進めたときに自分の重みを通る変化が沈める側の unit の割合。')
    md('')
    md('| seed | t10: ⟨W^c,F⟩ / κ / cos² / 補題 / s_load / 1 歩 | t30: 同 | t60: cos² / s_load | t100: cos² / s_load | 判定 |')
    md('|---|---|---|---|---|---|')
    f1 = []
    for seed in SEEDS:
        d = load(PF / f'j5_S_seed{seed}.json')
        if not d:
            md(f'| {seed} | — | — | — | — | — |'); f1.append(None); continue
        R = {r['task']: r for r in d['results']}
        ok = all(R[t]['inner'] > 0 and R[t]['lemma_holds'] and R[t]['s_load'] >= TH['s_load'] for t in (10, 30) if t in R) and all(t in R for t in (10, 30))
        f1.append(bool(ok))
        cell = lambda t: (f"{R[t]['inner']:,.0f} / {R[t]['kappa']:.1e} / {R[t]['cos2']:.2f} / {fmt(bool(R[t]['lemma_holds']))} / {R[t]['s_load']:.2f} / "
                          f"{fmt(R[t].get('first_step', {}).get('frac_adam_sinks'))}") if t in R else '—'
        short = lambda t: f"{R[t]['cos2']:.2f} / {R[t]['s_load']:.2f}" if t in R else '—'
        md(f'| {seed} | {cell(10)} | {cell(30)} | {short(60)} | {short(100)} | {fmt(ok)} |')
    J['F1'] = dict(flags=f1, label='LOADED' if count(f1) >= 3 else 'not LOADED', count=count(f1))
    md('')
    md(f"**J F1: {label(f1, 3)} → {J['F1']['label']}**")
    md('')

    # ---------- J F2 / J F3 ----------
    md('## J F2 `SINKS_VIA_INPUT`（R1 の対照の更新 25: 生きた unit のうち沈む割合 ≥ 0.78 かつ Δz̄ の入力の分 ≥ 0.74。t10→11 と t30→31 の両方。3/5）')
    md('')
    md('Δz̄ は切替時に生きていた unit（固定 probe の token の 0.5% 以上で開く）の、固定 probe（全 token）での平均前活性の変化の中央値。自分の重み dw·m₀、入力 w₀·dm、交差 dw·dm、bias の分に厳密に割る。')
    md('')
    md('| seed | 切替 | 生 | 更新 25: 沈む割合 / Δz̄ 中央値（自分・入力・入力の分） | 更新 100: 沈む割合 / Δz̄（自分・入力・入力の分） | 死 0 → 25 → 100 | ‖m‖ 0 → 25 → 100 | 損失 0 → 25 → 100 |')
    md('|---|---|---|---|---|---|---|---|')
    f2, r1 = [], {}
    for seed in SEEDS:
        okseed = []
        for t in (10, 30):
            for mode in ('base', 'fix_ffn0'):
                r1[(seed, t, mode)] = load(PF / f'r1_S_seed{seed}_t{t}_{mode}.json')
            d = r1[(seed, t, 'base')]
            if not d:
                md(f'| {seed} | {t}→{t + 1} | — | — | — | — | — | — |'); okseed.append(None); continue
            pr = {q['step']: q for q in d['probes']}
            rec = {q['step']: q for q in d['records']}
            L25, L100 = pr[25]['ledger'], pr[100]['ledger']
            okseed.append(L25['sink_frac'] >= TH['sink'] and L25['input_share'] >= TH['input_share'])
            md(f"| {seed} | {t}→{t + 1} | {L25['alive']:,} | {L25['sink_frac']:.2f} / {L25['total_median']:+.2f}（{L25['weight_median']:+.3f}・{L25['input_median']:+.2f}・{L25['input_share']:.2f}）"
               f" | {L100['sink_frac']:.2f} / {L100['total_median']:+.2f}（{L100['weight_median']:+.3f}・{L100['input_median']:+.2f}・{L100['input_share']:.2f}）"
               f" | {pr[0]['dead005']:.2f} → {pr[25]['dead005']:.2f} → {pr[100]['dead005']:.2f} | {pr[0]['m_norm']:,.0f} → {pr[25]['m_norm']:,.0f} → {pr[100]['m_norm']:,.0f}"
               f" | {rec[0]['loss']:.1f} → {rec[25]['loss']:.2f} → {rec[100]['loss']:.2f} |")
        f2.append(None if any(o is None for o in okseed) else all(okseed))
    J['F2'] = dict(flags=f2, label='SINKS_VIA_INPUT' if count(f2) >= 3 else 'not SINKS_VIA_INPUT', count=count(f2))
    md('')
    md(f"**J F2: {label(f2, 3)} → {J['F2']['label']}**")
    md('')
    md('## J F3 `UPSTREAM_CARRIES`（R1: FFN0 の共通成分を固定した腕の沈みの減り = 1 − 固定/対照 が t10 で ≥ 0.4。4/5）・`PATH_BLOCKED`（SC の t30: 最終 FFN の死 ≤ 0.2 かつ |z̄| ≤ 0.2 × S。3/5）')
    md('')
    md('固定は FFN0 の出力 f を各 forward で f − mean(f) + c₀ に置き換える（c₀ は切替時の網の写しが同じ batch に出す平均。更新 1 の forward は対照と同じ）。')
    md('')
    md('| seed | 切替 | 対照 Δz̄ 25 / 100 | 固定 Δz̄ 25 / 100 | 減り 25 / 100 | 固定の ‖m‖ 0 → 25 → 100 | 固定の損失 0 → 25 → 100（batch 損失 2・5・10） | 固定の死 100 |')
    md('|---|---|---|---|---|---|---|---|')
    f3a = []
    for seed in SEEDS:
        for t in (10, 30):
            b, f = r1.get((seed, t, 'base')), r1.get((seed, t, 'fix_ffn0'))
            if not b or not f:
                md(f'| {seed} | {t}→{t + 1} | — | — | — | — | — | — |')
                if t == 10:
                    f3a.append(None)
                continue
            pb, pf = {q['step']: q for q in b['probes']}, {q['step']: q for q in f['probes']}
            rf, tl = {q['step']: q for q in f['records']}, f['train_loss']
            red = {s: 1 - pf[s]['ledger']['total_median'] / pb[s]['ledger']['total_median'] for s in (25, 100)}
            if t == 10:
                f3a.append(red[25] >= TH['r1_reduction'])
            md(f"| {seed} | {t}→{t + 1} | {pb[25]['ledger']['total_median']:+.2f} / {pb[100]['ledger']['total_median']:+.2f} | {pf[25]['ledger']['total_median']:+.2f} / {pf[100]['ledger']['total_median']:+.2f}"
               f" | {red[25]:+.2f} / {red[100]:+.2f} | {pf[0]['m_norm']:,.0f} → {pf[25]['m_norm']:,.0f} → {pf[100]['m_norm']:,.0f}"
               f" | {rf[0]['loss']:.1f} → {rf[25]['loss']:.1f} → {rf[100]['loss']:.1f}（{tl[1]:.0f}・{tl[4]:.0f}・{tl[9]:.0f}） | {pf[100]['dead005']:.2f} |")
    md('')
    md('| seed | SC t30: 死 / z̄ / |z̄| ÷ S / 幅 / 入口の共通成分 ‖m‖（S） | SC t10 死 | SC t60 死 / z̄ | SC t100 死 | 判定 |')
    md('|---|---|---|---|---|---|')
    f3b = []
    for seed in SEEDS:
        S, Cc = layer2('S', seed), layer2('SC', seed)
        ok = Cc[30]['dead005'] <= TH['path_dead'] and abs(Cc[30]['mu_med']) <= TH['path_zbar'] * abs(S[30]['mu_med'])
        f3b.append(bool(ok))
        md(f"| {seed} | {Cc[30]['dead005']:.3f} / {Cc[30]['mu_med']:+.2f} / {abs(Cc[30]['mu_med']) / abs(S[30]['mu_med']):.3f} / {Cc[30]['sigma_med']:.1f} / {Cc[30]['in_mean_norm']:.2f}（{S[30]['in_mean_norm']:,.0f}）"
           f" | {Cc[10]['dead005']:.3f} | {Cc[60]['dead005']:.3f} / {Cc[60]['mu_med']:+.2f} | {Cc[100]['dead005']:.3f} | {fmt(ok)} |")
    J['F3'] = dict(upstream_flags=f3a, upstream=label(f3a, 4), path_flags=f3b, path=label(f3b, 3))
    md('')
    md(f"**J F3 `UPSTREAM_CARRIES`（減り 25 更新、t10）: {label(f3a, 4)}。`PATH_BLOCKED`: {label(f3b, 3)}。**")
    md('')

    # ---------- R1 補: 書き手の分解（事後の測定、判定には使わない） ----------
    md('## R1 の補い: 切替の直後に最終 FFN の入力の共通成分 m を動かす書き手（事後の測定、判定には使わない）')
    md('')
    md('対照（と固定）の軌道で、最終 FFN の入力 u₂ = e + a₀ + f₀ + a₁（埋め込み、注意 0、FFN0、注意 1）の各部分の token 平均の変化 Δm_part を取り、'
       '生きた unit の切替時の行 w₀ を通した入力の分 w₀·Δm_part の中央値を出す（和は w₀·Δm に厳密に一致）。「m の分解」は切替時の m への各部分の射影 ⟨m_part, m⟩/‖m‖（和が ‖m‖）。')
    md('')
    md('| seed | 切替 | 条件 | 切替時の m の分解 e / a₀ / f₀ / a₁（‖m‖） | 更新 25: 入力の分の合計 = e + a₀ + f₀ + a₁ | 注意 1 の割合 / FFN0 の割合 | 更新 100: 合計 = e + a₀ + f₀ + a₁ | ‖Δm‖ 25: f₀ / a₁ | 損失 0 → 25 → 100 |')
    md('|---|---|---|---|---|---|---|---|---|')
    wr = {}
    for seed in SEEDS:
        for t in (10, 30):
            for mode in ('base', 'fix_ffn0'):
                d = load(PF / f'writers_S_seed{seed}_t{t}_{mode}.json')
                if not d:
                    continue
                pr = {q['step']: q for q in d['probes']}
                p0, p25, p100 = pr[0], pr[25], pr[100]
                wr[f'{seed}_t{t}_{mode}'] = dict(share25=p25['share_of_input'], total25=p25['input_total_median'], total100=p100['input_total_median'],
                                                 static=p0['parts_proj'], m_norm=p0['m_norm'])
                part = lambda q: ' + '.join(f"{q['input_median'][k]:+.2f}" for k in ('e', 'a0', 'f0', 'a1'))
                md(f"| {seed} | {t}→{t + 1} | {'対照' if mode == 'base' else '固定'} | {p0['parts_proj']['e']:+.1f} / {p0['parts_proj']['a0']:+.1f} / {p0['parts_proj']['f0']:+.1f} / {p0['parts_proj']['a1']:+.1f}（{p0['m_norm']:,.0f}）"
                   f" | {p25['input_total_median']:+.2f} = {part(p25)} | {p25['share_of_input']['a1']:.2f} / {p25['share_of_input']['f0']:.2f}"
                   f" | {p100['input_total_median']:+.2f} = {part(p100)} | {p25['dm_norm']['f0']:,.0f} / {p25['dm_norm']['a1']:,.0f} | {p0['loss']:.1f} → {p25['loss']:.2f} → {p100['loss']:.2f} |")
    J['writers'] = wr
    md('')

    # ---------- J F4 ----------
    md('## J F4 `WIDTH_DEEPENS`（S: 幅 t30/t1 ≥ 5、課題末 t1〜t60 の |z̄| と幅の Spearman ≥ 0.8、r ≤ −1 を t10 以降保つ。3/5）')
    md('')
    md('幅は最終 FFN の unit ごとの前活性の std の中央値、z̄ は unit ごとの平均の中央値、r = μ/σ の unit ごとの中央値（固定 probe、全 token、課題末）。')
    md('')
    md('| seed | 幅 t1 → t10 → t30 → t60 → t100 | z̄ t10 / t30 / t60 / t100 | 死 t10 / t30 / t60 / t100 | 幅比 t30/t1 | ρ（t1〜60） | r: t10〜100 で ≤ −1 の課題の割合 / 中央値 / 範囲 | 判定（文字どおり） |')
    md('|---|---|---|---|---|---|---|---|')
    f4, f4_soft = [], []
    for seed in SEEDS:
        L = layer2('S', seed)
        ts = [t for t in range(1, 61) if t in L]
        rho = spearman([abs(L[t]['mu_med']) for t in ts], [L[t]['sigma_med'] for t in ts])
        rs = [L[t]['r_med'] for t in range(10, 101) if t in L]
        ratio = L[30]['sigma_med'] / L[1]['sigma_med']
        strict = ratio >= TH['width_ratio'] and rho >= TH['spearman'] and all(v <= TH['r_depth'] for v in rs)
        soft = ratio >= TH['width_ratio'] and rho >= TH['spearman'] and St.median(rs) <= TH['r_depth']
        f4.append(bool(strict)); f4_soft.append(bool(soft))
        md(f"| {seed} | {L[1]['sigma_med']:.1f} → {L[10]['sigma_med']:.1f} → {L[30]['sigma_med']:.1f} → {L[60]['sigma_med']:,.0f} → {L[100]['sigma_med']:,.0f}"
           f" | {L[10]['mu_med']:.1f} / {L[30]['mu_med']:.1f} / {L[60]['mu_med']:,.0f} / {L[100]['mu_med']:,.0f}"
           f" | {L[10]['dead005']:.2f} / {L[30]['dead005']:.2f} / {L[60]['dead005']:.2f} / {L[100]['dead005']:.2f}"
           f" | {ratio:.1f} | {rho:.3f} | {sum(v <= -1 for v in rs) / len(rs):.2f} / {St.median(rs):.2f} / [{min(rs):.2f}, {max(rs):.2f}] | {fmt(strict)} |")
    J['F4'] = dict(flags=f4, soft_flags=f4_soft, strict=label(f4, 3), soft=label(f4_soft, 3))
    md('')
    md(f"**J F4（文字どおり、r ≤ −1 を全課題で）: {label(f4, 3)}。幅比と ρ は 5/5。r の条件を「t10〜100 の中央値 ≤ −1」と読めば {label(f4_soft, 3)}（事後の読み替え）。**")
    md('')

    # ---------- J F5 ----------
    md('## J F5 応答と学び直し（主）')
    md('')
    md('読み方（F.11 で SN を外した後）: 前提は Q_S ≥ 1.5 が 3/5。`RESPONSE_BYPASSED` の SC の側 = SC の t30 の死 ≤ 0.2 かつ Q_SC ≥ 1.5（4/5）。`RESPONSE_CARRIES` の SC の側 = Q_SC ≤ 1.3（4/5）。'
       '副（R2・R3、S の t30 末と t60 末から次の課題を同じ batch 列で 5,000 更新、eager、対照と同じ機械）: 中心化の T90 の変化 ≤ 10% かつ尺度の T90 の変化 ≥ 40% なら BYPASSED と整合、逆なら CARRIES と整合。どちらでもなければ `MIXED`。')
    md('')
    md('| seed | Q_S | Q_SC | SC t30 死 | BYPASSED の SC 側 | CARRIES の SC 側 |')
    md('|---|---|---|---|---|---|')
    prem, byp, car = [], [], []
    for seed in SEEDS:
        qs, qc, dc = Q[('S', seed)]['q'], Q[('SC', seed)]['q'], layer2('SC', seed)[30]['dead005']
        prem.append(qs >= TH['q_loss']); byp.append(dc <= TH['path_dead'] and qc >= TH['q_loss']); car.append(qc <= TH['q_noloss'])
        md(f'| {seed} | {qs:.2f} | {qc:.2f} | {dc:.3f} | {fmt(byp[-1])} | {fmt(car[-1])} |')
    md('')
    md('| seed | 切替 | 対照 T90 / 死 (0 → T90) / ‖h‖ 中央値 | 中心化 T90 / 変化 / 死 | 尺度 ×倍率 T90 / 変化 / 死 (T90) | 新品 ‖h‖（目標） | 副の読み |')
    md('|---|---|---|---|---|---|---|')
    sec = {}
    for seed in SEEDS:
        for t in (30, 60):
            d = {m: load(PF / f'r2r3_S_seed{seed}_t{t}_{m}.json') for m in ('base', 'center', 'scale')}
            if not all(d.values()):
                md(f'| {seed} | {t}→{t + 1} | — | — | — | — | — |'); sec[(seed, t)] = None; continue
            b, c, s = d['base'], d['center'], d['scale']
            ch = {m: (d[m]['T90'] - b['T90']) / b['T90'] for m in ('center', 'scale')}
            read = ('BYPASSED と整合' if abs(ch['center']) <= TH['r2_change'] and -ch['scale'] >= TH['r3_change'] else
                    'CARRIES と整合' if -ch['center'] >= TH['r3_change'] and abs(ch['scale']) <= TH['r2_change'] else 'どちらでもない')
            sec[(seed, t)] = dict(base=b['T90'], center=c['T90'], scale=s['T90'], change_center=ch['center'], change_scale=ch['scale'], read=read,
                                  scale_factor=s['feature_scale'], dead_base=(b['probes'][0]['dead005'], b['probes'][-1]['dead005']),
                                  dead_scale=s['probes'][-1]['dead005'])
            md(f"| {seed} | {t}→{t + 1} | {b['T90']:,.0f} / {b['probes'][0]['dead005']:.2f} → {b['probes'][-1]['dead005']:.2f} / {s['scale_ref']['s_h_median']:,.0f}"
               f" | {c['T90']:,.0f} / {ch['center']:+.0%} / {c['probes'][-1]['dead005']:.2f}"
               f" | ×{s['feature_scale']:.2f} {s['T90']:,.0f} / {ch['scale']:+.0%} / {s['probes'][-1]['dead005']:.2f} | {s['scale_ref']['target']:,.0f} | {read} |")
    n_sec = [v for v in sec.values() if v]
    byp_sec = sum(v['read'] == 'BYPASSED と整合' for v in n_sec); car_sec = sum(v['read'] == 'CARRIES と整合' for v in n_sec)
    premise_ok = count(prem) >= 3
    if not premise_ok:
        lab = 'INSUFFICIENT'
    elif count(byp) >= 4 and not count(car) >= 4:
        lab = 'RESPONSE_BYPASSED（SC の側）'
    elif count(car) >= 4:
        lab = 'RESPONSE_CARRIES（SC の側）'
    else:
        lab = 'MIXED'
    J['F5'] = dict(premise=label(prem, 3), bypassed_sc=label(byp, 4), carries_sc=label(car, 4), label=lab,
                   secondary=dict(n=len(n_sec), bypassed=byp_sec, carries=car_sec))
    md('')
    md(f"**J F5: 前提（Q_S ≥ 1.5）{label(prem, 3)}。BYPASSED の SC 側 {label(byp, 4)}、CARRIES の SC 側 {label(car, 4)} → ラベル {lab}。副（R2・R3、{len(n_sec)} 組）: BYPASSED と整合 {byp_sec}、CARRIES と整合 {car_sec}。**")
    md('')

    # ---------- R5: head を 0 に戻す（事後に足した回し直し） ----------
    md('## R5（事後に足した回し直し）: S の t30・t60 末で head だけ 0 に戻す（Adam の状態は残す）。対照は R2 の base')
    md('')
    md('走る前の予測（Fable）: t30 で head 0 の T90 が対照の半分以下になる seed が 4/5（0.65）、t60 で 4/5（0.5）。新品は同 seed の fresh の課題 50（本走、compile）。')
    md('')
    md('| seed | 切替 | 対照 T90 | head 0 T90 | head 0 ÷ 対照 | 新品 T90 | head 0 ÷ 新品 | 死 0 → 末（head 0） | head の行ノルム（戻す前） |')
    md('|---|---|---|---|---|---|---|---|---|')
    r5 = {}
    for seed in SEEDS:
        fr50 = per_task('fresh', seed, 50)[50]
        for t in (30, 60):
            b, h = load(PF / f'r2r3_S_seed{seed}_t{t}_base.json'), load(PF / f'r5_S_seed{seed}_t{t}_head0.json')
            if not b or not h:
                md(f'| {seed} | {t}→{t + 1} | {fmt(b["T90"]) if b else "—"} | {fmt(h["T90"]) if h else "—"} | — | {fr50:,.0f} | — | — | — |'); r5[f'{seed}_t{t}'] = None; continue
            ratio = h['T90'] / b['T90']
            r5[f'{seed}_t{t}'] = dict(base=b['T90'], head0=h['T90'], ratio=ratio, fresh50=fr50, half=ratio <= .5)
            md(f"| {seed} | {t}→{t + 1} | {b['T90']:,.0f} | {h['T90']:,.0f} | {ratio:.2f} | {fr50:,.0f} | {h['T90'] / fr50:.2f} | {h['probes'][0]['dead005']:.2f} → {h['probes'][-1]['dead005']:.2f} | {h['head_norm_before']:.1f} |")
    for t in (30, 60):
        got = [v for k, v in r5.items() if k.endswith(f'_t{t}') and v]
        md('')
        md(f"t{t}: head 0 が対照の半分以下 {sum(v['half'] for v in got)}/{len(got)}（予測は 4/5）。")
    J['R5'] = r5
    md('')
    md('### R5 の続き: SC の対照と head 0（新品は freshC の課題 50）、S の体を凍結した head 0（R4 の新 head の分岐と同じ形）')
    md('')
    md('走る前の予測（Fable）: SC は t30・t60 とも head 0 の T90 が freshC の課題 50 の 1.3 倍以内（4/5、0.7）。S の t60 で体を凍結した新 head は 500 更新以内に届く（4/5、0.7）。')
    md('')
    md('| seed | 切替 | SC 対照 T90 | SC head 0 T90 | SC head 0 ÷ 新品 C | 新品 C（課題 50） | S head 0・体を凍結 T90 | S head 0（体も動く） T90 |')
    md('|---|---|---|---|---|---|---|---|')
    r5b = {}
    for seed in SEEDS:
        frc = per_task('freshC', seed, 50)[50]
        for t in (30, 60):
            cb, ch, sf = (load(PF / f'r5_SC_seed{seed}_t{t}_base.json'), load(PF / f'r5_SC_seed{seed}_t{t}_head0.json'),
                          load(PF / f'r5_S_seed{seed}_t{t}_head0_frozen.json'))
            sh = load(PF / f'r5_S_seed{seed}_t{t}_head0.json')
            r5b[f'{seed}_t{t}'] = dict(sc_base=cb['T90'] if cb else None, sc_head0=ch['T90'] if ch else None, freshC50=frc,
                                       sc_ok=(ch['T90'] <= 1.3 * frc) if ch else None, s_frozen=sf['T90'] if sf else None,
                                       s_frozen_ok=(sf['T90'] <= 500) if sf else None)
            md(f"| {seed} | {t}→{t + 1} | {fmt(cb['T90']) if cb else '—'} | {fmt(ch['T90']) if ch else '—'} | {fmt(ch['T90'] / frc) if ch else '—'} | {frc:,.0f}"
               f" | {fmt(sf['T90']) if sf else '—'} | {fmt(sh['T90']) if sh else '—'} |")
    for t in (30, 60):
        got = [v for k, v in r5b.items() if k.endswith(f'_t{t}')]
        a = [v['sc_ok'] for v in got if v['sc_ok'] is not None]; b = [v['s_frozen_ok'] for v in got if v['s_frozen_ok'] is not None]
        md('')
        md(f"t{t}: SC の head 0 が新品 C の 1.3 倍以内 {sum(a)}/{len(a)}（予測 4/5）。S の体を凍結した新 head が 500 以内 {sum(b)}/{len(b)}（予測 4/5）。")
    J['R5b'] = r5b
    md('')
    md('### R5d・R5e（事後、1104 17:10 起動）: S の t60 末、head 0 に Adam 新品（adamreset）／head の入力の中心化（hcenter）を足す')
    md('')
    md('走る前の予測（Fable）: adamreset は届く（T90 < 5,025）seed が 1 本以下（0.6）、4/5 届けば最適化器の状態の寄与（0.3）。hcenter は届くのが 4/5（0.6）、そのうち新品の課題 50 の 1.3 倍以内が 4/5（0.45）。')
    md('')
    md('| seed | 対照 T90 | head 0 T90 | head 0 ＋ Adam 新品 T90 | head 0 ＋ 入力中心化 T90 | 新品（課題 50） | 中心化前の ‖m̂‖ / ‖h − m̂‖ 中央値 | 死 末（対照 / head 0 / Adam 新品 / 中心化） |')
    md('|---|---|---|---|---|---|---|---|')
    r5de = {}
    for seed in SEEDS:
        fr50 = per_task('fresh', seed, 50)[50]
        b, h = load(PF / f'r2r3_S_seed{seed}_t60_base.json'), load(PF / f'r5_S_seed{seed}_t60_head0.json')
        a, c = load(PF / f'r5_S_seed{seed}_t60_head0_adamreset.json'), load(PF / f'r5_S_seed{seed}_t60_head0_hcenter.json')
        st = (c or {}).get('head_input_stats') or {}
        r5de[str(seed)] = dict(base=b['T90'] if b else None, head0=h['T90'] if h else None, adamreset=a['T90'] if a else None, hcenter=c['T90'] if c else None,
                               fresh50=fr50, adamreset_reached=(a['T90'] < UNR) if a else None, hcenter_reached=(c['T90'] < UNR) if c else None,
                               hcenter_ok=(c['T90'] <= 1.3 * fr50) if c else None, head_input_stats=st or None)
        dead = ' / '.join(f"{z['probes'][-1]['dead005']:.2f}" if z else '—' for z in (b, h, a, c))
        ratio = f"{st['mean_norm']:.0f} / {st['median_centered_norm']:.0f} = {st['mean_norm'] / st['median_centered_norm']:.2f}" if st else '—'
        md(f"| {seed} | {fmt(b['T90']) if b else '—'} | {fmt(h['T90']) if h else '—'} | {fmt(a['T90']) if a else '—'} | {fmt(c['T90']) if c else '—'} | {fr50:,.0f} | {ratio} | {dead} |")
    got = list(r5de.values())
    ar = [v['adamreset_reached'] for v in got if v['adamreset_reached'] is not None]
    hr = [v['hcenter_reached'] for v in got if v['hcenter_reached'] is not None]
    ho = [v['hcenter_ok'] for v in got if v['hcenter_ok'] is not None]
    md('')
    md(f"adamreset が届く {sum(ar)}/{len(ar)}（予測 ≤ 1/5）。hcenter が届く {sum(hr)}/{len(hr)}（予測 4/5）、新品の 1.3 倍以内 {sum(ho)}/{len(ho)}（予測 4/5）。")
    J['R5de'] = r5de
    md('')

    # ---------- J FR0 ----------
    md('## J FR0 `DEATH_NOT_LOSS`（R0 の t60 の死 ≥ 0.6 かつ後期の未到達 ≥ 10 課題（3/5）、かつ R4 の凍結分岐の最後 1,000 更新の平均 ≥ 0.9 かつ新 head の T90 ≤ 500（4/5））')
    md('')
    md('R4: R0 の t60 末から head lr 1/100（体 1e-4）で課題 61 を回し、暗記率が初めて 0.9 に届いた評価点で分岐。凍結 = 体を全部止めて残りを回す。新 head = 同じ体（凍結）に 0 から head を載せ lr 1e-4 で 5,000 更新。seed 10252010 は依頼 4 の走（posthoc_r0_late）。')
    md('')
    md('| seed | t60 死 / CLS 死 | 未到達 41〜60 / 81〜100 | 前提 | R4 元: 届いた更新 / 末 / 最後 1,000 の平均 | 凍結: 最後 1,000 の平均 / 最小 | 新 head: T90 / 末 | R4 判定 |')
    md('|---|---|---|---|---|---|---|---|')
    fr_prem, fr_r4 = [], []
    for seed in SEEDS:
        L, q = layer2('R0', seed), Q[('R0', seed)]
        p_ok = L[60]['dead005'] >= TH['r0_dead'] and q['late_unreached'] >= TH['r0_unreached']
        fr_prem.append(bool(p_ok))
        if seed == 10252010:
            src = RAW / 'posthoc_r0_late'
            base, body, new = (load(src / f'fable_R0_t60_hl001_{k}.json') for k in ('base', 'br_body', 'br_newhead1'))
        else:
            base, body, new = (load(PF / f'r4_R0_seed{seed}_t60_{k}.json') for k in ('base', 'br_body', 'br_newhead1'))
        if base and body and new and 'summary' in body and 'summary' in new:
            r4_ok = body['summary']['tail_mean'] >= TH['freeze_tail'] and (new['summary']['T90'] is not None and new['summary']['T90'] <= TH['newhead_t90'])
            fr_r4.append(bool(r4_ok))
            bs = base.get('summary', {})
            md(f"| {seed} | {L[60]['dead005']:.2f} / {L[60]['cls_dead0']:.2f} | {q['late_unreached']} / {q['tail_unreached']} | {fmt(p_ok)}"
               f" | {fmt(bs.get('T90'))} / {fmt(bs.get('final_acc'))} / {fmt(bs.get('tail_mean'))} | {body['summary']['tail_mean']:.3f} / {body['summary']['tail_min']:.3f}"
               f" | {fmt(new['summary']['T90'])} / {new['summary']['final_acc']:.3f} | {fmt(r4_ok)} |")
        else:
            fr_r4.append(None)
            bs = (base or {}).get('summary', {})
            md(f"| {seed} | {L[60]['dead005']:.2f} / {L[60]['cls_dead0']:.2f} | {q['late_unreached']} / {q['tail_unreached']} | {fmt(p_ok)} | {fmt(bs.get('T90'))} / {fmt(bs.get('final_acc'))} / {fmt(bs.get('tail_mean'))} | — | — | — |")
    J['FR0'] = dict(premise=label(fr_prem, 3), r4=label(fr_r4, 4), label='DEATH_NOT_LOSS' if count(fr_prem) >= 3 and count(fr_r4) >= 4 else 'not DEATH_NOT_LOSS')
    md('')
    md(f"**J FR0: 前提（死 ≥ 0.6 かつ未到達 ≥ 10）{label(fr_prem, 3)}、R4 {label(fr_r4, 4)} → {J['FR0']['label']}。**")
    md('')

    # ---------- 採点 ----------
    md('## 予測の採点（F.6、Fable の確率）')
    md('')
    md('| 記号 | 予測 | Fable | 結果 | 採点 |')
    md('|---|---|---|---|---|')
    rows = [('PF0', 'S は新 seed でも喪失する（Q_S ≥ 1.5 が 3/5）', '0.75', J['F5']['premise'], count(prem) >= 3),
            ('PF1', '装填は 5 seed で揃う', '0.85', J['F1']['label'], J['F1']['label'] == 'LOADED'),
            ('PF2', '切替の押しは入力が運ぶ', '0.80', J['F2']['label'], J['F2']['label'] == 'SINKS_VIA_INPUT'),
            ('PF3a', 'FFN0 の共通成分を固定すると切替の沈みが 4 割減る', '0.65', J['F3']['upstream'], count(f3a) >= 4),
            ('PF3b', '入口の中心化で最終 FFN は死なない', '0.75', J['F3']['path'], count(f3b) >= 3),
            ('PF4', '幅が伸び、絶対の深さは幅に従う', '0.80', J['F4']['strict'] + '（文字どおり）', count(f4) >= 3),
            ('PF5', '応答の喪失は残差が迂回し、学び直しを削るのは尺度', 'BYPASSED 0.55 / CARRIES 0.15 / MIXED 0.30', J['F5']['label'], None),
            ('PFR0', 'R0 の崩れは死ではない', '0.75', J['FR0']['label'], J['FR0']['label'] == 'DEATH_NOT_LOSS')]
    for sym, text, p, res, ok in rows:
        md(f"| {sym} | {text} | {p} | {res} | {'当たり' if ok else ('外れ' if ok is False else 'ラベルで採点')} |")
    md('')
    (OUT / 'judge.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'judge.json').write_text(json.dumps(dict(J=J, Q={f'{a}_{s}': v for (a, s), v in Q.items()},
                                                   secondary={f'{s}_t{t}': v for (s, t), v in sec.items()}, thresholds=TH),
                                              ensure_ascii=False, indent=1, default=str))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
