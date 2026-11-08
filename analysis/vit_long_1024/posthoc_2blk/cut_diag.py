"""追補 H.4（1108、平均置換を足した版）: 切除の前の診断（学び直し無し、CPU）。

S の課題 t の末（t = 60, 30）の体と head のまま、課題 t のラベル（C.task_dataset の task = t、C.loss_classes の行）で:
  - 訓練精度（2,500 枚）: 切除なし、各枝を 1 つずつ切ったとき。切り方は 2 種類（cut_replay と同じ hook）:
      零: 0 * output。
      平均置換: 出力を、同じ 2,500 枚での平均（token 位置ごとの定数ベクトル。切除なしの網で取る）に置き換える。
  - 各枝の出力の CLS 位置（token 0）の RMS ノルムと、その枝に入る残差の CLS 位置の RMS ノルムとの比（切除なしの forward）。
    RMS は 2,500 枚 × 384 次元の全要素で取る（pooled）。比 = RMS(枝の出力) / RMS(入る残差)。画像ごとの比の中央値、
    平均ベクトルの CLS 位置の RMS、CLS 位置の出力の平均を除いた RMS（入力ごとの違いの大きさ）も json に残す。
    LN なしの箱なので枝の入力（norm1・norm2 が Identity）が入る残差そのもの。

    cd <worktree> && PYTHONPATH=. VIT_LONG_PRODUCTION=production_2blk_rlfix.json python analysis/vit_long_1024/posthoc_2blk/cut_diag.py
"""
import json

import torch

from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.cut_replay import CUTS, RUNS, branch, const_output, zero_output

SEEDS = (10252010, 10252011, 10252012, 10252013, 10252014)
T_ENDS = (60, 30)
OUT = C.REPO / 'results/vit_long_1024_2blk_rlfix/addendum_h'
LABEL = {'cut_attn0': 'attn0', 'cut_ffn0': 'FFN0', 'cut_attn1': 'attn1', 'cut_ffn1': 'FFN1'}
ORDER = ('cut_attn0', 'cut_ffn0', 'cut_attn1', 'cut_ffn1')     # forward の順


def load(seed, t):
    m = C.build_arm_model('S', seed, 'cpu', 0.0, 2, 8.0)
    m.load_state_dict(torch.load(RUNS / 'S' / f'seed{seed}' / f'ckpt/model_t{t:02d}.pt', map_location='cpu', weights_only=False))
    for i in range(m.depth):
        for nm in ('norm1', 'norm2'):
            assert isinstance(m.layers[f'block_{i}'].layers[nm], torch.nn.Identity)
    return m.eval()


@torch.no_grad()
def accuracy(model, x, y, rows):
    c = 0
    for s in range(0, len(y), 250):
        c += int((model(x[s:s + 250]).index_select(1, rows).argmax(1) == y[s:s + 250]).sum())
    return c / len(y)


@torch.no_grad()
def stats(model, x):
    """切除なしの forward 1 回で、枝ごとの出力の全画像平均 [tokens, dim]・CLS 位置の入出力。"""
    store = {c: dict(inp=[], out=[], sum=None) for c in ORDER}
    hs = []
    for c in ORDER:
        def rec(mod, inp, out, c=c):              # None を返す（出力は変えない）
            store[c]['inp'].append(inp[0][:, 0].double())
            store[c]['out'].append(out[:, 0].double())
            v = out.double().sum(0)
            store[c]['sum'] = v if store[c]['sum'] is None else store[c]['sum'] + v
        hs.append(branch(model, *CUTS[c]).register_forward_hook(rec))
    for s in range(0, len(x), 250):
        model(x[s:s + 250])
    for h in hs:
        h.remove()
    res, means = {}, {}
    for c in ORDER:
        I, O = torch.cat(store[c]['inp']), torch.cat(store[c]['out'])
        means[c] = (store[c]['sum'] / len(x)).to(model.cls_token.dtype)
        rin, rout = float(I.pow(2).mean().sqrt()), float(O.pow(2).mean().sqrt())
        per = O.pow(2).mean(1).sqrt() / I.pow(2).mean(1).sqrt()
        res[c] = dict(rms_out=rout, rms_in=rin, ratio=rout / rin, ratio_median_per_image=float(per.median()),
                      rms_mean_cls=float(means[c][0].double().pow(2).mean().sqrt()),
                      rms_out_centered=float((O - O.mean(0)).pow(2).mean().sqrt()))
    return res, means


@torch.no_grad()
def acc_with(model, hook, c, x, y, rows):
    h = branch(model, *CUTS[c]).register_forward_hook(hook)
    try:
        return accuracy(model, x, y, rows)
    finally:
        h.remove()


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    source = C.image_source('real')
    rows_all = []
    for t in T_ENDS:
        for seed in SEEDS:
            data, _ = C.task_dataset(source, seed, t, 10, 250, augment=False)
            x, y = C.eval_tensor(data.images, 'cpu'), data.labels
            rows = torch.tensor(C.loss_classes('S', seed, t, 10, 250))
            model = load(seed, t)
            acc0 = accuracy(model, x, y, rows)
            norms, means = stats(model, x)
            r = dict(t_end=t, seed=seed, acc_none=acc0, branches={})
            for c in ORDER:
                az = acc_with(model, zero_output, c, x, y, rows)
                am = acc_with(model, const_output(means[c]), c, x, y, rows)
                r['branches'][LABEL[c]] = dict(acc_zero=az, d_acc=az - acc0, d_acc_zero=az - acc0,
                                               acc_mean=am, d_acc_mean=am - acc0, **norms[c])
            rows_all.append(r)
            print(t, seed, f'acc {acc0:.3f}', {k: (round(v['d_acc_zero'], 3), round(v['d_acc_mean'], 3), round(v['ratio'], 3))
                                               for k, v in r['branches'].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'diag.json').write_text(json.dumps(dict(
        what='追補 H.4 切除前の診断: S の課題 t 末の状態・課題 t のラベル・2,500 枚・eval・CPU。零の切除と平均置換',
        rms='CLS 位置（token 0）、2,500 枚 × 384 次元の pooled RMS。ratio = rms_out / rms_in（入る残差）。'
            'rms_mean_cls = 平均ベクトルの CLS 位置の RMS、rms_out_centered = CLS 位置の出力から全画像平均を引いた RMS',
        keys='d_acc と d_acc_zero は同じ（零の切除、旧版との互換）。d_acc_mean は平均置換',
        rows=rows_all), indent=1, ensure_ascii=False))
    br = [LABEL[c] for c in ORDER]
    L = ['# 追補 H.4 切除の前の診断（学び直し無し）', '',
         'S（LN なし・共有 head）の課題 t の末の状態を、課題 t のラベルのまま 2,500 枚で評価（eval、CPU）。',
         '枝を 1 つずつ切る。零 = forward hook で 0 * output。平均 = 出力を同じ 2,500 枚での平均（token 位置ごとの定数ベクトル、切除なしの網で取る）に置き換える。'
         'どちらも cut_replay と同じ hook。Δ精度 = 切除後 − 切除なし。',
         'RMS は CLS 位置（token 0）の、2,500 枚 × 384 次元の全要素の RMS。比 = 枝の出力の RMS ÷ その枝に入る残差の RMS（切除なしの forward）。'
         '「平均を除いた RMS」は CLS 位置の出力から全画像平均を引いた RMS（平均置換で消える部分の大きさ）。',
         '生成: `analysis/vit_long_1024/posthoc_2blk/cut_diag.py`。数値の全体は `diag.json`。', '']
    for t in T_ENDS:
        rs = [r for r in rows_all if r['t_end'] == t]
        L += [f'## 課題 {t} の末', '', '### 訓練精度と Δ精度（零 ／ 平均置換）', '',
              '| seed | 切除なし | ' + ' | '.join(f'Δ −{b} 零' for b in br) + ' | ' + ' | '.join(f'Δ −{b} 平均' for b in br) + ' |',
              '|---|---|' + '---|' * (2 * len(br))]
        for r in rs:
            L.append(f"| {r['seed']} | {r['acc_none']:.3f} | " + ' | '.join(f"{r['branches'][b]['d_acc_zero']:+.3f}" for b in br)
                     + ' | ' + ' | '.join(f"{r['branches'][b]['d_acc_mean']:+.3f}" for b in br) + ' |')
        L += ['', '### 枝の出力の CLS 位置の RMS（入る残差の RMS）／ 平均を除いた RMS', '',
              '| seed | ' + ' | '.join(br) + ' |', '|---|' + '---|' * len(br)]
        for r in rs:
            L.append(f"| {r['seed']} | " + ' | '.join(
                f"{r['branches'][b]['rms_out']:.3g}（{r['branches'][b]['rms_in']:.3g}）／ {r['branches'][b]['rms_out_centered']:.3g}" for b in br) + ' |')
        L += ['', '### 比（枝の出力 ÷ 入る残差、CLS 位置）', '',
              '| seed | ' + ' | '.join(br) + ' |', '|---|' + '---|' * len(br)]
        for r in rs:
            L.append(f"| {r['seed']} | " + ' | '.join(f"{r['branches'][b]['ratio']:.3g}" for b in br) + ' |')
        L.append('')
    (OUT / 'diag.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
