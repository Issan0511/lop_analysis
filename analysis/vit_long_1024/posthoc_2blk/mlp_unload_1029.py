"""MLP（RC/raw/ReLU、3072-100-100-10、1,200 枚、Adam 1e-3）の共有 head の解き直しの費用を、
ch_chb_200_0919 の課題末 snapshot から次の課題の最初の N 更新を回し直して測る（Fable、1029）。

腕 CH・CHB × head 共有／head を 0 に戻す × 課題 t の末（t+1 を回す）× seed 0〜9（束ねて R=10）。
量: 初期損失 L0（旧 head の自信）、η Σ|a2|（head の 1 歩の logit の歩幅の目安）、‖W3‖、
損失が一様 ln10 まで戻る更新数 n_chance、T50・T90（全 1,200 枚、10 更新ごと）。
Adam のモーメントは snapshot に無いので 0 から（課題末の収束状態なら v は小さく、近似）。
ckpt.pt（t200、実モーメント付き）で t200→201 を実モーメント／0 から の両方で回して近似の差を見る。
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/issan/Projects/claude/proj_004_drift')
from src import relu_doors_0919 as E
from src import pmnist_rlcifar_0907 as RC
from src import pmnist_0905 as H

RAW = Path.home() / 'Projects/obsidian-research-data/ch_chb_200_0919/results/ch_chb_200_0919'
OUT = Path.home() / "Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_mlp_unload"
LAM = {'CH': 0.0, 'CHB': 0.1713}
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
SEEDS = list(range(10))
TASKS = [1, 2, 5, 10, 20, 30, 50, 100, 150, 199]
N_STEPS, EVAL_EVERY = 3000, 10
dev = torch.device('cuda')
torch.backends.cudnn.deterministic = True
cifar = RC.Cifar10()
LN10 = math.log(10)


def labels_for_task(seed, task):
    g = H.stream('rlc_labels', seed)
    for _ in range(task):
        y = RC.task_labels(g)
    return y


def orders_for_task(seed, task, epochs_needed):
    g = H.stream('rlc_batch', seed)
    for _ in range((task - 1) * 400):
        torch.randperm(RC.N_IMAGES, generator=g)
    return [torch.randperm(RC.N_IMAGES, generator=g) for _ in range(epochs_needed)]


@torch.no_grad()
def full_eval(P, X, act, Y):
    z1, a1, z2, a2, z3 = E.forward(P, X, act, train=False)
    loss = F.cross_entropy(z3.reshape(-1, 10), Y.reshape(-1), reduction='none').view(Y.shape).mean(1)
    acc = (z3.argmax(-1) == Y).float().mean(1)
    return loss, acc, a2


def replay(arm, task, head_reset, moments=None, P0=None, act_state=None, labels=None, orders=None):
    """task = the task to train (its labels); start from the snapshot of task-1 (or given P0)."""
    act = E.make_act(arm, lam=LAM[arm], beta=0.01)
    R = len(SEEDS)
    X = torch.stack([E.slot_inputs(cifar, s, 'raw', dev, center=act.door_c) for s in SEEDS])
    if P0 is None:
        snaps = [np.load(RAW / arm / 'snap' / f'{arm}_raw_seed{s}' / f't{task-1:02d}.npz') for s in SEEDS]
        P = [torch.stack([torch.from_numpy(sn[k]) for sn in snaps]).to(dev) for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
        act.init_state(R, dev, key='x')
        act.load_state({'m': [torch.stack([torch.from_numpy(sn['m1']) for sn in snaps]).to(dev),
                              torch.stack([torch.from_numpy(sn['m2']) for sn in snaps]).to(dev)]})
    else:
        P = [q.clone().to(dev) for q in P0]
        act.init_state(R, dev, key='x')
        act.load_state(act_state)
    if head_reset:
        P[4].zero_(); P[5].zero_()
    P = [q.contiguous().requires_grad_(True) for q in P]
    m = [torch.zeros_like(q) for q in P]; v = [torch.zeros_like(q) for q in P]
    tc0 = 0
    if moments is not None:
        for mi, vi, ms, vs in zip(m, v, moments['m'], moments['v']):
            mi.copy_(ms.to(dev)); vi.copy_(vs.to(dev))
        tc0 = moments['tc']
    Y = (torch.stack([labels_for_task(s, task) for s in SEEDS]) if labels is None else labels).to(dev)
    epochs = math.ceil(N_STEPS / RC.STEPS_PER_EPOCH)
    ORD = (torch.stack([torch.stack(orders_for_task(s, task, epochs)) for s in SEEDS]) if orders is None else orders).to(dev)  # (R, epochs, 1200)
    ar = torch.arange(R, device=dev)[:, None]
    loss0, acc0, a2 = full_eval(P, X, act, Y)
    rec = dict(L0=loss0.tolist(), acc0=acc0.tolist(),
               eta_sum_abs_a2=(LR * a2.abs().sum(-1).mean(1)).tolist(),
               a2_norm=a2.norm(dim=-1).mean(1).tolist(),
               W3_norm=P[4].detach().flatten(1).norm(dim=1).tolist(),
               evals=[], batch_loss=[])
    step = 0
    tc = tc0
    for e in range(epochs):
        for j in range(RC.STEPS_PER_EPOCH):
            if step >= N_STEPS:
                break
            idx = ORD[:, e, j * RC.BATCH:(j + 1) * RC.BATCH]
            xb, yb = X[ar, idx], Y[ar, idx]
            z1, a1, z2, a2b, z3 = E.forward(P, xb, act, train=True)
            lossv = F.cross_entropy(z3.reshape(-1, 10), yb.reshape(-1), reduction='none').view(R, RC.BATCH).mean(1)
            grads = torch.autograd.grad(lossv.sum(), P)
            tc += 1
            c1, c2 = 1.0 / (1 - B1 ** tc), 1.0 / (1 - B2 ** tc)
            with torch.no_grad():
                for p, gr, mi, vi in zip(P, grads, m, v):
                    mi.mul_(B1).add_(gr, alpha=1 - B1)
                    vi.mul_(B2).addcmul_(gr, gr, value=1 - B2)
                    p.sub_(LR * (mi * c1) / ((vi * c2).sqrt() + EPS))
                act.post_update(P, LR)
                act.update(z1.detach(), z2.detach())
            step += 1
            if step <= 300:
                rec['batch_loss'].append(lossv.detach().tolist())
            if step % EVAL_EVERY == 0:
                l, a, _ = full_eval(P, X, act, Y)
                rec['evals'].append([step] + l.tolist() + a.tolist())
    ev = np.array(rec['evals'])  # (n, 1+2R)
    steps = ev[:, 0]; L = ev[:, 1:1 + R]; A = ev[:, 1 + R:]
    def first(cond):
        out = []
        for r in range(R):
            w = np.nonzero(cond[:, r])[0]
            out.append(int(steps[w[0]]) if len(w) else None)
        return out
    rec['n_chance'] = first(L <= LN10)
    rec['T50'] = first(A >= 0.5)
    rec['T90'] = first(A >= 0.9)
    rec['acc_end'] = A[-1].tolist(); rec['loss_end'] = L[-1].tolist()
    return rec


def main():
    out = {}
    t_start = time.time()
    for arm in ('CHB', 'CH'):
        for task_end in TASKS:
            for head_reset in (False, True):
                key = f'{arm}_t{task_end}_{"reset" if head_reset else "shared"}'
                rec = replay(arm, task_end + 1, head_reset)
                out[key] = rec
                med = lambda k: float(np.median([x if x is not None else N_STEPS + EVAL_EVERY for x in rec[k]]))
                print(f'{key:18s} L0 {np.median(rec["L0"]):6.2f} ηΣ|a2| {np.median(rec["eta_sum_abs_a2"]):6.3f} ‖W3‖ {np.median(rec["W3_norm"]):5.2f} '
                      f'n_chance {med("n_chance"):6.0f} T50 {med("T50"):6.0f} T90 {med("T90"):6.0f} acc_end {np.median(rec["acc_end"]):.3f} ({time.time()-t_start:.0f}s)', flush=True)
                json.dump(out, open(OUT / 'mlp_unload.json', 'w'))
    # 実モーメント付きの ckpt（t200）で t201 を回し、0 からのモーメントとの差を見る
    for arm in ('CHB', 'CH'):
        ck = torch.load(RAW / arm / 'ckpt.pt', map_location='cpu', weights_only=False)
        assert ck['t'] == 200, ck['t']
        seeds_ck = ck['meta']['seeds']; assert seeds_ck == SEEDS, seeds_ck
        P0 = [q for q in ck['P'][:6]]
        g_lab = {s: H.stream('rlc_labels', s) for s in SEEDS}
        for s in SEEDS: g_lab[s].set_state(ck['g_lab'][s])
        g_batch = {s: H.stream('rlc_batch', s) for s in SEEDS}
        for s in SEEDS: g_batch[s].set_state(ck['g_batch'][s])
        labels = torch.stack([RC.task_labels(g_lab[s]) for s in SEEDS])
        epochs = math.ceil(N_STEPS / RC.STEPS_PER_EPOCH)
        orders = torch.stack([torch.stack([torch.randperm(RC.N_IMAGES, generator=g_batch[s]) for _ in range(epochs)]) for s in SEEDS])
        act_state = ck['act_state']
        for tag, mom in (('real', dict(m=ck['m'][:6], v=ck['v'][:6], tc=ck['tc'])), ('zero', None)):
            for head_reset in (False, True):
                key = f'{arm}_ckpt200_{tag}_{"reset" if head_reset else "shared"}'
                rec = replay(arm, 201, head_reset, moments=mom, P0=P0, act_state=act_state, labels=labels, orders=orders)
                out[key] = rec
                med = lambda k: float(np.median([x if x is not None else N_STEPS + EVAL_EVERY for x in rec[k]]))
                print(f'{key:28s} L0 {np.median(rec["L0"]):6.2f} n_chance {med("n_chance"):6.0f} T50 {med("T50"):6.0f} T90 {med("T90"):6.0f} acc_end {np.median(rec["acc_end"]):.3f}', flush=True)
                json.dump(out, open(OUT / 'mlp_unload.json', 'w'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
