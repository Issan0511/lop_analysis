"""cumulative_push_1013: bit-exact replay of three saved 1-hidden-layer RL-MNIST leaky runs (seed 0, a = 0.1 / 0.3 / 0.5)
to recover what the originals did not store: the Adam state at each task end, the first update of each task,
within-task per-unit pre-activation series, and per-unit accumulators of the push on m_i = w_i . mu + b_i.

Box (code copied, not imported, so nothing is written into the reference tree):
  784-100-10 MLP on 1200 fixed MNIST inputs, hard leaky ReLU with negative slope a (custom autograd: forward
  where(z>0, z, a z), backward grad*where(z>0, 1, a)), CE, Adam lr 1e-3 betas (0.9, 0.999) eps 1e-8, batch 16,
  4000 updates per task, labels iid uniform over 10 classes redrawn per task.
  stream / read_idx / initial / N / BATCH / H: push_direction_proof_0930/gen_states.py
  Act, loss, batch order, Adam, task loop:     push_direction_proof_0930/review/opus_lrslope/run_slope.py
  (a = 0.1 is the same arithmetic as push_direction_proof_0930/review/opus_bwdfloor/run_bwd.py --arm LR_ref)
References (read only, relative to /home/issan/Projects/obsidian-research-data/push_direction_proof_0930/):
  a = 0.1  review/opus_bwdfloor/out/LR_ref_s{seed}.npz
  a != 0.1 review/opus_lrslope/out/LRa{a:g}_s{seed}.npz   (LRa0.3_s0.npz, LRa0.5_s0.npz)
Bit check: subset and init0..3 must match the reference at start; at every task end the 4 params must equal ref
  end0..3[t-1] bitwise (uint32 view) and the labels must equal ref y[t-1]. Otherwise the run logs BIT MISMATCH and
  exits with code 3 (that task's npz is not written).

Saved per task t to OUT/a{a:g}/task_{t:03d}.npz. "s" = updates completed in task t; s = 0 is the state at the end of
task t-1 (the initialization for t = 1). Weights are NOT saved (they are ref end0..3).
  check_steps [851] int64  CHECK = {0, 1, ..., 500} U {510, 520, ..., 4000}; the last row (s = 4000) is the task end
  m_series, s_series [851, 100] float64
                           per unit i at each CHECK step: mean_n z_in and population std_n z_in over all 1200
                           inputs, z = X64 @ W1.T + b1 computed in float64
  m_layer [4000] float64   after update s = 1..4000: mean over all 100 units of (W1 @ mu64 + b1) (run_bwd.py formula)
  m_alive [4000] float64   same, mean over the units alive at the end of task t-1 (run_bwd.py / run_slope.py `mm`)
  alive_prev [100] bool    that alive set ((z > 0).any over inputs at the end of task t-1; all True for t = 1)
  bottom, bottom_ok        argmin(m_alive) + 1 (first minimum), and whether it equals ref bottom[t-1]
  first_batch_idx [16]     order[0]: input indices of the first batch of task t
  g1_0..g1_3 float32       p.grad of the first batch of task t (W1, b1, W2, b2), copied after backward, before step
  dtheta1_0..dtheta1_3     float32: theta(s=1) - theta(s=0), float32 subtraction
  adam_m_0..3, adam_v_0..3 float32: Adam exp_avg / exp_avg_sq at the end of task t
  adam_step int64          Adam step count at the end of task t (= 4000 t)
  bit_ok bool              end-of-task bit check (a saved file always has True; a mismatch stops the run)
  y [1200] int64           labels of task t
  accF_series, accFU_series, accFG_series, accF0_series [851, 100] float64
                           per-unit accumulators (reset to 0 at task start; row s = sum over updates 1..s). Update k
                           uses the state before update k (float64 copies W1, b1, W2, b2 of the float32 params) and
                           its batch idx (16): z_b = X64[idx] @ W1.T + b1, D_b = where(z_b>0, 1, a),
                           p_b = softmax(where(z_b>0, z_b, a z_b) @ W2.T + b2), Yb one-hot, u = 1/10, V = W2.T,
                           K_n = X64[n] @ mu64 + 1:
                             F  += (1/16) sum_n K_n D_b[n,i] <V[i], p_b[n] - Yb[n]>   (= W1.grad[i] @ mu + b1.grad[i])
                             FU += (1/16) sum_n K_n D_b[n,i] <V[i], p_b[n] - u 1>     (confidence part)
                             FG += (1/16) sum_n K_n D_b[n,i] <V[i], Yb[n] - u 1>      (label part; F = FU - FG)
                             F0 += (1/16) sum_n K_n D0[n,i] <V0[i], p_b[n] - Yb[n]>   (D0 [1200,100], V0 = W2.T frozen
                                                                                       at the task start)
  gate_flips int64         entries (n, i) over the 4000 batches where the float64 gate z_b > 0 differs from the gate
                           of the float32 training forward (the F formula then differs from the float32 gradient)
  gradcheck_steps, gradcheck_maxrel, gradcheck_maxrel_elem
                           over the first gradcheck_steps updates of the task: the per-update F increment vs the
                           float32 backward W1.grad @ mu64 + b1.grad (float64 dot of the float32 grads);
                           maxrel = max over updates of max_i|diff| / max_i|float32|; _elem = max over updates and i
                           of |diff_i| / |float32_i|
  identity_ratio float64   max over CHECK steps and units of |F - (FU - FG)| / bound, bound = 2 (4000 + 64) 2^-53 S_i,
                           S_i = sum over updates and batch entries of |K_n D_b/16| sum_k (|p-Y| + |p-u| + |Y-u|)|W2_ki|
                           (worst-case rounding of the three sums). > 1 stops the run (exit 5).
  a, task, seed
Also: OUT/a{a:g}/replay.log (one line per task with timing), OUT/a{a:g}/run_info.json (versions, sha256 of this
script and of the reference npz, result summary when finished).
usage: replay_adam.py --a 0.1 --tasks 100 --out DIR [--seed 0] [--ref PATH] [--gradcheck-steps 50] [--strict-gradcheck]
  --strict-gradcheck: exit 6 if gradcheck_maxrel > 1e-4 in any task (self-test); otherwise it is only recorded.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, datetime, gzip, hashlib, json, math, platform, socket, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)

REF_ROOT = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH, H, K, STEPS, LR = 1200, 16, 100, 10, 4000, 1e-3
CHECK = np.array(list(range(0, 501)) + list(range(510, STEPS + 1, 10)), dtype=np.int64)
GRADCHECK_TOL = 1e-4
ID_BOUND = 2.0 * (STEPS + 64) * 2.0 ** -53


# ---- copied from push_direction_proof_0930/gen_states.py ----
def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def initial(seed, dims):
    g = stream("init", seed); p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p


# ---- copied from push_direction_proof_0930/review/opus_lrslope/run_slope.py ----
class Act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, a):
        out = torch.where(z > 0, z, z * a)
        d = torch.where(z > 0, 1.0, a)
        ctx.save_for_backward(d)
        return out

    @staticmethod
    def backward(ctx, gout):
        d, = ctx.saved_tensors
        return gout * d, None


def default_ref(a, seed):
    if a == 0.1:
        return REF_ROOT / f"review/opus_bwdfloor/out/LR_ref_s{seed}.npz"
    return REF_ROOT / f"review/opus_lrslope/out/LRa{a:g}_s{seed}.npz"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def bits_equal(x, r):
    x, r = np.ascontiguousarray(x), np.ascontiguousarray(r)
    return x.shape == r.shape and x.dtype == r.dtype == np.float32 and np.array_equal(x.view(np.uint32), r.view(np.uint32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--gradcheck-steps", type=int, default=50)
    ap.add_argument("--strict-gradcheck", action="store_true")
    args = ap.parse_args()
    slope, seed, tasks, gc_steps = args.a, args.seed, args.tasks, args.gradcheck_steps
    out = Path(args.out).resolve() / f"a{slope:g}"
    if out == REF_ROOT.resolve() or REF_ROOT.resolve() in out.parents:
        sys.exit("refusing to write inside the reference tree")
    out.mkdir(parents=True, exist_ok=True)
    rp = Path(args.ref) if args.ref else default_ref(slope, seed)
    logf = open(out / "replay.log", "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # reference (read only), loaded into memory once
    with np.load(rp) as R:
        ref = {k: R[k] for k in ["subset", "y", "bottom"] + [f"{p}{j}" for p in ("init", "end") for j in range(4)]}
        ref_arm = str(R["arm"])
        ref_slope = float(R["slope"]) if "slope" in R.files else (0.1 if ref_arm == "LR_ref" else None)
    if ref_slope != slope or int(ref["y"].shape[0]) < tasks:
        sys.exit(f"reference {rp} has slope {ref_slope}, {ref['y'].shape[0]} tasks; asked a={slope}, tasks={tasks}")
    me = Path(__file__).resolve()
    try:
        git_head = subprocess.run(["git", "-C", str(me.parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git_head = None
    info = dict(script=str(me), script_sha256=sha256(me), git_head=git_head, git_note="script uncommitted at launch",
                argv=sys.argv, a=slope, seed=seed, tasks=tasks, out=str(out), ref=str(rp), ref_arm=ref_arm,
                ref_sha256=sha256(rp), check_steps=len(CHECK), gradcheck_steps=gc_steps, strict_gradcheck=args.strict_gradcheck,
                torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                host=socket.gethostname(), started=datetime.datetime.now().isoformat(timespec="seconds"), pid=os.getpid(),
                threads=torch.get_num_threads(), env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    json.dump(info, open(out / "run_info.json", "w"), indent=1)

    # ---- setup: identical to run_slope.py ----
    torch.manual_seed(seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    X64 = torch.tensor(ax[subset].astype(np.float64)); mu64 = X64.mean(0)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(seed, (784, H, K))]
    optim = torch.optim.Adam(P, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    snap = lambda: [q.detach().numpy().copy() for q in P]
    # ---- float64 side quantities (never touch P) ----
    Kvec = X64 @ mu64 + 1.0                                   # K_n = x_n . mu + 1   [1200]
    one64 = torch.tensor(1.0, dtype=torch.float64); a64 = torch.tensor(slope, dtype=torch.float64)
    u = 1.0 / K

    init_ok = np.array_equal(ref["subset"], subset) and all(bits_equal(x, ref[f"init{j}"]) for j, x in enumerate(snap()))
    log(f"replay a={slope:g} seed {seed} tasks {tasks} ref {rp.name} ({ref_arm}) subset+init bit {init_ok} "
        f"torch {torch.__version__} threads {torch.get_num_threads()} check_steps {len(CHECK)} gradcheck_steps {gc_steps}")
    if not init_ok:
        log("BIT MISMATCH at init (subset or init0..3)"); sys.exit(3)

    nchk = len(CHECK)
    row_of = np.full(STEPS + 1, -1, dtype=np.int64); row_of[CHECK] = np.arange(nchk)
    alive = torch.ones(H, dtype=torch.bool)
    t_all = time.monotonic(); n_bottom_ok = 0; worst_gc = 0.0; worst_id = 0.0; total_flips = 0
    for task in range(1, tasks + 1):
        t0 = time.monotonic()
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(STEPS * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:STEPS * BATCH].reshape(STEPS, BATCH)
        first_batch_idx = order[0].numpy().copy()
        Y64 = Y.double()
        m_ser = np.empty((nchk, H)); s_ser = np.empty((nchk, H))
        acc_ser = np.empty((4, nchk, H))                       # F, FU, FG, F0
        m_layer = np.empty(STEPS); m_alive = np.empty(STEPS)
        alive_prev = alive.numpy().copy()
        acc = torch.zeros(4, H, dtype=torch.float64); S_id = torch.zeros(H, dtype=torch.float64)
        flips = 0; gc_rel = 0.0; gc_rel_el = 0.0; id_ratio = 0.0

        def check(s, W64):
            nonlocal id_ratio
            with torch.no_grad():
                z64 = X64 @ W64[0].T + W64[1]
            r = row_of[s]
            m_ser[r] = z64.mean(0).numpy(); s_ser[r] = z64.std(0, correction=0).numpy()
            acc_ser[:, r] = acc.numpy()
            if s > 0:
                resid = (acc[0] - (acc[1] - acc[2])).abs()
                id_ratio = max(id_ratio, float((resid / (ID_BOUND * S_id).clamp_min(1e-300)).max()))
            return z64

        W64 = [q.detach().double() for q in P]                 # state at s = 0
        z0 = check(0, W64)
        D0 = torch.where(z0 > 0, one64, a64)                     # frozen gates [1200, 100]
        W2_0 = W64[2].clone()                                    # frozen readout, V0 = W2_0.T
        theta0 = snap()
        for it in range(STEPS):
            idx = order[it]; xb, yb = X[idx], Y[idx]
            z = xb @ P[0].T + P[1]; h = Act.apply(z, slope); logits = h @ P[2].T + P[3]
            loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
            optim.zero_grad(set_to_none=True); loss.backward()
            if it == 0:
                g1 = [q.grad.detach().numpy().copy() for q in P]
            with torch.no_grad():                                # float64 accumulators at the state before update it+1
                zb = X64[idx] @ W64[0].T + W64[1]
                pos = zb > 0
                Db = torch.where(pos, one64, a64)
                pb = torch.softmax(torch.where(pos, zb, zb * slope) @ W64[2].T + W64[3], -1)
                Yb = Y64[idx]; Kb = Kvec[idx][:, None] / BATCH
                c = Kb * Db
                eF, eU, eG = pb - Yb, pb - u, Yb - u
                incF = (c * (eF @ W64[2])).sum(0)
                acc[0] += incF
                acc[1] += (c * (eU @ W64[2])).sum(0)
                acc[2] += (c * (eG @ W64[2])).sum(0)
                acc[3] += ((Kb * D0[idx]) * (eF @ W2_0)).sum(0)
                S_id += (c.abs() * ((eF.abs() + eU.abs() + eG.abs()) @ W64[2].abs())).sum(0)
                flips += int((pos != (z.detach() > 0)).sum())
                if it < gc_steps:
                    f32 = P[0].grad.double() @ mu64 + P[1].grad.double()
                    diff = (incF - f32).abs()
                    gc_rel = max(gc_rel, float(diff.max() / f32.abs().max()))
                    gc_rel_el = max(gc_rel_el, float((diff / f32.abs()).max()))
            optim.step()
            with torch.no_grad():
                W64 = [q.detach().double() for q in P]
                mv = W64[0] @ mu64 + W64[1]
            m_layer[it] = float(mv.mean()); m_alive[it] = float(mv[alive].mean())
            s = it + 1
            if s == 1:
                theta1 = snap(); dtheta1 = [theta1[j] - theta0[j] for j in range(4)]
            if row_of[s] >= 0:
                z64 = check(s, W64)
        alive = (z64 > 0).any(0)          # z64 is the s = 4000 check: same expression as run_slope.py's task-end z64
        e = snap()
        per_param = [bits_equal(e[j], ref[f"end{j}"][task - 1]) for j in range(4)]
        y_ok = np.array_equal(y_new.numpy(), ref["y"][task - 1])
        bit_ok = all(per_param) and y_ok
        bottom = int(np.argmin(m_alive)) + 1; ref_bottom = int(ref["bottom"][task - 1]); bottom_ok = bottom == ref_bottom
        n_bottom_ok += bottom_ok
        if not bit_ok:
            nd = [int((e[j].view(np.uint32) != ref[f"end{j}"][task - 1].view(np.uint32)).sum()) for j in range(4)]
            log(f"BIT MISMATCH a={slope:g} task {task}: params equal {per_param} (differing elements {nd}) labels equal {y_ok}")
            info.update(finished=False, failed_task=task, seconds=time.monotonic() - t_all)
            json.dump(info, open(out / "run_info.json", "w"), indent=1)
            sys.exit(3)
        if not id_ratio <= 1.0:
            log(f"IDENTITY F = FU - FG VIOLATED a={slope:g} task {task}: ratio to rounding bound {id_ratio:.3e}"); sys.exit(5)
        if args.strict_gradcheck and not gc_rel <= GRADCHECK_TOL:
            log(f"GRADCHECK FAILED a={slope:g} task {task}: maxrel {gc_rel:.3e} > {GRADCHECK_TOL} (elem {gc_rel_el:.3e}, flips {flips})"); sys.exit(6)
        st = [optim.state[q] for q in P]
        adam_m = [x["exp_avg"].numpy().copy() for x in st]
        adam_v = [x["exp_avg_sq"].numpy().copy() for x in st]
        steps_seen = {int(x["step"].item()) if torch.is_tensor(x["step"]) else int(x["step"]) for x in st}
        if steps_seen != {STEPS * task}:
            log(f"ADAM STEP COUNT {steps_seen} != {STEPS * task} at task {task}"); sys.exit(4)
        adam_step = STEPS * task
        worst_gc = max(worst_gc, gc_rel); worst_id = max(worst_id, id_ratio); total_flips += flips
        d = dict(check_steps=CHECK, m_series=m_ser, s_series=s_ser, m_layer=m_layer, m_alive=m_alive,
                 alive_prev=alive_prev, bottom=np.int64(bottom), bottom_ok=np.bool_(bottom_ok),
                 first_batch_idx=first_batch_idx, adam_step=np.int64(adam_step), bit_ok=np.bool_(bit_ok),
                 y=y_new.numpy().copy(), a=np.float64(slope), task=np.int64(task), seed=np.int64(seed),
                 accF_series=acc_ser[0], accFU_series=acc_ser[1], accFG_series=acc_ser[2], accF0_series=acc_ser[3],
                 gate_flips=np.int64(flips), gradcheck_steps=np.int64(gc_steps), gradcheck_maxrel=np.float64(gc_rel),
                 gradcheck_maxrel_elem=np.float64(gc_rel_el), identity_ratio=np.float64(id_ratio))
        for j in range(4):
            d[f"g1_{j}"] = g1[j]; d[f"dtheta1_{j}"] = dtheta1[j]; d[f"adam_m_{j}"] = adam_m[j]; d[f"adam_v_{j}"] = adam_v[j]
        tmp = out / f"task_{task:03d}.tmp.npz"
        np.savez_compressed(tmp, **d); os.replace(tmp, out / f"task_{task:03d}.npz")
        log(f"a={slope:g} task {task:3d} bit_ok {bit_ok} bottom {bottom} (ref {ref_bottom}) bottom_ok {bottom_ok} "
            f"adam_step {adam_step} m s0 {float(m_ser[0].mean()):+.3f} end {m_layer[-1]:+.3f} alive {int(alive.sum())} "
            f"gradchk {gc_rel:.1e}/elem {gc_rel_el:.1e} flips {flips} ident {id_ratio:.1e} "
            f"task_sec {time.monotonic() - t0:.2f} total {time.monotonic() - t_all:.0f}s")
    info.update(finished=True, all_bit_ok=True, bottom_ok_tasks=int(n_bottom_ok), worst_gradcheck_maxrel=worst_gc,
                worst_identity_ratio=worst_id, total_gate_flips=total_flips, seconds=time.monotonic() - t_all,
                ended=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    log(f"DONE a={slope:g} tasks {tasks} all bit_ok True, bottom_ok {n_bottom_ok}/{tasks}, worst gradcheck {worst_gc:.1e}, "
        f"worst identity ratio {worst_id:.1e}, gate flips {total_flips} ({time.monotonic() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
