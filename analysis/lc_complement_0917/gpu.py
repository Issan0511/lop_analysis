"""C1 replay: same engine membership, RNG streams, blocks and device as neff_pred.

Only adds read-only measurements. A mismatch with any recorded online value or
endpoint mean z aborts the run; a replay on a different GPU is not silently reused.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import neff_pred_0917 as N
from src import lc_complement_0917 as C


@torch.no_grad()
def fields(box):
    e = box.e
    if box.mod is N.B:
        z1, a1, z2, a2, _ = N.B.forward(e.p, e.cx, e.A)
        return (z1, a1, N.B.gate(z1, e.A)), (z2, a2, N.B.gate(z2, e.A))
    z1, a1, z2, a2, _ = N.L.forward(e.p, e.cx, e.elu1, e.elu2)
    return (z1, a1, N.L.gate(z1, e.elu1)), (z2, a2, N.L.gate(z2, e.elu2))


def reference_zmeans(field_values):
    """Use the reference's batched reduction, including its rounding order.

    A separate mean for each model launches a different CUDA reduction kernel
    and can differ by a float64 ULP even when the float32 field is identical.
    The equality guard stays exact; this fixes the diagnostic computation.
    """
    return [z.double().mean(dim=(1, 2)).cpu().numpy() for z, _, _ in field_values]


def run(out, tasks=150, device="cuda", smoke=False):
    if not smoke and (tasks != 150 or device != "cuda"):
        raise ValueError("Main replay requires the registered 150-task CUDA protocol")
    out = Path(out)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    prov = C.provenance(out, "c1-gpu", smoke, tasks=tasks, device=device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recorded GPU trajectory")
    ref_path = C.ROOT / "results/neff_pred_0917/gpu/rows.csv"
    ref = pd.read_csv(ref_path, float_precision="round_trip").set_index(
        ["engine", "seed", "env", "act1", "act2", "task"])
    boxes = [N.Box(name, mod, mod.MODELS, device) for name, mod in (("B", N.B), ("L", N.L))]
    seeds = range(3)
    gp = {s: N.B.stream("env_perm_0913", s) for s in seeds}
    gl = {s: N.B.stream("env_labels_0913", s) for s in seeds}
    gb = {s: N.B.stream("env_batch_0913", s) for s in seeds}
    ax = N.B.read_idx(N.B.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    ay = N.B.read_idx(N.B.DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
    idx = {s: torch.randperm(len(ax), generator=N.B.stream("rl_subset", s))[:1200].numpy() for s in seeds}
    xs = {s: torch.tensor(ax[idx[s]], device=device) for s in seeds}
    ys = {s: torch.tensor(ay[idx[s]], device=device) for s in seeds}
    for box in boxes:
        box.e.capture()
    rows = []
    for task in range(1, tasks + 1):
        orders = {s: torch.stack([torch.randperm(1200, generator=gb[s]) for _ in range(80)]).reshape(6000, 16)
                  for s in seeds}
        perms = {s: torch.randperm(784, generator=gp[s]).to(device) for s in seeds}
        labels = {s: torch.randint(10, (1200,), generator=gl[s]).to(device) for s in seeds}
        for box in boxes:
            e = box.e
            orderstack = torch.stack([orders[m["seed"]] for m in box.models], 1).to(device)
            for j, m in enumerate(box.models):
                s = m["seed"]
                e.cx[j].copy_(xs[s][:, perms[s]] if m["env"] == "PM" else xs[s])
                e.cy[j].copy_(torch.nn.functional.one_hot(ys[s] if m["env"] == "PM" else labels[s], 10))
            pending = []
            start_fields = fields(box)
            start_zmeans = reference_zmeans(start_fields)
            for layer, (z, a, g) in enumerate(start_fields, 1):
                for j, m in enumerate(box.models):
                    act1, act2 = N.model_key(m)
                    family = "ELU" if (act1, act2)[layer - 1] == "ELU1" else (act1, act2)[layer - 1]
                    stats, _ = C.confusion(a[j], g[j], family)
                    pending.append(dict(engine=box.name, seed=m["seed"], env=m["env"],
                                        act1=act1, act2=act2, task=task, branch=task-1,
                                        layer=layer, phase="start", zbar=float(start_zmeans[layer-1][j]), **stats))
            e.acc.zero_()
            e.ce.zero_()
            for step in range(0, 6000, N.B.BLOCK):
                e.indices.copy_(orderstack[step:step + N.B.BLOCK])
                e.replay_block()
            acc = (e.acc / 6000).cpu().numpy()
            final = fields(box)
            final_zmeans = reference_zmeans(final)
            for j, m in enumerate(box.models):
                acts = N.model_key(m)
                rr = ref.loc[(box.name, m["seed"], m["env"], *acts, task)]
                values = [float(acc[j]), *(float(zm[j]) for zm in final_zmeans)]
                expected = [rr.online_acc, rr.end_zbar_l1, rr.end_zbar_l2]
                if device == "cuda" and values != expected:
                    raise ValueError(f"GPU replay differs at {box.name}/{m}/task{task}: {values} != {expected}")
                for row in pending:
                    if (row["seed"], row["env"], row["act1"], row["act2"]) == (m["seed"], m["env"], *acts):
                        row["E"] = float(acc[j])
                for layer, (z, a, g) in enumerate(final, 1):
                    family = "ELU" if acts[layer-1] == "ELU1" else acts[layer-1]
                    stats, _ = C.confusion(a[j], g[j], family)
                    # No E at task end: pairing it with this task's online would leak the outcome.
                    pending.append(dict(engine=box.name, seed=m["seed"], env=m["env"],
                                        act1=acts[0], act2=acts[1], task=task, branch=task,
                                        layer=layer, phase="end", zbar=float(final_zmeans[layer-1][j]), **stats))
            rows.extend(pending)
        C.R.write_csv(out / "rows.csv", rows)
        print(f"C1 GPU replay task={task}/{tasks}", flush=True)
    C.complete(out, prov, reference_sha256=C.sha(ref_path),
               engines_sha256={mod.__name__: C.sha(mod.__file__) for mod in (N, N.B, N.L)},
               device_name=torch.cuda.get_device_name() if device == "cuda" else device,
               data_sha256={f: C.sha(N.B.DATA / f) for f in N.B.DATA_FILES})
