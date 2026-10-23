#!/usr/bin/env python3
"""chargpt_detlabel_1023 の学習の側の検査（spec §8・§12.3。担当 compute_W）。結果は results/chargpt_detlabel_1023/checks/W/。

  scode      : src/chargpt_cl_1019.py・src/chargpt_cl_1019_kune.py の git blob と RUN_ID
  sstream    : α 0.01・seed 0〜9・課題 1〜3 の rule/data/probe_sha と A2 の provenance.json、init_sha（seed 0〜9）と A2
  spi        : 3 組の (α, seed, 課題) で、先計算の π と ChainM の π のビット一致、data_sha の一致
  ssoft      : S2 の損失を float64 の手計算と照合（相対 1e−6）、グラフの 1 更新が記録するサンプル CE と別に計算した値の照合
  srecorder  : 記録器あり（wrapper）となし（kune.py そのまま）の走（α 0.01・seed 0・課題 1〜2）のビット一致と A2 との差
  timing     : 1 課題の時間（1 本だけ・4 本並べたとき）
  trial      : S2 と A4 の短い試走の出力の確かめ
使い方: CHARGPT_NLAYER=2 python checks_W.py <name>
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CG = Path("/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019")
A2 = CG / "results/chargpt_cl_1019/M2/C3"
OUT = REPO / "results/chargpt_detlabel_1023/checks/W"
PICACHE = REPO / "results/chargpt_detlabel_1023/pi_cache"
PY = str(CG / ".venv/bin/python")
BLOBS = {"src/chargpt_cl_1019.py": "cc524f85d5f2d9813a272abb6b9476b065f1c17b",
         "src/chargpt_cl_1019_kune.py": "272e75548fef16d364c464327c8e44d034b9cbb2"}
ENV = dict(os.environ, CHARGPT_NLAYER="2", CHARGPT_DECOMP_CKPT="1")


def blob(p):
    b = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def dump(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, ensure_ascii=False, default=str))


def scode():
    rows = {p: dict(blob=blob(REPO / p), expected=b, ok=blob(REPO / p) == b) for p, b in BLOBS.items()}
    run_id = [l for l in (REPO / "src/chargpt_cl_1019.py").read_text().splitlines() if l.startswith("RUN_ID")]
    ok = all(r["ok"] for r in rows.values()) and run_id == ['RUN_ID = "chargpt_cl_1019"']
    dump("scode", dict(PASS=ok, files=rows, run_id_line=run_id))


def W():
    sys.path.insert(0, str(REPO / "src"))
    import chargpt_detlabel_1023 as Wm
    return Wm


def sstream():
    Wm = W()
    C = Wm.C
    import torch
    assert C.L == 2
    rows, ok = [], True
    for seed in range(10):
        prov = json.loads((A2 / f"seed{seed}/provenance.json").read_text())
        for t in (1, 2, 3):
            ref = prov["tasks_info"][t - 1]
            c = C.task_cpu_M(seed, t, 10000, 0.01)
            r = dict(seed=seed, task=t, **{k: c[k] == ref[k] for k in ("rule_sha", "data_sha", "probe_sha")})
            rows.append(r)
            ok &= all(r[k] for k in ("rule_sha", "data_sha", "probe_sha"))
        m = C.build_model("C3", "M", seed, torch.device("cuda"))
        r = dict(seed=seed, init_sha=C.state_sha(m) == prov["init_sha"], probe_uniform_sha=C.tensor_sha(C.uniform_probe_M(seed, "cpu")) == prov["probe_uniform_sha"])
        rows.append(r)
        ok &= r["init_sha"] and r["probe_uniform_sha"]
        del m
    n_task = sum(1 for r in rows if "task" in r)
    dump("sstream", dict(PASS=ok, n_task_rows=n_task, n_task_ok=sum(1 for r in rows if "task" in r and r["rule_sha"] and r["data_sha"] and r["probe_sha"]),
                         n_init_ok=sum(1 for r in rows if "init_sha" in r and r["init_sha"]), rows=rows,
                         note="init は build_model（α を取らない）から作るので、A4・A3・S2 の init_sha は同じ seed の A2 と同じになる。試走の provenance でも確かめる（trial）。"))


def spi():
    Wm = W()
    C = Wm.C
    import numpy as np
    combos = [(1e-4, 0, 1), (1e-3, 0, 1), (0.01, 0, 2)]
    rows, ok = [], True
    for alpha, seed, t in combos:
        f = PICACHE / Wm.alpha_tag(alpha) / f"seed{seed}" / f"t{t}.npz"
        z = np.load(f)
        Wm.load_pi_dir(PICACHE, alpha, seed)
        t0 = time.time()
        c_with = C.task_cpu_M(seed, t, 10000, alpha)                     # 先計算の π を使う
        sec_with = time.time() - t0
        src = Wm._DL["pi_src"].get(c_with["rule_sha"])
        C.stationary = Wm._orig_stationary                              # 使わない（元の関数で計算）
        t0 = time.time()
        c_wo = C.task_cpu_M(seed, t, 10000, alpha)
        sec_wo = time.time() - t0
        C.stationary = Wm.stationary_cached
        r = dict(alpha=alpha, seed=seed, task=t, pi_src_with=src, pi_iters=int(z["pi_iters"]),
                 pi_bitwise=bool(np.array_equal(z["pi"], c_wo["chain"].pi)) and z["pi"].tobytes() == c_wo["chain"].pi.tobytes(),
                 pi_iters_equal=int(z["pi_iters"]) == int(c_wo["chain"].pi_iters),
                 data_sha_equal=c_with["data_sha"] == c_wo["data_sha"], probe_sha_equal=c_with["probe_sha"] == c_wo["probe_sha"],
                 rule_sha_equal=c_with["rule_sha"] == c_wo["rule_sha"] == str(z["rule_sha"]),
                 sec_with_cache=sec_with, sec_without_cache=sec_wo)
        r["PASS"] = bool(str(src).startswith("file:") and r["pi_bitwise"] and r["pi_iters_equal"] and r["data_sha_equal"]
                         and r["probe_sha_equal"] and r["rule_sha_equal"])
        ok &= r["PASS"]
        rows.append(r)
    dump("spi", dict(PASS=ok, n=len(rows), n_pass=sum(r["PASS"] for r in rows), rows=rows))


def ssoft():
    Wm = W()
    C = Wm.C
    import numpy as np
    import torch
    import torch.nn.functional as F
    C.setup_torch()
    dev = torch.device("cuda")
    seed, t = 0, 1
    c = C.task_cpu_M(seed, t, 50, 0.01)
    ctx = C.TaskCtx("M", seed, t, 50, dev, alpha=0.01, cpu=c)
    Wm._DL["target"] = "soft"
    Wm._SOFT["P"] = torch.zeros(C.V * C.V, C.V, dtype=torch.float32, device=dev)
    m = C.build_model("C3", "M", seed, dev)
    eng = C.Engine(m, "C3", "M", graph=True)
    Wm.load_soft_P(ctx)
    P64 = torch.from_numpy(c["chain"].P.reshape(C.V * C.V, C.V)).to(dev)
    buf_l, buf_a = torch.zeros(20, device=dev), torch.zeros(20, device=dev)
    rows = []
    for u in range(20):
        eng.set_lr(1e-3 * (u + 1) / C.WARMUP)
        x, y = ctx.batch(u)
        x, y = x.long(), y.long()                                   # Engine は静的な long の tensor に写す（データは uint8）
        with torch.no_grad():
            out = m(x)
            lg = out[:, 1:].reshape(-1, C.V)
            ci = (x[:, :-1] * C.V + x[:, 1:]).reshape(-1)
            soft64 = float(-(P64[ci] * torch.log_softmax(lg.double(), -1)).sum(-1).mean())
            samp_ref = F.cross_entropy(lg, y[:, 1:].reshape(-1))
            samp64 = float(F.cross_entropy(lg.double(), y[:, 1:].reshape(-1)))
            soft32, samp32 = Wm.soft_losses(out, x, y)
        eng.step(x, y, buf_l, buf_a, u)
        torch.cuda.synchronize()
        rows.append(dict(u=u, soft_f64=soft64, soft_eager_f32=float(soft32), soft_graph=float(buf_a[u]),
                         rel_soft_graph=abs(float(buf_a[u]) - soft64) / abs(soft64),
                         rel_soft_eager=abs(float(soft32) - soft64) / abs(soft64),
                         samp_graph=float(buf_l[u]), samp_ref=float(samp_ref), samp_f64=samp64,
                         samp_bitwise=bool(float(buf_l[u]) == float(samp_ref)),
                         rel_samp_graph=abs(float(buf_l[u]) - samp64) / samp64))
    mx_soft = max(r["rel_soft_graph"] for r in rows)
    mx_samp = max(r["rel_samp_graph"] for r in rows)
    ok = mx_soft <= 1e-6 and mx_samp <= 1e-6
    dump("ssoft", dict(PASS=ok, n_updates=len(rows), max_rel_soft_vs_f64=mx_soft, max_rel_samp_vs_f64=mx_samp,
                       n_samp_bitwise_vs_eager_f32=sum(r["samp_bitwise"] for r in rows),
                       max_rel_soft_eager_vs_f64=max(r["rel_soft_eager"] for r in rows), rows=rows,
                       note="グラフの 1 更新が記録した値（更新前の網）を、同じ網に別に forward した値と照合。soft は勾配に使う損失、samp は online.npz の loss"))


def run_cmd(args, log):
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(args, cwd=REPO, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    return rc, time.time() - t0


def srecorder():
    import numpy as np
    import torch
    base = ["--box", "M", "--arm", "C3", "--seed", "0", "--tasks", "2", "--n_per_task", "10000", "--alpha", "0.01", "--fresh_tasks"]
    dw, dn = OUT / "srec_with", OUT / "srec_without"
    rc1, s1 = run_cmd([PY, "src/chargpt_detlabel_1023.py", "run", *base, "--out", str(dw)], OUT / "logs/srec_with.log")
    rc2, s2 = run_cmd([PY, "src/chargpt_cl_1019_kune.py", "run", *base, "--out", str(dn)], OUT / "logs/srec_without.log")
    res = dict(rc_with=rc1, rc_without=rc2, sec_with=s1, sec_without=s2)
    ow, on = np.load(dw / "online.npz"), np.load(dn / "online.npz")
    res["online_loss_bitwise"] = bool(np.array_equal(ow["loss"], on["loss"]))
    res["online_aux_bitwise"] = bool(np.array_equal(ow["aux"], on["aux"]))
    ck = {}
    for f in sorted(p.name for p in (dn / "ckpts").glob("*.pt")):
        a = torch.load(dw / "ckpts" / f, map_location="cpu", weights_only=False)
        b = torch.load(dn / "ckpts" / f, map_location="cpu", weights_only=False)
        ck[f] = (dw / "ckpts" / f).read_bytes() == (dn / "ckpts" / f).read_bytes() or _tree_equal(a, b)
    res["ckpt_bitwise"] = ck
    res["ckpt_files_with_extra"] = sorted(set(p.name for p in (dw / "ckpts").glob("*.pt")) - set(ck))
    a2 = np.load(A2 / "seed0/online.npz")["loss"][:2]
    dd = np.abs(ow["loss"].astype(np.float64) - a2.astype(np.float64))
    res["vs_A2"] = dict(bitwise=bool(np.array_equal(ow["loss"], a2)), first_diff_update=[int(np.argmax(dd[i] > 0)) if (dd[i] > 0).any() else None for i in range(2)],
                        max_absdiff_per_task=[float(dd[i].max()) for i in range(2)], mean_absdiff_per_task=[float(dd[i].mean()) for i in range(2)],
                        n_equal_per_task=[int((dd[i] == 0).sum()) for i in range(2)],
                        mean_loss_A2=[float(a2[i].mean()) for i in range(2)], mean_loss_byakko=[float(ow["loss"][i].mean()) for i in range(2)])
    pw = json.loads((dw / "provenance.json").read_text())
    res["task_secs_with"] = pw["task_secs"]
    res["task_secs_without"] = json.loads((dn / "provenance.json").read_text())["task_secs"]
    res["PASS"] = bool(rc1 == 0 and rc2 == 0 and res["online_loss_bitwise"] and res["online_aux_bitwise"]
                       and all(ck.get(f, False) for f in ("task1.pt", "task2.pt")) and all(ck.values()))
    dump("srecorder", res)


def _tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def timing():
    """1 課題の時間: 本体の課題（within なし・fresh なし・記録器あり）。1 本だけと、4 本並べたとき。"""
    base = ["--box", "M", "--arm", "C3", "--tasks", "2", "--n_per_task", "10000", "--alpha", "0.01", "--fresh_tasks", "--within_tasks"]
    res = {}
    rc, s = run_cmd([PY, "src/chargpt_detlabel_1023.py", "run", "--seed", "0", *base, "--out", str(OUT / "timing1/seed0")], OUT / "logs/timing1.log")
    res["single"] = dict(rc=rc, wall=s, task_secs=json.loads((OUT / "timing1/seed0/provenance.json").read_text())["task_secs"])
    env4 = dict(ENV, DETLABEL_GPU_CONCURRENCY="4")
    t0 = time.time()
    ps = [subprocess.Popen([PY, "src/chargpt_detlabel_1023.py", "run", "--seed", str(k), *base, "--out", str(OUT / f"timing4/seed{k}")],
                           cwd=REPO, env=env4, stdout=open(OUT / f"logs/timing4_s{k}.log", "w"), stderr=subprocess.STDOUT) for k in range(4)]
    rcs = [p.wait() for p in ps]
    res["four"] = dict(rc=rcs, wall=time.time() - t0,
                       task_secs=[json.loads((OUT / f"timing4/seed{k}/provenance.json").read_text())["task_secs"] for k in range(4)])
    dump("timing", res)


def trial():
    import numpy as np
    res = {}
    for name, alpha, target in (("S2", "0.01", "soft"), ("A4", "1e-4", "sample")):
        d = OUT / f"trial_{name}/seed0"
        rc, s = run_cmd([PY, "src/chargpt_detlabel_1023.py", "run", "--box", "M", "--arm", "C3", "--seed", "0", "--tasks", "3", "--n_per_task", "1000",
                         "--alpha", alpha, "--target", target, "--fresh_tasks", "2", "--within_tasks", "1", "--pi_dir", str(PICACHE), "--out", str(d)],
                        OUT / f"logs/trial_{name}.log")
        r = dict(rc=rc, wall=s)
        if rc == 0:
            pv = json.loads((d / "provenance.json").read_text())
            a2 = json.loads((A2 / "seed0/provenance.json").read_text())
            on = np.load(d / "online.npz")
            mo = np.load(d / "within_motion.npz")
            fo = np.load(d / "fresh_online_t2.npz")
            r.update(complete=pv.get("detlabel_complete"), init_sha_equal_A2=pv["init_sha"] == a2["init_sha"],
                     pi_source_counts=pv["detlabel"]["pi_source_counts"], chain_fail_tasks=pv["detlabel"]["chain_fail_tasks"],
                     online_finite=bool(np.isfinite(on["loss"]).all() and np.isfinite(on["aux"]).all()),
                     loss_eq_aux=bool(np.array_equal(on["loss"], on["aux"])), fresh_loss_eq_aux=bool(np.array_equal(fo["loss"], fo["aux"])),
                     online_mean_last100=[float(on["loss"][i, -100:].mean()) for i in range(3)],
                     aux_mean_last100=[float(on["aux"][i, -100:].mean()) for i in range(3)],
                     motion_shape=list(mo["motion"].shape), motion_finite=bool(np.isfinite(mo["motion"]).all()),
                     motion_L0_net5k_med=[float(np.median(mo["motion"][i, 0, :, 0])) for i in range(3)],
                     alive_L0=[int(mo["alive"][i, 0, :, 1].sum()) for i in range(3)],
                     files=sorted(p.name for p in d.iterdir()), n_ckpt=len(list((d / "ckpts").glob("*.pt"))),
                     task_secs=pv["task_secs"], ckpt_tasks=pv["ckpt_tasks"],
                     chain_stats_rows=len((d / "chain_stats.csv").read_text().splitlines()) - 1,
                     tasks_info_sha_equal_A2_alpha001=[pv["tasks_info"][i]["data_sha"] == a2["tasks_info"][i]["data_sha"] for i in range(3)]
                     if name == "S2" else None)
            r["PASS"] = bool(r["complete"] and r["init_sha_equal_A2"] and r["online_finite"] and r["motion_finite"]
                             and not r["chain_fail_tasks"] and (r["loss_eq_aux"] == (target == "sample")))
        else:
            r["PASS"] = False
        res[name] = r
    res["PASS"] = all(res[k]["PASS"] for k in ("S2", "A4"))
    dump("trial", res)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
