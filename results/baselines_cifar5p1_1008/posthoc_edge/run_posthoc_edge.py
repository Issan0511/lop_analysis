"""Post-hoc (NOT registered): one step beyond each grid edge, calibration seeds 100-109 only."""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, ".")
from src import baselines_cifar5p1_1008 as E, cifar5p1_mlp_0920 as C, pmnist_0905 as H
torch.set_num_threads(2)
dev = H.setup("cuda")
cif = C.Cifar100()
out = Path(sys.argv[1])
cells = [E.make_cfg("snp", eps="1e-6", sigma="1e-2"), E.make_cfg("snp", eps="1e-5", sigma="3e-2"),
         E.make_cfg("snp", eps="1e-6", sigma="3e-2"),
         E.make_cfg("cbp", rho="3e-6"), E.make_cfg("cbp", rho="1e-6"),
         E.make_cfg("redo", tau="0.3", period=1560), E.make_cfg("redo", tau="0.1", period=3900),
         E.make_cfg("redo", tau="0.3", period=3900)]
for cfg in cells:
    t0 = time.time()
    E.run(cfg, E.CALIB_SEEDS, C.N_TASKS, dev, out / E.cfg_tag(cfg), cifar=cif, progress=lambda m: None)
    print(f"{time.strftime('%T')} done {E.cfg_tag(cfg)} {time.time() - t0:.0f}s", flush=True)
