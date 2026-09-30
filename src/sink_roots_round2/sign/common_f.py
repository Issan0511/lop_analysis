"""検算の共通部品（float64）。"""
import gzip, math
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
SQRT2 = math.sqrt(2.0); INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
FLOOR = {"ELU": -1.0, "GELU": -0.16997, "SILU": -0.27846, "R": 0.0, "LR": 0.0, "SNK": 0.0}
_X = None

def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)

def load_X(subset):
    global _X
    if _X is None:
        _X = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    return torch.tensor(_X[subset], dtype=torch.float32).double()

def activ(z, act):
    if act == "LR":   return torch.where(z > 0, z, z * 0.1)
    if act == "ELU":  return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    if act == "R":    return torch.relu(z)
    if act == "GELU": return z * 0.5 * (1.0 + torch.erf(z / SQRT2))
    if act == "SILU": return z * torch.sigmoid(z)
    if act == "SNK":  return z + torch.sin(z) ** 2
def gate(z, act):
    if act == "LR":   return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
    if act == "ELU":  return torch.where(z > 0, torch.ones_like(z), z.clamp_max(0).exp())
    if act == "R":    return (z > 0).to(z.dtype)
    if act == "GELU": return 0.5 * (1.0 + torch.erf(z / SQRT2)) + z * torch.exp(-0.5 * z * z) * INV_SQRT_2PI
    if act == "SILU":
        s = torch.sigmoid(z); return s * (1.0 + z * (1.0 - s))
    if act == "SNK":  return 1.0 + torch.sin(2 * z)

def load_state(path):
    d = np.load(path, allow_pickle=True)
    L = int(d["L"]); act = str(d["act"]); K = int(d["K"])
    P = [torch.tensor(d[f"p{i}"]).double() for i in range(2 * L + 2)]
    X = load_X(d["subset"])
    return dict(P=P, X=X, L=L, act=act, K=K, y_old=torch.tensor(d["y_old"]).long(), y_new=torch.tensor(d["y_new"]).long(),
                dm=d["dm"], ks=d["ks"], mA=[torch.tensor(d[f"m{i}"]).double() for i in range(2 * L + 2)],
                vA=[torch.tensor(d[f"v{i}"]).double() for i in range(2 * L + 2)], step=float(d["step"]))

def p_F_positive(rho, d, n=400000, seed=0):
    """P(sum_{j=1}^d X_j Z_j > 0), (X_j, Z_j) iid 2 変量正規・相関 rho。= P(F_{d,d} > (1-rho)/(1+rho))。"""
    rng = np.random.default_rng(seed)
    A = rng.chisquare(d, n); B = rng.chisquare(d, n)
    rho = np.atleast_1d(rho)
    return np.array([np.mean((1 + r) * A - (1 - r) * B > 0) for r in rho])
