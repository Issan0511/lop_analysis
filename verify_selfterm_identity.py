"""task_080 / blindspot_0820 §1: 自己項の恒等式の数値検証（手検算 selfterm_identity_0820.md の突き合わせ）

実行: リポジトリルートで  python verify_selfterm_identity.py
依存: torch, numpy（リポの src/ を import する）

検証項目（ノートの式番号に対応）:
  [I1] 恒等式: 勾配の self 部分 2·f_i·∂f_i/∂θ = ∂(f_i²)/∂θ が点毎に成立
       （θ = w_i, b_i, v_i。autograd の ∇E[f_i²] と閉形式が一致）
  [I2] 完全性: 閉形式 self + rest = autograd ∇E[δ²]（分解に取りこぼしがない）
  [I3] コード経路: src.ratchet_log.exact_record の F_self が
       −η·(∇_{w_i}E[f_i²])·µ̂_u と一致（本物の測定コードに対する検証）
  [S1] condA 符号: s_i(x) = a_i·gate_i·(x·µ̂_u) ≥ 0 が全パターン点毎に成立
       ⇒ F_self ≤ 0 が任意バッチ（batch=1 含む）で決定論的に成立
  [S2] 等号条件: dead（p̂=0）で F_self = 0、alive（p̂>0, v≠0, flip≠0）で厳密 < 0
  [K]  ReLU の角（pre=0 を厳密に踏む）でも恒等式が成立し、gate 規約（>0 / ≥0）に非依存。
       中心差分 d E[f_i²]/db とも一致
  [C1] 反例: ガウス入力（condB 型）では符号保証が壊れる（F_self > 0 が実在）
  [C2] 反例: LeakyReLU (α>0) では condA でも壊れる。sigmoid（σ≥0, σ'≥0）では保たれる
       ⇒ 真の条件は「σ·σ' ≥ 0」×「x·µ̂_u ≥ 0（点毎）」
  [R]  ±1 再符号化した condA では壊れない（x·µ̂ = f/‖µ‖ > 0 の定数になるため）。
       条件の本体は x ≥ 0 ではなく x·µ̂ ≥ 0（点毎）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.nets import VecMLP
from src.envs import SCREnv, LTUTarget
from src import ratchet_log

torch.manual_seed(0)
DEV = "cpu"
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- 共通部品

def forward_units(W, b, v, X, act="relu", alpha=0.1):
    """X:[P,R,m] -> pre,a,gate,f  各 [P,R,h]。float64 前提。"""
    pre = torch.einsum("rhd,prd->prh", W, X) + b
    if act == "relu":
        a = torch.relu(pre)
        gate = (pre > 0).double()
    elif act == "leaky":
        a = torch.where(pre > 0, pre, alpha * pre)
        gate = (pre > 0).double() + alpha * (pre <= 0).double()
    elif act == "sigmoid":
        a = torch.sigmoid(pre)
        gate = a * (1 - a)
    f = v[None] * a
    return pre, a, gate, f


def selfgrad_formula(W, b, v, X, act="relu"):
    """self 勾配の閉形式（係数2規約）: gW=2 v² a·gate·x, gb=2 v² a·gate, gv=2 v a²。
    期待値（パターン一様平均）を返す。"""
    pre, a, gate, f = forward_units(W, b, v, X, act)
    ag = a * gate                                     # [P,R,h]
    gW = 2.0 * (v ** 2)[None, :, :, None] * ag[..., None] * X[:, :, None, :]
    gb = 2.0 * (v ** 2)[None] * ag
    gv = 2.0 * v[None] * a ** 2
    return gW.mean(0), gb.mean(0), gv.mean(0)


def autograd_Ef2(W, b, v, X, act="relu"):
    """autograd による ∇_{θ_i} E[f_i²]（Σ_i E[f_i²] の backward はユニット毎に分離する）。"""
    W = W.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    _, _, _, f = forward_units(W, b, v, X, act)
    S = (f ** 2).mean(0).sum()
    S.backward()
    return W.grad, b.grad, v.grad


def autograd_Edelta2(W, b, v, c, X, y, act="relu"):
    """autograd による ∇ E[δ²]。"""
    W = W.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    c = c.clone().requires_grad_(True)
    pre, a, gate, f = forward_units(W, b, v, X, act)
    delta = f.sum(-1) + c - y
    S = (delta ** 2).mean(0).sum()
    S.backward()
    return W.grad, b.grad, v.grad


def restgrad_formula(W, b, v, c, X, y, act="relu"):
    """rest 勾配の閉形式: δ_rest = δ − f_i を δ の位置に置いたもの。"""
    pre, a, gate, f = forward_units(W, b, v, X, act)
    delta = f.sum(-1) + c - y                          # [P,R]
    d_rest = delta[..., None] - f                      # [P,R,h]
    vg = v[None] * gate
    gW = 2.0 * (d_rest * vg)[..., None] * X[:, :, None, :]
    gb = 2.0 * d_rest * vg
    gv = 2.0 * d_rest * a
    return gW.mean(0), gb.mean(0), gv.mean(0)


def relerr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    den = max(np.abs(x).max(), np.abs(y).max(), 1e-300)
    return np.abs(x - y).max() / den


def condA_support(R, m, f_bits, gen):
    """flip_state をランダムに引き、32 パターン全サポート X:[P,R,m] と µ̂_u を返す。"""
    flip = torch.randint(0, 2, (R, f_bits), generator=gen).double()
    P = 2 ** (m - f_bits)
    pat = ((torch.arange(P)[:, None] >> torch.arange(m - f_bits)) & 1).double()
    X = torch.cat([flip[None].expand(P, -1, -1), pat[:, None, :].expand(-1, R, -1)], dim=2)
    mu = X.mean(0)
    # µ = [flip ‖ 0.5·1] の解析形の確認（ratchet_log.py L104 のコメント）
    mu_ref = torch.cat([flip, 0.5 * torch.ones(R, m - f_bits).double()], dim=1)
    assert torch.equal(mu, mu_ref), "µ = [flip ‖ 0.5·1] が成立していない"
    mu_u = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return X, mu_u, flip


# ================================================================ [I1][I2] 恒等式（一般入力・autograd）

print("\n[I1][I2] 恒等式と分解の完全性（ガウス入力、float64、autograd 突き合わせ）")
gen = torch.Generator().manual_seed(1)
R, h, m, P = 5, 17, 12, 400
W = (torch.rand(R, h, m, generator=gen).double() * 2 - 1)
b = (torch.rand(R, h, generator=gen).double() * 2 - 1)
v = (torch.rand(R, h, generator=gen).double() * 2 - 1)      # 両陣営を含む
c = (torch.rand(R, generator=gen).double() * 2 - 1)
X = torch.randn(P, R, m, generator=gen).double()
y = torch.randn(P, R, generator=gen).double()

sW, sb, sv = selfgrad_formula(W, b, v, X)
aW, ab, av = autograd_Ef2(W, b, v, X)
e = max(relerr(sW, aW), relerr(sb, ab), relerr(sv, av))
check("I1: self閉形式 = ∇E[f_i²] (autograd)", e < 1e-13, f"max relerr {e:.2e}")

rW, rb, rv = restgrad_formula(W, b, v, c, X, y)
tW, tb, tv = autograd_Edelta2(W, b, v, c, X, y)
e = max(relerr(sW + rW, tW), relerr(sb + rb, tb), relerr(sv + rv, tv))
check("I2: self+rest = ∇E[δ²] (autograd)", e < 1e-13, f"max relerr {e:.2e}")

# ================================================================ [I3] 本物の測定コード経路

print("\n[I3] src.ratchet_log.exact_record の F_self = −η·∇_{w_i}E[f_i²]·µ̂_u（condA 実走状態）")
gen_env = torch.Generator().manual_seed(11)
gen_tch = torch.Generator().manual_seed(12)
gen_ini = torch.Generator().manual_seed(13)
R, m, f_bits, h = 4, 20, 15, 100
env = SCREnv(R, m, f_bits, torch.full((R,), 10000, dtype=torch.long), gen_env, DEV)
teacher = LTUTarget(R, m, hidden=100, beta=0.7, gen=gen_tch, device=DEV)
net = VecMLP(R, h, m, gen_ini, DEV)
lr = torch.full((R,), 0.01)

for step in range(3000):                     # 実際の学習則で初期化から離す（batch=1 SGD）
    x = env.step()
    pre, a, yhat = net.forward(x)
    delta = yhat - teacher(x)
    net.sgd_step(lr, *net.grads(x, pre, a, delta))

st = dict(env=env, net=net, teacher=teacher, lr=lr,
          centered=torch.zeros(R, dtype=torch.bool),
          running_mean=torch.zeros(R, m))
rec = ratchet_log.exact_record(st, as_f64=True)          # ← 本物の測定関数

X = ratchet_log.full_support_ro(env).double()
mu = X.mean(0)
mu_u = mu / mu.norm(dim=1, keepdim=True)
aW, ab, av = autograd_Ef2(net.W.double(), net.b.double(), net.v.double(), X)
F_self_check = (-lr.double()[:, None] * torch.einsum("rhd,rd->rh", aW, mu_u)).numpy()
e = relerr(rec["F_self"], F_self_check)
check("I3: exact_record F_self = −η·(∇E[f_i²])·µ̂_u", e < 1e-12, f"max relerr {e:.2e}")

e = relerr(rec["F_self"] + rec["F_rest"], rec["F_gate"])
check("I3b: F_self + F_rest = F_gate（測定コード内の分解完全性）", e < 1e-12, f"max relerr {e:.2e}")

sgn_ok = float(rec["F_self"].max()) <= 0.0
check("I3c: 実走状態で F_self ≤ 0（全 R×h ユニット）", sgn_ok, f"max = {rec['F_self'].max():.3e}")

# ================================================================ [S1][S2] condA 符号の全数チェック

print("\n[S1][S2] condA 符号: s_i(x) ≥ 0 点毎 ⇒ 任意バッチで F_self ≤ 0（決定論）")
gen = torch.Generator().manual_seed(2)
n_nets, n_pointwise, n_pos, n_alive, n_alive_strict = 0, 0, 0, 0, 0
min_s = np.inf
for trial in range(60):
    R, h = 6, 25
    X, mu_u, flip = condA_support(R, m=12, f_bits=7, gen=gen)      # 32 パターン
    W = (torch.rand(R, h, 12, generator=gen).double() * 2 - 1) * (2.0 if trial % 3 else 0.5)
    b = (torch.rand(R, h, generator=gen).double() * 3 - 2)          # 深い負バイアス含む
    v = (torch.rand(R, h, generator=gen).double() * 2 - 1)
    pre, a, gate, f = forward_units(W, b, v, X)
    xdm = (X * mu_u[None]).sum(-1)                                  # [P,R] = x·µ̂_u
    s = a * gate * xdm[..., None]                                   # [P,R,h] 点毎 self スカラー
    min_s = min(min_s, float(s.min()))
    n_pointwise += s.numel()
    F_self = -2.0 * 0.01 * (v ** 2)[None] * s
    F_self_mean = F_self.mean(0)                                    # 厳密期待値（=フルバッチ）
    n_pos += int((F_self_mean > 0).sum()) + int((F_self > 0).sum()) # バッチ1(点毎)も同時に検査
    p_hat = gate.mean(0)
    alive = (p_hat > 0) & (v.abs() > 0) & (flip.sum(1, keepdim=True) > 0)
    n_alive += int(alive.sum())
    n_alive_strict += int((F_self_mean[alive] < 0).sum())
    n_nets += R * h
check(f"S1: 点毎 s_i(x) ≥ 0（{n_pointwise:,} 点、min = {min_s:.1e}）", min_s >= 0.0)
check(f"S1b: F_self > 0 の発生 0 件（期待値と batch=1 の両方、{n_nets:,} ユニット）", n_pos == 0)
check(f"S2: alive（p̂>0, v≠0, flip≠0）では厳密 F_self < 0（{n_alive_strict}/{n_alive}）",
      n_alive_strict == n_alive)

# dead ユニットの等号
W0 = torch.zeros(1, 1, 12).double(); W0[0, 0, 0] = 1.0
b0 = torch.full((1, 1), -100.0).double()                            # 全サポートで pre < 0
v0 = torch.ones(1, 1).double()
X0, mu_u0, _ = condA_support(1, 12, 7, gen)
pre, a, gate, f = forward_units(W0, b0, v0, X0)
F0 = (-2 * 0.01 * v0 ** 2 * (a * gate * (X0 * mu_u0[None]).sum(-1)[..., None]).mean(0))
check("S2b: dead（p̂=0）で F_self = 0（力の消灯 = 恒等式の平坦領域）", float(F0.abs().max()) == 0.0)

# ================================================================ [K] ReLU の角

print("\n[K] ReLU の角（pre = 0 を厳密に踏むパターン）での恒等式と規約非依存性")
gen = torch.Generator().manual_seed(3)
R, h = 3, 8
X, mu_u, _ = condA_support(R, 12, 7, gen)
W = (torch.rand(R, h, 12, generator=gen).double() * 2 - 1)
v = (torch.rand(R, h, generator=gen).double() * 2 - 1)
b = -torch.einsum("rhd,rd->rh", W, X[5])               # パターン5で全ユニット pre=0 厳密
pre = torch.einsum("rhd,prd->prh", W, X) + b
assert float(pre[5].abs().max()) == 0.0, "角の構成に失敗"

a = torch.relu(pre)
P_n = X.shape[0]

def Ef2(bb):
    aa = torch.relu(torch.einsum("rhd,prd->prh", W, X) + bb)
    return ((v[None] * aa) ** 2).mean(0)

# 角では E[f_i²] は C¹ だが C² でない (h''(z)=2·1[z>0] が跳ぶ) ため、中心差分は O(ε²) でなく
# O(ε) 収束になる。しかもその離散化誤差は手計算で厳密に予言できる:
#   角パターン1個あたり fd − 真値 = v_i²·ε/(2P)  (b±ε で σ² が ε² と 0 に割れるため)
# 予言した誤差を引いた上で閉形式と比較する (= 誤差構造ごと恒等式の確認になる)。
for gname, gate in [("gate=1[pre>0]", (pre > 0).double()), ("gate=1[pre≥0]", (pre >= 0).double())]:
    ag = a * gate
    gb = (2.0 * (v ** 2)[None] * ag).mean(0)           # self の b 成分 (閉形式)
    eps = 1e-6
    fd = (Ef2(b + eps) - Ef2(b - eps)) / (2 * eps)
    fd_corr = fd - (v ** 2) * eps / (2 * P_n)          # 角1個ぶんの予言誤差を除去
    e = relerr(gb.numpy(), fd_corr.numpy())
    check(f"K: {gname} で self閉形式 = 数値微分 dE[f_i²]/db（予言誤差 v²ε/2P を控除後）",
          e < 1e-8, f"max relerr {e:.2e}")
# 収束次数の確認: 生の誤差が O(ε) で線形に落ちる (ε を 1/10 → 誤差も 1/10)
gb = (2.0 * (v ** 2)[None] * (a * (pre > 0).double())).mean(0)
errs = []
for eps in (1e-5, 1e-6, 1e-7):
    fd = (Ef2(b + eps) - Ef2(b - eps)) / (2 * eps)
    errs.append(float((fd - gb).abs().max()))
r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
check(f"K3: 生の中心差分誤差が O(ε) 線形収束（比 {r1:.2f}, {r2:.2f} ≈ 10）",
      7 < r1 < 13 and 5 < r2 < 15)
diff = float((a * (pre > 0).double() - a * (pre >= 0).double()).abs().max())
check("K2: a·gate が gate 規約に厳密非依存（差 = 0）", diff == 0.0)

# ================================================================ [C1][C2][R] 反例と真の条件

print("\n[C1][C2][R] 符号保証のスコープ（何が条件を担っているか）")
gen = torch.Generator().manual_seed(4)

# C1: ガウス入力（condB 型）+ 固定射影方向 → F_self > 0 が実在するはず
R, h, m, P = 20, 30, 12, 512
W = (torch.rand(R, h, m, generator=gen).double() * 2 - 1)
b = (torch.rand(R, h, generator=gen).double() * 2 - 1)
v = (torch.rand(R, h, generator=gen).double() * 2 - 1)
Xg = torch.randn(P, R, m, generator=gen).double()
u = torch.randn(R, m, generator=gen).double()
u = u / u.norm(dim=1, keepdim=True)
pre, a, gate, f = forward_units(W, b, v, Xg)
Fg = (-2 * 0.01 * (v ** 2)[None] * a * gate * (Xg * u[None]).sum(-1)[..., None]).mean(0)
frac_pos = float((Fg > 0).double().mean())
check(f"C1: ガウス入力では F_self > 0 が実在（正符号率 {frac_pos:.3f}）", frac_pos > 0.1)

# C2a: LeakyReLU + condA → 壊れる（σσ' < 0 が可能）
X, mu_u, _ = condA_support(R, 12, 7, gen)
pre, a, gate, f = forward_units(W, b, v, X, act="leaky")
xdm = (X * mu_u[None]).sum(-1)
Fl = (-2 * 0.01 * (v ** 2)[None] * a * gate * xdm[..., None]).mean(0)
frac_pos_l = float((Fl > 0).double().mean())
check(f"C2a: LeakyReLU(α=0.1) は condA でも F_self > 0 が実在（正符号率 {frac_pos_l:.3f}）",
      frac_pos_l > 0.0)
# leaky でも恒等式自体は成立（壊れるのは符号だけ）
sW, sb, sv = selfgrad_formula(W, b, v, X, act="leaky")
aW, ab, av = autograd_Ef2(W, b, v, X, act="leaky")
e = max(relerr(sW, aW), relerr(sb, ab), relerr(sv, av))
check("C2a': 恒等式は LeakyReLU でも成立（符号のみが ReLU 特有）", e < 1e-13, f"max relerr {e:.2e}")

# C2b: sigmoid（σ≥0, σ'≥0）+ condA → 保たれる
pre, a, gate, f = forward_units(W, b, v, X, act="sigmoid")
Fs = (-2 * 0.01 * (v ** 2)[None] * a * gate * xdm[..., None]).mean(0)
check(f"C2b: sigmoid では符号が保たれる（max = {float(Fs.max()):.2e} ≤ 0）",
      float(Fs.max()) <= 0.0)

# R: ±1 再符号化 condA → x·µ̂ = f/‖µ‖ の正定数になり、符号はむしろ保たれる
Xpm = 2 * X - 1
mu_pm = Xpm.mean(0)
mu_pm_u = mu_pm / mu_pm.norm(dim=1, keepdim=True).clamp_min(1e-300)
xdm_pm = (Xpm * mu_pm_u[None]).sum(-1)
check(f"R: ±1 符号化でも x·µ̂_u ≥ 0 点毎（min = {float(xdm_pm.min()):.3e}）"
      "→ 条件の本体は x≥0 ではなく x·µ̂≥0", float(xdm_pm.min()) >= 0.0)

# ================================================================ まとめ

print()
if FAILURES:
    print(f"== {len(FAILURES)} 件 FAIL: {FAILURES}")
    sys.exit(1)
print("== 全項目 PASS: 恒等式は成立、condA での F_self ≤ 0 は構造的（任意バッチで決定論）")
