"""ベクトル化環境: 条件A (Slowly-Changing Regression) / 条件B (ガウス入力・教師切り替え)。

R 本の独立系列をバッチ次元で並列に進める。学習自体は各系列 batch=1 のオンライン学習。
乱数 generator は「入力」「教師」「初期化」で分離 (仕様書 §8)。
"""
import math
import torch


# ---------------------------------------------------------------- 条件A: SCR

class LTUTarget:
    """[D] FixLTUNet のベクトル化版 (R 系列分の独立教師)。重み・バイアス ±1、
    tau = beta*(m+1) - S, S = (m - sum(W) + b)/2  (公式実装 fix_ltu_net.py に一致)。"""

    def __init__(self, R, m, hidden, beta, gen, device):
        pm = lambda *s: (torch.randint(0, 2, s, generator=gen, device=device).float() * 2 - 1)
        self.W = pm(R, hidden, m)          # [R,H,m]
        self.b = pm(R, hidden)             # [R,H]
        self.v = pm(R, hidden)             # [R,H]
        self.cout = pm(R)                  # [R]
        S = (m - self.W.sum(dim=2) + self.b) / 2
        self.tau = beta * (m + 1) - S      # [R,H]

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in ("W", "b", "v", "cout", "tau")}

    def load_state(self, s):
        for k in ("W", "b", "v", "cout", "tau"):
            setattr(self, k, s[k].clone())

    def __call__(self, x):
        """x: [..., R, m] -> y: [..., R]"""
        pre = torch.einsum("rhm,...rm->...rh", self.W, x) + self.b
        h = (pre >= self.tau).float()
        return (h * self.v).sum(dim=-1) + self.cout


class SCREnv:
    """[D] 入力生成: f 個の flipping bits (系列別周期 T_r で 1 ビット反転) +
    (m-f) 個の毎ステップ U{0,1}。初期 flipping bits は randint{0,1} (公式実装準拠)。"""

    def __init__(self, R, m, f, T, gen, device):
        self.R, self.m, self.f = R, m, f
        self.T = T                          # [R] long (CPU tensor)
        self.gen, self.device = gen, device
        self.flip_state = torch.randint(0, 2, (R, f), generator=gen, device=device).float()
        # ランダムビット部の全サポート列挙 [2^(m-f), m-f] (full-batch GD 用)
        self.patterns = ((torch.arange(2 ** (m - f), device=device)[:, None]
                          >> torch.arange(m - f, device=device)) & 1).float()
        self.t = 0

    def state_dict(self):
        return {"flip_state": self.flip_state.clone(), "t": self.t}

    def load_state(self, s):
        self.flip_state = s["flip_state"].clone()
        self.t = s["t"]

    def maybe_flip(self):
        """現在時刻 t が系列 r の周期境界 (t>0, t % T_r == 0) なら 1 ビット反転。"""
        if self.t > 0 and (self.t % 100 == 0):   # 全 T は 100 の倍数
            mask = (self.t % self.T) == 0        # [R] bool (CPU)
            if mask.any():
                idx = torch.randint(0, self.f, (self.R,), generator=self.gen, device=self.device)
                rows = torch.nonzero(torch.as_tensor(mask, device=self.device)).squeeze(1)
                self.flip_state[rows, idx[rows]] = 1 - self.flip_state[rows, idx[rows]]

    def step(self):
        """1 ステップ分の生入力 x_raw [R, m] を返す (flip 処理込み)。"""
        self.maybe_flip()
        rnd = torch.randint(0, 2, (self.R, self.m - self.f),
                            generator=self.gen, device=self.device).float()
        self.t += 1
        return torch.cat([self.flip_state, rnd], dim=1)

    def step_batch(self, B):
        """1 ステップ分の iid ミニバッチ [B, R, m] (flip 処理は 1 ステップ分)。"""
        self.maybe_flip()
        rnd = torch.randint(0, 2, (B, self.R, self.m - self.f),
                            generator=self.gen, device=self.device).float()
        self.t += 1
        flip = self.flip_state.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([flip, rnd], dim=2)

    def full_support(self):
        """現在の flip_state 下の入力分布の全サポート [2^(m-f), R, m] (一様重み)。
        乱数を消費しない厳密フルバッチ (flip 選択のみ gen を使う)。"""
        self.maybe_flip()
        self.t += 1
        P = self.patterns.shape[0]
        flip = self.flip_state.unsqueeze(0).expand(P, -1, -1)
        rnd = self.patterns[:, None, :].expand(-1, self.R, -1)
        return torch.cat([flip, rnd], dim=2)

    def segment(self, C):
        """C ステップ分をまとめて生成 [C,R,m]。セグメント内に周期境界が無いことが前提
        (呼び出し側が境界でセグメントを切る)。先頭で flip 処理。凍結測定用。"""
        self.maybe_flip()
        rnd = torch.randint(0, 2, (C, self.R, self.m - self.f),
                            generator=self.gen, device=self.device).float()
        flip = self.flip_state.unsqueeze(0).expand(C, -1, -1)
        self.t += C
        return torch.cat([flip, rnd], dim=2)


# ------------------------------------------------- 条件B: ガウス入力・教師切替

def kaiming_mlp_params(R, h, d, gen, device):
    """学習器と同じ初期化則 ([D] ffnn.py 準拠): 入力層 kaiming_uniform(relu), bias=0;
    出力層 kaiming_uniform(linear), bias=0。"""
    bw = math.sqrt(6.0 / d)      # gain sqrt(2) * sqrt(3/fan_in)
    bv = math.sqrt(3.0 / h)      # gain 1 * sqrt(3/fan_in)
    W = (torch.rand(R, h, d, generator=gen, device=device) * 2 - 1) * bw
    v = (torch.rand(R, h, generator=gen, device=device) * 2 - 1) * bv
    b = torch.zeros(R, h, device=device)
    c = torch.zeros(R, device=device)
    return W, b, v, c


class MLPTeacher:
    """条件B教師: 学習器と同構造のランダム ReLU MLP。系列別周期 K_r で再サンプル。"""

    def __init__(self, R, h, d, K, gen, device):
        self.R, self.h, self.d = R, h, d
        self.K = K                        # [R] long (CPU)
        self.gen, self.device = gen, device
        self.W, self.b, self.v, self.c = kaiming_mlp_params(R, h, d, gen, device)
        self.t = 0

    def state_dict(self):
        return {"W": self.W.clone(), "b": self.b.clone(), "v": self.v.clone(),
                "c": self.c.clone(), "t": self.t}

    def load_state(self, s):
        self.W, self.b, self.v, self.c = (s[k].clone() for k in ("W", "b", "v", "c"))
        self.t = s["t"]

    def maybe_resample(self):
        if self.t > 0 and (self.t % 100 == 0):   # 全 K は 100 の倍数
            mask = (self.t % self.K) == 0
            if mask.any():
                Wn, bn, vn, cn = kaiming_mlp_params(self.R, self.h, self.d, self.gen, self.device)
                mdev = torch.as_tensor(mask, device=self.device)
                self.W = torch.where(mdev[:, None, None], Wn, self.W)
                self.v = torch.where(mdev[:, None], vn, self.v)
                # b, c は常に 0 なので置換不要

    def __call__(self, x):
        """x: [..., R, d] -> y: [..., R]"""
        pre = torch.einsum("rhd,...rd->...rh", self.W, x) + self.b
        return (torch.relu(pre) * self.v).sum(dim=-1) + self.c


class GaussEnv:
    """x = mu + Sigma^{1/2} z, z ~ N(0, I_d), mu = c/sqrt(d) * 1 (系列別 c)。

    Sigma = I + (kappa-1) u u^T (スパイク型) [NEW]。
    Sigma^{1/2} = I + (sqrt(kappa)-1) u u^T なので行列平方根は不要。
    kappa=None または全て 1 なら等方 (従来と乱数消費含め bit 一致)。

    spike_dir: "ones" = u ∝ 1 (mu と平行, 従来) /
               "alt"  = u ∝ (+1,-1,+1,...,0) (1 と直交 -> mu ⊥ u, 決定的ベクトル)。"""

    def __init__(self, R, d, c, gen, device, kappa=None, spike_dir="ones"):
        self.R, self.d = R, d
        self.mu = (torch.as_tensor(c, device=device).float() / math.sqrt(d)).unsqueeze(1) \
            .expand(R, d).contiguous()     # [R,d]
        if spike_dir == "ones":
            self.u = torch.full((d,), 1.0 / math.sqrt(d), device=device)
        elif spike_dir == "alt":
            n = d - (d % 2)
            u = torch.zeros(d, device=device)
            u[:n] = torch.tensor([1.0, -1.0], device=device).repeat(n // 2)
            self.u = u / u.norm()
        else:
            raise ValueError(f"unknown spike_dir: {spike_dir}")
        self.sk = None                     # sqrt(kappa) - 1 [R], None なら等方
        if kappa is not None:
            k = torch.as_tensor(kappa, device=device).float()
            sk = k.expand(R).sqrt() - 1.0
            if bool((sk != 0).any()):
                self.sk = sk.contiguous()
        self.gen, self.device = gen, device
        self.t = 0

    def state_dict(self):
        return {"t": self.t}

    def load_state(self, s):
        self.t = s["t"]

    def _transform(self, z):
        """z [..., R, d] -> Sigma^{1/2} z = z + (sqrt(kappa)-1)(u^T z) u。"""
        if self.sk is None:
            return z
        proj = torch.einsum("...rd,d->...r", z, self.u)
        return z + (self.sk * proj)[..., None] * self.u

    def step(self):
        self.t += 1
        z = torch.randn(self.R, self.d, generator=self.gen, device=self.device)
        return self.mu + self._transform(z)

    def step_batch(self, B):
        self.t += 1
        z = torch.randn(B, self.R, self.d, generator=self.gen, device=self.device)
        return self.mu.unsqueeze(0) + self._transform(z)

    def segment(self, C):
        z = torch.randn(C, self.R, self.d, generator=self.gen, device=self.device)
        self.t += C
        return self.mu.unsqueeze(0) + self._transform(z)
