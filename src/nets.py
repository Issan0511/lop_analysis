"""ベクトル化学習器: 1 隠れ層 ReLU MLP (R 系列分を並列)。

勾配は閉形式で計算する (autograd 不要):
  a = relu(W x + b),  yhat = v.a + c,  delta = yhat - y,  L = delta^2 / 2 ... ではなく
  [D] 公式実装 (bp.py) は loss = (yhat - y)^2 の squared error をそのまま SGD に流すため
  dL/dyhat = 2*delta。本実装も公式に合わせ係数 2 を含める。
    dL/dv = 2 delta a,  dL/dc = 2 delta
    dL/dW = 2 delta (v * gate) x^T,  dL/db = 2 delta (v * gate),  gate = 1[Wx+b > 0]
"""
import math
import torch

from .envs import kaiming_mlp_params


class VecMLP:
    def __init__(self, R, h, d, gen, device):
        self.R, self.h, self.d = R, h, d
        self.W, self.b, self.v, self.c = kaiming_mlp_params(R, h, d, gen, device)

    def params(self):
        return {"W": self.W, "b": self.b, "v": self.v, "c": self.c}

    def state_dict(self):
        return {k: p.clone() for k, p in self.params().items()}

    def load_state(self, s):
        self.W, self.b, self.v, self.c = (s[k].clone() for k in ("W", "b", "v", "c"))

    def forward(self, x):
        """x: [R,d] -> (pre [R,h], a [R,h], yhat [R])"""
        pre = torch.einsum("rhd,rd->rh", self.W, x) + self.b
        a = torch.relu(pre)
        yhat = (a * self.v).sum(dim=1) + self.c
        return pre, a, yhat

    def forward_batch(self, x):
        """x: [N,R,d] or [C,R,d] -> (pre, a, yhat) with leading batch dim."""
        pre = torch.einsum("rhd,nrd->nrh", self.W, x) + self.b
        a = torch.relu(pre)
        yhat = (a * self.v).sum(dim=-1) + self.c
        return pre, a, yhat

    def grads(self, x, pre, a, delta):
        """batch=1 オンライン勾配 (係数 2 込み)。x:[R,d], delta:[R]"""
        d2 = 2.0 * delta
        gv = d2[:, None] * a                        # [R,h]
        gc = d2                                     # [R]
        gate = (pre > 0).float()
        gb = d2[:, None] * self.v * gate            # [R,h]
        gW = gb[:, :, None] * x[:, None, :]         # [R,h,d]
        return gW, gb, gv, gc

    def grads_batch(self, x, pre, a, delta):
        """時間バッチ版 (凍結測定用)。x:[C,R,d], delta:[C,R] -> 各勾配 [C,R,...]"""
        d2 = 2.0 * delta
        gv = d2[..., None] * a
        gc = d2
        gate = (pre > 0).float()
        gb = d2[..., None] * self.v * gate
        gW = gb[..., None] * x[:, :, None, :]
        return gW, gb, gv, gc

    def sgd_step(self, lr, gW, gb, gv, gc):
        """lr: [R]"""
        self.W -= lr[:, None, None] * gW
        self.b -= lr[:, None] * gb
        self.v -= lr[:, None] * gv
        self.c -= lr * gc
