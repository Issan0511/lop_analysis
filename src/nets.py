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
    def __init__(self, R, h, d, gen, device, act_alpha=0.0, freeze_bias=False):
        self.R, self.h, self.d = R, h, d
        self.act_alpha = act_alpha        # Leaky ReLU 負側勾配 (0.0 = ReLU, 既存互換)
        self.freeze_bias = freeze_bias    # True で b を初期値 0 に固定 [bias_margin_0814]
        self.W, self.b, self.v, self.c = kaiming_mlp_params(R, h, d, gen, device)

    def _act(self, pre):
        if self.act_alpha == 0.0:
            return torch.relu(pre)        # 既存実験と bit 一致を保つ
        return torch.where(pre > 0, pre, self.act_alpha * pre)

    def _gate(self, pre):
        gate = (pre > 0).float()
        if self.act_alpha == 0.0:
            return gate
        return gate + self.act_alpha * (1.0 - gate)

    def params(self):
        return {"W": self.W, "b": self.b, "v": self.v, "c": self.c}

    def state_dict(self):
        return {k: p.clone() for k, p in self.params().items()}

    def load_state(self, s):
        self.W, self.b, self.v, self.c = (s[k].clone() for k in ("W", "b", "v", "c"))

    def forward(self, x):
        """x: [R,d] -> (pre [R,h], a [R,h], yhat [R])"""
        pre = torch.einsum("rhd,rd->rh", self.W, x) + self.b
        a = self._act(pre)
        yhat = (a * self.v).sum(dim=1) + self.c
        return pre, a, yhat

    def forward_batch(self, x):
        """x: [N,R,d] or [C,R,d] -> (pre, a, yhat) with leading batch dim."""
        pre = torch.einsum("rhd,nrd->nrh", self.W, x) + self.b
        a = self._act(pre)
        yhat = (a * self.v).sum(dim=-1) + self.c
        return pre, a, yhat

    def grads(self, x, pre, a, delta):
        """batch=1 オンライン勾配 (係数 2 込み)。x:[R,d], delta:[R]"""
        d2 = 2.0 * delta
        gv = d2[:, None] * a                        # [R,h]
        gc = d2                                     # [R]
        gate = self._gate(pre)
        gb = d2[:, None] * self.v * gate            # [R,h]
        gW = gb[:, :, None] * x[:, None, :]         # [R,h,d]
        return gW, gb, gv, gc

    def grads_batch(self, x, pre, a, delta):
        """時間バッチ版 (凍結測定用)。x:[C,R,d], delta:[C,R] -> 各勾配 [C,R,...]"""
        d2 = 2.0 * delta
        gv = d2[..., None] * a
        gc = d2
        gate = self._gate(pre)
        gb = d2[..., None] * self.v * gate
        gW = gb[..., None] * x[:, :, None, :]
        return gW, gb, gv, gc

    def sgd_step(self, lr, gW, gb, gv, gc):
        """lr: [R]。freeze_bias が真なら b の更新のみ止める (勾配計算・乱数消費は不変、
        b は初期値 0 のまま) [bias_margin_0814 §2.2]。"""
        self.W -= lr[:, None, None] * gW
        if not self.freeze_bias:
            self.b -= lr[:, None] * gb
        self.v -= lr[:, None] * gv
        self.c -= lr * gc


class VecMLPL:
    """Vectorized depth-``L`` ReLU MLP with closed-form gradients.

    ``hidden`` is the width of each hidden layer.  The class deliberately keeps
    the ``L=1`` execution path and state-dict schema identical to :class:`VecMLP`:
    it calls :func:`kaiming_mlp_params` directly and exposes ``W``, ``b``, ``v``
    and ``c``.  This is the compatibility contract used by
    ``mlp2_phase0_0829`` S0.

    For ``L>1``, ``forward_layers`` returns all preactivations and activations;
    ``grads_layers`` backpropagates the squared-error derivative
    ``dL/dyhat = 2 * delta`` without autograd.
    """

    def __init__(self, R, hidden, d, gen, device):
        if isinstance(hidden, int):
            hidden = [hidden]
        hidden = [int(h) for h in hidden]
        if not hidden or any(h <= 0 for h in hidden):
            raise ValueError("hidden must contain one or more positive widths")

        self.R, self.d = int(R), int(d)
        self.hidden = tuple(hidden)
        self.L = len(hidden)

        if self.L == 1:
            # Do not refactor this branch: S0 requires the exact same generator
            # consumption and arithmetic as VecMLP.
            W, b, self.v, self.c = kaiming_mlp_params(
                self.R, hidden[0], self.d, gen, device
            )
            self.Ws = [W]
            self.bs = [b]
        else:
            self.Ws, self.bs = [], []
            fan_in = self.d
            for fan_out in hidden:
                bound = math.sqrt(6.0 / fan_in)
                W = ((torch.rand(self.R, fan_out, fan_in, generator=gen,
                                 device=device) * 2 - 1) * bound)
                self.Ws.append(W)
                self.bs.append(torch.zeros(self.R, fan_out, device=device))
                fan_in = fan_out
            out_bound = math.sqrt(3.0 / hidden[-1])
            self.v = ((torch.rand(self.R, hidden[-1], generator=gen,
                                  device=device) * 2 - 1) * out_bound)
            self.c = torch.zeros(self.R, device=device)

        # Legacy aliases are useful to the exact one-layer instrumentation.  No
        # code should interpret them as "all layers" when L > 1.
        self.W = self.Ws[0]
        self.b = self.bs[0]
        self.h = self.hidden[0]

    def params(self):
        if self.L == 1:
            return {"W": self.Ws[0], "b": self.bs[0], "v": self.v, "c": self.c}
        out = {f"W{i + 1}": W for i, W in enumerate(self.Ws)}
        out.update({f"b{i + 1}": b for i, b in enumerate(self.bs)})
        out.update(v=self.v, c=self.c)
        return out

    def state_dict(self):
        return {k: p.clone() for k, p in self.params().items()}

    def load_state(self, state):
        if self.L == 1 and "W" in state:
            self.Ws = [state["W"].clone()]
            self.bs = [state["b"].clone()]
        else:
            self.Ws = [state[f"W{i + 1}"].clone() for i in range(self.L)]
            self.bs = [state[f"b{i + 1}"].clone() for i in range(self.L)]
        self.v, self.c = state["v"].clone(), state["c"].clone()
        self.W, self.b = self.Ws[0], self.bs[0]

    def forward_layers(self, x):
        """``x:[R,d] -> (pres, acts, yhat)`` for an online step."""
        pres, acts = [], []
        cur = x
        for W, b in zip(self.Ws, self.bs):
            pre = torch.einsum("rhd,rd->rh", W, cur) + b
            cur = torch.relu(pre)
            pres.append(pre)
            acts.append(cur)
        yhat = (acts[-1] * self.v).sum(dim=1) + self.c
        return pres, acts, yhat

    def forward_layers_batch(self, x):
        """``x:[N,R,d] -> (pres, acts, yhat)`` with a leading batch dim."""
        pres, acts = [], []
        cur = x
        for W, b in zip(self.Ws, self.bs):
            pre = torch.einsum("rhd,nrd->nrh", W, cur) + b
            cur = torch.relu(pre)
            pres.append(pre)
            acts.append(cur)
        yhat = (acts[-1] * self.v).sum(dim=-1) + self.c
        return pres, acts, yhat

    def forward(self, x):
        pres, acts, yhat = self.forward_layers(x)
        if self.L == 1:
            return pres[0], acts[0], yhat
        return pres, acts, yhat

    def forward_batch(self, x):
        pres, acts, yhat = self.forward_layers_batch(x)
        if self.L == 1:
            return pres[0], acts[0], yhat
        return pres, acts, yhat

    def grads_layers(self, x, pres, acts, delta):
        """Closed-form online gradients, including the squared-loss factor 2."""
        d2 = 2.0 * delta
        gv = d2[:, None] * acts[-1]
        gc = d2
        dz = d2[:, None] * self.v * (pres[-1] > 0).float()
        gWs = [None] * self.L
        gbs = [None] * self.L
        for layer in range(self.L - 1, -1, -1):
            inp = x if layer == 0 else acts[layer - 1]
            gbs[layer] = dz
            gWs[layer] = dz[:, :, None] * inp[:, None, :]
            if layer:
                dz = (torch.einsum("rhi,rh->ri", self.Ws[layer], dz)
                      * (pres[layer - 1] > 0).float())
        return gWs, gbs, gv, gc

    def grads_layers_batch(self, x, pres, acts, delta):
        """Closed-form gradients with leading time/sample batch dimension."""
        d2 = 2.0 * delta
        gv = d2[..., None] * acts[-1]
        gc = d2
        dz = d2[..., None] * self.v * (pres[-1] > 0).float()
        gWs = [None] * self.L
        gbs = [None] * self.L
        for layer in range(self.L - 1, -1, -1):
            inp = x if layer == 0 else acts[layer - 1]
            gbs[layer] = dz
            gWs[layer] = dz[..., None] * inp[:, :, None, :]
            if layer:
                dz = (torch.einsum("rhi,nrh->nri", self.Ws[layer], dz)
                      * (pres[layer - 1] > 0).float())
        return gWs, gbs, gv, gc

    def grads(self, x, pre, a, delta):
        if self.L != 1:
            raise ValueError("use grads_layers for L > 1")
        gWs, gbs, gv, gc = self.grads_layers(x, [pre], [a], delta)
        return gWs[0], gbs[0], gv, gc

    def grads_batch(self, x, pre, a, delta):
        if self.L != 1:
            raise ValueError("use grads_layers_batch for L > 1")
        gWs, gbs, gv, gc = self.grads_layers_batch(x, [pre], [a], delta)
        return gWs[0], gbs[0], gv, gc

    def sgd_step_layers(self, lr, gWs, gbs, gv, gc):
        for i in range(self.L):
            self.Ws[i] -= lr[:, None, None] * gWs[i]
            self.bs[i] -= lr[:, None] * gbs[i]
        self.v -= lr[:, None] * gv
        self.c -= lr * gc

    def sgd_step(self, lr, gW, gb, gv, gc):
        if self.L != 1:
            raise ValueError("use sgd_step_layers for L > 1")
        self.sgd_step_layers(lr, [gW], [gb], gv, gc)
