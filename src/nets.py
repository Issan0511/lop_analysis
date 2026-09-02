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


def _check_wd_b(wd_b, freeze_bias):
    """bias 専用 weight decay 係数の検証 [bias_wd_0901 §6]。

    負値と非有限値を弾き、`freeze_bias` との同時指定をエラーにする (b を固定する
    介入と b を減衰させる介入は同時に意味を持たない)。既定 0.0 は恒等で、
    `gb + 0.0*b` は有限な b に対し `gb` と bit 一致する (S1 で実測確認する)。"""
    wd_b = float(wd_b)
    if not math.isfinite(wd_b) or wd_b < 0.0:
        raise ValueError(f"wd_b must be a finite non-negative float, got {wd_b!r}")
    if freeze_bias and wd_b > 0.0:
        raise ValueError("freeze_bias=True and wd_b>0 cannot be combined")
    return wd_b


class VecMLP:
    def __init__(self, R, h, d, gen, device, act_alpha=0.0, freeze_bias=False,
                 wd_b=0.0):
        self.R, self.h, self.d = R, h, d
        self.act_alpha = act_alpha        # Leaky ReLU 負側勾配 (0.0 = ReLU, 既存互換)
        self.freeze_bias = freeze_bias    # True で b を初期値 0 に固定 [bias_margin_0814]
        # 隠れ層 bias だけに掛ける素の L2 勾配係数 (decoupled ではない)
        # [bias_wd_0901 §6]。W・v・出力 bias c には掛けない。
        self.wd_b = _check_wd_b(wd_b, freeze_bias)
        self.W, self.b, self.v, self.c = kaiming_mlp_params(R, h, d, gen, device)

    def set_weight_decay_b(self, wd_b):
        """構築後に wd_b を差し替える。乱数も状態も消費しない [bias_wd_0901 §6]。"""
        self.wd_b = _check_wd_b(wd_b, self.freeze_bias)
        return self

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
        b は初期値 0 のまま) [bias_margin_0814 §2.2]。

        wd_b > 0 のとき b の更新式だけが `b -= lr*(gb + wd_b*b)` になる
        [bias_wd_0901 §6]。分岐を置かないのは、wd_b=0 の腕が WD コード経路を
        通したうえで無 WD 実装と bit 一致することを S1 で検査可能にするため。"""
        self.W -= lr[:, None, None] * gW
        if not self.freeze_bias:
            self.b -= lr[:, None] * (gb + self.wd_b * self.b)
        self.v -= lr[:, None] * gv
        self.c -= lr * gc


class VecMLPL:
    """Vectorized depth-``L`` MLP with closed-form gradients.

    ``hidden`` is the width of each hidden layer.  The class deliberately keeps
    the ``L=1`` execution path and state-dict schema identical to :class:`VecMLP`:
    it calls :func:`kaiming_mlp_params` directly and exposes ``W``, ``b``, ``v``
    and ``c``.  This is the compatibility contract used by
    ``mlp2_phase0_0829`` S0.

    For ``L>1``, ``forward_layers`` returns all preactivations and activations;
    ``grads_layers`` backpropagates the squared-error derivative
    ``dL/dyhat = 2 * delta`` without autograd.

    ``act`` selects the hidden nonlinearity [elu_swamp_0830 §4.3].  ``"relu"``
    is the default and its execution path is untouched: :meth:`act_fn` returns
    ``torch.relu(pre)`` and :meth:`act_grad` returns ``(pre > 0)``, the exact
    expressions the frozen modules inline.  ``"elu"`` adds Clevert et al. 2015
    with ``phi(z) = alpha*(e^z - 1)`` on ``z <= 0``; the derivative reuses the
    forward activation (``phi(z) + alpha = alpha*e^z``) so ``exp`` is evaluated
    once and forward/backward cannot disagree numerically.  ``"leaky_relu"``
    keeps the ReLU kink while replacing the negative-side zero derivative with
    the constant ``act_alpha`` slope.  The choice consumes no randomness, so
    arms differing only in ``act`` share init, teacher, input stream and flip
    trajectory bit for bit.

    ``"bwd_leaky"`` and ``"fwd_leaky"`` split the ReLU kink's two directions
    [bwd_leak_0902 §4.3].  ``bwd_leaky`` is forward-strict-ReLU with a leaky
    *surrogate* backward slope (output 0, gradient ``a``); ``fwd_leaky`` is the
    mirror (output ``a*z``, gradient 0).  Neither is the gradient of its own
    forward map, so a finite-difference check is meaningless for them and
    ``S-bwd`` replaces it.  Both are built by **reusing the existing ``relu``
    and ``leaky_relu`` expressions verbatim** rather than writing new
    arithmetic, so ``S-cross`` can assert bit identity against the halves they
    are made of.
    """

    ACTIVATIONS = ("relu", "elu", "leaky_relu", "bwd_leaky", "fwd_leaky")
    SURROGATE_ACTIVATIONS = ("bwd_leaky", "fwd_leaky")

    def __init__(self, R, hidden, d, gen, device, act="relu", act_alpha=1.0,
                 act_grad_form="alpha_exp", wd_b=0.0):
        self.act_grad_form = "alpha_exp"
        # 全隠れ層の bias に掛ける素の L2 勾配係数 [bias_wd_0901 §6]。出力 bias c は
        # 対象外。既定 0.0 は恒等 (乱数消費も算術も無 WD 実装と bit 一致)。
        self.freeze_bias = False
        self.wd_b = _check_wd_b(wd_b, False)
        self.set_activation(act, act_alpha, act_grad_form)
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

    def set_activation(self, act, act_alpha=1.0, act_grad_form=None):
        """Switch the nonlinearity after construction.

        ``__init__`` never consults the activation, so this yields exactly the
        tensors a net constructed with ``act=`` would hold.  It is the hook
        ``elu_swamp_0830`` uses to vary phi while arm setup stays on the frozen
        ``mlp2_phase0.setup_arm`` path.
        """
        if act not in self.ACTIVATIONS:
            raise ValueError(f"unknown activation {act!r}")
        if act == "elu" and not float(act_alpha) >= 0.0:
            raise ValueError("ELU alpha must be non-negative")
        if act == "leaky_relu" and not 0.0 <= float(act_alpha) <= 1.0:
            raise ValueError("leaky ReLU slope must be in [0, 1]")
        if act in self.SURROGATE_ACTIVATIONS and not 0.0 <= float(act_alpha) <= 1.0:
            raise ValueError(f"{act} slope must be in [0, 1]")
        if act_grad_form is not None:
            if act_grad_form not in self.GRAD_FORMS:
                raise ValueError(f"unknown ELU derivative form {act_grad_form!r}")
            self.act_grad_form = str(act_grad_form)
        self.act = str(act)
        self.act_alpha = float(act_alpha)
        return self

    def act_fn(self, pre):
        """phi(pre).  The ReLU branch is literally ``torch.relu``."""
        if self.act == "relu":
            return torch.relu(pre)
        if self.act == "leaky_relu":
            return torch.where(pre > 0, pre, self.act_alpha * pre)
        if self.act == "bwd_leaky":
            # forward は厳密 ReLU。上の "relu" 分岐と同一の式を書く（S-cross）。
            return torch.relu(pre)
        if self.act == "fwd_leaky":
            # forward は leaky。上の "leaky_relu" 分岐と同一の式を書く（S-cross）。
            return torch.where(pre > 0, pre, self.act_alpha * pre)
        # expm1 keeps the small-|z| negative branch accurate; the positive
        # branch of `where` is selected before any overflow of expm1 matters.
        return torch.where(pre > 0, pre, self.act_alpha * torch.expm1(pre))

    GRAD_FORMS = ("alpha_exp", "activation_plus_alpha")

    def act_grad(self, pre, a):
        """phi'(pre).  ReLU: ``1[pre > 0]``.  ELU: ``alpha*e^z`` on ``z <= 0``.

        The two ELU forms are algebraically identical (``phi(z) + alpha =
        alpha*e^z``) but not numerically.  ``activation_plus_alpha`` reuses the
        forward activation and so calls ``exp`` once, but it cancels
        catastrophically as ``|phi| -> alpha``: in the float32 training dtype it
        is 4e-4 wrong at ``z=-10``, 6% wrong at ``z=-16`` and **exactly zero
        below z ~ -17.3**, which would silently turn ELU into an absorbing
        activation in the deep tail.  ``alpha_exp`` is accurate to ~1e-8
        relative at every depth and is what ``elu_swamp_0830`` §6 requires
        (1e-6 at ``z=-30``); it is the default.  The rejected form stays
        reachable so S-grad can put both numbers on the record.
        """
        if self.act == "relu":
            return (pre > 0).to(pre.dtype)
        if self.act == "leaky_relu":
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.full_like(pre, self.act_alpha))
        if self.act == "bwd_leaky":
            # backward は leaky。上の "leaky_relu" 分岐と同一の式（S-cross）。
            # これは phi' ではなく代替勾配なので、有限差分照合は成立しない。
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.full_like(pre, self.act_alpha))
        if self.act == "fwd_leaky":
            # backward は厳密 ReLU。上の "relu" 分岐と同一の式（S-cross）。
            return (pre > 0).to(pre.dtype)
        if self.act_grad_form == "activation_plus_alpha":
            return torch.where(pre > 0, torch.ones_like(a), a + self.act_alpha)
        return torch.where(pre > 0, torch.ones_like(pre),
                           self.act_alpha * torch.exp(pre))

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
            cur = self.act_fn(pre)
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
            cur = self.act_fn(pre)
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
        dz = d2[:, None] * self.v * self.act_grad(pres[-1], acts[-1])
        gWs = [None] * self.L
        gbs = [None] * self.L
        for layer in range(self.L - 1, -1, -1):
            inp = x if layer == 0 else acts[layer - 1]
            gbs[layer] = dz
            gWs[layer] = dz[:, :, None] * inp[:, None, :]
            if layer:
                dz = (torch.einsum("rhi,rh->ri", self.Ws[layer], dz)
                      * self.act_grad(pres[layer - 1], acts[layer - 1]))
        return gWs, gbs, gv, gc

    def grads_layers_batch(self, x, pres, acts, delta):
        """Closed-form gradients with leading time/sample batch dimension."""
        d2 = 2.0 * delta
        gv = d2[..., None] * acts[-1]
        gc = d2
        dz = d2[..., None] * self.v * self.act_grad(pres[-1], acts[-1])
        gWs = [None] * self.L
        gbs = [None] * self.L
        for layer in range(self.L - 1, -1, -1):
            inp = x if layer == 0 else acts[layer - 1]
            gbs[layer] = dz
            gWs[layer] = dz[..., None] * inp[:, :, None, :]
            if layer:
                dz = (torch.einsum("rhi,nrh->nri", self.Ws[layer], dz)
                      * self.act_grad(pres[layer - 1], acts[layer - 1]))
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

    def set_weight_decay_b(self, wd_b):
        """構築後に wd_b を差し替える。乱数も状態も消費しないので、arm 設定は
        凍結済みの ``mlp2_phase0.setup_arm`` 経路のままでよい [bias_wd_0901 §6]
        (``set_activation`` と同じ hook 方式)。"""
        self.wd_b = _check_wd_b(wd_b, self.freeze_bias)
        return self

    def sgd_step_layers(self, lr, gWs, gbs, gv, gc):
        """wd_b > 0 のとき**全隠れ層の** bias だけが `b -= lr*(gb + wd_b*b)` になる
        [bias_wd_0901 §6]。Ws・v・出力 bias c の更新式は wd_b に依らない。"""
        for i in range(self.L):
            self.Ws[i] -= lr[:, None, None] * gWs[i]
            self.bs[i] -= lr[:, None] * (gbs[i] + self.wd_b * self.bs[i])
        self.v -= lr[:, None] * gv
        self.c -= lr * gc

    def sgd_step(self, lr, gW, gb, gv, gc):
        if self.L != 1:
            raise ValueError("use sgd_step_layers for L > 1")
        self.sgd_step_layers(lr, [gW], [gb], gv, gc)
