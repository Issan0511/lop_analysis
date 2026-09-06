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
import torch.nn.functional as F      # softplus_b だけが使う [edge_law_0905 §3.2]

from .envs import kaiming_mlp_params


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _std_normal_cdf(t):
    """``Phi(t)``。GELU は exact(erf) で書く。tanh 近似は使わない。"""
    return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


def _std_normal_pdf(t):
    """``phi(t)``。float32 では ``|t| > ~13.2`` で厳密 0 になる（S-num が測る）。"""
    return torch.exp(-0.5 * t * t) * _INV_SQRT2PI


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

    ``"silu"`` and ``"gelu"`` are the two smooth families that converge to ReLU
    as their steepness ``act_alpha`` (= beta) grows [gate_dial_0902 §4.3].  Both
    are written in closed form like every other branch (no autograd) and both
    are genuine derivatives of their own forward map, so ``S-fd`` can check them
    against a float64 central difference.  ``silu`` is ``z*sigmoid(beta*z)``
    with derivative ``sigmoid(b z)*(1 + b z*(1 - sigmoid(b z)))``; ``gelu`` is
    the **exact** (erf) form ``z*Phi(beta*z)`` with derivative
    ``Phi(b z) + (b z)*phi(b z)``.  The tanh approximation is not used.  Unlike
    ReLU/leaky/ELU both have a negative-side zero of the derivative (the
    "valley"), which is why they are in the dial experiment at all.

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

    ACTIVATIONS = ("relu", "elu", "leaky_relu", "bwd_leaky", "fwd_leaky",
                   "silu", "gelu", "silu_clamp", "gelu_clamp",
                   "silu_clamp0", "gelu_clamp0",
                   "bwd_reflect", "bwd_quad", "bwd_leaky_proj",
                   "mirror_leaky",
                   "fold_leaky_d1", "fold_leaky_d2", "fold_leaky_dbig",
                   "band_leaky_d0", "band_leaky_d0p5", "band_leaky_d1",
                   "band_leaky_d2", "band_leaky_d4",
                   "ramp_leaky_d1",
                   "comb_binf", "comb_b5",
                   "comb1_flat", "comb1_leaky", "band_leaky_dpi", "snake")
    SURROGATE_ACTIVATIONS = ("bwd_leaky", "fwd_leaky", "bwd_reflect", "bwd_quad",
                             "bwd_leaky_proj")
    # 幻の壁 3 型 [phantom_wall_0902 §4.3]。どれも forward は厳密 ReLU で、
    # backward の負側だけが違う: -a（反転）・a_Q*z（深さ比例の反転）・+a（bwd_leaky と
    # 同一。区別する µ 射影は act_grad では書けず、組み立てた勾配へ
    # phantom_wall_0902.grads_phantom が掛ける）。
    #   * act_alpha は **常に +a を格納**し、符号は act_grad の分岐で付ける
    #     （set_activation の凍結済み [0,1] ガードを緩めないため）
    #   * bwd_quad の act_alpha は a_Q で **1/z の次元。傾きではない**
    #   * 反転分岐は `0.0 - act_alpha` と書く。`-act_alpha` だと a=0 で -0.0 になり、
    #     torch.equal は符号盲なので S-limit がバイトの違いを見逃す
    PHANTOM_ACTIVATIONS = ("bwd_reflect", "bwd_quad", "bwd_leaky_proj")
    # phi' の零点を負側に持つ族（谷）。act_alpha は傾きではなく鋭さ beta。
    STEEPNESS_ACTIVATIONS = ("silu", "gelu", "silu_clamp", "gelu_clamp",
                             "silu_clamp0", "gelu_clamp0")
    # 谷の向こうを谷底の値で埋めた版 [valley_clamp_0902]。谷底 z_c = -u*/beta より
    # 深い側では phi は定数 phi(z_c)、phi' は 0。谷の手前は silu / gelu と**同一の式を
    # 同一の入力に**当てるので bit 一致する。u* は phi' の負側で最初の零点（beta=1）。
    # 逃走（谷の向こうで phi' < 0）だけを消し、浅い硬い床を残す介入。
    VALLEY_ZERO = {"silu_clamp": 1.2784645427610739,
                   "gelu_clamp": 0.7517915246935645,
                   "silu_clamp0": 1.2784645427610739,
                   "gelu_clamp0": 0.7517915246935645}
    # 床ゼロ版 [valley_clamp0_0902]。谷底より深い側で phi ≡ 0・phi' ≡ 0（z_c に
    # phi(z_c) ぶんの段差がある）。谷の手前は clamp 版と同じ式・同じ入力なので bit 一致。
    # 凍結ユニットの出力が厳密 0 になるので v_i も凍る（ReLU の死ユニットと同型）。
    # clamp 版（床 = phi(z_c) ≠ 0・v_i に勾配が流れ続ける）との差が床の値の効果。

    # 謎関数ダイヤル 5 族 [weird_act_0903 §2]。どれも z >= 0 で恒等（ReLU と同じ正側）で、
    # **負側の形を 1 か所だけ**設計する。5 族とも **自分の forward の真の導関数**なので
    # 有限差分で検算できる（折れ目は測度 0。bwd_leak の代替勾配とは違う）。
    #   * act_alpha は leaky 由来 4 族で傾き a、櫛で振動数 alpha
    #   * 第 2 母数（帯の深さ d・包絡 beta）は **活性化名に埋め込み**、下の
    #     クラス定数辞書で引く（VALLEY_ZERO と同じ流儀。set_activation の署名を変えない）
    #   * 反転・0 倍の分岐は `0.0 - a` / `0.0 +` と書く（a=0 や z=-2d で -0.0 を作らない。
    #     torch.equal は符号盲なので、これが無いと S-limit がバイトの違いを見逃す。追補 9）
    #   * `*_dbig` / `*_d0` は S-limit 専用の退化点（d=1e6 で leaky、d=0 で leaky）。
    #     本走の腕には使わない [weird_act_0903 §10 追補 9]
    WEIRD_SLOPE_ACTIVATIONS = ("mirror_leaky", "fold_leaky_d1", "fold_leaky_d2",
                               "fold_leaky_dbig", "band_leaky_d0",
                               "band_leaky_d0p5", "band_leaky_d1",
                               "band_leaky_d2", "band_leaky_d4", "ramp_leaky_d1",
                               "band_leaky_dpi")
    # 櫛は act_alpha が振動数で [0,1] に入らない（CB_a2 は alpha=2）。有限正だけを課す。
    # comb1_* と snake も act_alpha は振動数 alpha（漏れ a はクラス定数）。
    WEIRD_FREQ_ACTIVATIONS = ("comb_binf", "comb_b5",
                              "comb1_flat", "comb1_leaky", "snake")
    # 折り返しの折れ目の深さ d。phi の零点が 0 と -2d の 2 か所（分水嶺 d・極小 2d）。
    FOLD_DEPTH = {"fold_leaky_d1": 1.0, "fold_leaky_d2": 2.0,
                  "fold_leaky_dbig": 1.0e6}
    # 死帯の幅 d。d=0 は leaky、d=inf は ReLU。
    BAND_WIDTH = {"band_leaky_d0": 0.0, "band_leaky_d0p5": 0.5,
                  "band_leaky_d1": 1.0, "band_leaky_d2": 2.0,
                  "band_leaky_d4": 4.0,
                  # 幅 pi の死帯 [comb_isolate_0903]。comb1_leaky の葉を平坦に置き換えた
                  # 対照なので、幅は葉の端 pi/alpha（alpha=1）に合わせる。
                  "band_leaky_dpi": math.pi}
    # 滑り出しの二次区間の深さ d。kappa = a/(2d) は分岐内で解き直す（VALLEY_ZERO と同じで、
    # forward と backward のあいだで第 2 母数がずれないようにキャッシュしない）。
    RAMP_DEPTH = {"ramp_leaky_d1": 1.0}
    # 櫛の包絡 beta。inf は包絡なし（env = 1.0・1/beta = 0.0 を定数で書き exp を呼ばない）。
    COMB_ENVELOPE = {"comb_binf": float("inf"), "comb_b5": 5.0}
    # 櫛の分離 [comb_isolate_0903 §2]。負側の**最初の 1 葉だけ**を残し、その先を
    # ReLU の平坦（comb1_flat）か leaky（comb1_leaky）にする。葉の端は u = pi/alpha で
    # そこは二重零点（phi = phi' = 0）なので、平坦との接続は C^1 で、leaky との接続は
    # C^0（折れ目・S-fd で除外する）。
    #   * act_alpha は **振動数 alpha**（漏れ a ではない）
    #   * comb1_leaky の漏れ a は LR_1216 に揃えたクラス定数（下）
    COMB1_ACTIVATIONS = ("comb1_flat", "comb1_leaky")
    COMB1_LEAK = {"comb1_leaky": 0.1}

    # ---- 上端則 11 名 [edge_law_0905 §3.2]（逐語登録・等価な別形に書き換えない）----
    # どれも **自分の forward の真の導関数**を持ち、`act_curv` にも分岐がある
    # （§5 S-fd / S-curv / S-fallthrough）。既存の名前・式・タプルは 1 行も
    # 書き換えず、**足すだけ**にする（既存腕の実行経路を byte 一致に保つため。
    # ACTIVATIONS 等は下でタプル連結して伸ばす）。
    #   * `flip_leaky` … leaky の奇鏡像 -phi(-z)。述語は **`< 0`**
    #   * `shelf_leaky_d*` … 折れ目を深さ d に置いた leaky（`d0` は S-limit 専用）
    #   * `steep_shelf_d*` … 折れ目位置と上側傾きは棚と同じで、下側傾きだけ 2
    #   * `softplus_b` / `tanh_b` … 滑らかな 2 族（beta = 1）
    EDGE_LAW_ACTIVATIONS = ("flip_leaky",
                            "shelf_leaky_d0", "shelf_leaky_d0p5",
                            "shelf_leaky_d1", "shelf_leaky_d2",
                            "shelf_leaky_d3", "shelf_leaky_d30",
                            "steep_shelf_d1", "steep_shelf_d2",
                            "softplus_b", "tanh_b")
    # 棚の折れ目の深さ d（第 2 母数は名前に埋め、クラス定数辞書で引く。
    # BAND_WIDTH / FOLD_DEPTH と同じ流儀で set_activation の署名を変えない）。
    # d=0 は leaky と bit 一致する退化点で **S-limit 専用**（本走に使わない）。
    SHELF_DEPTH = {"shelf_leaky_d0": 0.0, "shelf_leaky_d0p5": 0.5,
                   "shelf_leaky_d1": 1.0, "shelf_leaky_d2": 2.0,
                   "shelf_leaky_d3": 3.0, "shelf_leaky_d30": 30.0}
    # 傾きを反転した棚の折れ目の深さ d。下側の傾き 2 は **act_alpha に入れない**
    # （WEIRD_SLOPE_ACTIVATIONS の [0,1] ガードに触れるため）。
    STEEP_DEPTH = {"steep_shelf_d1": 1.0, "steep_shelf_d2": 2.0}
    STEEP_SLOPE = 2.0
    SOFTPLUS_BETA = 1.0
    TANH_BETA = 1.0
    # act_alpha を**使わない**族。dial が誤って 0.1 などで渡ると、名前だけ合って
    # いて実際の関数が別物になる事故が黙って通るので、`act_alpha == 1.0` を要求する。
    UNIT_ALPHA_ACTIVATIONS = ("steep_shelf_d1", "steep_shelf_d2",
                              "softplus_b", "tanh_b")
    # phi'' が恒等的に 0 な区分線形族（折れ目は測度 0）。線形は leaky の a=1。
    ZERO_CURVATURE_ACTIVATIONS = (
        ("relu", "leaky_relu", "flip_leaky")
        + tuple(SHELF_DEPTH) + tuple(STEEP_DEPTH))
    ACTIVATIONS = ACTIVATIONS + EDGE_LAW_ACTIVATIONS
    # 傾き a を act_alpha に持つ族は [0,1] ガードへ（S-guard）。
    WEIRD_SLOPE_ACTIVATIONS = (WEIRD_SLOPE_ACTIVATIONS
                               + ("flip_leaky",) + tuple(SHELF_DEPTH))
    # Snake の「反転」を切り分ける 2 族 [snake_flip_0906 §3]。act_alpha は振動数 α。
    #   * snake1: 第 1 罠までの 1 葉 [-3π/4α, +π/4α] だけ Snake、外は z + 1/(2α)
    #     （連続・傾き 1。切る点は φ' の極大なので継ぎ目で φ' が 2→1 に跳ぶ C^0）
    #   * snake_amp{A}: φ = z + A sin²(αz)/α（φ' = 1 + A sin 2αz・零点なし・反転位置は同じ）。
    #     A=1 は S-limit 専用で snake と bit 一致（1.0*x は x と bit 一致）
    SNAKE_AMP = {"snake_amp0p25": 0.25, "snake_amp0p5": 0.5, "snake_amp1": 1.0}
    SNAKE_FLIP_ACTIVATIONS = ("snake1",) + tuple(SNAKE_AMP)
    ACTIVATIONS = ACTIVATIONS + SNAKE_FLIP_ACTIVATIONS
    WEIRD_FREQ_ACTIVATIONS = WEIRD_FREQ_ACTIVATIONS + SNAKE_FLIP_ACTIVATIONS
    # φ に定数 c を足した 2 族 [act_offset_0906 §3]。φ′・φ″ は元の族と**同一**で、
    # 変わるのは |φ| の偏り（E_支持[φ²] の最小点の位置）だけ。
    #   * leaky_off_{c}: φ = leaky_relu(z; a) + c。act_alpha は傾き a
    #     （[0,1] ガードは WEIRD_SLOPE_ACTIVATIONS で leaky と共有）。φ″ ≡ 0。
    #     **c=0 は加算せず leaky_relu の式を逐語で返す**（`x + 0.0` は -0.0 の
    #     符号だけ +0.0 に変え得るので、S-limit の bit 一致はこの書き方で担保する）
    #   * elu_off_{c}: φ = elu(z; α) + c。act_alpha は α（≥0 ガード）。導関数は
    #     `alpha_exp` 形のみ（`activation_plus_alpha` は φ+c を見るので使えない）
    #   名前と定数は明示 dict（SNAKE_AMP と同じ流儀）。config の `offset` と S-const で
    #   突き合わせる。既存の名前・式・タプルは 1 行も書き換えず、**足すだけ**。
    LEAKY_OFFSET = {"leaky_off_m2": -2.0, "leaky_off_m0p5": -0.5,
                    "leaky_off_0": 0.0, "leaky_off_p0p5": 0.5, "leaky_off_p2": 2.0}
    ELU_OFFSET = {"elu_off_m1": -1.0, "elu_off_p1": 1.0}
    ACT_OFFSET_ACTIVATIONS = tuple(LEAKY_OFFSET) + tuple(ELU_OFFSET)
    ACTIVATIONS = ACTIVATIONS + ACT_OFFSET_ACTIVATIONS
    WEIRD_SLOPE_ACTIVATIONS = WEIRD_SLOPE_ACTIVATIONS + tuple(LEAKY_OFFSET)
    ZERO_CURVATURE_ACTIVATIONS = ZERO_CURVATURE_ACTIVATIONS + tuple(LEAKY_OFFSET)

    def __init__(self, R, hidden, d, gen, device, act="relu", act_alpha=1.0,
                 act_grad_form="alpha_exp", wd_b=0.0):
        self.act_grad_form = "alpha_exp"
        # 全隠れ層の bias に掛ける素の L2 勾配係数 [bias_wd_0901 §6]。出力 bias c は
        # 対象外。既定 0.0 は恒等 (乱数消費も算術も無 WD 実装と bit 一致)。
        self.freeze_bias = False
        self.wd_b = _check_wd_b(wd_b, False)
        # 隠れ層の W だけに掛ける素の L2 勾配係数 [phantom_wall_0902 §4.3]。
        # v・b・出力 bias c は対象外。既定 0.0 は恒等（`gW + 0.0*W` は有限な W に
        # 対し `gW` と bit 一致する。S-limit-w が実測する）。
        self.wd_w = 0.0
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
        if act in self.STEEPNESS_ACTIVATIONS:
            beta = float(act_alpha)
            if not (math.isfinite(beta) and beta > 0.0):
                raise ValueError(f"{act} beta must be a finite positive float")
        if act in self.WEIRD_SLOPE_ACTIVATIONS and not 0.0 <= float(act_alpha) <= 1.0:
            raise ValueError(f"{act} slope must be in [0, 1]")
        if act in self.WEIRD_FREQ_ACTIVATIONS:
            freq = float(act_alpha)
            if not (math.isfinite(freq) and freq > 0.0):
                raise ValueError(f"{act} frequency must be a finite positive float")
        if act in self.UNIT_ALPHA_ACTIVATIONS and float(act_alpha) != 1.0:
            # 傾き 2 も beta も act_alpha には入っていない [edge_law_0905 §3.2]。
            # dial が 1.0 以外で渡ったら、名前は合っているのに別の関数を走らせて
            # いる（腕表の写し間違い）ので落とす（S-guard）。
            raise ValueError(f"{act} requires act_alpha == 1.0, got {act_alpha!r}")
        if act in self.ELU_OFFSET and not float(act_alpha) >= 0.0:
            # elu_off_* は elu と同じ α ガード [act_offset_0906 §3]（S-guard）。
            raise ValueError(f"{act} alpha must be non-negative")
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
        if self.act in ("bwd_reflect", "bwd_quad", "bwd_leaky_proj"):
            # 幻の 3 型はどれも forward が厳密 ReLU。"relu" 分岐と同一の式（S-cross）。
            return torch.relu(pre)
        if self.act == "silu":
            # torch.sigmoid は両裾で安定。exp(-beta*z) を裸で書かない（S-num）。
            return pre * torch.sigmoid(self.act_alpha * pre)
        if self.act == "gelu":
            return pre * _std_normal_cdf(self.act_alpha * pre)
        if self.act == "silu_clamp":
            zc = -self.VALLEY_ZERO["silu_clamp"] / self.act_alpha
            zz = torch.clamp(pre, min=zc)      # 谷底より深い側は谷底の値
            return zz * torch.sigmoid(self.act_alpha * zz)
        if self.act == "gelu_clamp":
            zc = -self.VALLEY_ZERO["gelu_clamp"] / self.act_alpha
            zz = torch.clamp(pre, min=zc)
            return zz * _std_normal_cdf(self.act_alpha * zz)
        if self.act == "silu_clamp0":
            zc = -self.VALLEY_ZERO["silu_clamp0"] / self.act_alpha
            return torch.where(pre > zc, pre * torch.sigmoid(self.act_alpha * pre),
                               torch.zeros_like(pre))
        if self.act == "gelu_clamp0":
            zc = -self.VALLEY_ZERO["gelu_clamp0"] / self.act_alpha
            return torch.where(pre > zc, pre * _std_normal_cdf(self.act_alpha * pre),
                               torch.zeros_like(pre))
        if self.act == "mirror_leaky":
            # leaky の負側を z 軸で折り返す。phi^2 は leaky と同一、phi' の符号だけ逆。
            # `0.0 +` は a=0 のとき負側が -0.0 になるのを防ぐ（追補 9）。
            return torch.where(pre > 0, pre,
                               0.0 + (0.0 - self.act_alpha) * pre)
        if self.act in self.FOLD_DEPTH:
            d = self.FOLD_DEPTH[self.act]
            # 深さ d で傾きが反転する V 字。phi の零点は 0 と -2d。
            return torch.where(pre > 0, pre,
                               torch.where(pre > -d, self.act_alpha * pre,
                                           0.0 + (0.0 - self.act_alpha)
                                           * (pre + 2.0 * d)))
        if self.act in self.BAND_WIDTH:
            d = self.BAND_WIDTH[self.act]
            # 幅 d の死帯（ReLU の壁）の先に leaky。d=0 で leaky の式に退化する。
            return torch.where(pre > 0, pre,
                               torch.where(pre > -d, torch.zeros_like(pre),
                                           self.act_alpha * (pre + d)))
        if self.act in self.RAMP_DEPTH:
            d = self.RAMP_DEPTH[self.act]
            kappa = self.act_alpha / (2.0 * d)
            # C^1。壁で phi'=0 だが深さ d で leaky に接続する（値も傾きも連続）。
            return torch.where(pre > 0, pre,
                               torch.where(pre > -d,
                                           0.0 - kappa * pre * pre,
                                           self.act_alpha * pre
                                           + self.act_alpha * d / 2.0))
        if self.act in self.COMB1_ACTIVATIONS:
            lobe = math.pi / self.act_alpha
            leaf = 0.0 - torch.sin(self.act_alpha * pre) ** 2
            beyond = (torch.zeros_like(pre) if self.act == "comb1_flat"
                      else self.COMB1_LEAK["comb1_leaky"] * (pre + lobe))
            return torch.where(pre > 0, pre,
                               torch.where(pre > -lobe, leaf, beyond))
        if self.act == "snake":
            # ゲートを持たない周期活性化 [Ziyin et al. 2020]。**正側も恒等ではない**。
            # 単調（phi' = 1 + sin 2az >= 0）で、負側の可動度の平均は 1。
            return pre + torch.sin(self.act_alpha * pre) ** 2 / self.act_alpha
        if self.act == "snake1":
            # 1 葉だけの Snake [snake_flip_0906 §3]。葉の内側は snake と同一の式。
            a = self.act_alpha
            lo, hi = -3.0 * math.pi / (4.0 * a), math.pi / (4.0 * a)
            inside = pre + torch.sin(a * pre) ** 2 / a
            return torch.where((pre >= lo) & (pre <= hi), inside, pre + 0.5 / a)
        if self.act in self.SNAKE_AMP:
            A = self.SNAKE_AMP[self.act]
            return pre + A * torch.sin(self.act_alpha * pre) ** 2 / self.act_alpha
        if self.act == "comb_binf":
            # 包絡なしの櫛。env = 1.0 を定数で書き、exp を呼ばない（S-num）。
            return torch.where(pre > 0, pre,
                               0.0 - torch.sin(self.act_alpha * pre) ** 2)
        if self.act == "comb_b5":
            beta = self.COMB_ENVELOPE["comb_b5"]
            env = torch.exp(-pre / beta)
            return torch.where(pre > 0, pre,
                               0.0 - env * torch.sin(self.act_alpha * pre) ** 2)
        if self.act == "flip_leaky":
            # leaky の奇鏡像 -phi(-z) [edge_law_0905 §3.2]。**述語は `< 0`**:
            # `> 0` で書くと phi'(±0.0) が 1 になり、鏡像の要請
            # phi~'(u) = phi'(-u) = a と食い違う（S-flip がバイトで落とす）。
            return torch.where(pre < 0, pre, self.act_alpha * pre)
        if self.act in self.SHELF_DEPTH:
            d = self.SHELF_DEPTH[self.act]
            # z >= -d で恒等・折れ目より下で傾き a。折れ目ちょうどで恒等枝と
            # 厳密に連続（`a*pre + (a-1)*d` は丸めが 2 回入り -d を厳密に返さない）。
            return torch.where(pre > -d, pre, self.act_alpha * (pre + d) - d)
        if self.act in self.STEEP_DEPTH:
            d = self.STEEP_DEPTH[self.act]
            # 棚と折れ目位置・上側傾きを揃え、下側の傾きだけ 2（曲率の符号が逆）。
            return torch.where(pre > -d, pre, self.STEEP_SLOPE * (pre + d) - d)
        if self.act == "softplus_b":
            # beta は SOFTPLUS_BETA（= 1.0・S-const が config と突き合わせる）。
            return F.softplus(pre, beta=1.0)
        if self.act == "tanh_b":
            return torch.tanh(pre)
        if self.act == "leaky_off_0":
            # c=0 は加算せず "leaky_relu" 分岐と**同一の式**を返す [act_offset_0906 §3]
            # （S-limit: bit 一致）。`+ 0.0` は -0.0 を +0.0 に変えるので書かない。
            return torch.where(pre > 0, pre, self.act_alpha * pre)
        if self.act in self.LEAKY_OFFSET:
            # leaky_relu の式（上と同一）に定数 c を足す。φ′ は leaky と同一。
            return (torch.where(pre > 0, pre, self.act_alpha * pre)
                    + self.LEAKY_OFFSET[self.act])
        if self.act in self.ELU_OFFSET:
            # ELU の式（末尾の fallthrough と同一）に定数 c を足す。
            return (torch.where(pre > 0, pre, self.act_alpha * torch.expm1(pre))
                    + self.ELU_OFFSET[self.act])
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
        if self.act == "bwd_reflect":
            # 負側ゲートを **反転** する。`0.0 - a` と書くのは a=0 で -0.0 を
            # 作らないため（追補 9）。0.0 - 0.1 は厳密に -0.1 なので a=0.1 は不変。
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.full_like(pre, 0.0 - self.act_alpha))
        if self.act == "bwd_quad":
            # 負側ゲートが深さ比例。act_alpha は a_Q で 1/z の次元。
            # a_Q=0 かつ pre<0 で -0.0 を作らないよう 0.0 + を噛ませる（追補 9）。
            return torch.where(pre > 0, torch.ones_like(pre),
                               0.0 + self.act_alpha * pre)
        if self.act == "bwd_leaky_proj":
            # act_grad は bwd_leaky と同一の式。µ 射影は勾配を組み立てたあとに
            # phantom_wall_0902.grads_phantom が掛ける（act_grad では書けない）。
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.full_like(pre, self.act_alpha))
        if self.act == "silu":
            s = torch.sigmoid(self.act_alpha * pre)
            return s * (1.0 + self.act_alpha * pre * (1.0 - s))
        if self.act == "gelu":
            t = self.act_alpha * pre
            return _std_normal_cdf(t) + t * _std_normal_pdf(t)
        if self.act == "silu_clamp":
            zc = -self.VALLEY_ZERO["silu_clamp"] / self.act_alpha
            s = torch.sigmoid(self.act_alpha * pre)
            g = s * (1.0 + self.act_alpha * pre * (1.0 - s))
            return torch.where(pre > zc, g, torch.zeros_like(g))
        if self.act == "gelu_clamp":
            zc = -self.VALLEY_ZERO["gelu_clamp"] / self.act_alpha
            t = self.act_alpha * pre
            g = _std_normal_cdf(t) + t * _std_normal_pdf(t)
            return torch.where(pre > zc, g, torch.zeros_like(g))
        if self.act == "silu_clamp0":
            zc = -self.VALLEY_ZERO["silu_clamp0"] / self.act_alpha
            s = torch.sigmoid(self.act_alpha * pre)
            g = s * (1.0 + self.act_alpha * pre * (1.0 - s))
            return torch.where(pre > zc, g, torch.zeros_like(g))
        if self.act == "gelu_clamp0":
            zc = -self.VALLEY_ZERO["gelu_clamp0"] / self.act_alpha
            t = self.act_alpha * pre
            g = _std_normal_cdf(t) + t * _std_normal_pdf(t)
            return torch.where(pre > zc, g, torch.zeros_like(g))
        if self.act == "mirror_leaky":
            # 負側ゲートを反転する。`0.0 - a` と書くのは a=0 で -0.0 を作らないため（追補 9）。
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.full_like(pre, 0.0 - self.act_alpha))
        if self.act in self.FOLD_DEPTH:
            d = self.FOLD_DEPTH[self.act]
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.where(pre > -d,
                                           torch.full_like(pre, self.act_alpha),
                                           torch.full_like(pre,
                                                           0.0 - self.act_alpha)))
        if self.act in self.BAND_WIDTH:
            d = self.BAND_WIDTH[self.act]
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.where(pre > -d, torch.zeros_like(pre),
                                           torch.full_like(pre, self.act_alpha)))
        if self.act in self.RAMP_DEPTH:
            d = self.RAMP_DEPTH[self.act]
            kappa = self.act_alpha / (2.0 * d)
            # 二次区間の傾きは -2*kappa*z（z<0 なので正）。`0.0 +` は z=-0.0 で
            # -0.0 を作らないため（bwd_quad と同じ理由・追補 9）。
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.where(pre > -d,
                                           0.0 + (0.0 - 2.0 * kappa) * pre,
                                           torch.full_like(pre, self.act_alpha)))
        if self.act in self.COMB1_ACTIVATIONS:
            lobe = math.pi / self.act_alpha
            leaf = 0.0 - self.act_alpha * torch.sin(2.0 * self.act_alpha * pre)
            beyond = (torch.zeros_like(pre) if self.act == "comb1_flat"
                      else torch.full_like(pre, self.COMB1_LEAK["comb1_leaky"]))
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.where(pre > -lobe, leaf, beyond))
        if self.act == "snake":
            return 1.0 + torch.sin(2.0 * self.act_alpha * pre)
        if self.act == "snake1":
            a = self.act_alpha
            lo, hi = -3.0 * math.pi / (4.0 * a), math.pi / (4.0 * a)
            return torch.where((pre >= lo) & (pre <= hi),
                               1.0 + torch.sin(2.0 * a * pre), torch.ones_like(pre))
        if self.act in self.SNAKE_AMP:
            A = self.SNAKE_AMP[self.act]
            return 1.0 + A * torch.sin(2.0 * self.act_alpha * pre)
        if self.act == "comb_binf":
            return torch.where(pre > 0, torch.ones_like(pre),
                               0.0 - self.act_alpha
                               * torch.sin(2.0 * self.act_alpha * pre))
        if self.act == "comb_b5":
            beta = self.COMB_ENVELOPE["comb_b5"]
            env = torch.exp(-pre / beta)
            g = env * (torch.sin(self.act_alpha * pre) ** 2 / beta
                       - self.act_alpha
                       * torch.sin(2.0 * self.act_alpha * pre))
            return torch.where(pre > 0, torch.ones_like(pre), 0.0 + g)
        if self.act == "flip_leaky":
            # act_fn と同じ述語 `< 0`（S-flip: phi~'(z) == phi'(-z) がバイト一致）。
            return torch.where(pre < 0, torch.ones_like(pre),
                               torch.full_like(pre, self.act_alpha))
        if self.act in self.SHELF_DEPTH:
            d = self.SHELF_DEPTH[self.act]
            return torch.where(pre > -d, torch.ones_like(pre),
                               torch.full_like(pre, self.act_alpha))
        if self.act in self.STEEP_DEPTH:
            d = self.STEEP_DEPTH[self.act]
            # 下側は STEEP_SLOPE（act_alpha ではない）。
            return torch.where(pre > -d, torch.ones_like(pre),
                               torch.full_like(pre, self.STEEP_SLOPE))
        if self.act == "softplus_b":
            return torch.sigmoid(pre)
        if self.act == "tanh_b":
            t = torch.tanh(pre)
            return 1.0 - t ** 2
        if self.act in self.LEAKY_OFFSET:
            # 定数 c は微分に入らない。"leaky_relu" 分岐と同一の式 [act_offset_0906 §3]。
            return torch.where(pre > 0, torch.ones_like(pre),
                               torch.full_like(pre, self.act_alpha))
        if self.act in self.ELU_OFFSET:
            # ELU の alpha_exp 形と同一の式（activation_plus_alpha は φ+c を見るので使わない）。
            return torch.where(pre > 0, torch.ones_like(pre),
                               self.act_alpha * torch.exp(pre))
        if self.act_grad_form == "activation_plus_alpha":
            return torch.where(pre > 0, torch.ones_like(a), a + self.act_alpha)
        return torch.where(pre > 0, torch.ones_like(pre),
                           self.act_alpha * torch.exp(pre))

    def act_curv(self, pre):
        """phi''(pre) [edge_law_0905 §3.2]。**未登録名は `NotImplementedError`**。

        `act_fn`/`act_grad` の if 連鎖と違い、最後を ELU に落とさない。落とすと
        分岐の書き忘れが例外にならず、`m_dphiddphi` 列（§4.5-g の停留残差
        $G_i = 2E[\\varphi\\varphi'] + 2\\kappa E[\\varphi'\\varphi'']$ が全部乗る列）が
        黙って ELU の曲率になる。代替勾配の族（`bwd_*`/`fwd_leaky`）は
        `act_grad` が自分の forward の導関数ではないので phi'' も定義しない。
        """
        if self.act in self.ZERO_CURVATURE_ACTIVATIONS:
            # 区分線形（relu・leaky・線形 = leaky a=1・flip・棚・傾き反転棚）は
            # 折れ目が測度 0 なので恒等的に 0 と登録する。
            return torch.zeros_like(pre)
        if self.act == "elu":
            return torch.where(pre > 0, torch.zeros_like(pre),
                               self.act_alpha * torch.exp(pre))
        if self.act == "softplus_b":
            s = torch.sigmoid(pre)
            return s * (1.0 - s)
        if self.act == "tanh_b":
            t = torch.tanh(pre)
            return -2.0 * t * (1.0 - t ** 2)
        if self.act == "snake":
            # φ'' = 2α cos 2αz [snake_flip_0906 §3]。零点（φ'=0）は φ''=0 の変曲点。
            return 2.0 * self.act_alpha * torch.cos(2.0 * self.act_alpha * pre)
        if self.act == "snake1":
            a = self.act_alpha
            lo, hi = -3.0 * math.pi / (4.0 * a), math.pi / (4.0 * a)
            return torch.where((pre >= lo) & (pre <= hi),
                               2.0 * a * torch.cos(2.0 * a * pre), torch.zeros_like(pre))
        if self.act in self.SNAKE_AMP:
            A = self.SNAKE_AMP[self.act]
            return 2.0 * self.act_alpha * A * torch.cos(2.0 * self.act_alpha * pre)
        if self.act in self.ELU_OFFSET:
            # φ″ は elu と同一 [act_offset_0906 §3]（leaky_off_* は ZERO_CURVATURE で 0）。
            return torch.where(pre > 0, torch.zeros_like(pre),
                               self.act_alpha * torch.exp(pre))
        raise NotImplementedError(f"act_curv is not registered for {self.act!r}")

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

    def set_weight_decay_w(self, wd_w):
        """構築後に wd_w を差し替える。乱数も状態も消費しない。

        ``set_weight_decay_b`` の鏡像で、掛かる相手が**隠れ層の W だけ**である点
        だけが違う [phantom_wall_0902 §4.3]。
        """
        wd_w = float(wd_w)
        if not math.isfinite(wd_w) or wd_w < 0.0:
            raise ValueError(f"wd_w must be a finite non-negative float, got {wd_w!r}")
        self.wd_w = wd_w
        return self
    def set_weight_decay_b(self, wd_b):
        """構築後に wd_b を差し替える。乱数も状態も消費しないので、arm 設定は
        凍結済みの ``mlp2_phase0.setup_arm`` 経路のままでよい [bias_wd_0901 §6]
        (``set_activation`` と同じ hook 方式)。"""
        self.wd_b = _check_wd_b(wd_b, self.freeze_bias)
        return self

    def sgd_step_layers(self, lr, gWs, gbs, gv, gc):
        """wd_b > 0 のとき**全隠れ層の** bias だけが `b -= lr*(gb + wd_b*b)` に、
        wd_w > 0 のとき**全隠れ層の** W だけが `W -= lr*(gW + wd_w*W)` になる
        [bias_wd_0901 §6・phantom_wall_0902 §4.3]。v・出力 bias c はどちらにも
        依らない。既定 0.0 では両方とも恒等で既存の走と bit 一致する（分岐を
        置かないのは wd=0 の腕が WD コード経路を通したうえで無 WD 実装と
        bit 一致することを検査可能にするため）。"""
        for i in range(self.L):
            self.Ws[i] -= lr[:, None, None] * (gWs[i] + self.wd_w * self.Ws[i])
            self.bs[i] -= lr[:, None] * (gbs[i] + self.wd_b * self.bs[i])
        self.v -= lr[:, None] * gv
        self.c -= lr * gc

    def sgd_step(self, lr, gW, gb, gv, gc):
        if self.L != 1:
            raise ValueError("use sgd_step_layers for L > 1")
        self.sgd_step_layers(lr, [gW], [gb], gv, gc)
