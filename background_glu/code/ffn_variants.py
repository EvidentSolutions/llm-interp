"""Grouped k-in-1-out FFN variants for GPT-NeoX.

The standard FFN writes through `fc2` of shape (d_ff, d): d_ff = 4d write
vectors living in R^d, a 4x overcomplete dictionary.  That overcompleteness
is the substrate that lets writes superpose.

A grouped variant partitions the d_ff pre-activations into d_ff/k contiguous
groups of k, collapses each group to one unit with a fixed operator, and
writes through a narrower (d_ff/k, d) matrix.  At k=4 the write dictionary
becomes d vectors in R^d, at most a basis, so the layer cannot superpose on
the write side by construction.

Structurally this is maxout (Goodfellow et al. 2013, arXiv:1302.4389), whose
defining property is exactly the shared outgoing weight per block.  What is
not standard is the operator: the maxout/LWTA family is uniformly `max`.
`min` is the Goedel t-norm, the conjunction whose gradient goes to the
*weakest* branch, which is the correct credit assignment for an AND and is
self-balancing rather than winner-take-all.

Operators:
    min     Goedel t-norm on raw affines (concave piecewise-linear unit)
    max     maxout, the published control
    prod    product t-norm on sigmoid-bounded memberships (logic semantics)
    mean    linear control: no gate at all, just averaging within the group

Config key consumed by train.py:

    "ffn": {"kind": "min", "group": 4}                # d_ff stays 3072
    "ffn": {"kind": "standard", "intermediate_size": 1920}   # param control
"""
import torch
import torch.nn as nn

OPS = ("min", "max", "prod", "mean", "minmax", "swiglu", "standard",
       "bilinear", "relu2", "ncffn", "doublegate")


class GroupedMLP(nn.Module):
    """d -> d_ff -> (group-collapse) -> d_ff/group -> d."""

    def __init__(self, hidden_size, intermediate_size, group, kind):
        super().__init__()
        if intermediate_size % group:
            raise ValueError(
                f"intermediate_size {intermediate_size} not divisible by "
                f"group {group}")
        self.group = group
        self.kind = kind
        self.n_units = intermediate_size // group
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        # minmax outputs both min and max per group → 2x width
        out_units = self.n_units * 2 if kind == "minmax" else self.n_units
        self.dense_4h_to_h = nn.Linear(out_units, hidden_size)

    def forward(self, x):
        h = self.dense_h_to_4h(x)
        if self.kind in ("min", "max", "minmax"):
            from ffn_kernels import group_reduce
            u = group_reduce(h, self.n_units, self.group, self.kind)
        elif self.kind == "prod":
            h = h.view(*h.shape[:-1], self.n_units, self.group)
            u = torch.sigmoid(h).prod(dim=-1)
        elif self.kind == "mean":
            h = h.view(*h.shape[:-1], self.n_units, self.group)
            u = h.mean(dim=-1)
        else:
            raise ValueError(self.kind)
        return self.dense_4h_to_h(u)


class SwiGLUMLP(nn.Module):
    """SwiGLU FFN: gate=swish(W_gate x), up=W_up x, out=W_down(gate * up).

    Three weight matrices: W_gate (d, d_ff), W_up (d, d_ff), W_down (d_ff, d).
    Total params: 3 * d * d_ff (vs standard FFN's 2 * d * d_ff).
    To param-match standard FFN at 4d width: d_ff = 8d/3.
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size)
        self.w_up = nn.Linear(hidden_size, intermediate_size)
        self.w_down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class BilinearMLP(nn.Module):
    """Pure bilinear FFN: out = W_down(W_a x * W_b x). A PRODUCT with NO activation.

    The H1-vs-H2 discriminator. It has SwiGLU's factorised multiplication but no
    nonlinearity and therefore NO off-state -- silu(gate) can drive a SwiGLU unit to
    zero, W_a x cannot. If bilinear tracks SwiGLU, the product structure is what
    matters (H1). If it lags while relu2 keeps up, the sparse heavy-tailed code is
    what matters (H2). Same three-matrix parameter cost as SwiGLU.
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w_a = nn.Linear(hidden_size, intermediate_size)
        self.w_b = nn.Linear(hidden_size, intermediate_size)
        self.w_down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.w_down(self.w_a(x) * self.w_b(x))


class DoubleGatedMLP(nn.Module):
    """BOTH branches gated: out = W_down(silu(W_a x) * silu(W_b x)).

    Same 3-matrix shape as SwiGLU/Bilinear, and w_a/w_b are created in the same
    order as SwiGLU's w_gate/w_up, so from the shared init seed the triple
    swiglu / bilinear / doublegate is BIT-IDENTICAL at step 0 and differs ONLY in
    how many branches carry the SiLU (1 / 0 / 2). The b_L-localization question:
    with two symmetric thresholding branches, does the shared reference coupling
    SPLIT across both, stay concentrated in one (symmetry breaking), or relocate?
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w_a = nn.Linear(hidden_size, intermediate_size)
        self.w_b = nn.Linear(hidden_size, intermediate_size)
        self.w_down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        silu = torch.nn.functional.silu
        return self.w_down(silu(self.w_a(x)) * silu(self.w_b(x)))


class ReLU2MLP(nn.Module):
    """Squared-ReLU FFN: out = W_down(relu(W_in x)^2).

    The other half of the discriminator: self-gated (x*x is a product of a value with
    ITSELF) and heavy-tailed, but NOT factorised -- there is no separate control
    terminal. Two matrices, so it param-matches the standard FFN at the same width,
    unlike the gated arms. Used in Primer (So et al. 2021).
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w_in = nn.Linear(hidden_size, intermediate_size)
        self.w_down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.w_down(torch.nn.functional.relu(self.w_in(x)) ** 2)


class NCFFNMLP(nn.Module):
    """Oskin's negation-capable FFN (arXiv:2606.31845) as a 2-matrix arm.

    The d_ff pre-activations split into n_gelu = round(rho*d_ff) standard GELU
    units (the trainability "gradient highway", rho=0.75 is Oskin's main config;
    a fully-operator FFN diverges) and n_op = d_ff - n_gelu operand units forming
    n_op/2 pairs: A=sigmoid, B=sigmoid, outputs A*B ("and") and A*(1-B)
    ("and-not"). Same two Linear shapes as the standard FFN, so parameter-
    IDENTICAL to gelu/relu2 at the same width (4,722,432/layer at d_ff 3072).

    The unique point in the GLU 2x2: BOUNDED support (sigmoid) -> a two-input
    conjunctive detector (fires only when both operands are high), and NO
    designated value branch -- releasing NEXT_ARMS properties 1 (asymmetry) and
    2 (unbounded support), which no swept GLU arm does.
    """

    def __init__(self, hidden_size, intermediate_size, rho=0.75):
        super().__init__()
        n_gelu = int(round(rho * intermediate_size))
        n_op = intermediate_size - n_gelu
        if n_op % 2:
            n_gelu += 1
            n_op -= 1
        self.n_gelu, self.half, self.rho = n_gelu, n_op // 2, rho
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.dense_h_to_4h(x)
        g = self.act(h[..., :self.n_gelu])
        rest = h[..., self.n_gelu:]
        A = torch.sigmoid(rest[..., :self.half])
        B = torch.sigmoid(rest[..., self.half:self.half * 2])
        return self.dense_4h_to_h(torch.cat([g, A * B, A * (1.0 - B)], -1))


def apply_ffn_variant(model, spec):
    """Replace every layer's MLP. Returns a description dict for logging."""
    kind = spec.get("kind", "standard")
    if kind not in OPS:
        raise ValueError(f"kind must be one of {OPS}, got {kind}")
    cfg = model.config
    d = cfg.hidden_size
    d_ff = spec.get("intermediate_size", cfg.intermediate_size)
    group = spec.get("group", 1)
    # match whatever the rest of the model is in, before we swap anything out
    ref = next(model.parameters())
    device, dtype = ref.device, ref.dtype

    if kind == "standard":
        # width control: standard GELU FFN at a chosen intermediate size
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
        import copy
        sub = copy.deepcopy(cfg)
        sub.intermediate_size = d_ff
        for layer in model.gpt_neox.layers:
            layer.mlp = GPTNeoXMLP(sub)
        writers = d_ff
        per_layer = d * d_ff + d_ff * d + d_ff + d
    elif kind == "swiglu":
        for layer in model.gpt_neox.layers:
            layer.mlp = SwiGLUMLP(d, d_ff)
        writers = d_ff
        # 3 matrices: gate(d,d_ff) + up(d,d_ff) + down(d_ff,d), plus biases
        per_layer = 2 * d * d_ff + d_ff * d + 2 * d_ff + d
    elif kind == "bilinear":
        for layer in model.gpt_neox.layers:
            layer.mlp = BilinearMLP(d, d_ff)
        writers = d_ff
        per_layer = 2 * d * d_ff + d_ff * d + 2 * d_ff + d
    elif kind == "doublegate":
        for layer in model.gpt_neox.layers:
            layer.mlp = DoubleGatedMLP(d, d_ff)
        writers = d_ff
        per_layer = 2 * d * d_ff + d_ff * d + 2 * d_ff + d       # = swiglu/bilinear
    elif kind == "relu2":
        for layer in model.gpt_neox.layers:
            layer.mlp = ReLU2MLP(d, d_ff)
        writers = d_ff
        per_layer = d * d_ff + d_ff * d + d_ff + d
    elif kind == "ncffn":
        rho = spec.get("rho", 0.75)
        for layer in model.gpt_neox.layers:
            layer.mlp = NCFFNMLP(d, d_ff, rho)
        writers = d_ff
        per_layer = d * d_ff + d_ff * d + d_ff + d          # 2-matrix, = standard
    else:
        for layer in model.gpt_neox.layers:
            layer.mlp = GroupedMLP(d, d_ff, group, kind)
        writers = (d_ff // group) * 2 if kind == "minmax" else d_ff // group
        per_layer = d * d_ff + writers * d + d_ff + d

    model.to(device=device, dtype=dtype)
    return {"kind": kind, "group": group, "intermediate_size": d_ff,
            "units": writers, "write_vectors": writers,
            "overcomplete": round(writers / d, 3),
            "ffn_params_per_layer": per_layer}
