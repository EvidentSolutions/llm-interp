"""LADDER PROBE -- the cross-arm-valid structural instruments, run against a ladder snapshot.

Two instruments the in-run probes do NOT provide, both required by the survey in
`glu_paper/INSTRUMENTATION.md`:

1. CETT sparsity (2402.03804, which compares ReLU/SwiGLU/ReGLU/ReLU^2 at 1.3B matched).
   The in-run L1/L2 = (E|a|)^2/E[a^2] is scale-invariant but NOT threshold-calibrated, and the
   bilinear arm has no pre-activation to threshold at all -- so L1/L2 cannot be compared across
   our four arms and cannot describe bilinear even in principle. CETT fixes both by defining
   deadness on the neuron's OUTPUT VECTOR norm:
       n_i(x) = a_i(x) * W_down[:, i]        contribution of neuron i to the FFN output
       D      = {i : ||n_i(x)||_2 < eps}
       CETT   = ||sum_{i in D} n_i(x)||_2 / ||FFN(x)||_2
   CETT is monotone in eps, so eps is found by binary search to a fixed CETT target; the reported
   quantity is then the FRACTION OF NEURONS IN D at matched output-error, which is comparable
   across architectures because the error budget, not the threshold, is held fixed.

2. R^2_lin per block (2606.19379). Least-squares linear map from FFN input to FFN output, scored
   on held-out positions. Activation-agnostic and scale-invariant, and it measures exactly what
   the bilinear arm exists to isolate: how much of a block is NOT recoverable by a linear map.
   2606.19379 reports recoverability is "a learned property of individual trained blocks, not an
   architectural one" -- a null our design can attack properly, because its models differ in data,
   tokenizer and seed and ours differ ONLY in the nonlinearity.

[!!] Everything here is reported PER LAYER. Layer means have hidden depth-dependent sign flips
three times in this project; see the CORRECTION 2026-08-25c block in glu_paper/PLAN.md.

[!!] d_ff differs by arm (3072 gelu/relu2, 2048 swiglu/bilinear). CETT sparsity is a FRACTION so
it is already n-normalised; R^2_lin is a variance ratio so it is too. Do not add raw-count
statistics here without a per-arm null.

Usage:
  ./.venv/Scripts/python.exe minout/code/ladder_probe.py minout/data/ladder/gelu_step800.pt
  SMOKE=1 ... (2 sequences instead of 24)
"""
import os
import sys
import json
import glob
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffn_variants import apply_ffn_variant                      # noqa: E402

# DEVICE=cpu keeps this off a GPU that another job is using. bf16 autocast is a CUDA win but a
# CPU pessimisation, so it is enabled only on cuda; CPU runs fp32.
DEV = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
if DEV == "cpu":
    torch.set_num_threads(int(os.environ.get("THREADS", str(max(1, (os.cpu_count() or 4) - 2)))))


def amp():
    return torch.amp.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda"))


def empty_cache():
    if DEV == "cuda":
        torch.cuda.empty_cache()


SMOKE = bool(int(os.environ.get("SMOKE", "0")))
NSEQ = 2 if SMOKE else 24
SEQ = 512
CETT_TARGET = float(os.environ.get("CETT_TARGET", "0.2"))
# NLL_SEQ=0 skips the held-out loss entirely. On CPU that is the right call: the loss work is done
# better by paired_nll.py (paired SE ~0.0028 vs unpaired ~0.008 on the same tokens), so computing
# it here is duplicated effort, and 256 sequences of forward pass dominates CPU runtime.
NLL_SEQ = int(os.environ.get("NLL_SEQ", "256"))
RESUME = bool(int(os.environ.get("RESUME", "1")))   # skip snapshots already in OUT
CORPUS = os.environ.get("CORPUS", "minout/data/pile-5b.bin")
OUT = os.environ.get("OUT", "minout/data/ladder-probe.json")

ARMS = {"gelu": {"kind": "standard", "intermediate_size": 3072},
        "swiglu": {"kind": "swiglu", "intermediate_size": 2048},
        "bilinear": {"kind": "bilinear", "intermediate_size": 2048},
        "relu2": {"kind": "relu2", "intermediate_size": 3072},
        # 5th arm, trained 2026-08-26/27 to step 60000 (983M tok) by the concurrent session.
        # Oskin's NC-FFN as a 2-MATRIX arm at d_ff 3072: the SAME dense_h_to_4h/dense_4h_to_h
        # shapes as gelu and relu2, so it draws BIT-IDENTICAL init with both -- a three-way
        # controlled set. rho=0.75 of the pre-activations go through GELU as usual; the other 25%
        # form sigmoid operand pairs computing A*B and A*(1-B). So it differs from gelu ONLY in
        # what happens to a quarter of its units: a DOSE-RESPONSE test of the frame result.
        "ncffn": {"kind": "ncffn", "intermediate_size": 3072, "rho": 0.75},
        # double-gated arm (SiLU on both branches), trained 2026-09-09; bit-identical
        # init to swiglu/bilinear at 2048.
        "doublegate": {"kind": "doublegate", "intermediate_size": 2048}}
DOWN = {"standard": "dense_4h_to_h", "swiglu": "w_down",
        "bilinear": "w_down", "relu2": "w_down", "ncffn": "dense_4h_to_h",
        "doublegate": "w_down"}


def load_snapshot(path):
    """Rebuild the arm's architecture and load a bf16 model-only ladder snapshot into it."""
    from transformers import AutoConfig, AutoModelForCausalLM
    st = torch.load(path, map_location="cpu", weights_only=False)
    arm, step = st["arm"], st["step"]
    cfg = AutoConfig.from_pretrained("EleutherAI/pythia-160m-deduped")
    cfg.torch_dtype = torch.float32          # see run_glu_matched.py: fp16 masters NaN AdamW
    model = AutoModelForCausalLM.from_config(cfg).to(DEV, dtype=torch.float32)
    apply_ffn_variant(model, ARMS[arm])
    model.load_state_dict({k: v.float() for k, v in st["model"].items()})
    model.to(DEV).eval()
    return model, arm, step


def eval_batch(n=NSEQ):
    """Fixed held-out slice, taken from the FAR END of the corpus so it is unseen at 5B tokens."""
    toks = np.memmap(CORPUS, dtype=np.uint16, mode="r")
    start = len(toks) - (n + 1) * SEQ - 1
    ids = np.asarray(toks[start:start + n * SEQ], dtype=np.int64).reshape(n, SEQ)
    return torch.from_numpy(ids).to(DEV)


@torch.no_grad()
def collect(model, kind, ids):
    """Per layer: FFN input X, FFN output Y, and post-activation coefficients A."""
    lay = [b.mlp for b in model.gpt_neox.layers]
    X, Y, A = {}, {}, {}
    cur = {}

    def pre(L):
        def fn(mod, inp):
            cur[L] = inp[0].detach()          # coefficients entering the down-projection
        return fn

    def io(L):
        def fn(mod, inp, out):
            X.setdefault(L, []).append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
            Y.setdefault(L, []).append(out.detach().float().reshape(-1, out.shape[-1]).cpu())
            a = cur[L].float().reshape(-1, cur[L].shape[-1])
            A.setdefault(L, []).append(a.cpu())
        return fn

    hks = []
    for L, m in enumerate(lay):
        hks.append(getattr(m, DOWN[kind]).register_forward_pre_hook(pre(L)))
        hks.append(m.register_forward_hook(io(L)))
    for i in range(0, ids.shape[0], 2):
        model(ids[i:i + 2])
    for h in hks:
        h.remove()
    return ({L: torch.cat(v) for L, v in X.items()},
            {L: torch.cat(v) for L, v in Y.items()},
            {L: torch.cat(v) for L, v in A.items()})


def cett_sparsity(a, wdown, target=CETT_TARGET):
    """Fraction of neurons droppable at a matched FFN-output error budget.

    a: (T, n) post-activation coefficients. wdown: (d, n) down-projection.
    ||n_i(x)||_2 = |a_i| * ||W_down[:, i]||_2, so the per-neuron contribution norms are a
    rank-1 rescaling of |a| -- no need to materialise T x n x d.
    """
    a = a.to(DEV)
    cn = wdown.to(DEV).norm(dim=0)                       # (n,) column norms
    contrib = a.abs() * cn                               # (T, n) ||n_i(x)||
    full = a @ wdown.T.to(DEV)                           # (T, d) FFN output (pre-bias)
    fn = full.norm(dim=1).clamp_min(1e-9)
    lo, hi = 0.0, float(contrib.max()) + 1e-6
    frac = 0.0
    for _ in range(24):                                  # CETT is monotone in eps -> bisect
        eps = 0.5 * (lo + hi)
        mask = (contrib < eps)
        dropped = (a * mask) @ wdown.T.to(DEV)
        cett = float((dropped.norm(dim=1) / fn).mean())
        if cett < target:
            lo, frac = eps, float(mask.float().mean())
        else:
            hi = eps
    return frac, lo


@torch.no_grad()
def held_out_nll(model, n=256):
    """Mean NLL over n held-out sequences from the far end of the corpus.

    [!!] The in-run measure() uses 16 sequences = 8,192 tokens. At ~2-3 nats of per-token spread
    that is a standard error near 0.03 nats, and the entire four-arm loss spread is ~0.03 -- so the
    in-run number cannot resolve the arms at all, and it visibly cannot: at step 75000->80000 three
    arms' losses ROSE on a fixed eval set under a decaying schedule, which is impossible except as
    noise. 256 sequences = 131,072 tokens brings SE to ~0.008, four times finer than the effect
    instead of three times coarser. This matters specifically because the survey's reframe makes
    the LOSS-CROSSING POINT the headline measurement (2605.20749), and a crossing cannot be located
    with an instrument whose noise exceeds the gap.
    """
    toks = np.memmap(CORPUS, dtype=np.uint16, mode="r")
    start = len(toks) - (n + 1) * SEQ - 1
    ids = np.asarray(toks[start:start + n * SEQ], dtype=np.int64).reshape(n, SEQ)
    tot, cnt = 0.0, 0
    for i in range(0, n, 8):
        b = torch.from_numpy(ids[i:i + 8]).to(DEV)
        with amp():
            lg = model(b[:, :-1]).logits.float()
        l = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="sum")
        tot += float(l)
        cnt += b[:, 1:].numel()
    return tot / cnt


def r2_lin(x, y, drop_massive=0):
    """Held-out R^2 of the best affine map x -> y. Ridge-stabilised; train/test split 75/25.

    drop_massive: if >0, exclude that many highest-variance OUTPUT coordinates before scoring.
    [!!] R^2 is a variance ratio, so it is dominated by the loudest output dimensions -- which in
    a transformer are the massive activations. Participation ratio collapses to ~1.3 in this
    project for exactly that reason. An uncorrected R^2_lin may be reporting "the massive dims are
    linearly predictable" rather than anything about the block, so both versions are reported and
    a large gap between them is a warning, not a detail.
    """
    T = x.shape[0]
    ntr = int(T * 0.75)
    xtr = torch.cat([x[:ntr], torch.ones(ntr, 1)], 1).to(DEV).double()
    ytr = y[:ntr].to(DEV).double()
    xte = torch.cat([x[ntr:], torch.ones(T - ntr, 1)], 1).to(DEV).double()
    yte = y[ntr:].to(DEV).double()
    G = xtr.T @ xtr
    G += 1e-6 * torch.eye(G.shape[0], device=DEV, dtype=G.dtype) * float(G.diagonal().mean())
    Wm = torch.linalg.solve(G, xtr.T @ ytr)
    keep = torch.ones(y.shape[1], dtype=torch.bool, device=DEV)
    if drop_massive:
        loud = ytr.var(0).topk(drop_massive).indices
        keep[loud] = False
    resid = ((yte[:, keep] - (xte @ Wm)[:, keep]) ** 2).sum()
    tot = ((yte[:, keep] - ytr.mean(0)[keep]) ** 2).sum()
    return float(1.0 - resid / tot.clamp_min(1e-12))


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob("minout/data/ladder/*.pt"))
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    ids = eval_batch()
    for p in paths:
        t0 = time.time()
        import re
        m = re.match(r"(\w+)_step(\d+)\.pt", os.path.basename(p))
        if RESUME and m and m.group(2) in res.get(m.group(1), {}):
            continue
        model, arm, step = load_snapshot(p)
        kind = ARMS[arm]["kind"]
        nll = held_out_nll(model, n=8 if SMOKE else NLL_SEQ) if NLL_SEQ else float("nan")
        X, Y, A = collect(model, kind, ids)
        lay = [b.mlp for b in model.gpt_neox.layers]
        cett, r2, r2x = [], [], []
        for L in range(len(lay)):
            wd = getattr(lay[L], DOWN[kind]).weight.detach().float()
            f, _ = cett_sparsity(A[L], wd)
            cett.append(round(f * 100, 2))
            r2.append(round(r2_lin(X[L], Y[L]), 4))
            r2x.append(round(r2_lin(X[L], Y[L], drop_massive=8), 4))
        rec = res.setdefault(arm, {})[str(step)] = {
            "nll": round(nll, 5),
            "cett_drop_pct": cett, "r2_lin": r2, "r2_lin_exmassive": r2x,
            "cett_mean": round(float(np.mean(cett)), 2),
            "r2_mean": round(float(np.mean(r2)), 4),
            "r2x_mean": round(float(np.mean(r2x)), 4)}
        print("[{:>8s}] step {:>6d}  NLL {:.4f}  CETT-droppable {:5.1f}%  R2_lin {:.3f}  "
              "ex-massive {:.3f}  ({:.0f}s)".format(
                  arm, step, rec["nll"], rec["cett_mean"], rec["r2_mean"], rec["r2x_mean"],
                  time.time() - t0), flush=True)
        print("   CETT/layer  " + " ".join("{:5.1f}".format(v) for v in cett), flush=True)
        print("   R2lin/layer " + " ".join("{:5.3f}".format(v) for v in r2), flush=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        del model, X, Y, A
        empty_cache()


if __name__ == "__main__":
    main()
