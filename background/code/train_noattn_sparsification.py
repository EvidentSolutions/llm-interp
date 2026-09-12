"""DOES GATE SPARSIFICATION NEED ATTENTION?  A no-attention training arm.

Hypothesis under test (Olli 2026-09-11). Gate sparsification is not an information-free early
structure; it is driven by the PRESSURE to exploit the trickle of information that even a
near-untrained attention carries. Noisy gates (duty ~0.5) drown a weak signal, so the model quiets
them (pushes them off-by-default) to make the trickle usable. The measurable induction head is a
later crystallisation of the same attention substrate.

Clean test: remove the DRIVER and watch the downstream form. Train a model with attention
DISABLED and see whether the gates sparsify at all. This targets the driver, not the form, so
there is no "route around the constraint" escape.
  - duty stays ~0.5  -> sparsification needs cross-token (attention) information  => supports it.
  - duty drops like normal gelu -> sparsification arises from token-local prediction pressure too
    => the attention trickle is not the unique driver (reframes, does not kill, the idea).

How attention is disabled: zero AND freeze each layer's attention output projection
(`attention.dense`). The attention forward still runs (so the return signature is untouched) but
its contribution is identically 0, and Q/K/V receive no gradient (d out / d context = dense.W = 0),
so they stay at init -- inert. Under Pythia's parallel residual (hidden = h + attn(ln1 h) +
mlp(ln2 h)) this leaves a purely POSITION-WISE model: the MLP and embeddings are the only paths,
rotary position enters only through attention, so every position is processed in isolation. That is
the cleanest "no information from other tokens" condition.

Two arms, bit-identical init (seed 0), identical data order / schedule (the run_glu_matched
constants), differing ONLY in whether attention is on:
  gelu     attention ON   -- positive control; must reproduce the known duty drop (0.5 -> ~0.07)
  noattn   attention OFF  -- the test

Measured inline at each checkpoint, mid-stack (layers 4-8), on the gelu read branch:
  duty      = mean(pre-activation > 0)            sparsification (low = gate formed)
  z         = -mu_g / sd(g)                        operating-point depth in sd of drive
  frac_rest = |mu_x . w| / |mu_g|                  share of the operating point from the mean-input
                                                   (b_L) coupling rather than the bias

GPU (bf16 autocast) if available, else CPU. Short run (STEPS=3000 covers the whole sparsification
window; normal gelu is done by ~800). Resumable per arm. Writes noattn-sparsification.json.

Usage:
  .venv/Scripts/python.exe minout/code/train_noattn_sparsification.py
  STEPS=3000 ARMS=gelu,noattn
"""
import os
import sys
import json
import time
import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoConfig, AutoModelForCausalLM               # noqa: E402
from ffn_variants import apply_ffn_variant                             # noqa: E402
from corpus import BinPileDataset                                      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GLUDATA", os.path.join(REPO, "..", "data"))
CORPUS = os.path.join(DATA, os.environ.get("CORPUS", "pile-5b.bin"))
CKDIR = os.path.join(DATA, os.environ.get("CKDIR", "noattn_ckpt"))
OUT = os.path.join(DATA, os.environ.get("OUTNAME", "noattn-sparsification.json"))

# same knobs as run_glu_matched.py so the gelu control matches the existing ladder
BATCH, SEQ, ACC = 16, 512, 2
LR, WARMUP, WD = 6e-4, 200, 0.01
DATA_SEED, INIT_SEED = 42, 0
STEPS = int(os.environ.get("STEPS", "3000"))
CKPT = [0, 50, 100, 200, 400, 800, 1500, 3000]
MID = [4, 5, 6, 7, 8]
ARMS = os.environ.get("ARMS", "gelu,noattn").split(",")
GELU_SPEC = {"kind": "standard", "intermediate_size": 3072}


def disable_attention(model):
    """Zero + freeze each layer's attention output projection -> attention contributes 0,
    Q/K/V get no gradient. Returns the step-0 max |attn-output-weight| for a sanity print."""
    n = 0
    for layer in model.gpt_neox.layers:
        attn = layer.attention
        dense = getattr(attn, "dense", None) or getattr(attn, "o_proj", None)
        if dense is None:
            raise RuntimeError("could not find attention output projection (dense/o_proj)")
        with torch.no_grad():
            dense.weight.zero_()
            if dense.bias is not None:
                dense.bias.zero_()
        dense.weight.requires_grad_(False)
        if dense.bias is not None:
            dense.bias.requires_grad_(False)
        n += 1
    return n


@torch.no_grad()
def attn_contribution_norm(model, ids):
    """Mean L2 norm of each layer's attention output -- should be ~0 when disabled."""
    norms = []

    def hook(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        norms.append(float(o.float().norm(dim=-1).mean()))

    hks = [lay.attention.register_forward_hook(hook) for lay in model.gpt_neox.layers]
    model(ids[:2])
    for h in hks:
        h.remove()
    return round(float(np.mean(norms)), 6)


@torch.no_grad()
def gate_duty(model, ids):
    """Mid-stack median duty / z / frac_rest on the standard gelu read branch."""
    layers = model.gpt_neox.layers
    X = {}

    def cap(L):
        def fn(mod, inp):
            X.setdefault(L, []).append(
                inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
        return fn

    hks = [lay.mlp.register_forward_pre_hook(cap(L)) for L, lay in enumerate(layers)]
    for i in range(0, ids.shape[0], 2):
        model(ids[i:i + 2])
    for h in hks:
        h.remove()

    per = {}
    for L, chunks in X.items():
        x = torch.cat(chunks)
        mu = x.mean(0)
        lin = layers[L].mlp.dense_h_to_4h
        W = lin.weight.detach().float().cpu()
        b = lin.bias.detach().float().cpu() if lin.bias is not None else torch.zeros(W.shape[0])
        g = x @ W.T + b
        mu_g = mu @ W.T + b
        sd = g.std(0).clamp_min(1e-9)
        duty = (g > 0).float().mean(0)
        z = -mu_g / sd
        off_rest = mu @ W.T
        frac_rest = off_rest.abs() / mu_g.abs().clamp_min(1e-9)
        per[L] = (float(duty.median()), float(z.median()), float(frac_rest.median()))
    def agg(j):
        vals = [per[L][j] for L in MID if L in per]
        return round(float(np.median(vals)), 4) if vals else None
    return agg(0), agg(1), agg(2)


def main():
    t0 = time.time()
    ds = BinPileDataset(CORPUS, seq_len=SEQ, seed=DATA_SEED)
    NB = len(ds)
    print("device {}  corpus {:,} chunks ({:.0f}M tok)  arms {}  steps {}".format(
        DEV, NB, NB * SEQ / 1e6, ARMS, STEPS), flush=True)
    ev = torch.stack([ds[i] for i in range(NB - 32, NB)]).to(DEV)
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.setdefault("config", {"steps": STEPS, "batch": BATCH, "seq": SEQ, "accum": ACC,
                              "lr": LR, "warmup": WARMUP, "wd": WD, "data_seed": DATA_SEED,
                              "init_seed": INIT_SEED, "mid_layers": MID,
                              "tokens_per_step": BATCH * SEQ * ACC})
    res.setdefault("arms", {})
    os.makedirs(CKDIR, exist_ok=True)

    def get_batch(bi):
        idx = [(bi * BATCH + k) % NB for k in range(BATCH)]
        return torch.stack([ds[j] for j in idx])

    def build(name):
        torch.manual_seed(INIT_SEED)
        cfg = AutoConfig.from_pretrained("EleutherAI/pythia-160m-deduped")
        cfg.torch_dtype = torch.float32
        model = AutoModelForCausalLM.from_config(cfg).to(DEV, dtype=torch.float32)
        apply_ffn_variant(model, GELU_SPEC)            # both arms are standard gelu FFN
        if name == "noattn":
            disable_attention(model)
        model.to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=LR, weight_decay=WD, betas=(0.9, 0.95))
        step = 0
        cp = os.path.join(CKDIR, name + ".pt")
        if os.path.exists(cp):
            st = torch.load(cp, map_location=DEV, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            step = st["step"]
        return model, opt, step

    def measure(name, model, rec, st):
        model.eval()
        with torch.no_grad(), torch.amp.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            lg = model(ev[:16]).logits[:, :-1].float()
            loss = float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ev[:16, 1:].reshape(-1)))
        duty, z, fr = gate_duty(model, ev[:16])
        rec["traj"][str(st)] = {"loss": round(loss, 4), "duty": duty, "z": z,
                                "frac_rest": fr, "tokens": st * BATCH * SEQ * ACC}
        print("    [{:>7s}] step {:>6d}  loss {:6.3f}  duty {:.3f}  z {:+.3f}  frac_rest {}".format(
            name, st, loss, duty, z, fr), flush=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        model.train()

    for name in ARMS:
        model, opt, step = build(name)
        rec = res["arms"].setdefault(name, {"traj": {}})
        if name == "noattn" and step == 0:
            print("  noattn: mean attention-output norm at init = {} (must be ~0)".format(
                attn_contribution_norm(model, ev)), flush=True)
        if step == 0 and "0" not in rec["traj"]:
            measure(name, model, rec, 0)
        model.train()
        while step < STEPS:
            opt.zero_grad(set_to_none=True)
            for j in range(ACC):
                b = get_batch(step * ACC + j).to(DEV)
                with torch.amp.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                    lg = model(b[:, :-1]).logits
                    loss = torch.nn.functional.cross_entropy(
                        lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1)) / ACC
                loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            lr = LR * (step / max(WARMUP, 1) if step < WARMUP else
                       0.1 + 0.45 * (1 + math.cos(math.pi * (step - WARMUP) /
                                                  max(STEPS - WARMUP, 1))))
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.step()
            step += 1
            if step in CKPT:
                measure(name, model, rec, step)
        tmp = os.path.join(CKDIR, name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, os.path.join(CKDIR, name + ".pt"))
        del model, opt
        if DEV == "cuda":
            torch.cuda.empty_cache()

    # side-by-side duty trajectory
    print("\n=== mid-stack duty trajectory (low = gates formed) ===", flush=True)
    steps_all = sorted({int(s) for a in ARMS for s in res["arms"].get(a, {}).get("traj", {})})
    print("{:>7s}".format("step") + "".join("{:>12s}".format(a) for a in ARMS))
    for s in steps_all:
        row = "{:>7d}".format(s)
        for a in ARMS:
            v = res["arms"].get(a, {}).get("traj", {}).get(str(s), {}).get("duty")
            row += "{:>12}".format("{:.3f}".format(v) if v is not None else "--")
        print(row)
    print("\nwrote {}  ({:.0f}s)".format(OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
