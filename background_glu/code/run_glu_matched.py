"""THE KEYSTONE: matched-pair training to make the activation->sign law CAUSAL.

Everything measured so far is correlational across pretrained families: GELU models anti-align a
neuron's write with its own read (rank ~0% of 98k), SwiGLU-up models align it (rank ~100%), the
gate is exempt (~0.000), ReLU is a depth crossover. The off-switch account PREDICTS that sign but
has never been tested by holding everything else fixed and changing only the nonlinearity.

FOUR ARMS, identical data / data order / init seed / steps / schedule, differing ONLY in the FFN:

  gelu      standard GPT-NeoX FFN, d_ff=3072              no off-state, one read
  swiglu    down(silu(gate) * up), d_ff=2048              product AND off-state
  bilinear  down(a * b), d_ff=2048, NO activation         product, NO off-state
  relu2     down(relu(in)^2), d_ff=3072                   off-state + heavy tails, NOT factorised

The 2x2 is the point. bilinear vs swiglu isolates the OFF-STATE (both are products); relu2 vs
swiglu isolates FACTORISATION (both have off-states and heavy tails). That separates H1
(factorised multiplication is what matters) from H2 (the sparse heavy-tailed code is what matters).

PRIMARY ENDPOINT -- the sign trajectory, not the loss:
    cos(own-read, own-write) per layer, and the RANK of the own-read among ALL reads in the model
    (0% = most negative, 100% = most positive), at every checkpoint.
SECONDARY: eval loss; and the L1/L2 emission sparsity that split the architectures cleanly on
pretrained models (gelu 0.51-0.59 vs swiglu 0.36-0.42, no overlap).

PREDICTIONS, RECORDED BEFORE THE FIRST STEP so the run can falsify them (written into the run
config and echoed at startup):
  P1  every arm starts at rank ~50% (random init has no alignment) -- the step0 null
  P2  every arm builds SUPPRESSION first (rank -> 0%), including the gated ones. This is the
      OLMo finding: a gated model with the off-switch available from birth still spends its first
      ~17B tokens building GELU-style anti-alignment.
  P3  only the PRODUCT arms (swiglu, bilinear) later convert toward enrichment (rank -> 100%);
      gelu and relu2 do not. If bilinear converts, the off-state is NOT required and H1 wins.
      If bilinear stalls with gelu while relu2 converts, H2 wins.
  P4  L1/L2 sparsity orders swiglu < bilinear < relu2 < gelu at the final checkpoint.
A null result on P2 (nobody builds suppression) would mean the juvenile phase is a property of
large-scale training, not of the architecture, and would scope every developmental claim in the
arc to >=1B-parameter models.

Usage: .venv/Scripts/python.exe minout/code/run_glu_matched.py [STEPS=8000] [ARMS=gelu,swiglu,...]
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ffn_variants import apply_ffn_variant
from corpus import BinPileDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GLUDATA", os.path.join(REPO, "..", "data"))
CORPUS = os.path.join(DATA, os.environ.get("CORPUS", "pile-5b.bin"))
CKDIR = os.path.join(DATA, os.environ.get("CKDIR", "glu_ckpt"))
# per-arm output so parallel one-arm-per-GPU processes never clobber each other's JSON
OUT = os.path.join(DATA, os.environ.get("OUTNAME", "glu-matched.json"))

ARMS = {"gelu": {"kind": "standard", "intermediate_size": 3072},
        "swiglu": {"kind": "swiglu", "intermediate_size": 2048},
        "bilinear": {"kind": "bilinear", "intermediate_size": 2048},
        "relu2": {"kind": "relu2", "intermediate_size": 3072},
        # 5th arm: Oskin NC-FFN, 2-matrix at 3072 -> exact param match to gelu.
        # bounded 2-input conjunctive detector; releases asymmetry + support.
        "ncffn": {"kind": "ncffn", "intermediate_size": 3072, "rho": 0.75},
        # both branches gated; bit-identical init to swiglu/bilinear at 2048.
        "doublegate": {"kind": "doublegate", "intermediate_size": 2048}}
STEPS = int(os.environ.get("STEPS", 8000))
# LR-anneal horizon: defaults to STEPS, but set larger to reproduce a longer run's
# schedule when stopping early (e.g. a seed replicate to step 12000 that must match
# the original 150k-horizon LR trajectory rather than annealing to floor by 12000).
SCHED_STEPS = int(os.environ.get("SCHED_STEPS", STEPS))
BATCH, SEQ, ACC = 16, 512, 2
LR, WARMUP, WD = 6e-4, 200, 0.01
DATA_SEED = int(os.environ.get("DATA_SEED", 42))
INIT_SEED = int(os.environ.get("INIT_SEED", 0))
BLOCK = int(os.environ.get("BLOCK", 500))   # round-robin block: all arms stay within
                                           # BLOCK steps of each other, so stopping at
                                           # ANY moment leaves a valid comparison
CKPT = [0, 50, 100, 200, 400, 800, 1500, 3000, 5000, 8000, 12000, 20000, 32000,
        50000, 75000, 100000, 150000, 200000]
PREDICTIONS = {
    "P1": "every arm starts at own-read rank ~50% (random init null)",
    "P2": "every arm builds suppression first (rank -> 0%), including gated",
    "P3": "only product arms (swiglu, bilinear) convert toward rank 100%",
    "P4": "final L1/L2 sparsity orders swiglu < bilinear < relu2 < gelu"}


def read_write(mlp, kind):
    """(label -> read rows), write rows; all (n_units, d_model)."""
    if kind == "standard":
        return {"read": mlp.dense_h_to_4h.weight}, mlp.dense_4h_to_h.weight.T
    if kind == "swiglu":
        return {"up": mlp.w_up.weight, "gate": mlp.w_gate.weight}, mlp.w_down.weight.T
    if kind == "bilinear":
        return {"a": mlp.w_a.weight, "b": mlp.w_b.weight}, mlp.w_down.weight.T
    if kind == "doublegate":
        return {"g1": mlp.w_a.weight, "g2": mlp.w_b.weight}, mlp.w_down.weight.T
    if kind == "ncffn":
        # one read per write column: value=A branch, gate=B branch (gelu units
        # use their own read for both). A is shared by A*B and A*(1-B), so this
        # is provisional per NEXT_ARMS s3 -- report both and flag the shared A.
        ng, hf = mlp.n_gelu, mlp.half
        Wr = mlp.dense_h_to_4h.weight
        n = Wr.shape[0]
        val = torch.arange(n, device=Wr.device)          # gelu + AB use col=A row
        gate = torch.arange(n, device=Wr.device)
        ab = torch.arange(ng, ng + hf, device=Wr.device)
        gate[ab] = ng + hf + (ab - ng)                    # AB gate = B row
        anb = torch.arange(ng + hf, ng + 2 * hf, device=Wr.device)
        val[anb] = ng + (anb - (ng + hf))                 # A*(1-B) value = A row
        gate[anb] = ng + hf + (anb - (ng + hf))           # A*(1-B) gate = B row
        return {"value": Wr[val], "gate": Wr[gate]}, mlp.dense_4h_to_h.weight.T
    return {"read": mlp.w_in.weight}, mlp.w_down.weight.T


def unit(M):
    return M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def sign_probe(model, kind, ns=96):
    """cos(own-read, own-write) per layer + own-read RANK among ALL reads in the model."""
    lay = [b.mlp for b in model.gpt_neox.layers]
    nL = len(lay)
    W = [unit(read_write(m, kind)[1].detach().float()) for m in lay]
    out = {}
    for lab in read_write(lay[0], kind)[0]:
        R = [unit(read_write(m, kind)[0][lab].detach().float()) for m in lay]
        ALL = torch.cat(R, 0)
        N, n = ALL.shape[0], R[0].shape[0]
        g = torch.Generator(device=DEV).manual_seed(7)
        cs, rk = [], []
        for L in range(nL):
            cs.append(float((R[L] * W[L]).sum(1).median()))
            idx = torch.randperm(n, generator=g, device=DEV)[:ns]
            C = W[L][idx] @ ALL.T
            own = C[torch.arange(ns, device=DEV), L * n + idx]
            rk.append(float(((C < own[:, None]).sum(1).float() + 1).median()) / N * 100)
            del C
        out[lab] = {"cos": [round(v, 4) for v in cs], "rank_pct": [round(v, 2) for v in rk],
                    "cos_mean": round(float(np.mean(cs)), 4),
                    "rank_mean": round(float(np.mean(rk)), 2)}
        del ALL, R
    return out


@torch.no_grad()
def emission_probe(model, kind, ids):
    """median per-neuron L1/L2 = (E|a|)^2 / E[a^2] of the post-activation coefficient."""
    lay = [b.mlp for b in model.gpt_neox.layers]
    down = {"standard": "dense_4h_to_h", "swiglu": "w_down",
            "bilinear": "w_down", "relu2": "w_down",
            "ncffn": "dense_4h_to_h", "doublegate": "w_down"}[kind]
    cur, acc = {}, {}

    def mk(L):
        def fn(mod, inp):
            cur[L] = inp[0].detach()
        return fn
    hk = [getattr(lay[L], down).register_forward_pre_hook(mk(L)) for L in range(len(lay))]
    for i in range(0, ids.shape[0], 4):
        model(ids[i:i + 4])
        for L in cur:
            a = cur[L].float().reshape(-1, cur[L].shape[-1])[:, :]
            s = acc.setdefault(L, {"n": 0, "sa": 0, "s2": 0})
            s["n"] += a.shape[0]
            s["sa"] = s["sa"] + a.abs().sum(0).double()
            s["s2"] = s["s2"] + (a ** 2).sum(0).double()
    for h in hk:
        h.remove()
    v = []
    for L, s in acc.items():
        e1 = (s["sa"] / s["n"]).float()
        e2 = (s["s2"] / s["n"]).float().clamp_min(1e-12)
        v.append(float(((e1 ** 2) / e2).median()))
    return round(float(np.median(v)), 4)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
    ds = BinPileDataset(CORPUS, seq_len=SEQ, seed=DATA_SEED)
    print(f"corpus: {len(ds):,} chunks of {SEQ} = {len(ds)*SEQ/1e6:.0f}M tokens")
    ev = torch.stack([ds[i] for i in range(len(ds) - 64, len(ds))]).to(DEV)
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.setdefault("config", {"steps": STEPS, "batch": BATCH, "seq": SEQ, "accum": ACC,
                              "lr": LR, "warmup": WARMUP, "wd": WD,
                              "data_seed": DATA_SEED, "init_seed": INIT_SEED,
                              "tokens_per_step": BATCH * SEQ * ACC,
                              "predictions": PREDICTIONS})
    res.setdefault("arms", {})
    print("\nPREDICTIONS ON RECORD:")
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    want = os.environ.get("ARMS", ",".join(ARMS)).split(",")
    os.makedirs(CKDIR, exist_ok=True)

    def build(name):
        """model + optimizer for `name`, resumed from disk if a checkpoint exists."""
        spec = ARMS[name]
        torch.manual_seed(INIT_SEED)
        cfg = AutoConfig.from_pretrained("EleutherAI/pythia-160m-deduped")
        # [!!] the published config declares torch_dtype=float16 and from_config HONOURS it, so
        # the model is built with fp16 MASTER WEIGHTS -- AdamW on fp16 masters NaNs at step 1 for
        # every architecture and every LR (diagnosed: loss 10.9 / gradnorm 6.03 at step 0, NaN at
        # step 1, invariant to lr 6e-4 vs 2e-4). Masters must be fp32; bf16 only via autocast.
        cfg.torch_dtype = torch.float32
        model = AutoModelForCausalLM.from_config(cfg).to(DEV, dtype=torch.float32)
        info = apply_ffn_variant(model, spec)
        model.to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
        step = 0
        cp = os.path.join(CKDIR, name + ".pt")
        if os.path.exists(cp):
            st = torch.load(cp, map_location=DEV, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            step = st["step"]
        return model, opt, step, info, spec

    def save(name, model, opt, step):
        tmp = os.path.join(CKDIR, name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, os.path.join(CKDIR, name + ".pt"))   # atomic: never a torn checkpoint

    def measure(name, model, spec, rec, st):
        model.eval()
        sp = sign_probe(model, spec["kind"])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = model(ev[:16]).logits[:, :-1].float()
            lo = float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ev[:16, 1:].reshape(-1)))
            l12 = emission_probe(model, spec["kind"], ev[:16])
        rec["traj"][str(st)] = {"loss": round(lo, 4), "l1l2": l12, "sign": sp,
                                "tokens": st * BATCH * SEQ * ACC}
        print("    [{:>8s}] step {:>7d}  {:7.1f}M tok  loss {:6.3f}  l1l2 {:.4f}  ".format(
            name, st, st * BATCH * SEQ * ACC / 1e6, lo, l12) +
            "  ".join("{}: cos {:+.3f} rank {:5.1f}%".format(
                k, sp[k]["cos_mean"], sp[k]["rank_mean"]) for k in sp), flush=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        model.train()

    # [!!] NO DataLoader. On Windows num_workers>0 spawns fresh processes on every iter(),
    # which round-robin calls once per arm per rotation -- each respawn re-imports torch and
    # transformers, and the smoke test hung ~15 min unresponsive. The dataset is a memmap; direct
    # indexing is faster, needs no workers, and makes resume exact arithmetic rather than
    # replaying skipped batches.
    NB = len(ds)

    def get_batch(bi):
        """batch `bi`, deterministic in the index alone -- so arm N at step S always sees
        exactly the batch arm M at step S saw."""
        idx = [(bi * BATCH + k) % NB for k in range(BATCH)]
        return torch.stack([ds[j] for j in idx])
    print("\nround-robin: {} arms x {} steps per rotation, target {} steps "
          "({:.2f}B tokens each)".format(len(want), BLOCK, STEPS,
                                         STEPS * BATCH * SEQ * ACC / 1e9))
    print("interrupt any time -- every arm resumes from its last block boundary\n")
    finished = False
    while not finished:
        finished = True
        for name in want:
            model, opt, step, info, spec = build(name)
            rec = res["arms"].setdefault(
                name, {"info": info, "params": sum(p.numel() for p in model.parameters()),
                       "traj": {}})
            if step >= STEPS:
                del model, opt
                torch.cuda.empty_cache()
                continue
            finished = False
            if step == 0 and "0" not in rec["traj"]:
                measure(name, model, spec, rec, 0)
            model.train()
            tgt = min(step + BLOCK, STEPS)
            while step < tgt:
                opt.zero_grad(set_to_none=True)
                for j in range(ACC):
                    b = get_batch(step * ACC + j).to(DEV)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(b[:, :-1]).logits
                        loss = torch.nn.functional.cross_entropy(
                            lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1)) / ACC
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = LR * (step / max(WARMUP, 1) if step < WARMUP else
                           0.1 + 0.45 * (1 + math.cos(math.pi * (step - WARMUP) /
                                                      max(SCHED_STEPS - WARMUP, 1))))
                for pg in opt.param_groups:
                    pg["lr"] = lr
                opt.step()
                step += 1
                if step in CKPT:
                    measure(name, model, spec, rec, step)
            save(name, model, opt, step)
            if step not in CKPT:
                measure(name, model, spec, rec, step)
            del model, opt
            torch.cuda.empty_cache()
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
