# -*- coding: utf-8 -*-
"""Why does b_L have a different SHAPE across architectures? (Olli, 2026-09-12,
for background_glu.)

Observation: b_L (= mean post-norm MLP input) is DISTRIBUTED in Phi-2 mid-stack
(anchor: ~10-23% of norm on top outlier coords) but CONCENTRATED in Qwen2.5-3B
(control A: 66-75% on 8 coords). Two hypotheses, tested against each other here:

  H1  NORMALISATION TYPE. RMSNorm has no mean-centring, so a large ~constant
      coordinate passes into x_ln intact and b_L concentrates on it; LayerNorm
      mean-centres (and adds a learned bias), which can redistribute. TEST:
      compare b_L concentration POST-norm to m_L concentration PRE-norm (the raw
      residual mean at the block input). If the norm redistributes, post < pre
      for LayerNorm and post ~= pre for RMSNorm.

  H2  MASSIVE-ACTIVATION PROFILE. Concentration just tracks how peaked the
      model's activations are (a family/arch/training property, not norm type).
      TEST: does b_L concentration correlate with the massive-activation ratio
      max|x_ln|/median across ALL models, including within one norm type?

Discriminator: two LayerNorm models on opposite ends (Phi-2 distributed, OPT
concentrated) break a pure-H1 story and would support H2.

Per model x 3 mid-stack layers, static (one calibration pass), CPU/float32:
  conc_bL_8/32   top-K |b_L| share of ||b_L||^2         (post-norm concentration)
  conc_mL_8/32   same for m_L = mean pre-norm block input (pre-norm concentration)
  post_over_pre  conc_bL_8 / conc_mL_8                   (>1 norm concentrates)
  massive_ratio  max_i mabs / median_i mabs, mabs = mean_t |x_ln|  (peakedness)
  jaccard_bL_massive  overlap of b_L top-8 coords with mabs top-8 coords
  norm_type      RMSNorm | LayerNorm

Usage: .venv/Scripts/python.exe superposition/code/census_bL_shape_xarch.py
"""
import os
import sys
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # avoid surprise downloads; skip if uncached
import numpy as np
import torch

DEV = "cpu"
DTYPE = torch.float32
MAXLEN, SKIP = 128, 16
N_CAL = 16
MODELS = [
    # (name, norm-type-hint gate-type for the writeup; detected at runtime too)
    "Qwen/Qwen2.5-3B",              # RMSNorm, SwiGLU
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # RMSNorm, SwiGLU
    "HuggingFaceTB/SmolLM2-1.7B",   # RMSNorm, SwiGLU
    "microsoft/phi-2",              # LayerNorm, gelu-gated  (anchor: distributed)
    "gpt2-medium",                  # LayerNorm, gelu
    "facebook/opt-350m",            # LayerNorm, relu        (sweep: concentrated)
    "EleutherAI/pythia-410m",       # LayerNorm, gelu
]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-bL-shape-xarch.json")


def find_layers(model):
    for path in ("model.layers", "transformer.h", "model.decoder.layers",
                 "gpt_neox.layers"):
        obj, ok = model, True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("could not locate decoder layers")


def find_mlp(layer):
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "fc1"):          # OPT
        return layer.fc1
    raise RuntimeError("could not locate MLP module in layer")


def norm_type(model):
    return ("RMSNorm" if any("RMSNorm" in type(m).__name__ for m in model.modules())
            else "LayerNorm")


def topk_share(vec, k):
    v2 = vec.double() ** 2
    tot = v2.sum().clamp(min=1e-20)
    return float(v2.topk(min(k, v2.numel())).values.sum() / tot)


def analyze(name, docs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    layers = find_layers(m)
    NL = len(layers)
    d = m.config.hidden_size
    nt = norm_type(m)
    tgt = sorted(set(int(round(f * NL)) for f in (0.25, 0.5, 0.75)))
    tgt = [min(max(L, 0), NL - 1) for L in tgt]

    cap = {}   # (L,'pre'|'post') -> running sums

    def pre_hook_block(L):
        def h(mod, args, kwargs):
            x = (args[0] if args else kwargs.get("hidden_states"))
            x = x.detach().float().reshape(-1, x.shape[-1])[SKIP:]
            a = cap.setdefault((L, "pre"), [torch.zeros(d, dtype=torch.float64),
                                            torch.zeros(d, dtype=torch.float64), 0])
            a[0] += x.sum(0); a[1] += x.abs().sum(0); a[2] += x.shape[0]
        return h

    def pre_hook_mlp(L):
        def h(mod, args, kwargs):
            x = (args[0] if args else kwargs.get("hidden_states"))
            x = x.detach().float().reshape(-1, x.shape[-1])[SKIP:]
            a = cap.setdefault((L, "post"), [torch.zeros(d, dtype=torch.float64),
                                             torch.zeros(d, dtype=torch.float64), 0])
            a[0] += x.sum(0); a[1] += x.abs().sum(0); a[2] += x.shape[0]
        return h

    hooks = []
    for L in tgt:
        hooks.append(layers[L].register_forward_pre_hook(
            pre_hook_block(L), with_kwargs=True))
        hooks.append(find_mlp(layers[L]).register_forward_pre_hook(
            pre_hook_mlp(L), with_kwargs=True))

    n = 0
    for text in docs[:N_CAL]:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        n += 1
    for h in hooks:
        h.remove()

    out = {"n_layers": NL, "d": d, "norm_type": nt, "layers": tgt, "per_layer": {}}
    for L in tgt:
        pre, post = cap.get((L, "pre")), cap.get((L, "post"))
        if pre is None or post is None:
            continue
        bL = (post[0] / max(post[2], 1)).float()          # mean post-norm
        mL = (pre[0] / max(pre[2], 1)).float()            # mean pre-norm
        mabs = (post[1] / max(post[2], 1)).float()        # mean |x_ln| per coord
        b_top8 = set(bL.abs().topk(8).indices.tolist())
        m_top8 = set(mabs.abs().topk(8).indices.tolist())
        jac = len(b_top8 & m_top8) / len(b_top8 | m_top8)
        e = {"conc_bL_8": round(topk_share(bL, 8), 4),
             "conc_bL_32": round(topk_share(bL, 32), 4),
             "conc_mL_8": round(topk_share(mL, 8), 4),
             "conc_mL_32": round(topk_share(mL, 32), 4),
             "post_over_pre_8": round(topk_share(bL, 8) /
                                      max(topk_share(mL, 8), 1e-9), 3),
             "massive_ratio": round(float(mabs.max() /
                                          mabs.median().clamp(min=1e-9)), 1),
             "jaccard_bL_massive": round(jac, 3)}
        out["per_layer"][f"L{L}"] = e
    del m
    return out


def main():
    t0 = time.time()
    docs = json.load(open(PILE, encoding="utf-8"))
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res = res.get("results", res) if isinstance(res, dict) else {}
    print(f"{'model':>34} {'norm':>10} {'L':>4} {'cBL8':>6} {'cBL32':>6} "
          f"{'cML8':>6} {'post/pre':>9} {'massive':>8} {'jac':>5}", flush=True)
    for name in MODELS:
        if name in res:
            continue
        try:
            r = analyze(name, docs)
        except Exception as ex:
            print(f"  {name}: SKIP ({type(ex).__name__}: {str(ex)[:70]})", flush=True)
            continue
        res[name] = r
        for L in r["layers"]:
            e = r["per_layer"].get(f"L{L}")
            if not e:
                continue
            print(f"{name[-32:]:>34} {r['norm_type']:>10} {L:>4} "
                  f"{e['conc_bL_8']:>6.3f} {e['conc_bL_32']:>6.3f} "
                  f"{e['conc_mL_8']:>6.3f} {e['post_over_pre_8']:>9.2f} "
                  f"{e['massive_ratio']:>8.1f} {e['jaccard_bL_massive']:>5.2f}",
                  flush=True)
        json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: H1 (norm redistributes) => post/pre << 1 for LayerNorm, ~1 for "
          "RMSNorm.\nH2 (massive profile) => conc_bL tracks massive_ratio across "
          "all models,\nincluding two LayerNorm models on opposite ends "
          "(phi-2 vs opt).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
