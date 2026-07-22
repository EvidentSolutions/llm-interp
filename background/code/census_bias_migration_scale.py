"""Cross-model bias migration (background paper, PLAN.md item 1 -- the
highest-value strengthener: attacks the "all causal results on one model"
limitation).

The Phi-2 result (paper Sec 5): a gate's resting pre-activation decomposes
exactly as <w.x_ln + b_n> = w.b_L + b_n, and training carries the resting
inhibition in the reference coupling w.b_L (ref-dominant >99.9%, 50-60x the
bias parameter), which rank-predicts duty at rho~0.95. This script re-runs
the STATIC decomposition + the cross-layer "one direction" check on 3-4 more
families, each against a from_config random-init twin.

Architecture note. The claim has two informative forms:
  - Models WITH an MLP gate bias (GPT-2 c_fc, Pythia dense_h_to_4h): the full
    ref-vs-bias comparison is meaningful -- frac_ref_dominant, |ref|/|bias|,
    rho(resting, duty).
  - Gated bias-free MLPs (Qwen2 / Llama SwiGLU: gate_proj has bias=False):
    ref-dominance is vacuous (b_n = 0). The informative statistics are then
    (a) the resting coupling w.b_L is negative (gates held off by the
    reference) and (b) it still rank-predicts duty. Reported as such.
Duty = frac(gate pre-activation > 0), which equals frac(act > 0) for GELU and
SiLU alike (both are >0 iff pre-activation >0).

Pre-registered branches:
  REPLICATES   -- on >=3 families: resting coupling negative and
                  rho(resting, duty) >= ~0.8 (and, where a bias exists,
                  ref-dominant >~95%), twin floored (resting ~0, duty ~0.5,
                  rho ~0). -> the mechanism is general; the paper's abstract
                  "one model" hedge on Sec 5 can be upgraded.
  HETEROGENEOUS -- spread across families -> report the distribution, keep
                  the causal claim Phi-2-specific, note which families carry
                  the coupling and which do not.
Also reported: cos(m_hat) between two mid-stack layers per model (the Sec 2
"nearly a single direction" structural claim) and its twin floor.

Weights/activations only, laptop-friendly (all models <= 1.5B). No Phi-2 here
(it is the paper's baseline; this script is the OTHER families, and avoids
its trust_remote_code path -- all models below are built-in HF archs).

Usage: .venv/Scripts/python.exe superposition/code/census_bias_migration_scale.py
       SMOKE=1 fast pass (fewer docs; first model only).
"""
import sys
import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

# (hf id, tied embeddings?) -- all built-in archs, laptop-sized
MODELS = [
    ("gpt2-medium", True),
    ("EleutherAI/pythia-1.4b-deduped", False),
    ("Qwen/Qwen2.5-1.5B", True),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", False),
]
if SMOKE:
    MODELS = MODELS[:1]

N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16
DEPTH_FRACS = [0.20, 0.35, 0.50, 0.65]   # mid-stack sampling per model

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")   # raw text list
OUT = os.path.join(DATA, "census-bias-migration-scale.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def get_layers(model):
    """Decoder layer ModuleList across the built-in architectures."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h                       # GPT-2
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers                     # Pythia
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                        # Llama / Qwen2
    raise RuntimeError("unknown architecture: no decoder layer list found")


def get_gate(layer):
    """(gate linear module, name). The linear whose input is the post-LN MLP
    input and whose output is the gate pre-activation."""
    mlp = layer.mlp
    for name in ("c_fc", "dense_h_to_4h", "gate_proj", "fc1"):
        if hasattr(mlp, name):
            return getattr(mlp, name), name
    raise RuntimeError("unknown MLP: no gate linear found")


def gate_W_b(gate_mod, name):
    """Return (W units-by-d, bias-or-None) with GPT-2 Conv1D handled."""
    W = gate_mod.weight.detach().float()
    if name == "c_fc":                 # transformers Conv1D: weight is (in,out)
        W = W.t().contiguous()         # -> (out=units, in=d)
    b = gate_mod.bias
    b = b.detach().float() if b is not None else None
    return W, b


@torch.no_grad()
def run_one(hf_id, which, docs_text, tok, sel_layers):
    """which in {trained, twin}. Returns per-layer static decomposition +
    cross-layer cosine."""
    cfg = AutoConfig.from_pretrained(hf_id)
    if which == "twin":
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, low_cpu_mem_usage=True)
    m = m.to(DEV).eval()
    layers = get_layers(m)
    d = m.config.hidden_size

    # hooks: gate-linear input (b_L material) + output (duty), per sel layer;
    # decoder-layer input (raw resid, for m_hat + cross-layer cos)
    ln_in = {}
    gate_out = {}
    resid_in = {}
    handles = []
    for L in sel_layers:
        gmod, gname = get_gate(layers[L])

        def mk_in(L=L):
            def f(mod, inp):
                ln_in[L] = inp[0][0, SKIP:].detach().float()
            return f

        def mk_out(L=L):
            def f(mod, inp, out):
                gate_out[L] = out[0, SKIP:].detach().float()
            return f

        def mk_res(L=L):
            def f(mod, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                resid_in[L] = hs[0, SKIP:].detach().float()
            return f
        handles.append(gmod.register_forward_pre_hook(mk_in()))
        handles.append(gmod.register_forward_hook(mk_out()))
        handles.append(layers[L].register_forward_pre_hook(
            mk_res(), with_kwargs=True))

    s_ln = {L: torch.zeros(d, device=DEV) for L in sel_layers}
    s_res = {L: torch.zeros(d, device=DEV) for L in sel_layers}
    duty_num = {L: None for L in sel_layers}
    npos = 0
    for txt in docs_text:
        ids = tok(txt, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] < SKIP + 48:
            continue
        m(input_ids=ids)
        T = ln_in[sel_layers[0]].shape[0]
        npos += T
        for L in sel_layers:
            s_ln[L] += ln_in[L].sum(0)
            s_res[L] += resid_in[L].sum(0)
            pc = (gate_out[L] > 0).float().sum(0)
            duty_num[L] = pc if duty_num[L] is None else duty_num[L] + pc
    for h in handles:
        h.remove()

    b_ln = {L: s_ln[L] / npos for L in sel_layers}
    m_raw = {L: s_res[L] / npos for L in sel_layers}
    mhat = {L: m_raw[L] / m_raw[L].norm().clamp(min=1e-9) for L in sel_layers}

    per_layer = {}
    for L in sel_layers:
        gmod, gname = get_gate(layers[L])
        W, b = gate_W_b(gmod, gname)
        ref = (W @ b_ln[L]).cpu()
        bnc = b.cpu() if b is not None else torch.zeros(ref.shape[0])
        resting = ref + bnc
        duty = (duty_num[L] / npos).cpu().numpy()
        per_layer[str(L)] = {
            "gate": gname,
            "has_bias": b is not None,
            "ref_med": round(float(ref.median()), 4),
            "bias_med": round(float(bnc.median()), 4),
            "resting_med": round(float(resting.median()), 4),
            "abs_ref_med": round(float(ref.abs().median()), 4),
            "abs_bias_med": round(float(bnc.abs().median()), 4),
            "frac_resting_neg": round(float((resting < 0).float().mean()), 4),
            "frac_ref_dominant": (round(float(
                (ref.abs() > bnc.abs()).float().mean()), 4)
                if b is not None else None),
            "ref_over_bias": (round(float(ref.abs().median()
                                          / bnc.abs().median().clamp(min=1e-9)),
                                    2) if b is not None else None),
            "spearman_resting_duty": round(spearman(resting.numpy(), duty), 3),
            "duty_med": round(float(np.median(duty)), 4),
        }

    # cross-layer "one direction" (Sec 2): cos(m_hat) of the 35% & 65% layers
    lo, hi = sel_layers[1], sel_layers[-1]
    xcos = round(float(F.cosine_similarity(mhat[lo], mhat[hi], dim=0)), 4)

    del m
    gc.collect()
    torch.cuda.empty_cache()
    return {"per_layer": per_layer, "cross_layer_cos": xcos,
            "cross_layer_pair": [lo, hi]}


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  models={len(MODELS)}")
    docs_text = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    rec = {"config": {"smoke": SMOKE, "n_docs": len(docs_text), "skip": SKIP,
                      "depth_fracs": DEPTH_FRACS, "seed": SEED}, "models": {}}

    for hf_id, tied in MODELS:
        print(f"\n==== {hf_id} (tied={tied}) ====")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            cfg = AutoConfig.from_pretrained(hf_id)
            n_layers = cfg.num_hidden_layers
            sel = sorted(set(min(max(int(n_layers * f), 1), n_layers - 1)
                             for f in DEPTH_FRACS))
            torch.manual_seed(SEED)
            twin = run_one(hf_id, "twin", docs_text, tok, sel)
            trained = run_one(hf_id, "trained", docs_text, tok, sel)
            rec["models"][hf_id] = {"tied": tied, "n_layers": n_layers,
                                    "sel_layers": sel,
                                    "trained": trained, "twin": twin}
            print(f"  layers {sel}  cross-layer cos(m_hat) trained="
                  f"{trained['cross_layer_cos']} twin={twin['cross_layer_cos']}")
            for L in sel:
                e = trained["per_layer"][str(L)]
                tw = twin["per_layer"][str(L)]
                print(f"  L{L} [{e['gate']}]: resting_med={e['resting_med']} "
                      f"ref_dom={e['frac_ref_dominant']} "
                      f"ref/bias={e['ref_over_bias']} "
                      f"rho(rest,duty)={e['spearman_resting_duty']} "
                      f"duty={e['duty_med']} | twin resting="
                      f"{tw['resting_med']} rho={tw['spearman_resting_duty']} "
                      f"duty={tw['duty_med']}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rec["models"][hf_id] = {"error": repr(ex)}
            print(f"  !! {hf_id} failed: {ex!r}")
            gc.collect()
            torch.cuda.empty_cache()

    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
