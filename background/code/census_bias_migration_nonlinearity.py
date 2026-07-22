"""Nonlinearity signature of the gate-threshold mechanism (background paper,
follow-up to census_bias_migration_scale.py / PLAN item 1).

Question (from the ReLU/GELU/SwiGLU discussion): the bias-migration mechanism
is nonlinearity-INDIFFERENT among sign-thresholding activations (a gate fires
iff its pre-activation > 0 for ReLU, GELU and SiLU alike). The only place the
activation SHAPE could leave a fingerprint is the operating point itself: a
sparser nonlinearity (ReLU is empirically the sparsest, cf. Szatkowski 2025)
needs a MORE NEGATIVE resting coupling to hold gates off. Prediction (3):
mean gate duty clusters by nonlinearity family at matched scale, ReLU lowest,
and resting magnitude tracks it (ReLU most negative).

This is OBSERVATIONAL, not a matched-twin experiment: these models differ in
data, scale, norm and tuning as well as the activation. A clean isolation
needs identical-architecture twins differing only in the activation (training,
out of laptop scope). What this script CAN do is break the "no ReLU in the
sample" gap (the committed sweep had only GELU + SiLU) and report whether the
family-level offset survives across several models per family. Twins are not
run: a random-init gate has ~50% duty for ANY sign-thresholding activation, so
the twin is uninformative for a sparsity comparison (it was the floor for the
migration claim, already established in census-bias-migration-scale.json).

Two scale tiers, ReLU vs GELU vs SiLU:
  ~350-410M : facebook/opt-350m (relu) | gpt2-medium, pythia-410m (gelu)
  ~1.1-1.7B : facebook/opt-1.3b (relu) | pythia-1.4b (gelu)
              | tinyllama-1.1b, qwen2.5-1.5b, smollm2-1.7b (silu)

Usage: .venv/Scripts/python.exe superposition/code/census_bias_migration_nonlinearity.py
       SMOKE=1 -> first model only, few docs.
Downloads facebook/opt-350m (~700MB) and facebook/opt-1.3b (~2.6GB) if absent.
"""
import sys
import os
import gc
import json
import time
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

# (hf id, expected nonlinearity family, param tier)
MODELS = [
    ("facebook/opt-350m", "relu", "350M"),
    ("gpt2-medium", "gelu", "355M"),
    ("EleutherAI/pythia-410m-deduped", "gelu", "410M"),
    ("facebook/opt-1.3b", "relu", "1.3B"),
    ("EleutherAI/pythia-1.4b-deduped", "gelu", "1.4B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "silu", "1.1B"),
    ("Qwen/Qwen2.5-1.5B", "silu", "1.5B"),
    ("HuggingFaceTB/SmolLM2-1.7B", "silu", "1.7B"),
]
if SMOKE:
    MODELS = MODELS[:2]

N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16
DEPTH_FRACS = [0.20, 0.35, 0.50, 0.65]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-bias-migration-nonlinearity.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def get_layers(model):
    """Decoder layer ModuleList across archs, incl. OPT."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h                          # GPT-2
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers                        # Pythia
    if hasattr(model, "model") and hasattr(model.model, "decoder") \
            and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers                   # OPT
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                           # Llama / Qwen
    raise RuntimeError("unknown architecture: no decoder layer list found")


def get_gate(layer):
    """(gate linear module, name). OPT keeps fc1 directly on the layer;
    others under layer.mlp."""
    holder = layer.mlp if hasattr(layer, "mlp") else layer
    for name in ("c_fc", "dense_h_to_4h", "gate_proj", "fc1"):
        if hasattr(holder, name):
            return getattr(holder, name), name
    raise RuntimeError("unknown MLP: no gate linear found")


def gate_W_b(gate_mod, name):
    """(W units-by-d, bias-or-None), GPT-2 Conv1D handled."""
    W = gate_mod.weight.detach().float()
    if name == "c_fc":                 # Conv1D weight is (in, out)
        W = W.t().contiguous()         # -> (out=units, in=d)
    b = gate_mod.bias
    b = b.detach().float() if b is not None else None
    return W, b


@torch.no_grad()
def run_one(hf_id, docs_text, tok, sel_layers):
    m = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    layers = get_layers(m)
    d = m.config.hidden_size

    ln_in, gate_out = {}, {}
    handles = []
    for L in sel_layers:
        gmod, _ = get_gate(layers[L])

        def mk_in(L=L):
            def f(mod, inp):
                ln_in[L] = inp[0][0, SKIP:].detach().float()
            return f

        def mk_out(L=L):
            def f(mod, inp, out):
                gate_out[L] = out[0, SKIP:].detach().float()
            return f
        handles.append(gmod.register_forward_pre_hook(mk_in()))
        handles.append(gmod.register_forward_hook(mk_out()))

    s_ln = {L: torch.zeros(d, device=DEV) for L in sel_layers}
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
            pc = (gate_out[L] > 0).float().sum(0)
            duty_num[L] = pc if duty_num[L] is None else duty_num[L] + pc
    for h in handles:
        h.remove()

    b_ln = {L: s_ln[L] / npos for L in sel_layers}
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
            "resting_med": round(float(resting.median()), 4),
            "abs_resting_med": round(float(resting.abs().median()), 4),
            "frac_resting_neg": round(float((resting < 0).float().mean()), 4),
            "frac_ref_dominant": (round(float(
                (ref.abs() > bnc.abs()).float().mean()), 4)
                if b is not None else None),
            "ref_over_bias": (round(float(ref.abs().median()
                                          / bnc.abs().median().clamp(min=1e-9)),
                                    2) if b is not None else None),
            "spearman_resting_duty": round(spearman(resting.numpy(), duty), 3),
            "duty_med": round(float(np.median(duty)), 4),
            "duty_mean": round(float(np.mean(duty)), 4),
        }
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return per_layer


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  models={len(MODELS)}")
    docs_text = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    rec = {"config": {"smoke": SMOKE, "n_docs": len(docs_text), "skip": SKIP,
                      "depth_fracs": DEPTH_FRACS, "seed": SEED,
                      "note": "trained-only; twin duty ~0.5 for any "
                              "sign-thresholding nonlinearity"},
           "models": {}}

    for hf_id, fam, tier in MODELS:
        print(f"\n==== {hf_id}  [{fam}, {tier}] ====")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            cfg = AutoConfig.from_pretrained(hf_id)
            act = getattr(cfg, "activation_function", None) \
                or getattr(cfg, "hidden_act", None) or "?"
            n_layers = cfg.num_hidden_layers
            sel = sorted(set(min(max(int(n_layers * f), 1), n_layers - 1)
                             for f in DEPTH_FRACS))
            torch.manual_seed(SEED)
            pl = run_one(hf_id, docs_text, tok, sel)
            duties = [pl[str(L)]["duty_mean"] for L in sel]
            rests = [pl[str(L)]["abs_resting_med"] for L in sel]
            agg = {"mean_duty": round(float(np.mean(duties)), 4),
                   "med_abs_resting": round(float(np.median(rests)), 4)}
            rec["models"][hf_id] = {"family": fam, "tier": tier,
                                    "config_act": act, "n_layers": n_layers,
                                    "sel_layers": sel, "per_layer": pl,
                                    "agg": agg}
            print(f"  cfg act={act}  layers {sel}")
            for L in sel:
                e = pl[str(L)]
                print(f"  L{L} [{e['gate']}]: duty_mean={e['duty_mean']} "
                      f"resting_med={e['resting_med']} "
                      f"|rest|={e['abs_resting_med']} "
                      f"ref_dom={e['frac_ref_dominant']} "
                      f"rho(rest,duty)={e['spearman_resting_duty']}")
            print(f"  AGG mean_duty={agg['mean_duty']} "
                  f"med|resting|={agg['med_abs_resting']}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rec["models"][hf_id] = {"family": fam, "error": repr(ex)}
            print(f"  !! {hf_id} failed: {ex!r}")
            gc.collect()
            torch.cuda.empty_cache()

    # family summary
    print("\n==== family summary (mean duty, lower = sparser) ====")
    by_fam = {}
    for hf_id, mrec in rec["models"].items():
        if "agg" not in mrec:
            continue
        by_fam.setdefault(mrec["family"], []).append(
            (hf_id, mrec["tier"], mrec["agg"]["mean_duty"],
             mrec["agg"]["med_abs_resting"]))
    fam_summary = {}
    for fam in ("relu", "gelu", "silu"):
        rows = by_fam.get(fam, [])
        if not rows:
            continue
        duties = [r[2] for r in rows]
        rests = [r[3] for r in rows]
        fam_summary[fam] = {"n": len(rows),
                            "mean_duty": round(float(np.mean(duties)), 4),
                            "mean_abs_resting": round(float(np.mean(rests)), 4)}
        print(f"  {fam:5s} n={len(rows)}  mean_duty="
              f"{fam_summary[fam]['mean_duty']}  "
              f"mean|resting|={fam_summary[fam]['mean_abs_resting']}")
        for hf_id, tier, du, re_ in rows:
            print(f"        {tier:5s} {hf_id:45s} duty={du} |rest|={re_}")
    rec["family_summary"] = fam_summary

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
