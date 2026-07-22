"""Is the carried reference residual-MAINTAINED (homeostatic) or outsourced to
the static LayerNorm bias? (background paper, PLAN revision thread; Olli
2026-07-21 -- 'does homeostasis connect to layer normalization?').

The paper's b_L is the mean POST-norm input to the gate linear. For a
LayerNorm model that input is gamma*normalized + beta, so b_L silently
INCLUDES the LN bias beta -- a static per-layer parameter injected fresh at
every norm, never separated from the residual-carried mean the Sec 6
maintenance story attributes it to. RMSNorm models have no beta, so their
reference is PURELY residual-maintained (the clean homeostatic case).

Decompose each gate's resting inhibition into three additive parts:
  resting = w.b_L + b_n = [w.(b_L - beta)] + [w.beta] + b_n
            residual-carried    LN-bias        gate-bias
            (homeostatic)       (static)       (static)
For RMSNorm beta = 0 -> residual-carried == resting by construction.

Per model x layer we report which part dominates the resting inhibition, the
share carried by the LN bias, and which part rank-predicts duty. Prediction:
  - residual-carried dominates and rho(resid_carried, duty) high -> the
    reference is genuinely maintained in the residual (homeostasis explains
    the thresholds), even in LayerNorm models; beta is a minor augmentation.
  - w.beta dominates -> in LN models the 'carried constant' is substantially
    the static LN bias, not a maintained quantity; the Sec 6 story is
    RMSNorm-clean but LN-augmented. May also explain OPT (operating point in
    a static bias rather than migrated into the residual coupling).

Static, weights + one forward pass for the means. Built-in archs only.
Usage: .venv/Scripts/python.exe superposition/code/census_reference_norm_decomp.py
       SMOKE=1 -> first two models, few docs.
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

# (hf id, norm family label)
MODELS = [
    ("gpt2-medium", "LN+bias"),
    ("EleutherAI/pythia-1.4b-deduped", "LN+bias"),
    ("facebook/opt-1.3b", "LN+bias"),
    ("Qwen/Qwen2.5-1.5B", "RMSNorm"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "RMSNorm"),
    ("HuggingFaceTB/SmolLM2-1.7B", "RMSNorm"),
]
if SMOKE:
    MODELS = MODELS[:2]

N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16
DEPTH_FRACS = [0.20, 0.35, 0.50, 0.65]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-norm-decomp.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def get_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") \
            and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("unknown architecture")


def get_gate(layer):
    holder = layer.mlp if hasattr(layer, "mlp") else layer
    for name in ("c_fc", "dense_h_to_4h", "gate_proj", "fc1"):
        if hasattr(holder, name):
            return getattr(holder, name), name
    raise RuntimeError("unknown MLP")


def get_mlp_norm(layer):
    """The norm module whose output feeds the gate linear."""
    for name in ("ln_2", "post_attention_layernorm", "final_layer_norm"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def gate_W_b(gate_mod, name):
    W = gate_mod.weight.detach().float().cpu()
    if name == "c_fc":
        W = W.t().contiguous()
    b = gate_mod.bias
    b = b.detach().float().cpu() if b is not None else None
    return W, b


def norm_beta(norm_mod):
    """LN bias beta (or None for RMSNorm / bias-free norm)."""
    if norm_mod is None:
        return None
    b = getattr(norm_mod, "bias", None)
    return b.detach().float().cpu() if b is not None else None


@torch.no_grad()
def run_one(hf_id, which, docs_text, tok, sel_layers):
    cfg = AutoConfig.from_pretrained(hf_id)
    if which == "twin":
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, low_cpu_mem_usage=True)
    m = m.to(DEV).eval()
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
    b_ln = {L: (s_ln[L] / npos).cpu() for L in sel_layers}

    per_layer = {}
    for L in sel_layers:
        gmod, gname = get_gate(layers[L])
        W, b = gate_W_b(gmod, gname)
        beta = norm_beta(get_mlp_norm(layers[L]))
        ref_full = W @ b_ln[L]                         # w.b_L
        beta_term = (W @ beta) if beta is not None \
            else torch.zeros(ref_full.shape[0])        # w.beta
        resid_carried = ref_full - beta_term           # w.(b_L - beta)
        bnc = b if b is not None else torch.zeros(ref_full.shape[0])
        resting = ref_full + bnc
        duty = (duty_num[L] / npos).cpu().numpy()

        comps = torch.stack([resid_carried.abs(), beta_term.abs(), bnc.abs()])
        winner = comps.argmax(0)                       # 0 resid,1 beta,2 gate
        per_layer[str(L)] = {
            "gate": gname,
            "has_ln_bias": beta is not None,
            "has_gate_bias": b is not None,
            "resting_med": round(float(resting.median()), 4),
            "abs_resid_carried_med": round(float(resid_carried.abs().median()), 4),
            "abs_beta_term_med": round(float(beta_term.abs().median()), 4),
            "abs_gate_bias_med": round(float(bnc.abs().median()), 4),
            "beta_share": round(float(beta_term.abs().median()
                                      / resting.abs().median().clamp(min=1e-9)), 3),
            "frac_resid_dominant": round(float((winner == 0).float().mean()), 4),
            "frac_beta_dominant": round(float((winner == 1).float().mean()), 4),
            "frac_gatebias_dominant": round(float((winner == 2).float().mean()), 4),
            "rho_resid_duty": round(spearman(resid_carried.numpy(), duty), 3),
            "rho_beta_duty": (round(spearman(beta_term.numpy(), duty), 3)
                              if beta is not None else None),
            "rho_resting_duty": round(spearman(resting.numpy(), duty), 3),
            "duty_med": round(float(np.median(duty)), 4),
        }
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return per_layer


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}")
    docs_text = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    rec = {"config": {"smoke": SMOKE, "n_docs": len(docs_text), "skip": SKIP,
                      "depth_fracs": DEPTH_FRACS}, "models": {}}

    for hf_id, fam in MODELS:
        print(f"\n==== {hf_id}  [{fam}] ====")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            cfg = AutoConfig.from_pretrained(hf_id)
            n_layers = cfg.num_hidden_layers
            sel = sorted(set(min(max(int(n_layers * f), 1), n_layers - 1)
                             for f in DEPTH_FRACS))
            torch.manual_seed(SEED)
            pl = run_one(hf_id, "trained", docs_text, tok, sel)
            rec["models"][hf_id] = {"family": fam, "n_layers": n_layers,
                                    "sel_layers": sel, "per_layer": pl}
            for L in sel:
                e = pl[str(L)]
                print(f"  L{L} [{e['gate']}] ln_bias={e['has_ln_bias']}: "
                      f"|resid|={e['abs_resid_carried_med']} "
                      f"|beta|={e['abs_beta_term_med']} "
                      f"|gate_b|={e['abs_gate_bias_med']}  "
                      f"beta_share={e['beta_share']}")
                print(f"      dominant: resid={e['frac_resid_dominant']} "
                      f"beta={e['frac_beta_dominant']} "
                      f"gate={e['frac_gatebias_dominant']} | "
                      f"rho(resid,duty)={e['rho_resid_duty']} "
                      f"rho(beta,duty)={e['rho_beta_duty']} "
                      f"rho(resting,duty)={e['rho_resting_duty']}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rec["models"][hf_id] = {"family": fam, "error": repr(ex)}
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
