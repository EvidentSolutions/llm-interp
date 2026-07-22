"""The real anchor at every scale (plan_mid_stack_empirical_basis.md 3ah,
3ag's named missing half, pre-registered 2026-07-17).

3ag located the BAR at every scale; this leg asks whether the SIGNAL
compensates. The 3af Leg-3 machinery, unchanged, at five scales: the
enveloped hot/cold-dog contrast, read at the final position at the layer
nearest 62.5% depth; f_real = least-squares span energy of the top-15
W_U rows on Delta; per model the measured bath ceiling s gives the
EMPIRICAL bar f_emp = s^2/(d + s^2); the margin is m = f_real/f_emp.

Branches: COMPENSATION (margin ~constant, within ~2x of Phi-2's) /
NO-COMPENSATION (margin falls as d falls; small-model margins < half of
Phi-2's) / INVERTED (margin grows at small d).

Gates: (1) Phi-2 reproduces 3af Leg 3 (f_real 0.0923 +/- 0.010, >= 8 of
the recorded top-15 tokens recur); (2) signal-present: a margin is only
interpreted where f_real >= 3x the model's own random-span null --
failing models are reported SIGNAL-ABSENT, not read-blocked.

Usage: .venv/Scripts/python.exe \
           superposition/code/census_visibility_anchor_scale.py
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

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
TOPK = 15
N_NULL = 40
N_BATH = 100

# (name, capture layer, n_layers) -- layer nearest 62.5% depth,
# pre-registered.
MODELS = [
    ("EleutherAI/pythia-70m-deduped", 4, 6),
    ("EleutherAI/pythia-160m-deduped", 8, 12),
    ("EleutherAI/pythia-410m-deduped", 15, 24),
    ("EleutherAI/pythia-1b-deduped", 10, 16),
    ("microsoft/phi-2", 20, 32),
]

PROMPT_HOT = "My grandmother said that the hot dog that she saw was"
PROMPT_COLD = "My grandmother said that the cold dog that she saw was"
FOOD_WORDS = ("food", "cook", "eat", "tast", "delici", "fried", "grill",
              "juic", "crisp", "flavor", "spic", "meat", "sausage",
              "charred", "season", "edible")

# 3af Leg-3 published record -- the Phi-2 reproduction gate.
AF_F_REAL = 0.0923
AF_F_TOL = 0.010
AF_TOP15 = [" euth", " hiber", " despair", " temperament", " wary",
            "winter", " recovering", " delicious", " enrolled",
            " deterrence", " breeds", " afraid", " tantal", " canine",
            " scared"]
AF_RECUR_MIN = 8

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "census-visibility-anchor-scale.json")


def get_layers(m):
    if hasattr(m, "gpt_neox"):
        return m.gpt_neox.layers
    return m.model.layers


def get_head(m):
    return getattr(m, "embed_out", None) or getattr(m, "lm_head", None)


@torch.no_grad()
def run_model(name, layer, n_layers, rng):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    layers = get_layers(m)
    assert len(layers) == n_layers, \
        f"{name}: {len(layers)} layers, pre-registered {n_layers}"
    WU = get_head(m).weight.detach().float()
    v_eff = min(len(tok), WU.shape[0])
    WU = WU[:v_eff].to(DEV)
    V, d = WU.shape
    WUn = WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
    print(f"\n=== {name}  d={d}  V={v_eff}  capture L{layer}/{n_layers}")

    # measured bath ceiling (3ag construction, seed 0) -> empirical bar
    brng = np.random.RandomState(SEED)
    bath = []
    for _ in range(N_BATH):
        v = torch.from_numpy(brng.randn(d).astype(np.float32)).to(DEV)
        v = v / v.norm()
        lg = WUn @ v
        bath.append(float(lg.abs().max() / lg.std()))
    s = float(np.median(bath))
    f_emp = s * s / (d + s * s)
    f_an = float(2 * np.log(2 * V) / (d + 2 * np.log(2 * V)))
    print(f"    bath {s:.2f} sigma -> empirical bar f_emp = "
          f"{f_emp*100:.2f}%  (analytic f* = {f_an*100:.2f}%)")

    # capture the contrast at the pre-registered layer
    caps = {}

    def hook(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        caps["h"] = hs[0, -1].detach().float()

    hnd = layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    for key, prompt in (("hot", PROMPT_HOT), ("cold", PROMPT_COLD)):
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        m(input_ids=ids)
        caps[key] = caps["h"].clone()
    hnd.remove()
    delta = caps["hot"] - caps["cold"]

    lg = WUn @ delta
    top = torch.topk(lg.abs(), TOPK).indices
    top_toks = [tok.decode([int(t)]) for t in top]
    food_hits = sum(1 for t in top_toks
                    if any(w in t.lower() for w in FOOD_WORDS))

    # aligned-energy fraction: top-15 span energy (3af Leg-3 machinery)
    D = WU[top].T
    coef = torch.linalg.lstsq(D, delta.unsqueeze(1)).solution
    recv = (D @ coef).squeeze(1)
    f_real = float((recv @ recv) / (delta @ delta))

    nulls = []
    for _ in range(N_NULL):
        sel = torch.as_tensor(rng.choice(V, TOPK, replace=False), device=DEV)
        Dn = WU[sel].T
        cn = torch.linalg.lstsq(Dn, delta.unsqueeze(1)).solution
        rn = (Dn @ cn).squeeze(1)
        nulls.append(float((rn @ rn) / (delta @ delta)))
    null_med = float(np.median(nulls))

    signal_present = f_real >= 3 * null_med
    margin_emp = f_real / f_emp
    margin_an = f_real / f_an
    print(f"    f_real = {f_real*100:.2f}%  (null {null_med*100:.2f}%, "
          f"ratio {f_real/null_med:.1f}x, "
          f"{'SIGNAL-PRESENT' if signal_present else 'SIGNAL-ABSENT'})")
    print(f"    margin vs empirical bar = {margin_emp:.2f}x   "
          f"(vs analytic {margin_an:.2f}x)   food-ish {food_hits}/15")
    print(f"    top: {top_toks}")

    del m, WU, WUn
    gc.collect()
    torch.cuda.empty_cache()
    return {"model": name, "d": d, "V_eff": v_eff,
            "layer": layer, "n_layers": n_layers,
            "bath_sigma": round(s, 3),
            "f_emp_bar": round(f_emp, 5),
            "f_star_analytic": round(f_an, 5),
            "f_real": round(f_real, 5),
            "null_span_med": round(null_med, 5),
            "signal_over_null": round(f_real / null_med, 2),
            "signal_present": bool(signal_present),
            "margin_vs_emp_bar": round(margin_emp, 3),
            "margin_vs_analytic": round(margin_an, 3),
            "food_hits_in_top15": food_hits,
            "top_tokens": top_toks,
            "secs": round(time.time() - t0, 1)}


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}")
    rng = np.random.RandomState(SEED)
    rec = {"config": {"topk": TOPK, "n_null": N_NULL, "n_bath": N_BATH,
                      "seed": SEED,
                      "prompts": [PROMPT_HOT, PROMPT_COLD],
                      "models": [(n, l, nl) for n, l, nl in MODELS]},
           "per_model": []}

    for name, layer, n_layers in MODELS:
        try:
            rec["per_model"].append(run_model(name, layer, n_layers, rng))
        except Exception as e:
            print(f"    !! FAILED {name}: {e}")
            rec["per_model"].append({"model": name, "error": str(e)})

    ok = [r for r in rec["per_model"] if "f_real" in r]

    # ---- Gate 1: Phi-2 reproduction of 3af Leg 3 ----
    phi = next((r for r in ok if r["model"] == "microsoft/phi-2"), None)
    g1 = {"pass": False}
    if phi:
        recur = len(set(phi["top_tokens"]) & set(AF_TOP15))
        g1 = {"f_real_published": AF_F_REAL, "f_real_measured": phi["f_real"],
              "dev": round(phi["f_real"] - AF_F_REAL, 4), "tol": AF_F_TOL,
              "top15_recurrence": recur, "recur_min": AF_RECUR_MIN,
              "pass": abs(phi["f_real"] - AF_F_REAL) <= AF_F_TOL
              and recur >= AF_RECUR_MIN}
    rec["gate_reproduction_3af_leg3"] = g1

    # ---- The reading: margins across scale ----
    interp = [r for r in ok if r["signal_present"]]
    if phi and phi["signal_present"] and len(interp) >= 2:
        pm = phi["margin_vs_emp_bar"]
        small = [r for r in interp if r["model"] != "microsoft/phi-2"]
        margins = {r["model"]: r["margin_vs_emp_bar"] for r in interp}
        if all(r["margin_vs_emp_bar"] >= pm / 2 for r in small) \
                and all(r["margin_vs_emp_bar"] <= pm * 2 for r in small):
            verdict = "COMPENSATION"
        elif all(r["margin_vs_emp_bar"] < pm / 2 for r in small):
            verdict = "NO-COMPENSATION"
        elif all(r["margin_vs_emp_bar"] > pm * 2 for r in small):
            verdict = "INVERTED"
        else:
            verdict = "MIXED"
        rec["reading"] = {
            "margins_vs_emp_bar": margins,
            "signal_absent": [r["model"] for r in ok
                              if not r["signal_present"]],
            "verdict": verdict,
            "branches": {
                "COMPENSATION": "margin ~constant: corollary refuted",
                "NO-COMPENSATION": "margin falls with d: corollary "
                                   "measured end to end",
                "INVERTED": "margin grows at small d: competence "
                            "confound flag"}}

        print("\n" + "=" * 72)
        print(f"{'model':<32} {'d':>5} {'bar_emp':>8} {'f_real':>8} "
              f"{'margin':>7} {'sig':>4}")
        for r in ok:
            print(f"{r['model']:<32} {r['d']:>5} "
                  f"{r['f_emp_bar']*100:>7.2f}% {r['f_real']*100:>7.2f}% "
                  f"{r['margin_vs_emp_bar']:>6.2f}x "
                  f"{'  ok' if r['signal_present'] else 'ABST'}")
        print("=" * 72)
        print(f"gate 1 (Phi-2 reproduces 3af Leg 3): "
              f"{'PASS' if g1['pass'] else 'FAIL'}  {g1}")
        print(f"VERDICT: {verdict}")

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
