"""Visibility-paper revision legs (visibility_paper/README.md,
pre-registered 2026-07-20, from external review).

Leg M: extend the planted-signal sweep to 6 more models (gpt2-medium,
bloom-560m, Qwen2.5-0.5B/1.5B, TinyLlama-1.1B) -- passers join the
log f_50 ~ log d regression; failers extend the anisotropy bound; the
V-span (32k-251k) gives the ceiling's V-term its first test.
Leg E: assumption (i) on real vectors -- remove the top-15 W_U-row span
from real hidden states (and the real contrast delta) at the anchor
layers of the three isotropic models; measure the remainder's bath
kurtosis + max-sigma through row-normalised W_U.
Leg A: an 8-pair anchor battery on the five original models at the
pre-registered layers (same f_real / margin / 3x-null machinery).

Branches per README. Reuses census_visibility_threshold_scale.sweep_model
and census_visibility_anchor_scale machinery unchanged.

Usage: .venv/Scripts/python.exe superposition/code/census_visibility_revision.py
       SMOKE=1 for a fast pass (fewer models/pairs/samples).
"""
import sys
import os
import gc
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import census_visibility_threshold_scale as e_scale
import census_visibility_anchor_scale as e_anchor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
np.random.seed(SEED); torch.manual_seed(SEED)

# ---- Leg M: new models (name, expected d, tied?) ----
NEW_MODELS = [
    ("gpt2-medium", 1024, True),
    ("bigscience/bloom-560m", 1024, True),
    ("Qwen/Qwen2.5-0.5B", 896, True),
    ("Qwen/Qwen2.5-1.5B", 1536, True),
    ("TinyLlama/TinyLlama_v1.1", 2048, False),
]
if SMOKE:
    NEW_MODELS = NEW_MODELS[:1]

# prior isotropic-subset rows (from census-visibility-threshold-scale.json)
PRIOR_ISO = [
    ("EleutherAI/pythia-410m-deduped", 1024, 0.01235),
    ("EleutherAI/pythia-1b-deduped", 2048, 0.00621),
    ("microsoft/phi-2", 2560, 0.00500),
]

# ---- Leg E config ----
ISO_MODELS = [  # (name, anchor layer)
    ("EleutherAI/pythia-410m-deduped", 15),
    ("EleutherAI/pythia-1b-deduped", 10),
    ("microsoft/phi-2", 20),
]
N_STATE_DOCS = 8 if SMOKE else 40

# ---- Leg A: the battery (label, prompt A, prompt B) ----
PAIRS = [
    ("hotcold-dog", "My grandmother said that the hot dog that she saw was",
     "My grandmother said that the cold dog that she saw was"),
    ("hotcold-soup", "My grandmother said that the hot soup that she tasted was",
     "My grandmother said that the cold soup that she tasted was"),
    ("food-tool", "I bought a pizza yesterday and it was",
     "I bought a hammer yesterday and it was"),
    ("animal-vehicle", "The dog that he bought last week was",
     "The car that he bought last week was"),
    ("negation", "She said the movie was not good and the",
     "She said the movie was very good and the"),
    ("king-queen", "The king entered the room and everyone",
     "The queen entered the room and everyone"),
    ("season", "The weather in summer made the trip",
     "The weather in winter made the trip"),
    ("drink", "He drank the coffee quickly because it was",
     "He drank the water quickly because it was"),
]
if SMOKE:
    PAIRS = PAIRS[:3]
BATTERY_MODELS = e_anchor.MODELS if not SMOKE else e_anchor.MODELS[-1:]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-visibility-revision.json")


# ---------------- Leg M ----------------
def leg_m():
    print("\n############ LEG M: additional models ############")
    rng = np.random.RandomState(SEED)
    rows = []
    for name, d_exp, tied in NEW_MODELS:
        try:
            r = e_scale.sweep_model(name, d_exp, rng)
            r["tied"] = tied
            r["bath_gate_pass"] = bool(
                r["bath_max_sigma_measured"]
                <= r["bath_max_sigma_pred"] + 0.5)
            rows.append(r)
        except Exception as e:
            print(f"    !! FAILED {name}: {e}")
            rows.append({"model": name, "error": str(e), "tied": tied})

    # joint regression: prior isotropic three + new passers
    pts = [(n, d, f50) for n, d, f50 in PRIOR_ISO]
    for r in rows:
        if r.get("bath_gate_pass") and r.get("f_50"):
            pts.append((r["model"], r["d"], r["f_50"]))
    reg = None
    if len(pts) >= 3:
        ld = np.log([p[1] for p in pts])
        lf = np.log([p[2] for p in pts])
        slope, icept = np.polyfit(ld, lf, 1)
        pred = icept + slope * ld
        ss_res = float(((lf - pred) ** 2).sum())
        ss_tot = float(((lf - lf.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        reg = {"n_points": len(pts),
               "points": [{"model": n, "d": d, "f_50": f} for n, d, f in pts],
               "slope_log_f50_vs_log_d": round(float(slope), 3),
               "r2": round(float(r2), 4)}
        print(f"\nJOINT regression (prior iso + new passers): n={len(pts)}  "
              f"slope={slope:.3f}  R2={r2:.4f}")

    # V-term check on ALL bath-gate passers (prior + new): measured ceiling
    # vs sqrt(2 ln 2V)
    vterm = []
    prior_ceilings = {"EleutherAI/pythia-410m-deduped": (50277, 4.410),
                      "EleutherAI/pythia-1b-deduped": (50277, 4.486),
                      "microsoft/phi-2": (50295, 4.463)}
    for name, (v, s) in prior_ceilings.items():
        vterm.append({"model": name, "V": v, "pred": round(
            float(np.sqrt(2 * np.log(2 * v))), 3), "measured": s})
    for r in rows:
        if r.get("bath_gate_pass"):
            vterm.append({"model": r["model"], "V": r["V_eff"],
                          "pred": r["bath_max_sigma_pred"],
                          "measured": r["bath_max_sigma_measured"]})
    if len(vterm) > 3:
        vs = np.log([x["V"] for x in vterm])
        ms = [x["measured"] for x in vterm]
        rho = float(np.corrcoef(vs, ms)[0, 1]) if len(set(
            [x["V"] for x in vterm])) > 1 else None
        print(f"V-term: passers span V={min(x['V'] for x in vterm)}-"
              f"{max(x['V'] for x in vterm)}; corr(ln V, ceiling)="
              f"{rho if rho is None else round(rho, 3)}")
    else:
        rho = None
    return {"new_models": rows, "joint_regression": reg,
            "v_term": {"passers": vterm,
                       "corr_lnV_ceiling": None if rho is None
                       else round(rho, 3)}}


# ---------------- Leg E ----------------
@torch.no_grad()
def leg_e():
    print("\n############ LEG E: assumption (i) on real vectors ############")
    docs = json.load(open(DOCS, encoding="utf-8"))
    out = []
    for name, layer in (ISO_MODELS if not SMOKE else ISO_MODELS[-1:]):
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
        layers = e_anchor.get_layers(m)
        WU = e_anchor.get_head(m).weight.detach().float()
        v_eff = min(len(tok), WU.shape[0])
        WU = WU[:v_eff].to(DEV)
        V, d = WU.shape
        WUn = WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sig_pred = float(np.sqrt(2 * np.log(2 * V)))
        caps = {}

        def hook(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            caps["h"] = hs[0].detach().float()

        hnd = layers[layer].register_forward_pre_hook(hook, with_kwargs=True)

        def remainder_stats(vec):
            """Remove top-15 W_U-row span; bath stats of the remainder."""
            lg = WUn @ vec
            top = torch.topk(lg.abs(), 15).indices
            D = WU[top].T
            coef = torch.linalg.lstsq(D, vec.unsqueeze(1)).solution
            r = vec - (D @ coef).squeeze(1)
            z = WUn @ (r / r.norm().clamp(min=1e-9))
            zs = z / z.std()
            kurt = float((zs ** 4).mean() - 3)
            mx = float(zs.abs().max())
            return kurt, mx

        # real hidden states at final positions
        kr, mr = [], []
        for t in docs[:N_STATE_DOCS]:
            ids = tok(t, truncation=True, max_length=128,
                      return_tensors="pt")["input_ids"].to(DEV)
            m(input_ids=ids)
            k, x = remainder_stats(caps["h"][-1])
            kr.append(k); mr.append(x)
        # the real contrast delta (the paper's central object)
        for key, prompt in (("hot", e_anchor.PROMPT_HOT),
                            ("cold", e_anchor.PROMPT_COLD)):
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
            m(input_ids=ids)
            caps[key] = caps["h"][-1].clone()
        kd, md = remainder_stats(caps["hot"] - caps["cold"])
        hnd.remove()

        row = {"model": name, "layer": layer, "d": d, "V": V,
               "sigma_pred": round(sig_pred, 3),
               "n_states": len(kr),
               "state_kurtosis_med": round(float(np.median(kr)), 3),
               "state_maxsigma_med": round(float(np.median(mr)), 3),
               "delta_kurtosis": round(kd, 3),
               "delta_maxsigma": round(md, 3)}
        out.append(row)
        print(f"  {name} L{layer}: states kurt {row['state_kurtosis_med']} "
              f"max {row['state_maxsigma_med']}s (pred {sig_pred:.2f}s) | "
              f"delta kurt {kd:.2f} max {md:.2f}s")
        del m, WU, WUn
        gc.collect(); torch.cuda.empty_cache()
    return out


# ---------------- Leg A ----------------
@torch.no_grad()
def leg_a():
    print("\n############ LEG A: anchor battery ############")
    rng = np.random.RandomState(SEED)
    out = []
    for name, layer, n_layers in BATTERY_MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
        layers = e_anchor.get_layers(m)
        WU = e_anchor.get_head(m).weight.detach().float()
        v_eff = min(len(tok), WU.shape[0])
        WU = WU[:v_eff].to(DEV)
        V, d = WU.shape
        WUn = WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
        brng = np.random.RandomState(SEED)
        bath = []
        for _ in range(100):
            v = torch.from_numpy(brng.randn(d).astype(np.float32)).to(DEV)
            v = v / v.norm()
            lg = WUn @ v
            bath.append(float(lg.abs().max() / lg.std()))
        s = float(np.median(bath))
        f_emp = s * s / (d + s * s)
        caps = {}

        def hook(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            caps["h"] = hs[0, -1].detach().float()

        hnd = layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
        rows = []
        for label, pa, pb in PAIRS:
            hs = {}
            for key, prompt in (("a", pa), ("b", pb)):
                ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
                m(input_ids=ids)
                hs[key] = caps["h"].clone()
            delta = hs["a"] - hs["b"]
            lg = WUn @ delta
            top = torch.topk(lg.abs(), 15).indices
            D = WU[top].T
            coef = torch.linalg.lstsq(D, delta.unsqueeze(1)).solution
            recv = (D @ coef).squeeze(1)
            f_real = float((recv @ recv) / (delta @ delta))
            nulls = []
            for _ in range(20):
                sel = torch.as_tensor(rng.choice(V, 15, replace=False),
                                      device=DEV)
                Dn = WU[sel].T
                cn = torch.linalg.lstsq(Dn, delta.unsqueeze(1)).solution
                rn = (Dn @ cn).squeeze(1)
                nulls.append(float((rn @ rn) / (delta @ delta)))
            null_med = float(np.median(nulls))
            present = f_real >= 3 * null_med
            rows.append({"pair": label, "f_real": round(f_real, 5),
                         "null_med": round(null_med, 5),
                         "margin_vs_emp": round(f_real / f_emp, 2),
                         "present": bool(present)})
        hnd.remove()
        pres = [r for r in rows if r["present"]]
        med_margin = float(np.median([r["margin_vs_emp"] for r in pres])) \
            if pres else None
        out.append({"model": name, "layer": layer, "d": d,
                    "bath_sigma": round(s, 3), "f_emp_bar": round(f_emp, 5),
                    "pairs": rows, "n_present": len(pres),
                    "n_pairs": len(rows),
                    "median_margin_present": None if med_margin is None
                    else round(med_margin, 2)})
        print(f"  {name}: {len(pres)}/{len(rows)} present, median margin "
              f"{med_margin if med_margin is None else round(med_margin, 1)}x")
        for r in rows:
            print(f"    {r['pair']:<15} f_real={r['f_real']*100:6.2f}%  "
                  f"margin={r['margin_vs_emp']:6.2f}x  "
                  f"{'present' if r['present'] else 'ABSENT'}")
        del m, WU, WUn
        gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}")
    rec = {"config": {"smoke": SMOKE, "seed": SEED,
                      "new_models": [n for n, _, _ in NEW_MODELS],
                      "pairs": [p[0] for p in PAIRS],
                      "n_state_docs": N_STATE_DOCS}}
    rec["leg_M"] = leg_m()
    rec["leg_E"] = leg_e()
    rec["leg_A"] = leg_a()

    # ---- verdicts (pre-registered in visibility_paper/README.md) ----
    # Leg A branch
    iso_names = {n for n, _ in ISO_MODELS}
    verds = {}
    battery_iso = [r for r in rec["leg_A"] if r["model"] in iso_names]
    if battery_iso and not SMOKE:
        anchors = {r["model"]: next((p["margin_vs_emp"] for p in r["pairs"]
                                     if p["pair"] == "hotcold-dog"), None)
                   for r in battery_iso}
        meds = {r["model"]: r["median_margin_present"] for r in battery_iso}
        ok = all(m is not None and a is not None and m / a < 2 and a / m < 2
                 for m, a in zip(meds.values(), anchors.values()))
        verds["leg_A"] = ("BATTERY-CONFIRMS" if ok else "HETEROGENEOUS")
    # Leg E branch
    if rec["leg_E"]:
        heavy = any(r["state_kurtosis_med"] > 0.5 for r in rec["leg_E"])
        delta_ok = all(abs(r["delta_kurtosis"]) < 0.5 for r in rec["leg_E"])
        verds["leg_E"] = ("ISO-OK" if not heavy else
                          ("HEAVY-TAILED-STATES-CLEAN-DELTA" if delta_ok
                           else "HEAVY-TAILED"))
    rec["verdicts"] = verds
    print(f"\nVERDICTS: {verds}")

    json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
