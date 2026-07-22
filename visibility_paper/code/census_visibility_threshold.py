"""The visibility threshold (plan_mid_stack_empirical_basis.md 3af,
Olli's puzzle made demonstrative, pre-registered 2026-07-16).

Why do contrastive reads come out clean over a ~90%-carpet stream? Because
the top-k readout is a max-statistic: an isotropic mass in d=2560 tops out
at the extreme-value ceiling ~sqrt(2 ln 2V) sigma over V=51,200 rows,
while an aligned component carrying energy fraction f projects at
sqrt(f)||x||. Analytic threshold f* = 2 ln(2V) / (d + 2 ln(2V)) ~ 0.9%.

  Leg 1: synthetic threshold curve -- hit-rate of planted tokens in the
         top-15 vs f, single-token and 5-token-family signals.
  Leg 2: triangulation's sqrt(N) rescue of a sub-threshold signal.
  Leg 3: one real contrast (hot dog vs cold dog, L20, enveloped) placed on
         the curve: measured aligned-energy fraction + read cleanliness.

Gates: f=0 hit-rate ~0 with bath max at the predicted sigma; f=0.3
hit-rate ~1; Leg 3 reproduces the known clean food read before its energy
fraction is quoted.

Usage: .venv/Scripts/python.exe superposition/code/census_visibility_threshold.py
       SMOKE=1 fast pass.
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
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

FS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
NS = [1, 2, 4, 8, 16]
F_TRI = 0.003
N_SAMP = 30 if SMOKE else 100
TOPK = 15
L_REAL = 20

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "census-visibility-threshold.json")

PROMPT_HOT = "My grandmother said that the hot dog that she saw was"
PROMPT_COLD = "My grandmother said that the cold dog that she saw was"
FOOD_WORDS = ("food", "cook", "eat", "tast", "delici", "fried", "grill",
              "juic", "crisp", "flavor", "spic", "meat", "sausage",
              "charred", "season", "edible")


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  n/f={N_SAMP}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    WU = m.lm_head.weight.detach().float().to(DEV)          # (V, d)
    V, d = WU.shape
    WUn = WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
    f_star = 2 * np.log(2 * V) / (d + 2 * np.log(2 * V))
    sig_ceiling = float(np.sqrt(2 * np.log(2 * V)))
    rec = {"config": {"smoke": SMOKE, "fs": FS, "ns": NS, "f_tri": F_TRI,
                      "n_samp": N_SAMP, "topk": TOPK, "V": V, "d": d,
                      "seed": SEED},
           "analytic": {"f_star": round(float(f_star), 5),
                        "bath_max_sigma": round(sig_ceiling, 3)}}
    print(f"analytic f* = {f_star:.4f}  bath ceiling = {sig_ceiling:.2f} "
          f"sigma")
    rng = np.random.RandomState(SEED)

    def rand_unit():
        v = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
        return v / v.norm()

    def topk_hits(delta, planted_ids):
        lg = WUn @ delta
        top = torch.topk(lg.abs(), TOPK).indices.tolist()
        hits = len(set(top) & set(planted_ids))
        best_rank = min((torch.argsort(lg.abs(), descending=True)
                         == t).nonzero()[0].item()
                        for t in planted_ids)
        return hits, best_rank, top

    # gate at f=0: bath max in sigma units
    bath = []
    for _ in range(N_SAMP):
        v = rand_unit()
        lg = WUn @ v
        bath.append(float(lg.abs().max() / lg.std()))
    rec["gate_f0_bath_max_sigma"] = round(float(np.median(bath)), 3)
    print(f"gate f=0: bath max = {np.median(bath):.2f} sigma "
          f"(predicted ~{sig_ceiling:.2f})")

    # content-word pool for planting (space-prefixed alphabetic)
    pool = []
    for t in range(0, V, 7):
        s = tok.decode([t])
        if s.startswith(" ") and s[1:].isalpha() and len(s) > 4:
            pool.append(t)
        if len(pool) >= 4000:
            break

    # ---- Leg 1: threshold curve ----
    leg1 = {}
    for mode in ("single", "family5"):
        leg1[mode] = {}
        for f in FS:
            hits_acc, rank_acc = [], []
            for i in range(N_SAMP):
                if mode == "single":
                    tid = int(rng.choice(pool))
                    sig = WUn[tid].clone()
                    planted = [tid]
                else:
                    seed_t = int(rng.choice(pool))
                    cs = WUn @ WUn[seed_t]
                    fam = torch.topk(cs, 5).indices.tolist()
                    s_ = WUn[torch.as_tensor(fam, device=DEV)].sum(0)
                    sig = s_ / s_.norm()
                    planted = fam
                delta = float(np.sqrt(1 - f)) * rand_unit() \
                    + float(np.sqrt(f)) * sig
                h, r, _ = topk_hits(delta, planted)
                hits_acc.append(h / len(planted))
                rank_acc.append(r)
            leg1[mode][str(f)] = {
                "hit_rate": round(float(np.mean(hits_acc)), 3),
                "best_rank_med": int(np.median(rank_acc))}
            print(f"  Leg1 {mode} f={f}: hit={np.mean(hits_acc):.3f} "
                  f"rank_med={int(np.median(rank_acc))}")
    rec["leg1"] = leg1

    # ---- Leg 2: triangulation rescue ----
    leg2 = {}
    for N in NS:
        hits_acc, rank_acc = [], []
        for i in range(N_SAMP):
            tid = int(rng.choice(pool))
            sig = WUn[tid].clone()
            deltas = [float(np.sqrt(1 - F_TRI)) * rand_unit()
                      + float(np.sqrt(F_TRI)) * sig for _ in range(N)]
            avg = torch.stack(deltas).mean(0)
            h, r, _ = topk_hits(avg, [tid])
            hits_acc.append(h)
            rank_acc.append(r)
        leg2[str(N)] = {"hit_rate": round(float(np.mean(hits_acc)), 3),
                        "best_rank_med": int(np.median(rank_acc))}
        print(f"  Leg2 N={N}: hit={np.mean(hits_acc):.3f} "
              f"rank_med={int(np.median(rank_acc))}")
    rec["leg2"] = leg2
    rec["leg2_predicted_rescue_N"] = round(float(f_star / F_TRI), 1)

    # ---- Leg 3: real-Delta anchor ----
    caps = {}

    def hook(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        caps["h"] = hs[0, -1].detach().float()
    hnd = m.model.layers[L_REAL].register_forward_pre_hook(
        hook, with_kwargs=True)
    ids_h = tok(PROMPT_HOT, return_tensors="pt")["input_ids"].to(DEV)
    m(input_ids=ids_h)
    h_hot = caps["h"].clone()
    ids_c = tok(PROMPT_COLD, return_tensors="pt")["input_ids"].to(DEV)
    m(input_ids=ids_c)
    h_cold = caps["h"].clone()
    hnd.remove()
    delta = h_hot - h_cold
    lg = WUn @ delta
    top = torch.topk(lg.abs(), TOPK).indices
    top_toks = [tok.decode([int(t)]) for t in top]
    food_hits = sum(1 for s in top_toks
                    if any(w in s.lower() for w in FOOD_WORDS))
    # aligned-energy fraction: top-15 span energy (3ae machinery)
    D = WU[top].T                                            # (d, 15)
    coef = torch.linalg.lstsq(D, delta.unsqueeze(1)).solution
    recv = (D @ coef).squeeze(1)
    f_real = float((recv @ recv) / (delta @ delta))
    # null span energy for 15 random rows
    nulls = []
    for _ in range(40):
        sel = torch.as_tensor(rng.choice(V, TOPK, replace=False),
                              device=DEV)
        Dn = WU[sel].T
        cn = torch.linalg.lstsq(Dn, delta.unsqueeze(1)).solution
        rn = (Dn @ cn).squeeze(1)
        nulls.append(float((rn @ rn) / (delta @ delta)))
    rec["leg3"] = {"top_tokens": top_toks, "food_hits_in_top15": food_hits,
                   "aligned_energy_frac": round(f_real, 4),
                   "null_span_frac_med": round(float(np.median(nulls)), 4),
                   "f_star": round(float(f_star), 4)}
    print(f"  Leg3 real Δ: food-ish {food_hits}/15, aligned-energy "
          f"f={f_real:.3f} (null {np.median(nulls):.3f}, f*={f_star:.3f})")
    print(f"    top: {top_toks}")

    del m, WU, WUn
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
