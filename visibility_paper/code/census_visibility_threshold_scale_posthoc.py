"""POST-HOC diagnostic for the 3ag bath-gate failures (2026-07-17).

NOT pre-registered. The 3ag sweep found pythia-70m and pythia-160m
violating the bath-ceiling gate (9.4 / 8.3 sigma vs the isotropic
prediction ~4.8) while 410m, 1b, and Phi-2 sit at 4.4-4.5. This script
asks WHY, testing the obvious suspect: rogue-dimension anisotropy of the
unembedding (Timkey & van Schijndel 2021; Puccetti et al. 2022 -- known
to be strongest in small models).

Per model, on the row-normalized W_U (the exact object the sweep used):
  1. column variance concentration: share of total row-energy carried by
     the top-1 / top-8 highest-variance coordinates (isotropic: ~1/d,
     ~8/d), plus a column-variance gini;
  2. excess kurtosis of the bath distribution W_U_n @ v for random unit
     v (isotropic: ~0 -- a heavy tail is what inflates max/std);
  3. the bath max recomputed after dropping the top-8 rogue coordinates
     from every row (re-normalized): if the inflation is carried by the
     rogue dims, the clipped bath should fall back toward ~4.8 sigma.

Prediction under the rogue-dimension account: all three statistics
separate 70m/160m from the gate-passing trio, and (3) restores the
isotropic ceiling.

Usage: .venv/Scripts/python.exe \
    superposition/code/census_visibility_threshold_scale_posthoc.py
"""
import sys
import os
import gc
import json
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
N_BATH = 100
N_DROP = 8

MODELS = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "microsoft/phi-2",
]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "census-visibility-threshold-scale-posthoc.json")


def get_unembed(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, low_cpu_mem_usage=True)
    head = getattr(m, "embed_out", None) or getattr(m, "lm_head", None)
    WU = head.weight.detach().float().clone()
    WU = WU[:min(len(tok), WU.shape[0])]
    del m, head, tok
    gc.collect()
    return WU.to(DEV)


def bath_stats(WUn, rng, d):
    """median over random unit v of max/std, plus pooled excess kurtosis."""
    maxes, kurts = [], []
    for _ in range(N_BATH):
        v = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
        v = v / v.norm()
        lg = WUn @ v
        s = lg.std()
        maxes.append(float(lg.abs().max() / s))
        z = (lg - lg.mean()) / s
        kurts.append(float((z ** 4).mean() - 3.0))
    return float(np.median(maxes)), float(np.median(kurts))


@torch.no_grad()
def main():
    rng = np.random.RandomState(SEED)
    rows = []
    for name in MODELS:
        WU = get_unembed(name)
        V, d = WU.shape
        WUn = WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)

        # 1. column variance concentration (rows unit norm => sum of
        #    per-column second moments == 1 on average)
        col_e = (WUn ** 2).mean(dim=0)          # per-coordinate energy share
        col_e = col_e / col_e.sum()
        srt = torch.sort(col_e, descending=True).values
        top1 = float(srt[0])
        top8 = float(srt[:N_DROP].sum())
        cs = torch.cumsum(torch.sort(col_e).values, 0)
        gini = float(1 - 2 * cs.mean() / cs[-1])

        # 2. bath max + kurtosis on the full matrix
        bath_full, kurt_full = bath_stats(WUn, np.random.RandomState(SEED), d)

        # 3. drop top-N_DROP rogue coordinates, renormalize, re-measure
        keep = torch.ones(d, dtype=torch.bool, device=DEV)
        keep[torch.argsort(col_e, descending=True)[:N_DROP]] = False
        Wc = WUn[:, keep]
        Wc = Wc / Wc.norm(dim=1, keepdim=True).clamp(min=1e-9)
        bath_clip, kurt_clip = bath_stats(
            Wc, np.random.RandomState(SEED), int(keep.sum()))

        pred = float(np.sqrt(2 * np.log(2 * V)))
        r = {"model": name, "d": d, "V": V,
             "iso_pred_sigma": round(pred, 3),
             "col_energy_top1": round(top1, 5),
             "col_energy_top8": round(top8, 5),
             "col_energy_top1_iso": round(1 / d, 5),
             "col_energy_top8_iso": round(N_DROP / d, 5),
             "col_gini": round(gini, 3),
             "bath_max_sigma": round(bath_full, 3),
             "bath_kurtosis": round(kurt_full, 3),
             "bath_max_sigma_top8_dropped": round(bath_clip, 3),
             "bath_kurtosis_top8_dropped": round(kurt_clip, 3)}
        rows.append(r)
        print(f"{name:<34} d={d:<5} top1 {top1:.4f} (iso {1/d:.4f})  "
              f"top8 {top8:.4f}  gini {gini:.3f}")
        print(f"{'':<34} bath {bath_full:.2f}s (kurt {kurt_full:+.2f})  "
              f"-> top8-dropped {bath_clip:.2f}s (kurt {kurt_clip:+.2f})  "
              f"pred {pred:.2f}s")
        del WU, WUn, Wc
        gc.collect()
        torch.cuda.empty_cache()

    json.dump({"note": "post-hoc, not pre-registered; see 3ag RESULT",
               "n_bath": N_BATH, "n_drop": N_DROP, "seed": SEED,
               "per_model": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
