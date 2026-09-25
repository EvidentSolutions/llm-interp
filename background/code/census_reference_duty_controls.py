"""Addendum to census_reference_statistic_nulls.py: what ELSE predicts duty?

Leg B found that layer 30's reference direction -- cosine only 0.24-0.30 to the
mid-stack reference -- still rank-predicts mid-stack duty at rho 0.69-0.77. Two
readings, and they need separating:

  (a) the reference is a broad band and the operating point is robust to which
      nearby direction carries it; or
  (b) the duty ordering is a property of the fc1 ROW POPULATION (row norms, a
      dominant shared component), and almost any direction with a systematic
      projection recovers it -- in which case b_L is not doing the work.

The discriminating control is the one Leg B lacks: a RANDOM direction at matched
norm. Plus the two simplest deflationary alternatives.

Per layer, trained and twin:
  rho_bL           spearman(w.b_L + b_n, duty)                 [the paper's stat]
  rho_random       spearman(w.(||b_L||*r_hat) + b_n, duty), K random r_hat;
                   median and p95 of |rho|. THE control.
  rho_rownorm      spearman(||w_n||, duty)      -- row norm alone
  rho_bias         spearman(b_n, duty)          -- bias parameter alone
  rho_coupling     spearman(w.b_hat_L, duty)    -- coupling without b_n (the
                                                   paper's per-gate c_n stat)
  rho_negrownorm   spearman(-||w_n||, duty)     -- sign check for rownorm

Usage: .venv/Scripts/python.exe superposition/code/census_reference_duty_controls.py
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
from scipy.stats import spearmanr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
N_DOCS = int(os.environ.get("NDOCS", "40"))
MAXLEN, SKIP = 256, 16
PAPER_LAYERS = [6, 10, 14, 22]
K_RAND = 50

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-duty-controls.json")


def sp(a, b):
    r = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return float(r) if np.isfinite(r) else float("nan")


@torch.no_grad()
def run(which, docs_ids, rng):
    print(f"\n=== {which} ===", flush=True)
    if which == "random":
        m = AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False).to(DEV).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    d, n_layers = m.config.hidden_size, len(m.model.layers)
    want = [L for L in PAPER_LAYERS if L < n_layers]

    sum_ln = {L: torch.zeros(d, device=DEV) for L in want}
    pos_cnt = {L: None for L in want}
    cnt = 0
    for ids in docs_ids:
        cap = {}
        hs = []
        for L in want:
            def mk(L=L):
                def f(mod, inp):
                    cap[L] = inp[0][0].detach().float()
                return f
            hs.append(m.model.layers[L].mlp.fc1.register_forward_pre_hook(mk()))
        m(input_ids=ids)
        for h in hs:
            h.remove()
        for L in want:
            x = cap[L][SKIP:]
            sum_ln[L] += x.sum(0)
            w = m.model.layers[L].mlp.fc1.weight.detach().float()
            b = m.model.layers[L].mlp.fc1.bias.detach().float()
            fired = ((x @ w.T + b) > 0).sum(0).float()
            pos_cnt[L] = fired if pos_cnt[L] is None else pos_cnt[L] + fired
        cnt += cap[want[0]].shape[0] - SKIP
        del cap

    out = {}
    for L in want:
        b_L = sum_ln[L] / cnt
        bh = b_L / b_L.norm().clamp(min=1e-9)
        nrm = b_L.norm()
        w = m.model.layers[L].mlp.fc1.weight.detach().float()
        bn = m.model.layers[L].mlp.fc1.bias.detach().float()
        duty = (pos_cnt[L] / cnt).cpu().numpy().astype(np.float64)

        rest = (w @ b_L + bn).cpu().numpy().astype(np.float64)
        coup = (w @ bh).cpu().numpy().astype(np.float64)
        rown = w.norm(dim=1).cpu().numpy().astype(np.float64)
        bnn = bn.cpu().numpy().astype(np.float64)

        rr = []
        for _ in range(K_RAND):
            r = torch.randn(d, device=DEV, generator=None)
            r = r / r.norm()
            rv = (w @ (nrm * r) + bn).cpu().numpy().astype(np.float64)
            rr.append(abs(sp(rv, duty)))
        rr = np.array(rr)

        out[str(L)] = {
            "duty_median": round(float(np.median(duty)), 4),
            "rho_bL": round(sp(rest, duty), 4),
            "rho_coupling_no_bias": round(sp(coup, duty), 4),
            "rho_random_median": round(float(np.median(rr)), 4),
            "rho_random_p95": round(float(np.percentile(rr, 95)), 4),
            "rho_random_max": round(float(rr.max()), 4),
            "rho_rownorm": round(sp(rown, duty), 4),
            "rho_bias_only": round(sp(bnn, duty), 4),
        }
        e = out[str(L)]
        print(f"  L{L}: rho_bL={e['rho_bL']:+.3f}  coupling={e['rho_coupling_no_bias']:+.3f}"
              f"  RANDOM med={e['rho_random_median']:.3f} p95={e['rho_random_p95']:.3f}"
              f" max={e['rho_random_max']:.3f}  rownorm={e['rho_rownorm']:+.3f}"
              f"  bias={e['rho_bias_only']:+.3f}", flush=True)

    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    docs_ids = []
    for t in docs:
        ids = tok(t, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] >= SKIP + 48:
            docs_ids.append(ids)
    print(f"{len(docs_ids)} docs usable; K_RAND={K_RAND}")
    rec = {"config": {"n_docs": len(docs_ids), "k_rand": K_RAND, "seed": SEED}}
    for which in ["trained", "random"]:
        rec[which] = run(which, docs_ids, rng)
    json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
