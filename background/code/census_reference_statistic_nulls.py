"""Are the background paper's two headline STATISTICS discriminating, or are they
restatements of their own construction? (Olli's challenge, 2026-08-13.)

Two worries, both fair on their face:

(1) TRIVIALITY. `resting = w.b_L + b_n` is an IDENTITY, not a measurement:
    b_L := <x_ln>, so resting == <w.x_ln + b_n> == the MEAN PRE-ACTIVATION.
    `duty` is the fraction of that same distribution above zero. rho(mean,
    P(>0)) is high for ANY family of distributions with comparable spread.
    The paper's twin does NOT test this: the twin has resting ~ 0 for every
    neuron, i.e. no spread to correlate, so it cannot distinguish "b_L is the
    reference" from "means vary and means predict duty".
    -> LEG A builds the null the twin does not: give a population per-neuron
       offsets that carry ZERO information about b_L, and see whether rho
       survives.

(2) COMPOSITION. If b_L's direction/content changes with depth, then "gates
    couple to THE carried reference" may really be "each layer's gates couple
    to the mean of their own input", which is a per-layer fact and a weaker
    claim than a single carried object. The paper tests one pair (L10 accepts
    L18's mean, cos 0.94) causally, and Limitations already concede the
    geometry is family-dependent (Pythia-1.4B 0.41, BELOW its own twin floor).
    -> LEG B does the full layer x layer version statically: cosine matrix, and
       functional transfer (does another layer's reference predict THIS layer's
       duty?).

LEG A -- per layer, trained and twin:
  identity check   resting (from weights) vs mu_n (from activations); must match
  rho_actual       spearman(resting, duty)                    [paper: ~0.95]
  rho_permuted     PERMUTE mu across neurons, keep each neuron's own centered
                   distribution, recompute duty' = P(centered + mu' > 0), then
                   spearman(mu', duty'). mu' is a random relabelling and carries
                   NO information about b_L. If this is also ~0.95, the
                   statistic is a location fact, not a b_L fact.
  rho_gauss        spearman(mu, Phi(mu/sigma)) -- the pure location model
  rho_snr          spearman(mu/sigma, duty) -- does spread add anything?
  twin_injected    twin pre-activations + offsets sampled from the TRAINED
                   resting distribution; spearman(offset, duty). Same test on a
                   model with no trained coupling at all.
  sigma_cv         coefficient of variation of per-neuron sigma. The permuted
                   null is exact when sigma is constant across neurons; sigma_cv
                   says how much room there was for the statistic to fail.

LEG B -- reference identity across depth:
  cos(b_hat_L, b_hat_L') for all layer pairs, trained and twin (twin = floor)
  transfer rho: resting_LL' = w^L . (||b_L^L|| * b_hat^{L'}) + b_n^L, then
                spearman(resting_LL', duty^L). Magnitude is held at layer L's own
                so only the DIRECTION changes -- the static analogue of the
                paper's replacement panel, run on every pair instead of one.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_statistic_nulls.py
       SMOKE=1 fast pass
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
from scipy.stats import spearmanr, norm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
N_DOCS = int(os.environ.get("NDOCS", "6" if SMOKE else "40"))
MAXLEN = 256
SKIP = 16
PAPER_LAYERS = [6, 10, 14, 22]      # the paper's Table 2 layers
N_SAMP = 3000                        # positions subsampled per layer for Leg A
WHICH = ["trained"] if SMOKE else ["trained", "random"]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-statistic-nulls.json")


@torch.no_grad()
def load_model(which):
    if which == "random":
        return AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False).to(DEV).eval()
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()


def sp(a, b):
    r = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return float(r) if np.isfinite(r) else float("nan")


@torch.no_grad()
def collect(m, docs_ids, n_layers, want_preact):
    """One pass: per-layer mean LN'd fc1 input (b_L), and for the requested
    layers a subsample of fc1 PRE-ACTIVATIONS."""
    d = m.config.hidden_size
    sum_ln = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    pre = {L: [] for L in want_preact}
    cnt = 0
    for ids in docs_ids:
        cap = {}
        hs = []
        for L in range(n_layers):
            def mk(L=L):
                def f(mod, inp):
                    cap[L] = inp[0][0].detach().float()
                return f
            hs.append(m.model.layers[L].mlp.fc1.register_forward_pre_hook(mk()))
        m(input_ids=ids)
        for h in hs:
            h.remove()
        for L in range(n_layers):
            x = cap[L][SKIP:]
            sum_ln[L] += x.sum(0)
            if L in want_preact:
                w = m.model.layers[L].mlp.fc1.weight.detach().float()
                b = m.model.layers[L].mlp.fc1.bias.detach().float()
                pre[L].append((x @ w.T + b).half().cpu())
        cnt += cap[0].shape[0] - SKIP
        del cap
    b_L = {L: sum_ln[L] / cnt for L in range(n_layers)}
    pre = {L: torch.cat(v, 0) for L, v in pre.items()}
    return b_L, pre, cnt


def leg_a(m, b_L, pre, rng, trained_resting_pool=None):
    """Tautology null for rho(resting, duty)."""
    out = {}
    for L, P in pre.items():
        w = m.model.layers[L].mlp.fc1.weight.detach().float()
        bn = m.model.layers[L].mlp.fc1.bias.detach().float()
        resting = (w @ b_L[L] + bn).cpu().numpy().astype(np.float64)

        idx = rng.choice(P.shape[0], size=min(N_SAMP, P.shape[0]), replace=False)
        A = P[idx].float().numpy().astype(np.float64)          # (n_pos, n_neur)
        mu = A.mean(0)
        sigma = A.std(0)
        duty = (A > 0).mean(0)

        # --- the identity: resting (weights) must equal mu (activations)
        ident = float(np.max(np.abs(resting - mu)))
        ident_rel = ident / float(np.abs(mu).mean())

        # --- permuted-offset null: same spreads, offsets carry no b_L info
        cen = A - mu
        perm = rng.permutation(len(mu))
        mu_p = mu[perm]
        duty_p = (cen + mu_p > 0).mean(0)

        # --- pure location model
        duty_g = norm.cdf(mu / np.maximum(sigma, 1e-9))

        out[str(L)] = {
            "n_neurons": int(len(mu)), "n_positions": int(A.shape[0]),
            "identity_max_abs_diff": round(ident, 5),
            "identity_rel": round(ident_rel, 6),
            "rho_actual": round(sp(resting, duty), 4),
            "rho_permuted_NULL": round(sp(mu_p, duty_p), 4),
            "rho_gauss_location": round(sp(mu, duty_g), 4),
            "rho_snr": round(sp(mu / np.maximum(sigma, 1e-9), duty), 4),
            "sigma_median": round(float(np.median(sigma)), 4),
            "sigma_cv": round(float(sigma.std() / max(sigma.mean(), 1e-9)), 4),
            "mu_median": round(float(np.median(mu)), 4),
            "duty_median": round(float(np.median(duty)), 4),
        }
        if trained_resting_pool is not None:
            inj = rng.choice(trained_resting_pool, size=len(mu), replace=True)
            duty_i = (cen + inj > 0).mean(0)
            out[str(L)]["rho_twin_injected"] = round(sp(inj, duty_i), 4)
    return out


def leg_b(m, b_L, pre, n_layers):
    """Reference identity across depth: cosine matrix + functional transfer."""
    bh = {L: (v / v.norm().clamp(min=1e-9)) for L, v in b_L.items()}
    probe = list(range(1, n_layers))
    cosmat = {}
    for L in PAPER_LAYERS:
        if L >= n_layers:
            continue
        cosmat[str(L)] = {str(Lp): round(float(bh[L] @ bh[Lp]), 4)
                          for Lp in probe}
    transfer = {}
    for L, P in pre.items():
        w = m.model.layers[L].mlp.fc1.weight.detach().float()
        bn = m.model.layers[L].mlp.fc1.bias.detach().float()
        nrm = b_L[L].norm()
        duty = (P.float().numpy() > 0).mean(0)
        transfer[str(L)] = {}
        for Lp in probe:
            rest = (w @ (nrm * bh[Lp]) + bn).cpu().numpy().astype(np.float64)
            transfer[str(L)][str(Lp)] = round(sp(rest, duty), 4)
    return {"cos": cosmat, "transfer_rho": transfer}


@torch.no_grad()
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print(f"Device: {DEV} SMOKE={SMOKE} docs={N_DOCS} layers={PAPER_LAYERS}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    docs_ids = []
    for t in docs:
        ids = tok(t, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] >= SKIP + 48:
            docs_ids.append(ids)
    print(f"{len(docs_ids)} docs usable")

    rec = {"config": {"smoke": SMOKE, "n_docs": len(docs_ids), "skip": SKIP,
                      "n_samp": N_SAMP, "paper_layers": PAPER_LAYERS,
                      "seed": SEED, "model": MODEL}}
    pool = None
    for which in WHICH:
        print(f"\n=== {which} ===", flush=True)
        m = load_model(which)
        n_layers = len(m.model.layers)
        want = [L for L in PAPER_LAYERS if L < n_layers]
        b_L, pre, cnt = collect(m, docs_ids, n_layers, want)
        print(f"  {cnt} far positions", flush=True)
        a = leg_a(m, b_L, pre, rng, trained_resting_pool=pool)
        if which == "trained":
            w = m.model.layers[10].mlp.fc1.weight.detach().float()
            bn = m.model.layers[10].mlp.fc1.bias.detach().float()
            pool = (w @ b_L[10] + bn).cpu().numpy().astype(np.float64)
        b = leg_b(m, b_L, pre, n_layers)
        rec[which] = {"leg_a": a, "leg_b": b}

        print(f"  {'L':>3} {'identity':>10} {'rho_act':>8} {'rho_PERM':>9} "
              f"{'rho_gauss':>10} {'rho_snr':>8} {'sig_cv':>7} {'inj':>7}")
        for L, e in a.items():
            print(f"  {L:>3} {e['identity_rel']:>10.2e} {e['rho_actual']:>8.3f} "
                  f"{e['rho_permuted_NULL']:>9.3f} {e['rho_gauss_location']:>10.3f} "
                  f"{e['rho_snr']:>8.3f} {e['sigma_cv']:>7.3f} "
                  f"{e.get('rho_twin_injected', float('nan')):>7.3f}")
        print("  cos(b_hat_L, b_hat_L') and transfer rho:")
        for L in a.keys():
            cs = b["cos"][L]
            tr = b["transfer_rho"][L]
            shown = [x for x in ["2", "6", "10", "14", "18", "22", "26", "30"]
                     if x in cs]
            print(f"    L{L}: cos " +
                  " ".join(f"{k}:{cs[k]:+.2f}" for k in shown))
            print(f"         rho " +
                  " ".join(f"{k}:{tr[k]:+.2f}" for k in shown))
        del m, pre
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
