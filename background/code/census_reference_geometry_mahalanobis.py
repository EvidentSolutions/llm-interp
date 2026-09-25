"""Variance-aware recomputation of the background paper's DISTINCTNESS claim.

The published paper (Zenodo 10.5281/zenodo.21498411, Related work) defends b_L
against the adjacent sink / massive-activation / outlier-dimension objects with
Euclidean cosines: cos(b_L, {sink, first-token ME, massive}) <= 0.27, "the
objects are distinct". That defence is what dodges the Shi et al. 2605.08504
"shared global reference vector" collision, so it carries weight.

THREAT (2606.19603, Ying/Hase/Kriegeskorte 2026): "Standard cosine similarity
treats all dimensions equally, yet variance is concentrated along a small number
of directions. Therefore, the thousands of low-variance dimensions can induce
noise, masking genuine alignment between probe directions." A low Euclidean
|cos| is weak evidence of distinctness in exactly this kind of space -- and
Phi-2 mid-stack is the extreme case (a handful of massive coordinates carry a
large share of the variance; cf. P10, where raw mean-diff directions were
hijacked by those same dims at mean|cos| 0.75).

This script recomputes the SAME direction pairs as census_reference_geometry.py
under variance-aware metrics, so any change is attributable to the metric alone.

METRICS (all on unit-normalised u, v):
  euclid  u.v                                        -- the published metric
  mcs     u'Sv / sqrt((u'Su)(v'Sv))                  -- Sigma-weighted; this is
          the threat's own prescription ("reweights the inner product by test
          data covariance"). Reads: do these directions give correlated readings
          on the data? Low-variance dims stop diluting the numerator.
  topr    cos after projecting both onto the top-r principal subspace of Sigma,
          r in {8,32,128,512}. Reads: are they distinct in the subspace that
          actually carries the variance? Sidesteps ill-conditioning entirely.
  whit    Sigma^-1-weighted with shrinkage lambda -- equalises variance instead
          of emphasising it. The opposite-direction check, so the verdict does
          not depend on which way the reweighting goes.

NULLS: chance |cos| = 1/sqrt(d) holds for euclid ONLY. Under every other metric
the chance level differs and is not derivable by hand, so each metric gets an
EMPIRICAL null from N_NULL random unit-direction pairs (median and p95 of
|cos|). A pair counts as distinct only if it sits at or near its own metric's
null. Reporting a reweighted cosine without its own null would be meaningless.

SPACES: the published comparison already mixes spaces (b_L is post-LN; the
sink/first/massive proxies are raw residual). Rather than change the vectors,
Sigma is estimated in BOTH and every metric reported twice:
  Sigma_ln[L]  -- LN'd fc1 input at L (where b_L lives; what gates read)
  Sigma_res[L] -- raw residual at L (where the proxies live)
The verdict must agree under both, or it is reported as space-dependent.

POSITIVE CONTROLS: b vs m_raw (published 0.82) and mhat10 vs mhat18 (published
0.9405) are pairs we expect to stay aligned under every metric. If a metric
destroys those too, the metric is broken, not the claim.

CONSTRUCTION-VALIDITY GATE: the euclid column must reproduce
census-reference-geometry.json -- cos(mhat10,mhat18)=0.9405 and trained
cos_b_sink = 0.1177/0.1476/0.1810/0.2739 at L6/10/14/18.

Usage: .venv/Scripts/python.exe \
         superposition/code/census_reference_geometry_mahalanobis.py
       SMOKE=1  fast pass (6 docs, trained only)
       NDOCS=n  override doc count (covariance-stability check)
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

MID_LAYERS = [6, 10, 14, 18]
N_DOCS = int(os.environ.get("NDOCS", "6" if SMOKE else "40"))
MAXLEN = 256
SKIP = 16
TOPK_OD = 20
TOPK_MASSIVE = 5
WHICH = ["trained"] if SMOKE else ["random", "trained"]

TOPR = [8, 32, 128, 512]
SHRINK = [0.01, 0.1, 0.5]
N_NULL = 300

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-geometry-mahalanobis.json")
PUBLISHED = os.path.join(DATA, "census-reference-geometry.json")


@torch.no_grad()
def load_model(which):
    if which == "random":
        m = AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    return m.to(DEV).eval()


class Capture:
    """Identical to census_reference_geometry.py: per-layer raw residual at the
    layer input and the fc1 input (the object gates read)."""

    def __init__(self, model, n_layers):
        self.model = model
        self.n = n_layers
        self.resid = {}
        self.lnin = {}
        self.handles = []

    def __enter__(self):
        for L in range(self.n):
            lyr = self.model.model.layers[L]

            def mk_res(L=L):
                def f(mod, args, kwargs):
                    hs = args[0] if args else kwargs["hidden_states"]
                    self.resid[L] = hs[0].detach().float()
                return f

            def mk_ln(L=L):
                def f(mod, inp):
                    self.lnin[L] = inp[0][0].detach().float()
                return f
            self.handles.append(lyr.register_forward_pre_hook(
                mk_res(), with_kwargs=True))
            self.handles.append(lyr.mlp.fc1.register_forward_pre_hook(mk_ln()))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- metrics ---

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cos_euclid(u, v):
    return float(np.dot(_unit(u), _unit(v)))


def cos_form(u, v, A):
    """cos under the bilinear form A (must be SPD): u'Av / sqrt(u'Au v'Av)."""
    au, av = A @ u, A @ v
    du, dv = float(u @ au), float(v @ av)
    if du <= 0 or dv <= 0:
        return float("nan")
    return float((u @ av) / np.sqrt(du * dv))


def cos_subspace(u, v, P):
    """cos after projecting onto the columns of P (d x r, orthonormal)."""
    pu, pv = P.T @ u, P.T @ v
    nu, nv = np.linalg.norm(pu), np.linalg.norm(pv)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float((pu @ pv) / (nu * nv))


def shrunk_inverse(S, lam):
    d = S.shape[0]
    tgt = np.trace(S) / d
    Sl = (1.0 - lam) * S + lam * tgt * np.eye(d)
    return np.linalg.inv(Sl)


def empirical_null(fn, d, rng, n=N_NULL):
    """|cos| between random unit direction pairs under an arbitrary metric.
    Chance is metric-dependent and not derivable by hand, so it is measured."""
    vals = []
    for _ in range(n):
        u = _unit(rng.standard_normal(d))
        v = _unit(rng.standard_normal(d))
        c = fn(u, v)
        if np.isfinite(c):
            vals.append(abs(c))
    if not vals:
        return {"median": None, "p95": None}
    return {"median": round(float(np.median(vals)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4)}


# ------------------------------------------------------------------- main ---

@torch.no_grad()
def run_model(which, docs_ids):
    t0 = time.time()
    print(f"\n=== {which} ===", flush=True)
    m = load_model(which)
    d = m.config.hidden_size
    n_layers = len(m.model.layers)

    sum_far = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_sink = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_first = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_lnfar = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    maxabs = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    # full second moments, MID_LAYERS only, both spaces
    out_res = {L: torch.zeros(d, d, device=DEV) for L in MID_LAYERS}
    out_ln = {L: torch.zeros(d, d, device=DEV) for L in MID_LAYERS}
    cnt_far = cnt_sink = cnt_first = 0

    for i, ids in enumerate(docs_ids):
        with Capture(m, n_layers) as cap:
            m(input_ids=ids)
        T = cap.resid[0].shape[0]
        for L in range(n_layers):
            r = cap.resid[L]
            far = r[SKIP:]
            sum_far[L] += far.sum(0)
            sum_sink[L] += r[:SKIP].sum(0)
            sum_first[L] += r[0]
            maxabs[L] = torch.maximum(maxabs[L], r.abs().max(0).values)
            ln = cap.lnin[L][SKIP:]
            sum_lnfar[L] += ln.sum(0)
            if L in MID_LAYERS:
                out_res[L] += far.T @ far
                out_ln[L] += ln.T @ ln
        cnt_far += T - SKIP
        cnt_sink += SKIP
        cnt_first += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(docs_ids)} docs", flush=True)

    m_far = {L: sum_far[L] / cnt_far for L in range(n_layers)}
    b_far = {L: sum_lnfar[L] / cnt_far for L in range(n_layers)}
    sink = {L: sum_sink[L] / cnt_sink for L in range(n_layers)}
    first = {L: sum_first[L] / cnt_first for L in range(n_layers)}

    me_layer = int(max(range(n_layers), key=lambda L: float(maxabs[L].max())))
    me_val = float(maxabs[me_layer].max())
    me_coords = torch.topk(maxabs[me_layer], TOPK_MASSIVE).indices
    massive_vec = torch.zeros(d, device=DEV)
    massive_vec[me_coords] = first[me_layer][me_coords]
    first_me = first[me_layer]

    npy = lambda t: t.double().cpu().numpy()
    rec = {
        "me_layer": me_layer,
        "me_max_abs": round(me_val, 2),
        "n_far_positions": cnt_far,
        "d": d,
        "cos_mhat10_mhat18_euclid": round(
            cos_euclid(npy(m_far[10]), npy(m_far[18])), 4)
        if 18 < n_layers else None,
        "per_layer": {},
    }

    rng = np.random.default_rng(SEED)

    for L in MID_LAYERS:
        if L >= n_layers:
            continue
        b = npy(b_far[L])
        pairs = {
            "b_vs_mraw_POSCTRL": npy(m_far[L]),
            "b_vs_sink": npy(sink[L]),
            "b_vs_first_me": npy(first_me),
            "b_vs_massive_me": npy(massive_vec),
        }
        bu = _unit(b)
        entry = {"euclid": {k: round(cos_euclid(b, v), 4)
                            for k, v in pairs.items()}}

        for space, outer, mean_t in (("ln", out_ln[L], b_far[L]),
                                     ("res", out_res[L], m_far[L])):
            mu = npy(mean_t)
            S = npy(outer) / cnt_far - np.outer(mu, mu)     # centered
            S = 0.5 * (S + S.T)
            evals, evecs = np.linalg.eigh(S)
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
            tot = float(evals.clip(min=0).sum())

            sp = {
                "var_frac_top8": round(float(evals[:8].clip(min=0).sum() / tot), 4),
                "var_frac_top32": round(float(evals[:32].clip(min=0).sum() / tot), 4),
                "cond_top_over_med": round(float(evals[0] / max(evals[d // 2], 1e-12)), 1),
                # variance carried along each object's own direction
                "var_along": {k: round(float(_unit(v) @ S @ _unit(v)), 3)
                              for k, v in [("b", b)] + list(pairs.items())},
                "mcs": {}, "mcs_null": {}, "topr": {}, "topr_null": {},
                "whit": {}, "whit_null": {},
            }
            _rd = [_unit(rng.standard_normal(d)) for _ in range(20)]
            sp["var_along"]["random_dir"] = round(
                float(np.mean([u @ S @ u for u in _rd])), 3)

            # -- MCS (Sigma-weighted): the threat's own prescription
            for k, v in pairs.items():
                sp["mcs"][k] = round(cos_form(bu, _unit(v), S), 4)
            sp["mcs_null"] = empirical_null(
                lambda u, v: cos_form(u, v, S), d, rng)

            # -- top-r principal subspace
            for r in TOPR:
                P = evecs[:, :r]
                sp["topr"][f"r{r}"] = {k: round(cos_subspace(bu, _unit(v), P), 4)
                                       for k, v in pairs.items()}
                sp["topr_null"][f"r{r}"] = empirical_null(
                    lambda u, v, P=P: cos_subspace(u, v, P), d, rng, n=120)

            # -- whitened (Sigma^-1), shrinkage swept
            for lam in SHRINK:
                Si = shrunk_inverse(S, lam)
                sp["whit"][f"lam{lam}"] = {k: round(cos_form(bu, _unit(v), Si), 4)
                                           for k, v in pairs.items()}
                sp["whit_null"][f"lam{lam}"] = empirical_null(
                    lambda u, v, Si=Si: cos_form(u, v, Si), d, rng, n=120)

            entry[f"sigma_{space}"] = sp

        rec["per_layer"][str(L)] = entry
        e = entry["euclid"]
        print(f"  L{L} euclid: sink={e['b_vs_sink']} first_me={e['b_vs_first_me']} "
              f"massive={e['b_vs_massive_me']} mraw={e['b_vs_mraw_POSCTRL']}",
              flush=True)
        for space in ("ln", "res"):
            sp = entry[f"sigma_{space}"]
            print(f"     sigma_{space}: MCS sink={sp['mcs']['b_vs_sink']} "
                  f"first={sp['mcs']['b_vs_first_me']} "
                  f"massive={sp['mcs']['b_vs_massive_me']} "
                  f"mraw={sp['mcs']['b_vs_mraw_POSCTRL']} "
                  f"| null med={sp['mcs_null']['median']} p95={sp['mcs_null']['p95']} "
                  f"| var_top8={sp['var_frac_top8']}", flush=True)

    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s) ME layer={me_layer}")
    return rec


@torch.no_grad()
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  mid={MID_LAYERS}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    docs_ids = []
    for t in docs:
        ids = tok(t, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] >= SKIP + 48:
            docs_ids.append(ids)
    print(f"{len(docs_ids)} docs usable")

    rec = {"config": {"smoke": SMOKE, "n_docs": len(docs_ids),
                      "mid_layers": MID_LAYERS, "skip": SKIP,
                      "topr": TOPR, "shrink": SHRINK, "n_null": N_NULL,
                      "seed": SEED, "model": MODEL}}
    for which in WHICH:
        rec[which] = run_model(which, docs_ids)

    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")

    # ---- construction-validity gate against the published JSON ----
    print("\n===== GATE: euclid must reproduce the published numbers =====")
    pub = json.load(open(PUBLISHED, encoding="utf-8"))
    ok = True
    for which in WHICH:
        p, n = pub.get(which, {}), rec.get(which, {})
        g_pub = p.get("cos_mhat10_mhat18")
        g_new = n.get("cos_mhat10_mhat18_euclid")
        if g_pub is not None and g_new is not None:
            hit = abs(g_pub - g_new) < 0.01
            ok &= hit
            print(f"  [{which}] cos(mhat10,mhat18): published {g_pub} "
                  f"vs new {g_new}  {'OK' if hit else 'MISMATCH'}")
        for L, pe in p.get("per_layer", {}).items():
            ne = n.get("per_layer", {}).get(L)
            if not ne:
                continue
            for pk, nk in (("cos_b_sink", "b_vs_sink"),
                           ("cos_b_first_me", "b_vs_first_me"),
                           ("cos_b_mraw", "b_vs_mraw_POSCTRL")):
                a, b = pe.get(pk), ne["euclid"].get(nk)
                hit = a is not None and b is not None and abs(a - b) < 0.01
                ok &= hit
                print(f"  [{which}] L{L} {pk}: published {a} vs new {b}  "
                      f"{'OK' if hit else 'MISMATCH'}")
    print(f"\nGATE {'PASSED' if ok else 'FAILED'} -- "
          f"{'metric deltas are attributable to the metric' if ok else 'DO NOT READ the new metrics'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
