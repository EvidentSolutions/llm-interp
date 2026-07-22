"""Reference geometry: is b_L the same object as the four adjacent "constant
direction / sink / outlier" priors, or a distinct one? (Background paper
positioning, decision #2 of the 2026-07-21 lit-check. Pre-registered branches
below.)

The background paper's b_L is the corpus mean of the layer-normalised MLP
input (the object gates read), measured at positions >= SKIP (away from the
attention-sink region). Four recent objects share vocabulary with it and must
be shown geometrically distinct (or, honestly, unified):

  (a) Macocco et al. 2503.21718 -- last-layer OUTLIER DIMENSIONS: a small set
      of coordinate axes with anomalous magnitude, favouring frequent tokens.
      Proxy: top-k coords by per-coordinate second moment E[x^2] at the final-
      LN input (activation-based), cross-checked weights-only against the
      final_layernorm gain gamma's own top-k coords (Macocco localise ODs to
      the final LN + last down-projection).
  (b) Shi et al. 2605.08504 -- ME-LAYER first-token MASSIVE ACTIVATION: at one
      intermediate layer a first-token activation emerges, proposed as "a
      stable and shared global reference vector" for attention. Proxy: the
      first-token (pos 0) residual direction at the layer whose max|coord| over
      positions peaks (the ME layer located here, not assumed).
  (c) Sun et al. 2603.05498 (The Spike, the Sparse and the Sink) / massive-
      activation "implicit parameters": a few spike coordinates acting as a
      carried constant. Proxy: the sparse massive-coordinate direction (top
      coords by max|value| over positions) at the ME layer.
  (d) Qiu et al. 2601.22966 -- RESIDUAL SINK: the persistent residual component
      concentrated at the sink region. Proxy: the mean raw residual over the
      sink positions (pos < SKIP, incl. pos 0) at each layer.

Statistic (per mid-stack layer L in {6,10,14,18}): cosine of b_L to each proxy
direction; fraction of ||b_L||^2 carried on the OD coordinate set; Jaccard of
b_L's own top-k coords with the OD set. All directions unit-normalised. The
raw-mean m_L (un-normalised) is included so the LN-normalisation effect on the
object is visible (cos(b_L, m_L)). A random-init twin is run as the floor
(massive/OD structure is trained; the twin should show none).

Pre-registered branches:
  DISTINCT   -- cos(b_L, {sink, massive, first-tok}) small (|cos| < ~0.3) and
                b_L's OD-coordinate mass near the isotropic share (k/d): b_L is
                a distributed mid-stack direction, NOT the sink / OD / massive
                object. Confirms the paper's "should not be conflated" and
                converts the four open geometric caveats into measured ones.
  OVERLAP    -- any cos > ~0.5 or OD mass >> k/d: b_L partly coincides with that
                object; report the coincidence and soften the corresponding
                related-work paragraph (esp. Shi ME-layer for (b)).
  UNIFIES    -- cos ~ 1 to one proxy: b_L IS that object under another name;
                merge the claim and re-cite.

Construction-validity gate: cos(m_hat_10, m_hat_18) must reproduce ~0.94 (the
paper's single-shared-direction result); the ME layer must be located (a coord
whose |value| >> the bulk) before its direction is used.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_geometry.py
       SMOKE=1 for a fast pass (fewer docs, trained only).
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

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

MID_LAYERS = [6, 10, 14, 18]      # where b_L is compared
N_DOCS = 6 if SMOKE else 40
MAXLEN = 256
SKIP = 16                          # sink/start exclusion, project convention
TOPK_OD = 20                       # outlier-dimension set size
TOPK_MASSIVE = 5                   # massive/spike coordinate set size
WHICH = ["trained"] if SMOKE else ["random", "trained"]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-geometry.json")


@torch.no_grad()
def load_model(which):
    if which == "random":
        m = AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    return m.to(DEV).eval()


def cos(a, b):
    return float(F.cosine_similarity(a.float(), b.float(), dim=0))


class Capture:
    """One forward pass: per-layer raw residual (layer input) split into
    far (pos>=SKIP) / sink (pos<SKIP) / first-token, LN MLP-input mean (far),
    per-coordinate second moment (far), and per-layer max|coord| locator."""

    def __init__(self, model, n_layers):
        self.model = model
        self.n = n_layers
        self.resid = {}     # L -> (T,d) raw residual at layer input
        self.lnin = {}      # L -> (T,d) fc1 input (the object gates read)
        self.final_ln = None
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

        def mk_final(mod, inp):
            self.final_ln = inp[0][0].detach().float()
        self.handles.append(
            self.model.model.final_layernorm.register_forward_pre_hook(
                mk_final))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_model(which, docs_ids, tok):
    t0 = time.time()
    print(f"--- {which} ---")
    m = load_model(which)
    d = m.config.hidden_size
    n_layers = len(m.model.layers)

    sum_far = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_sq = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_sink = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_first = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_lnfar = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    maxabs = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_finalsq = torch.zeros(d, device=DEV)
    cnt_far = cnt_sink = cnt_first = cnt_ln = 0
    cnt_final = 0

    for ids in docs_ids:
        with Capture(m, n_layers) as cap:
            m(input_ids=ids)
        T = cap.resid[0].shape[0]
        for L in range(n_layers):
            r = cap.resid[L]
            far = r[SKIP:]
            sum_far[L] += far.sum(0)
            sum_sq[L] += (far * far).sum(0)
            sum_sink[L] += r[:SKIP].sum(0)
            sum_first[L] += r[0]
            maxabs[L] = torch.maximum(maxabs[L], r.abs().max(0).values)
            ln = cap.lnin[L][SKIP:]
            sum_lnfar[L] += ln.sum(0)
        fl = cap.final_ln[SKIP:]
        sum_finalsq += (fl * fl).sum(0)
        cnt_far += T - SKIP
        cnt_sink += SKIP
        cnt_first += 1
        cnt_ln += T - SKIP
        cnt_final += T - SKIP

    # per-layer objects
    m_far = {L: sum_far[L] / cnt_far for L in range(n_layers)}
    b_far = {L: sum_lnfar[L] / cnt_ln for L in range(n_layers)}
    sink = {L: sum_sink[L] / cnt_sink for L in range(n_layers)}
    first = {L: sum_first[L] / cnt_first for L in range(n_layers)}
    var_far = {L: sum_sq[L] / cnt_far for L in range(n_layers)}
    var_final = sum_finalsq / cnt_final

    # ---- locate the ME layer (max single-coordinate |value| over positions) ----
    me_layer = int(max(range(n_layers),
                       key=lambda L: float(maxabs[L].max())))
    me_val = float(maxabs[me_layer].max())
    me_coords = torch.topk(maxabs[me_layer], TOPK_MASSIVE).indices
    # massive/spike direction (Sun): sparse vector on massive coords carrying
    # the first-token values there; first-token direction at ME (Shi)
    massive_vec = torch.zeros(d, device=DEV)
    massive_vec[me_coords] = first[me_layer][me_coords]
    first_me = first[me_layer]

    # ---- outlier dimensions (Macocco): top-k coords by final-LN E[x^2] ----
    od_coords = torch.topk(var_final, TOPK_OD).indices
    od_set = set(od_coords.tolist())
    gamma = m.model.final_layernorm.weight.detach().float().abs()
    gamma_coords = torch.topk(gamma, TOPK_OD).indices
    od_gamma_jaccard = len(od_set & set(gamma_coords.tolist())) / \
        len(od_set | set(gamma_coords.tolist()))
    iso_share = TOPK_OD / d               # isotropic expectation for OD mass

    rec = {
        "me_layer": me_layer,
        "me_max_abs": round(me_val, 2),
        "me_coords": me_coords.tolist(),
        "od_coords_final": od_coords.tolist(),
        "od_gamma_topk_jaccard": round(od_gamma_jaccard, 3),
        "od_isotropic_mass_share": round(iso_share, 5),
        "cos_mhat10_mhat18": round(cos(m_far[10], m_far[18]), 4)
        if 18 < n_layers else None,
        "per_layer": {},
    }

    for L in MID_LAYERS:
        if L >= n_layers:
            continue
        b = b_far[L]
        bu = b / b.norm().clamp(min=1e-9)
        b_sq = (b * b)
        od_mass = float(b_sq[od_coords].sum() / b_sq.sum().clamp(min=1e-9))
        b_top = set(torch.topk(b.abs(), TOPK_OD).indices.tolist())
        od_jac = len(b_top & od_set) / len(b_top | od_set)
        rec["per_layer"][str(L)] = {
            "cos_b_mraw": round(cos(b, m_far[L]), 4),
            "cos_b_sink": round(cos(b, sink[L]), 4),
            "cos_b_first_me": round(cos(b, first_me), 4),
            "cos_b_massive_me": round(cos(bu, massive_vec), 4),
            "b_mass_on_od": round(od_mass, 4),
            "b_od_coord_jaccard": round(od_jac, 3),
            "b_norm": round(float(b.norm()), 3),
        }
        print(f"  L{L}: cos(b,mraw)={rec['per_layer'][str(L)]['cos_b_mraw']} "
              f"cos(b,sink)={rec['per_layer'][str(L)]['cos_b_sink']} "
              f"cos(b,first_ME)={rec['per_layer'][str(L)]['cos_b_first_me']} "
              f"OD_mass={rec['per_layer'][str(L)]['b_mass_on_od']} "
              f"(iso {iso_share:.4f})")

    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s)  ME layer={me_layer} "
          f"max|coord|={me_val:.1f}")
    return rec


@torch.no_grad()
def main():
    t0 = time.time()
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
                      "topk_od": TOPK_OD, "topk_massive": TOPK_MASSIVE,
                      "seed": SEED}}
    for which in WHICH:
        rec[which] = run_model(which, docs_ids, tok)

    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")

    tr = rec.get("trained", {})
    print(f"\nGATE cos(mhat10,mhat18)={tr.get('cos_mhat10_mhat18')} "
          f"(expect ~0.94)")
    print(f"ME layer={tr.get('me_layer')} od_gamma_jaccard="
          f"{tr.get('od_gamma_topk_jaccard')}")
    for L, e in tr.get("per_layer", {}).items():
        print(f"  L{L}: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
