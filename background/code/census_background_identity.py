"""Background identity + co-removal (plan_mid_stack_empirical_basis.md 3v,
pre-registered 2026-07-16). Follow-ups to 3t:

Leg A (identity): REPLACE the background projection at L10 with candidate
directions -- x' = x - (x.mhat)mhat + (x.mhat)chat -- rand / coordinate-
shuffled mhat / another layer's background / a function-word W_U sum.
Direction-specific reference predicts release ~ pure removal for low-cos
candidates, preservation scaling with cos(chat, mhat).

Leg B (co-removal): alpha=0 at ALL census layers >= 6 (and >= 14)
simultaneously -- defeats the ~12-16-layer restoration 3t measured, exposing
the full downstream dependence.

Gates: alpha=1 exact no-op; single-site alpha=0 at L10 reproduces 3t
(dfrac05 ~ +0.086, KL ~ 0.51); clean duty cycle reproduces the audit.
Twin runs the full panel (its own mhat) as the floor.

Usage: .venv/Scripts/python.exe superposition/code/census_background_identity.py
       SMOKE=1 fast pass.
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

CENSUS_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
L_A = 10                     # Leg A site
L_B18 = 18                   # other-layer background candidate
N_DOCS = 6 if SMOKE else 24
MAXLEN = 256
SKIP = 16

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-background-identity.json")

STRUCTURAL = [" the", " a", " an", " and", " or", " not", " no", " of",
              " to", " in", " on", " at", " is", " was", " are", " were",
              " it", " he", " she", " they", " who", " that", " this",
              " but", " so", " if", " as", " than", " too", " more",
              " most", " some", " all", " any", " own", " same", " up",
              " down", " out", " into", " away", " over", " under", ".",
              ",", ";", ":", "?", "!", "'s", "'t", "'d", "'ll", "'re",
              "nt", "s", "ing", "ed", "er", "est", " however", " because",
              " although", "\n"]


def boot_ci(vals, nboot=2000, seed=0):
    """95% percentile document-bootstrap CI of the median (PLAN item 3).
    Additive -- leaves the point-estimate median alongside it unchanged."""
    v = np.asarray(vals, dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.RandomState(seed)
    n = v.size
    bs = np.median(v[rng.randint(0, n, size=(nboot, n))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


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
    def __init__(self, model, layers):
        self.model, self.layers = model, layers
        self.acts, self.resid = {}, {}
        self.handles = []

    def __enter__(self):
        for L in self.layers:
            lyr = self.model.model.layers[L]

            def mk_res(L=L):
                def f(mod, args, kwargs):
                    hs = args[0] if args else kwargs["hidden_states"]
                    self.resid[L] = hs[0, SKIP:].detach().float()
                return f

            def mk_fc2in(L=L):
                def f(mod, inp):
                    self.acts[L] = inp[0][0, SKIP:].detach().float()
                return f
            self.handles.append(lyr.register_forward_pre_hook(
                mk_res(), with_kwargs=True))
            self.handles.append(lyr.mlp.fc2.register_forward_pre_hook(
                mk_fc2in()))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


class MultiManip:
    """Per-layer residual manipulations. spec = {L: ('scale', alpha) |
    ('repl', chat)} -- scale: x' = x + (a-1)(x.mhat)mhat;
    repl: x' = x - (x.mhat)mhat + (x.mhat)chat. Positions >= SKIP."""

    def __init__(self, model, spec, mhats):
        self.model, self.spec, self.mhats = model, spec, mhats
        self.handles = []

    def __enter__(self):
        for L, action in self.spec.items():
            mh = self.mhats[L]

            def mk(L=L, action=action, mh=mh):
                def f(mod, args, kwargs):
                    hs = args[0] if args else kwargs["hidden_states"]
                    x = hs[0, SKIP:].float()
                    proj = x @ mh
                    if action[0] == "scale":
                        x = x + (action[1] - 1.0) * proj.unsqueeze(1) \
                            * mh.unsqueeze(0)
                    else:
                        x = x - proj.unsqueeze(1) * mh.unsqueeze(0) \
                            + proj.unsqueeze(1) * action[1].unsqueeze(0)
                    hs = hs.clone()
                    hs[0, SKIP:] = x.to(hs.dtype)
                    if args:
                        return (hs,) + args[1:], kwargs
                    kwargs["hidden_states"] = hs
                    return args, kwargs
                return f
            self.handles.append(
                self.model.model.layers[L].register_forward_pre_hook(
                    mk(), with_kwargs=True))
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
    rec = {}

    # ---- Phase 1: means + clean stats ----
    sums = {L: torch.zeros(d, device=DEV) for L in CENSUS_LAYERS}
    duty = {L: None for L in CENSUS_LAYERS}
    npos = 0
    for ids in docs_ids:
        with Capture(m, CENSUS_LAYERS) as cap:
            m(input_ids=ids)
        T = cap.acts[CENSUS_LAYERS[0]].shape[0]
        npos += T
        for L in CENSUS_LAYERS:
            sums[L] += cap.resid[L].sum(0)
            pc = (cap.acts[L] > 0).float().sum(0)
            duty[L] = pc if duty[L] is None else duty[L] + pc
    m_raw = {L: sums[L] / npos for L in CENSUS_LAYERS}
    mhat = {L: m_raw[L] / m_raw[L].norm().clamp(min=1e-9)
            for L in CENSUS_LAYERS}
    rec["duty_med"] = {str(L): round(float((duty[L] / npos).median()), 4)
                       for L in CENSUS_LAYERS}

    # ---- candidates ----
    rng = np.random.RandomState(SEED)
    r = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
    cand = {"rand": r / r.norm()}
    perm = torch.as_tensor(rng.permutation(d), device=DEV)
    bs = mhat[L_A][perm].clone()
    cand["b_shuf"] = bs / bs.norm().clamp(min=1e-9)
    cand["b_L18"] = mhat[L_B18].clone()
    WU = m.lm_head.weight.detach().float()
    fw_ids = []
    for s in STRUCTURAL:
        e = tok.encode(s)
        if len(e) == 1:
            fw_ids.append(e[0])
    WUn = WU[sorted(set(fw_ids))].to(DEV)
    WUn = WUn / WUn.norm(dim=1, keepdim=True).clamp(min=1e-9)
    fw = WUn.sum(0)
    cand["fw_sum"] = fw / fw.norm().clamp(min=1e-9)
    rec["cand_cos_mhat"] = {k: round(float(v @ mhat[L_A]), 4)
                            for k, v in cand.items()}
    print(f"  cand cos(m̂_{L_A}):", rec["cand_cos_mhat"])
    del WU, WUn

    # ---- variants ----
    variants = {"a1.0": {L_A: ("scale", 1.0)},
                "a0.0": {L_A: ("scale", 0.0)}}
    for k, v in cand.items():
        variants[f"repl_{k}"] = {L_A: ("repl", v)}
    variants["corem_ge6"] = {L: ("scale", 0.0)
                             for L in CENSUS_LAYERS if L >= 6}
    variants["corem_ge14"] = {L: ("scale", 0.0)
                              for L in CENSUS_LAYERS if L >= 14}

    agg = {v: {"kl": [], "dnll": [], "dfrac05": {}, "dfrac10": {},
               "dcosm": {}} for v in variants}
    clean_frac_acc = {L: [] for L in CENSUS_LAYERS}
    for di, ids in enumerate(docs_ids):
        with Capture(m, CENSUS_LAYERS) as cap:
            out = m(input_ids=ids)
        clean_logits = out.logits[0].float()
        tgt = ids[0, SKIP + 1:]
        clean_nll = float(F.cross_entropy(
            clean_logits[SKIP:-1], tgt, reduction="mean"))
        cf = {}
        ccos = {}
        for L in CENSUS_LAYERS:
            a = cap.acts[L]
            cf[L] = (float((a > 0.5).float().mean()),
                     float((a > 1.0).float().mean()))
            clean_frac_acc[L].append(cf[L])
            x = cap.resid[L]
            ccos[L] = float(F.cosine_similarity(
                x, mhat[L].unsqueeze(0).expand_as(x), dim=1).mean())
        for vname, spec in variants.items():
            with MultiManip(m, spec, mhat), \
                    Capture(m, CENSUS_LAYERS) as cap2:
                out2 = m(input_ids=ids)
            lg = out2.logits[0].float()
            kl = float(F.kl_div(
                F.log_softmax(lg[-1], dim=-1),
                F.log_softmax(clean_logits[-1], dim=-1),
                log_target=True, reduction="sum"))
            dnll = float(F.cross_entropy(
                lg[SKIP:-1], tgt, reduction="mean")) - clean_nll
            agg[vname]["kl"].append(kl)
            agg[vname]["dnll"].append(dnll)
            for L in CENSUS_LAYERS:
                a2 = cap2.acts[L]
                agg[vname]["dfrac05"].setdefault(L, []).append(
                    float((a2 > 0.5).float().mean()) - cf[L][0])
                agg[vname]["dfrac10"].setdefault(L, []).append(
                    float((a2 > 1.0).float().mean()) - cf[L][1])
                x2 = cap2.resid[L]
                c2 = float(F.cosine_similarity(
                    x2, mhat[L].unsqueeze(0).expand_as(x2), dim=1).mean())
                agg[vname]["dcosm"].setdefault(L, []).append(c2 - ccos[L])
        if di == 0:
            print(f"  doc0 done ({time.time()-t0:.0f}s)")

    rec["clean_frac05_med"] = {
        str(L): round(float(np.median([t[0] for t in clean_frac_acc[L]])), 4)
        for L in CENSUS_LAYERS}
    rec["clean_frac10_med"] = {
        str(L): round(float(np.median([t[1] for t in clean_frac_acc[L]])), 4)
        for L in CENSUS_LAYERS}
    mm = {}
    for vname, r_ in agg.items():
        mm[vname] = {
            "kl": round(float(np.median(r_["kl"])), 5),
            "dnll": round(float(np.median(r_["dnll"])), 5),
            "kl_ci": boot_ci(r_["kl"]),
            "dnll_ci": boot_ci(r_["dnll"]),
            "dfrac05": {str(L): round(float(np.median(v)), 5)
                        for L, v in r_["dfrac05"].items()},
            "dfrac10": {str(L): round(float(np.median(v)), 5)
                        for L, v in r_["dfrac10"].items()},
            "dcosm": {str(L): round(float(np.median(v)), 4)
                      for L, v in r_["dcosm"].items()}}
    rec["variants"] = mm
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s)")
    return rec


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  LA=L{L_A}")
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
                      "leg_a_layer": L_A, "seed": SEED}}
    rec["twin"] = run_model("random", docs_ids, tok)
    rec["trained"] = run_model("trained", docs_ids, tok)
    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")

    tr = rec["trained"]
    print("\ncand cos:", tr["cand_cos_mhat"])
    print("clean frac05:", tr["clean_frac05_med"])
    for v in ["a1.0", "a0.0", "repl_rand", "repl_b_shuf", "repl_b_L18",
              "repl_fw_sum", "corem_ge6", "corem_ge14"]:
        e = tr["variants"][v]
        print(f"  {v}: kl={e['kl']} dnll={e['dnll']} "
              f"dfrac05@10={e['dfrac05']['10']} "
              f"dfrac05@18={e['dfrac05']['18']} "
              f"dfrac05@26={e['dfrac05']['26']}")
    tw = rec["twin"]
    for v in ["a0.0", "corem_ge6"]:
        e = tw["variants"][v]
        print(f"  twin {v}: kl={e['kl']} dfrac05@10={e['dfrac05']['10']} "
              f"dfrac05@18={e['dfrac05']['18']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
