"""Background reference test: is the high-frequency-token background b_L the
gate population's reference frame? (plan_mid_stack_empirical_basis.md 3t,
Olli's hypothesis, pre-registered 2026-07-16.)

Leg A (anatomy): decode b_L through W_U per layer (premise: high-frequency
function words); exact source decomposition of the raw-residual mean in
Phi-2's parallel block (m_{L+1} = m_emb + sum attn_l + sum mlp_l); split-half
constancy.

Leg B (causal): scale the residual's projection onto m_hat_L at layer L,
alpha in {0,0.5,1,1.5,2}, plus a norm-matched random-direction control
(subtract |x.m_hat| r_hat). Measure population + quadrant firing rates at
downstream census layers, final-position KL, dNLL, and background
restoration (homeostasis). Prediction (gate-reference): alpha=0 raises
firing broadly, alpha=2 lowers it, dose-response, beyond the norm-matched
control; twin flat.

Gates: clean duty cycle must reproduce the frac_pos audit (trained ~0.076,
twin ~0.48); alpha=1 must be an exact no-op; cos(S_jumble,b) < 0 <
cos(S_readable,b) must reproduce 3f/3q.

Usage: .venv/Scripts/python.exe superposition/code/census_background_reference.py
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
MANIP_LAYERS = [6] if SMOKE else [6, 10, 14]
ALPHAS = [0.0, 1.0, 2.0] if SMOKE else [0.0, 0.5, 1.0, 1.5, 2.0]
N_DOCS = 6 if SMOKE else 40
MAXLEN = 256
SKIP = 16                      # sink/start exclusion, project convention
GATE3_POS = 24                 # sampled positions/doc for cos(S_g, b)

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-background-reference.json")


def boot_ci(vals, nboot=2000, seed=0):
    """95% percentile document-bootstrap CI of the median (PLAN item 3).
    Resamples the per-document values with replacement; additive -- does not
    touch the point-estimate median reported alongside it."""
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


@torch.no_grad()
def layer_write_metrics(model, WUn, L):
    """mc_wu_fc2 + conc16 (verbatim quadrant machinery, 3n-3s)."""
    f2 = model.model.layers[L].mlp.fc2.weight.detach()
    f2u = (f2.float() / f2.float().norm(dim=0, keepdim=True).clamp(min=1e-9)
           ).to(torch.float16)
    f2u_dev = f2u.to(DEV)
    mc = torch.empty(f2.shape[1])
    for s in range(0, f2.shape[1], 2048):
        mc[s:s + 2048] = (WUn @ f2u_dev[:, s:s + 2048]
                          ).abs().max(dim=0).values.float().cpu()
    later = [Lp for Lp in CENSUS_LAYERS if Lp > L]
    top16 = None
    for Lp in later:
        f1 = model.model.layers[Lp].mlp.fc1.weight.detach()
        f1u = (f1.float() / f1.float().norm(dim=1, keepdim=True
                                            ).clamp(min=1e-9)).to(DEV,
                                                                  torch.float16)
        cs = (f1u @ f2u_dev).abs()
        cand = torch.topk(cs, 16, dim=0).values
        top16 = cand if top16 is None else torch.topk(
            torch.cat([top16, cand], dim=0), 16, dim=0).values
        del f1u, cs, cand
    conc16 = top16.mean(dim=0).float().cpu()
    del f2u_dev, top16
    torch.cuda.empty_cache()
    return mc, conc16


class Capture:
    """Accumulating hooks for one forward pass."""

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.acts = {}          # L -> post-GELU acts (T, dff), pos>=SKIP
        self.resid = {}         # L -> raw resid (T, d), pos>=SKIP
        self.lnin = {}          # L -> fc1 input (T, d), pos>=SKIP
        self.handles = []

    def __enter__(self):
        act_fn = self.model.model.layers[0].mlp.activation_fn
        for L in self.layers:
            lyr = self.model.model.layers[L]

            def mk_res(L=L):
                def f(mod, args, kwargs):
                    hs = args[0] if args else kwargs["hidden_states"]
                    self.resid[L] = hs[0, SKIP:].detach().float()
                return f

            def mk_fc1in(L=L):
                def f(mod, inp):
                    self.lnin[L] = inp[0][0, SKIP:].detach().float()
                return f

            def mk_fc2in(L=L):
                def f(mod, inp):
                    self.acts[L] = inp[0][0, SKIP:].detach().float()
                return f
            self.handles.append(lyr.register_forward_pre_hook(
                mk_res(), with_kwargs=True))
            self.handles.append(lyr.mlp.fc1.register_forward_pre_hook(
                mk_fc1in()))
            self.handles.append(lyr.mlp.fc2.register_forward_pre_hook(
                mk_fc2in()))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


class Manip:
    """Residual manipulation at layer L: x' = x + (alpha-1)(x.mhat)mhat, or
    norm-matched random subtraction (mode='rand0'). Positions >= SKIP."""

    def __init__(self, model, L, mhat, alpha=None, rhat=None, mode="scale"):
        self.model, self.L = model, L
        self.mhat, self.alpha, self.rhat, self.mode = mhat, alpha, rhat, mode
        self.h = None

    def __enter__(self):
        def f(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            x = hs[0, SKIP:].float()
            proj = x @ self.mhat
            if self.mode == "scale":
                x = x + (self.alpha - 1.0) * proj.unsqueeze(1) \
                    * self.mhat.unsqueeze(0)
            else:                                   # rand0: norm-matched
                x = x - proj.abs().unsqueeze(1) * self.rhat.unsqueeze(0)
            hs = hs.clone()
            hs[0, SKIP:] = x.to(hs.dtype)
            if args:
                return (hs,) + args[1:], kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        self.h = self.model.model.layers[self.L].register_forward_pre_hook(
            f, with_kwargs=True)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def run_model(which, docs_ids, tok, quadrants=None):
    """Full protocol on one model. Returns record dict."""
    t0 = time.time()
    print(f"--- {which} ---")
    m = load_model(which)
    d = m.config.hidden_size
    WU = m.lm_head.weight.detach().float()
    WUn = (WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
           ).to(DEV, torch.float16)
    rec = {}

    # quadrants (trained only; population-only for twin)
    groups = {}
    if which == "trained":
        floors = quadrants                       # twin per-layer p95 floors
        for L in MANIP_LAYERS:
            mc, cc = layer_write_metrics(m, WUn, L)
            wr, rc_ = mc > floors[L][0], cc > floors[L][1]
            groups[L] = {
                "jumble_conc": torch.nonzero((~wr) & rc_).squeeze(1).to(DEV),
                "readable_conc": torch.nonzero(wr & rc_).squeeze(1).to(DEV)}
            print(f"  L{L}: jc={len(groups[L]['jumble_conc'])} "
                  f"rc={len(groups[L]['readable_conc'])}")
    twin_floors = {}
    if which == "random":
        for L in MANIP_LAYERS:
            mc, cc = layer_write_metrics(m, WUn, L)
            twin_floors[L] = (float(np.percentile(mc.numpy(), 95)),
                              float(np.percentile(cc.numpy(), 95)))

    # ---- Phase 1: means, anatomy, clean stats ----
    n_layers_all = len(m.model.layers)
    sum_resid = {L: torch.zeros(d, device=DEV) for L in CENSUS_LAYERS}
    sum_lnA = {L: torch.zeros(d, device=DEV) for L in CENSUS_LAYERS}
    sum_lnB = {L: torch.zeros(d, device=DEV) for L in CENSUS_LAYERS}
    cntA = cntB = 0
    cnt = 0
    duty_num = {L: None for L in CENSUS_LAYERS}   # per-neuron count(act>0)
    sum_attn = [torch.zeros(d, device=DEV) for _ in range(n_layers_all)]
    sum_mlp = [torch.zeros(d, device=DEV) for _ in range(n_layers_all)]
    sum_emb = torch.zeros(d, device=DEV)
    comp_handles = []

    def mk_attn(i):
        def f(mod, inp, out):
            sum_attn[i] += out[0, SKIP:].detach().float().sum(0)
        return f

    def mk_mlp(i):
        def f(mod, inp, out):
            sum_mlp[i] += out[0, SKIP:].detach().float().sum(0)
        return f
    for i, lyr in enumerate(m.model.layers):
        comp_handles.append(lyr.self_attn.dense.register_forward_hook(
            mk_attn(i)))
        comp_handles.append(lyr.mlp.fc2.register_forward_hook(mk_mlp(i)))

    def emb_hook(mod, args, kwargs):
        nonlocal sum_emb
        hs = args[0] if args else kwargs["hidden_states"]
        sum_emb += hs[0, SKIP:].detach().float().sum(0)
    comp_handles.append(m.model.layers[0].register_forward_pre_hook(
        emb_hook, with_kwargs=True))

    gate3_S = {L: {"jumble_conc": [], "readable_conc": []}
               for L in MANIP_LAYERS}
    gate3_ln = {L: [] for L in MANIP_LAYERS}
    for di, ids in enumerate(docs_ids):
        with Capture(m, CENSUS_LAYERS) as cap:
            m(input_ids=ids)
        T = cap.acts[CENSUS_LAYERS[0]].shape[0]
        cnt += T
        for L in CENSUS_LAYERS:
            sum_resid[L] += cap.resid[L].sum(0)
            if di % 2 == 0:
                sum_lnA[L] += cap.lnin[L].sum(0)
            else:
                sum_lnB[L] += cap.lnin[L].sum(0)
            pos_cnt = (cap.acts[L] > 0).float().sum(0)
            duty_num[L] = pos_cnt if duty_num[L] is None \
                else duty_num[L] + pos_cnt
        if di % 2 == 0:
            cntA += T
        else:
            cntB += T
        # GATE 3 material: per-position channel writes at manip layers
        if which == "trained":
            sel = np.linspace(0, T - 1, min(GATE3_POS, T)).astype(int)
            for L in MANIP_LAYERS:
                f2 = m.model.layers[L].mlp.fc2.weight.detach().float()
                for g in ("jumble_conc", "readable_conc"):
                    idx = groups[L][g]
                    S = cap.acts[L][sel][:, idx.cpu()].to(DEV) \
                        @ f2[:, idx].T                     # (P, d)
                    gate3_S[L][g].append(S)
                gate3_ln[L].append(cap.lnin[L][sel].to(DEV))
    for h in comp_handles:
        h.remove()

    m_raw = {L: sum_resid[L] / cnt for L in CENSUS_LAYERS}
    b_ln = {L: (sum_lnA[L] + sum_lnB[L]) / cnt for L in CENSUS_LAYERS}
    rec["duty_cycle_median"] = {
        str(L): round(float((duty_num[L] / cnt).median()), 4)
        for L in CENSUS_LAYERS}
    rec["split_half_cos_b"] = {
        str(L): round(float(F.cosine_similarity(
            sum_lnA[L] / max(cntA, 1), sum_lnB[L] / max(cntB, 1), dim=0)), 4)
        for L in CENSUS_LAYERS}

    # anatomy: decode b_L; component means
    anat = {}
    for L in CENSUS_LAYERS:
        b = b_ln[L]
        bu = (b / b.norm().clamp(min=1e-9)).half()
        cs = (WUn @ bu).float()
        top = torch.topk(cs, 10).indices
        R = WU.to(DEV)[torch.topk(cs.abs(), 30).indices]
        R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-9)
        G = (R @ R.T).abs()
        anat[str(L)] = {
            "b_top_tokens": [tok.decode([int(t)]) for t in top],
            "b_mc": round(float(cs.abs().max()), 4),
            "b_tipcoh": round(float((G.sum() - 30) / (30 * 29)), 4),
            "m_raw_norm": round(float(m_raw[L].norm()), 3)}
    m_deep = m_raw[CENSUS_LAYERS[-1]]
    m_deep_u = m_deep / m_deep.norm().clamp(min=1e-9)
    comp = {"emb": {"norm": round(float((sum_emb / cnt).norm()), 3),
                    "cos_mdeep": round(float(
                        F.cosine_similarity(sum_emb / cnt, m_deep, dim=0)),
                        4)}}
    for i in range(n_layers_all):
        for nm, s in (("attn", sum_attn[i]), ("mlp", sum_mlp[i])):
            v = s / cnt
            comp[f"{nm}{i}"] = {
                "norm": round(float(v.norm()), 3),
                "cos_mdeep": round(float(
                    F.cosine_similarity(v, m_deep, dim=0)), 4)}
    rec["anatomy"] = anat
    rec["components"] = comp

    # GATE 3 (trained): cos(S_g, b_L)
    if which == "trained":
        g3 = {}
        for L in MANIP_LAYERS:
            b = b_ln[L]
            g3[str(L)] = {}
            for g in ("jumble_conc", "readable_conc"):
                S = torch.cat(gate3_S[L][g], 0)
                cosb = F.cosine_similarity(
                    S, b.unsqueeze(0).expand_as(S), dim=1)
                g3[str(L)][g] = round(float(cosb.median()), 4)
        rec["gate3_cos_S_b"] = g3

    # ---- Phase 2: manipulation ----
    rng = np.random.RandomState(SEED)
    mhat = {L: (m_raw[L] / m_raw[L].norm().clamp(min=1e-9))
            for L in MANIP_LAYERS}
    rhat = {}
    for L in MANIP_LAYERS:
        r = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
        rhat[L] = r / r.norm()
    manip = {}
    for di, ids in enumerate(docs_ids):
        # clean reference for this doc
        with Capture(m, CENSUS_LAYERS) as cap:
            out = m(input_ids=ids)
        clean_logits = out.logits[0].float()
        T = ids.shape[1]
        tgt = ids[0, SKIP + 1:]
        clean_nll = float(F.cross_entropy(
            clean_logits[SKIP:-1], tgt, reduction="mean"))
        clean_frac = {}
        clean_cosm = {}
        for L in CENSUS_LAYERS:
            a = cap.acts[L]
            clean_frac[L] = (float((a > 0.5).float().mean()),
                             float((a > 1.0).float().mean()))
            x = cap.resid[L]
            mh = m_raw[L] / m_raw[L].norm().clamp(min=1e-9)
            clean_cosm[L] = float(F.cosine_similarity(
                x, mh.unsqueeze(0).expand_as(x), dim=1).mean())
        for L in MANIP_LAYERS:
            variants = [("a%.1f" % a, dict(alpha=a, mode="scale"))
                        for a in ALPHAS] + [("rand0", dict(mode="rand0"))]
            for vname, kw in variants:
                with Manip(m, L, mhat[L], rhat=rhat[L], **kw), \
                        Capture(m, [Lp for Lp in CENSUS_LAYERS
                                    if Lp >= L]) as cap2:
                    out2 = m(input_ids=ids)
                lg = out2.logits[0].float()
                kl = float(F.kl_div(
                    F.log_softmax(lg[-1], dim=-1),
                    F.log_softmax(clean_logits[-1], dim=-1),
                    log_target=True, reduction="sum"))
                dnll = float(F.cross_entropy(
                    lg[SKIP:-1], tgt, reduction="mean")) - clean_nll
                key = (L, vname)
                if key not in manip:
                    manip[key] = {"kl": [], "dnll": [], "dfrac": {},
                                  "dcosm": {}, "dfrac_g": {}}
                mrec = manip[key]
                mrec["kl"].append(kl)
                mrec["dnll"].append(dnll)
                for Lp in [x for x in CENSUS_LAYERS if x >= L]:
                    a2 = cap2.acts[Lp]
                    d05 = float((a2 > 0.5).float().mean()) \
                        - clean_frac[Lp][0]
                    d10 = float((a2 > 1.0).float().mean()) \
                        - clean_frac[Lp][1]
                    mrec["dfrac"].setdefault(Lp, []).append((d05, d10))
                    if Lp > L:
                        x2 = cap2.resid[Lp]
                        mh = m_raw[Lp] / m_raw[Lp].norm().clamp(min=1e-9)
                        c2 = float(F.cosine_similarity(
                            x2, mh.unsqueeze(0).expand_as(x2), dim=1
                        ).mean())
                        mrec["dcosm"].setdefault(Lp, []).append(
                            c2 - clean_cosm[Lp])
                    if which == "trained":
                        for g in ("jumble_conc", "readable_conc"):
                            if Lp in groups:
                                idx = groups[Lp][g].cpu()
                                ag = a2[:, idx]
                                cg = cap.acts[Lp][:, idx]
                                dg = float((ag > 1.0).float().mean()) \
                                    - float((cg > 1.0).float().mean())
                                mrec["dfrac_g"].setdefault(
                                    (Lp, g), []).append(dg)
        if di == 0:
            print(f"  doc0 done ({time.time()-t0:.0f}s)")

    mm = {}
    for (L, vname), r in manip.items():
        e = {"kl": round(float(np.median(r["kl"])), 5),
             "dnll": round(float(np.median(r["dnll"])), 5),
             "kl_ci": boot_ci(r["kl"]),
             "dnll_ci": boot_ci(r["dnll"]),
             "dfrac10": {str(Lp): round(float(np.median(
                 [t[1] for t in v])), 5) for Lp, v in r["dfrac"].items()},
             "dfrac05": {str(Lp): round(float(np.median(
                 [t[0] for t in v])), 5) for Lp, v in r["dfrac"].items()},
             "dcosm": {str(Lp): round(float(np.median(v)), 4)
                       for Lp, v in r["dcosm"].items()}}
        # CI on the headline self-layer duty shifts (Table 1: Lp == manip L)
        if L in r["dfrac"]:
            e["dfrac05_ci"] = boot_ci([t[0] for t in r["dfrac"][L]])
            e["dfrac10_ci"] = boot_ci([t[1] for t in r["dfrac"][L]])
        if r["dfrac_g"]:
            e["dfrac10_g"] = {f"L{Lp}_{g}": round(float(np.median(v)), 5)
                              for (Lp, g), v in r["dfrac_g"].items()}
        mm[f"L{L}_{vname}"] = e
    rec["manip"] = mm

    del m, WU, WUn
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s)")
    return rec, twin_floors


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  "
          f"manip={MANIP_LAYERS}  alphas={ALPHAS}")
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
                      "manip_layers": MANIP_LAYERS, "alphas": ALPHAS,
                      "skip": SKIP, "seed": SEED}}
    twin_rec, twin_floors = run_model("random", docs_ids, tok)
    rec["twin"] = twin_rec
    tr_rec, _ = run_model("trained", docs_ids, tok, quadrants=twin_floors)
    rec["trained"] = tr_rec

    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")

    # console summary of the headline
    print("\nGATES: duty trained", tr_rec["duty_cycle_median"],
          "\n       duty twin   ", twin_rec["duty_cycle_median"])
    if "gate3_cos_S_b" in tr_rec:
        print("       gate3", tr_rec["gate3_cos_S_b"])
    for L in MANIP_LAYERS:
        for v in ["a0.0", "a2.0", "rand0", "a1.0"]:
            k = f"L{L}_{v}"
            if k in tr_rec["manip"]:
                e = tr_rec["manip"][k]
                print(f"  trained {k}: kl={e['kl']} dnll={e['dnll']} "
                      f"dfrac10={e['dfrac10']}")
        for v in ["a0.0", "a2.0"]:
            k = f"L{L}_{v}"
            if k in twin_rec["manip"]:
                e = twin_rec["manip"][k]
                print(f"  twin    {k}: kl={e['kl']} dnll={e['dnll']} "
                      f"dfrac10={e['dfrac10']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
