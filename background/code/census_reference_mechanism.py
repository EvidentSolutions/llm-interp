"""Reference mechanism: bias migration, global knob, pilot tone
(plan_mid_stack_empirical_basis.md 3x, pre-registered 2026-07-16).

Leg A (exact): resting preactivation of every gate decomposes as
<w.x_ln + b_n> = w.b_L + b_n. Homogeneous-coordinates prediction: the
resting inhibition is carried by the reference coupling w.b_L, not the
parameter b_n. Consistency: resting preact predicts duty cycle; twin
resting ~ 0 with duty ~ 0.5.

Leg B: gentle alpha grid {0.8..1.2} on the reference projection at L10 vs
a matched random-direction transfer control; dH (mean output entropy
change) monotone + sign-consistent + high dH/KL purity = temperature knob.

Leg C: per-layer attention/MLP OUTPUT decomposition along mhat vs
orthogonal; carrier prediction: the mhat-channel is DC-dominant across
positions (steady feed), orthogonal AC-dominant; twin less dissociated.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_mechanism.py
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


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CI_NBOOT = 2000
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

CENSUS_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
L_B = 10
ALPHAS = [0.8, 1.0, 1.2] if SMOKE else [0.8, 0.9, 1.0, 1.1, 1.2]
N_DOCS = 6 if SMOKE else 20
MAXLEN = 256
SKIP = 16

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-mechanism.json")


@torch.no_grad()
def load_model(which):
    if which == "random":
        m = AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    return m.to(DEV).eval()


class Cap:
    """Per-forward capture: raw resid, ln'd fc1 input, post-GELU acts,
    attn-block and mlp-block outputs, at the census layers."""

    def __init__(self, model, layers, want=("resid", "ln", "act", "comp")):
        self.model, self.layers, self.want = model, layers, want
        self.resid, self.ln, self.act = {}, {}, {}
        self.attn_out, self.mlp_out = {}, {}
        self.h = []

    def __enter__(self):
        for L in self.layers:
            lyr = self.model.model.layers[L]
            if "resid" in self.want:
                def mk_r(L=L):
                    def f(mod, args, kwargs):
                        hs = args[0] if args else kwargs["hidden_states"]
                        self.resid[L] = hs[0, SKIP:].detach().float()
                    return f
                self.h.append(lyr.register_forward_pre_hook(
                    mk_r(), with_kwargs=True))
            if "ln" in self.want:
                def mk_l(L=L):
                    def f(mod, inp):
                        self.ln[L] = inp[0][0, SKIP:].detach().float()
                    return f
                self.h.append(lyr.mlp.fc1.register_forward_pre_hook(mk_l()))
            if "act" in self.want:
                def mk_a(L=L):
                    def f(mod, inp):
                        self.act[L] = inp[0][0, SKIP:].detach().float()
                    return f
                self.h.append(lyr.mlp.fc2.register_forward_pre_hook(mk_a()))
            if "comp" in self.want:
                def mk_ao(L=L):
                    def f(mod, inp, out):
                        self.attn_out[L] = out[0, SKIP:].detach().float()
                    return f

                def mk_mo(L=L):
                    def f(mod, inp, out):
                        self.mlp_out[L] = out[0, SKIP:].detach().float()
                    return f
                self.h.append(lyr.self_attn.dense.register_forward_hook(
                    mk_ao()))
                self.h.append(lyr.mlp.fc2.register_forward_hook(mk_mo()))
        return self

    def __exit__(self, *a):
        for hh in self.h:
            hh.remove()


class Knob:
    """x' = x + (alpha-1)(x.mhat)mhat, or the matched random-transfer
    control x' = x + (alpha-1)(x.mhat)rhat. Positions >= SKIP."""

    def __init__(self, model, L, mhat, alpha, rhat=None):
        self.model, self.L = model, L
        self.mhat, self.alpha, self.rhat = mhat, alpha, rhat
        self.h = None

    def __enter__(self):
        def f(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            x = hs[0, SKIP:].float()
            proj = x @ self.mhat
            tgt = self.rhat if self.rhat is not None else self.mhat
            x = x + (self.alpha - 1.0) * proj.unsqueeze(1) * tgt.unsqueeze(0)
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
def run_model(which, docs_ids):
    t0 = time.time()
    print(f"--- {which} ---")
    m = load_model(which)
    d = m.config.hidden_size
    rec = {}

    # ---- pass 1: means ----
    s_raw = {L: torch.zeros(d, device=DEV) for L in CENSUS_LAYERS}
    s_ln = {L: torch.zeros(d, device=DEV) for L in CENSUS_LAYERS}
    sln_doc = {L: [] for L in CENSUS_LAYERS}   # per-doc ln sums (PLAN item 3 CI)
    T_doc = []                                 # per-doc token counts
    npos = 0
    for ids in docs_ids:
        with Cap(m, CENSUS_LAYERS, want=("resid", "ln")) as c:
            m(input_ids=ids)
        npos += c.ln[CENSUS_LAYERS[0]].shape[0]
        T_doc.append(c.ln[CENSUS_LAYERS[0]].shape[0])
        for L in CENSUS_LAYERS:
            s_raw[L] += c.resid[L].sum(0)
            sd = c.ln[L].sum(0)
            s_ln[L] += sd
            sln_doc[L].append(sd.cpu())
    m_raw = {L: s_raw[L] / npos for L in CENSUS_LAYERS}
    b_ln = {L: s_ln[L] / npos for L in CENSUS_LAYERS}
    mhat = {L: m_raw[L] / m_raw[L].norm().clamp(min=1e-9)
            for L in CENSUS_LAYERS}

    # ---- Leg A: exact resting decomposition (weights + b_L) ----
    legA = {}
    duty_store = {}
    ci_store = {}                              # per-doc ref numerators for CI
    for L in CENSUS_LAYERS:
        W1 = m.model.layers[L].mlp.fc1.weight.detach().float()
        bn = m.model.layers[L].mlp.fc1.bias.detach().float()
        ref = (W1 @ b_ln[L]).cpu()
        bnc = bn.cpu()
        resting = ref + bnc
        # per-doc ref numerator: W1 @ (sum of ln over that doc), so a resample
        # reconstructs the exact mean ref = W1 @ b_L for the bootstrap docs.
        refnum = torch.stack([(W1 @ sd.to(W1.device)).cpu()
                              for sd in sln_doc[L]])   # (ndocs, hidden)
        ci_store[L] = {"refnum": refnum, "bnc": bnc}
        legA[str(L)] = {
            "ref_med": round(float(ref.median()), 4),
            "bias_med": round(float(bnc.median()), 4),
            "resting_med": round(float(resting.median()), 4),
            "abs_ref_med": round(float(ref.abs().median()), 4),
            "abs_bias_med": round(float(bnc.abs().median()), 4),
            "frac_resting_neg": round(float((resting < 0).float().mean()),
                                      4),
            "frac_ref_dominant": round(float(
                (ref.abs() > bnc.abs()).float().mean()), 4)}
        legA[str(L)]["_resting"] = resting  # for the duty correlation
        del W1

    # ---- pass 2: duty + Leg C component stats ----
    duty = {L: None for L in CENSUS_LAYERS}
    duty_doc = {L: [] for L in CENSUS_LAYERS}   # per-doc positive counts (CI)
    pjA = {L: {"attn": 0.0, "mlp": 0.0} for L in CENSUS_LAYERS}
    pjB = {L: {"attn": 0.0, "mlp": 0.0} for L in CENSUS_LAYERS}
    pj2 = {L: {"attn": 0.0, "mlp": 0.0} for L in CENSUS_LAYERS}
    vA = {L: {c: torch.zeros(d, device=DEV) for c in ("attn", "mlp")}
          for L in CENSUS_LAYERS}
    vB = {L: {c: torch.zeros(d, device=DEV) for c in ("attn", "mlp")}
          for L in CENSUS_LAYERS}
    v2 = {L: {"attn": 0.0, "mlp": 0.0} for L in CENSUS_LAYERS}
    csum = {L: {c: torch.zeros(d, device=DEV) for c in ("attn", "mlp")}
            for L in CENSUS_LAYERS}
    nA = nB = 0
    for di, ids in enumerate(docs_ids):
        with Cap(m, CENSUS_LAYERS, want=("act", "comp")) as c:
            m(input_ids=ids)
        T = c.act[CENSUS_LAYERS[0]].shape[0]
        if di % 2 == 0:
            nA += T
        else:
            nB += T
        for L in CENSUS_LAYERS:
            pc = (c.act[L] > 0).float().sum(0)
            duty[L] = pc if duty[L] is None else duty[L] + pc
            duty_doc[L].append(pc.cpu())
            for cn, o in (("attn", c.attn_out[L]), ("mlp", c.mlp_out[L])):
                proj = o @ mhat[L]                      # (T,)
                vperp = o - proj.unsqueeze(1) * mhat[L].unsqueeze(0)
                pj2[L][cn] += float((proj ** 2).sum())
                v2[L][cn] += float((vperp ** 2).sum())
                csum[L][cn] += o.sum(0)
                if di % 2 == 0:
                    pjA[L][cn] += float(proj.sum())
                    vA[L][cn] += vperp.sum(0)
                else:
                    pjB[L][cn] += float(proj.sum())
                    vB[L][cn] += vperp.sum(0)
    rec["duty_med"] = {str(L): round(float((duty[L] / npos).median()), 4)
                       for L in CENSUS_LAYERS}
    # Leg A duty correlation + cleanup
    for L in CENSUS_LAYERS:
        resting = legA[str(L)].pop("_resting")
        dl = (duty[L] / npos).cpu().numpy()
        rho = spearman(resting.numpy(), dl)
        legA[str(L)]["spearman_resting_duty"] = round(rho, 3)

        # ---- PLAN item 3: document-bootstrap CIs on the two headline stats ----
        # Resample the docs; reconstruct the exact resample-mean ref (via per-doc
        # ref numerators / token counts) and per-doc duty. Additive: the point
        # estimates above are untouched.
        refnum = ci_store[L]["refnum"].numpy()          # (nd, hidden)
        bnc = ci_store[L]["bnc"].numpy()                # (hidden,)
        dnum = torch.stack(duty_doc[L]).numpy()         # (nd, hidden)
        Tarr = np.asarray(T_doc, dtype=np.float64)      # (nd,)
        nd = refnum.shape[0]
        rng = np.random.RandomState(SEED)
        fr_bs = np.empty(CI_NBOOT)
        rho_bs = np.empty(CI_NBOOT)
        for b in range(CI_NBOOT):
            idx = rng.randint(0, nd, size=nd)
            denom = Tarr[idx].sum()
            ref_b = refnum[idx].sum(0) / denom
            duty_b = dnum[idx].sum(0) / denom
            resting_b = ref_b + bnc
            fr_bs[b] = float(np.mean(np.abs(ref_b) > np.abs(bnc)))
            rho_bs[b] = spearman(resting_b, duty_b)
        legA[str(L)]["frac_ref_dominant_ci"] = [
            round(float(np.percentile(fr_bs, 2.5)), 5),
            round(float(np.percentile(fr_bs, 97.5)), 5)]
        legA[str(L)]["spearman_resting_duty_ci"] = [
            round(float(np.percentile(rho_bs, 2.5)), 3),
            round(float(np.percentile(rho_bs, 97.5)), 3)]
    rec["legA"] = legA

    # Leg C summaries (+ the 3t anatomy gate)
    m_deep = m_raw[CENSUS_LAYERS[-1]]
    legC = {}
    for L in CENSUS_LAYERS:
        e = {}
        for cn in ("attn", "mlp"):
            muA, muB = pjA[L][cn] / max(nA, 1), pjB[L][cn] / max(nB, 1)
            const_m = (muA * muB) / (pj2[L][cn] / npos + 1e-9)
            wA, wB = vA[L][cn] / max(nA, 1), vB[L][cn] / max(nB, 1)
            const_p = float(wA @ wB) / (v2[L][cn] / npos + 1e-9)
            e[cn] = {"const_mchan": round(const_m, 4),
                     "const_perp": round(const_p, 4),
                     "cos_mean_mdeep": round(float(F.cosine_similarity(
                         csum[L][cn] / npos, m_deep, dim=0)), 4)}
        legC[str(L)] = e
    rec["legC"] = legC

    # ---- Leg B: the knob at L10 ----
    rng = np.random.RandomState(SEED)
    r = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
    rhat = r / r.norm()
    agg = {}
    for di, ids in enumerate(docs_ids):
        out = m(input_ids=ids)
        lg0 = out.logits[0, SKIP:].float()
        p0 = F.log_softmax(lg0, dim=-1)
        H0 = float((-p0.exp() * p0).sum(-1).mean())
        last0 = F.log_softmax(out.logits[0, -1].float(), dim=-1)
        for a in ALPHAS:
            for mode in ("ref", "rand"):
                if a == 1.0 and mode == "rand":
                    continue
                with Knob(m, L_B, mhat[L_B], a,
                          rhat=None if mode == "ref" else rhat):
                    o2 = m(input_ids=ids)
                lg = o2.logits[0, SKIP:].float()
                p = F.log_softmax(lg, dim=-1)
                dH = float((-p.exp() * p).sum(-1).mean()) - H0
                kl = float(F.kl_div(
                    F.log_softmax(o2.logits[0, -1].float(), dim=-1),
                    last0, log_target=True, reduction="sum"))
                k = f"{mode}_a{a:.1f}"
                agg.setdefault(k, {"dH": [], "kl": []})
                agg[k]["dH"].append(dH)
                agg[k]["kl"].append(kl)
    rec["legB"] = {k: {"dH": round(float(np.median(v["dH"])), 5),
                       "kl": round(float(np.median(v["kl"])), 5)}
                   for k, v in agg.items()}

    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s)")
    return rec


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  alphas={ALPHAS}")
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
                      "alphas": ALPHAS, "l_b": L_B, "seed": SEED}}
    rec["twin"] = run_model("random", docs_ids)
    rec["trained"] = run_model("trained", docs_ids)
    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")

    for w in ("trained", "twin"):
        r = rec[w]
        print(f"\n== {w} Leg A (ref_med / bias_med / resting_med / "
              f"fracRefDom / rho(rest,duty)):")
        for L in ["6", "10", "14", "22"]:
            a = r["legA"][L]
            print(f"  L{L}: {a['ref_med']} / {a['bias_med']} / "
                  f"{a['resting_med']} / {a['frac_ref_dominant']} / "
                  f"{a['spearman_resting_duty']}  duty="
                  f"{r['duty_med'][L]}")
        print(f"== {w} Leg B:")
        for k in sorted(r["legB"]):
            print(f"  {k}: dH={r['legB'][k]['dH']} kl={r['legB'][k]['kl']}")
        print(f"== {w} Leg C (const m-chan / const perp, attn|mlp) L10:")
        c = r["legC"]["10"]
        print(f"  attn {c['attn']['const_mchan']}/{c['attn']['const_perp']}"
              f"  mlp {c['mlp']['const_mchan']}/{c['mlp']['const_perp']}"
              f"  cos_mdeep attn={c['attn']['cos_mean_mdeep']} "
              f"mlp={c['mlp']['cos_mean_mdeep']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
