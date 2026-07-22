"""Cross-model CAUSAL background-removal + entropy-V (background paper, PLAN
item 5 + the promoted item 4, cross-architecture).

The Phi-2 causal result (paper Sec 6, Leg B): scaling the residual's
projection onto the corpus-mean direction m_hat_L at one mid-stack layer
moves downstream gate duty in dose-response (alpha=0 -> reference removed ->
gates fire MORE; alpha=2 -> gates held off MORE), beyond a norm-matched
random-direction control, twin flat. And output entropy has a MINIMUM at the
natural magnitude alpha=1 (the "V"): scaling the reference either way raises
entropy. Both legs were on ONE model. This script re-runs the SAME
architecture-neutral knob on two more families, each vs a from_config twin:

  - EleutherAI/pythia-1.4b-deduped : GELU MLP, gate bias present (like Phi-2).
  - Qwen/Qwen2.5-1.5B              : SwiGLU, gate bias FREE (b_n = 0).
  - microsoft/phi-2                : the paper baseline, as an in-script sanity
                                     replicate of Sec 6.

The knob x' = x + (alpha-1)(x.m_hat)m_hat at the decoder-layer input is
architecture-neutral (residual-stream projection scaling); m_hat_L is the
unit corpus-mean raw residual entering the manipulated layer. Manip layer is
picked at depth ~0.31 to match Phi-2 L10/32.

Two legs, one set of forward passes per (doc, alpha):
  DOSE (item 5): per census layer, d_duty = frac(gate_pre>0)_manip - clean;
    prediction gate-reference: alpha=0 raises duty, alpha=2 lowers it,
    monotone, beyond the norm-matched random control (rand0), twin flat. Plus
    last-position KL(clean||manip) and dNLL as output-damage dose curves.
  V (item 4): per-position output entropy dH = H_manip - H_clean, mean and
    three position terciles; prediction: dH>=0 with a minimum at alpha=1,0 in
    each tercile, ref arm separates from a matched random-scale control
    (mode='rand'), twin floored (|dH|~0).

Pre-registered branches (per family):
  V-REPLICATES  -- dH>=0, min at alpha=1.0, in all terciles, ref>rand at
                   matched |alpha-1|>=0.2 with disjoint CIs, twin |dH|<=~0.01.
  DOSE-REPLICATES -- d_duty(alpha=0)>0 and d_duty(alpha=2)<0 at the manip
                   layer, monotone across alpha, |d_duty| beyond rand0 with
                   disjoint CIs, twin |d_duty|~0.
  else NOISY/HETEROGENEOUS -- report the family's curve and CIs as-is.

Document-level percentile bootstrap CIs (2000) on every headline median.
Weights + forward passes only; all models <= 2.7B. Phi-2 built-in arch, no
trust_remote_code. from_config twins (no persistent twin dir needed).

Resumable: writes per-model keys into the output JSON after each model, and
frees the model (del/gc/empty_cache) between families. MODEL env selects a
single family by hf-id substring (e.g. MODEL=pythia) to run one at a time.

Usage: .venv/Scripts/python.exe superposition/code/census_crossmodel_causal.py
       SMOKE=1  -> phi-2 only, coarse grid, few docs, no twin.
       MODEL=qwen -> only the family whose id contains 'qwen'.
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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
ONLY = os.environ.get("MODEL", "").strip().lower()

# (hf id, trust_remote_code) -- all built-in archs, trust stays False
MODELS = [
    ("microsoft/phi-2", False),
    ("EleutherAI/pythia-1.4b-deduped", False),
    ("Qwen/Qwen2.5-1.5B", False),
]
if SMOKE:
    MODELS = MODELS[:1]
if ONLY:
    MODELS = [mm for mm in MODELS if ONLY in mm[0].lower()]

MANIP_DEPTH = 0.3125                          # Phi-2 L10/32
CENSUS_DEPTHS = [0.3125, 0.5, 0.7, 0.9]       # gate-duty read layers
ALPHAS = ([0.0, 1.0, 2.0] if SMOKE else
          [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
           1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0])
N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16
NBUCK = 3
CI_NBOOT = 200 if SMOKE else 2000

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-crossmodel-causal.json")


def boot_ci(vals, nboot=CI_NBOOT, seed=0):
    """95% percentile document-bootstrap CI of the median. Additive: leaves
    the point-estimate median untouched."""
    v = np.asarray(vals, dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.RandomState(seed)
    n = v.size
    bs = np.median(v[rng.randint(0, n, size=(nboot, n))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def get_layers(model):
    """Decoder layer ModuleList across the built-in architectures."""
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers                    # Pythia
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                       # Phi-2 / Qwen2 / Llama
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h                      # GPT-2
    raise RuntimeError("unknown architecture: no decoder layer list found")


def get_gate(layer):
    """(gate linear module, name): input is the post-LN MLP input, output is
    the gate pre-activation whose sign is the duty."""
    mlp = layer.mlp
    for name in ("fc1", "dense_h_to_4h", "gate_proj", "c_fc"):
        if hasattr(mlp, name):
            return getattr(mlp, name), name
    raise RuntimeError("unknown MLP: no gate linear found")


def pos_entropy(logits):
    """per-position Shannon entropy (nats) for positions >= SKIP, (T-SKIP,)."""
    lg = logits[SKIP:].float()
    lp = F.log_softmax(lg, dim=-1)
    return -(lp.exp() * lp).sum(-1)


class Knob:
    """Residual manipulation at layer L, positions >= SKIP.
      mode='scale' : x' = x + (alpha-1)(x.mhat)mhat   (reference scaling)
      mode='rand'  : x' = x + (alpha-1)(x.rhat)rhat   (matched random scale)
      mode='rand0' : x' = x - |x.rhat| rhat           (norm-matched removal)
    Architecture-neutral: hooks the decoder layer's hidden_states input."""

    def __init__(self, layer, mhat=None, rhat=None, alpha=None, mode="scale"):
        self.layer, self.mhat, self.rhat = layer, mhat, rhat
        self.alpha, self.mode, self.h = alpha, mode, None

    def __enter__(self):
        def f(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            x = hs[0, SKIP:].float()
            if self.mode == "rand0":
                proj = x @ self.rhat
                x = x - proj.abs().unsqueeze(1) * self.rhat.unsqueeze(0)
            else:
                v = self.mhat if self.mode == "scale" else self.rhat
                proj = x @ v
                x = x + (self.alpha - 1.0) * proj.unsqueeze(1) \
                    * v.unsqueeze(0)
            hs = hs.clone()
            hs[0, SKIP:] = x.to(hs.dtype)
            if args:
                return (hs,) + args[1:], kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        self.h = self.layer.register_forward_pre_hook(f, with_kwargs=True)
        return self

    def __exit__(self, *a):
        self.h.remove()


class DutyCap:
    """Capture gate pre-activation (>0 => firing) at census layers,
    positions >= SKIP, for one forward pass."""

    def __init__(self, layers, census):
        self.layers, self.census = layers, census
        self.pre = {}
        self.handles = []

    def __enter__(self):
        for L in self.census:
            gmod, _ = get_gate(self.layers[L])

            def mk(L=L):
                def f(mod, inp, out):
                    self.pre[L] = out[0, SKIP:].detach().float()
                return f
            self.handles.append(gmod.register_forward_hook(mk()))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def load(hf_id, which):
    cfg = AutoConfig.from_pretrained(hf_id)
    if which == "twin":
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, low_cpu_mem_usage=True)
    return m.to(DEV).eval()


@torch.no_grad()
def mean_resid(m, layers, docs_ids, L):
    """unit corpus-mean raw residual entering decoder layer L (m_hat_L)."""
    acc = {"s": None, "n": 0}

    def f(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        x = hs[0, SKIP:].detach().float().sum(0)
        acc["s"] = x if acc["s"] is None else acc["s"] + x
        acc["n"] += hs.shape[1] - SKIP
    h = layers[L].register_forward_pre_hook(f, with_kwargs=True)
    for ids in docs_ids:
        m(input_ids=ids)
    h.remove()
    v = acc["s"] / max(acc["n"], 1)
    return v / v.norm().clamp(min=1e-9)


def bucket_slices(T):
    """three position terciles over indices [0, T-SKIP)."""
    n = T - SKIP
    edges = [int(round(n * k / NBUCK)) for k in range(NBUCK + 1)]
    return [slice(edges[k], edges[k + 1]) for k in range(NBUCK)]


@torch.no_grad()
def run_model(hf_id, which, docs_ids, tok):
    t0 = time.time()
    print(f"--- {hf_id} [{which}] ---")
    m = load(hf_id, which)
    layers = get_layers(m)
    n = len(layers)
    d = m.config.hidden_size
    manip_L = min(max(int(round(n * MANIP_DEPTH)), 1), n - 1)
    census = sorted(set(min(max(int(round(n * f)), 1), n - 1)
                        for f in CENSUS_DEPTHS) | {manip_L})
    _, gname = get_gate(layers[manip_L])
    has_bias = get_gate(layers[manip_L])[0].bias is not None

    mhat = mean_resid(m, layers, docs_ids, manip_L)
    rng = np.random.RandomState(SEED)
    r = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
    rhat = r / r.norm()

    # per-doc accumulators
    kl = {a: [] for a in ALPHAS}
    dnll = {a: [] for a in ALPHAS}
    dduty = {a: {L: [] for L in census} for a in ALPHAS}     # ref scaling
    dH = {a: [] for a in ALPHAS}                             # ref entropy mean
    dH_b = {a: [[] for _ in range(NBUCK)] for a in ALPHAS}   # ref terciles
    dH_rand = {a: [] for a in ALPHAS}                        # rand ctrl mean
    dduty_r0 = {L: [] for L in census}                       # rand0 removal
    kl_r0, dnll_r0 = [], []

    for di, ids in enumerate(docs_ids):
        T = ids.shape[1]
        tgt = ids[0, SKIP + 1:]
        bslc = bucket_slices(T)
        with DutyCap(layers, census) as dc:
            out = m(input_ids=ids)
        clean_lg = out.logits[0].float()
        clean_nll = float(F.cross_entropy(
            clean_lg[SKIP:-1], tgt, reduction="mean"))
        clean_duty = {L: (dc.pre[L] > 0).float().mean(0) for L in census}
        clean_H = pos_entropy(clean_lg)

        for a in ALPHAS:
            # reference scaling: full metric set
            with Knob(layers[manip_L], mhat=mhat, alpha=a, mode="scale"), \
                    DutyCap(layers, census) as dc2:
                o2 = m(input_ids=ids)
            lg = o2.logits[0].float()
            kl[a].append(float(F.kl_div(
                F.log_softmax(lg[-1], dim=-1),
                F.log_softmax(clean_lg[-1], dim=-1),
                log_target=True, reduction="sum")))
            dnll[a].append(float(F.cross_entropy(
                lg[SKIP:-1], tgt, reduction="mean")) - clean_nll)
            for L in census:
                dd = float((dc2.pre[L] > 0).float().mean()
                           - clean_duty[L].mean())
                dduty[a][L].append(dd)
            H = pos_entropy(lg)
            dHv = (H - clean_H)
            dH[a].append(float(dHv.mean()))
            for k in range(NBUCK):
                dH_b[a][k].append(float(dHv[bslc[k]].mean()))

            # matched random-scale control (entropy V only)
            with Knob(layers[manip_L], rhat=rhat, alpha=a, mode="rand"):
                o3 = m(input_ids=ids)
            Hr = pos_entropy(o3.logits[0].float())
            dH_rand[a].append(float((Hr - clean_H).mean()))

        # norm-matched removal control (dose only), once per doc
        with Knob(layers[manip_L], rhat=rhat, mode="rand0"), \
                DutyCap(layers, census) as dc4:
            o4 = m(input_ids=ids)
        lg4 = o4.logits[0].float()
        kl_r0.append(float(F.kl_div(
            F.log_softmax(lg4[-1], dim=-1),
            F.log_softmax(clean_lg[-1], dim=-1),
            log_target=True, reduction="sum")))
        dnll_r0.append(float(F.cross_entropy(
            lg4[SKIP:-1], tgt, reduction="mean")) - clean_nll)
        for L in census:
            dduty_r0[L].append(float((dc4.pre[L] > 0).float().mean()
                                     - clean_duty[L].mean()))
        if di == 0:
            print(f"  doc0 done ({time.time()-t0:.0f}s) manip L{manip_L} "
                  f"[{gname}] census {census}")

    def med(x):
        return round(float(np.median(x)), 5)

    dose = {}
    for a in ALPHAS:
        dose[f"a{a}"] = {
            "kl": med(kl[a]), "kl_ci": boot_ci(kl[a]),
            "dnll": med(dnll[a]), "dnll_ci": boot_ci(dnll[a]),
            "dduty": {str(L): med(dduty[a][L]) for L in census},
            "dduty_ci": {str(L): boot_ci(dduty[a][L]) for L in census},
        }
    dose["rand0"] = {
        "kl": med(kl_r0), "kl_ci": boot_ci(kl_r0),
        "dnll": med(dnll_r0), "dnll_ci": boot_ci(dnll_r0),
        "dduty": {str(L): med(dduty_r0[L]) for L in census},
        "dduty_ci": {str(L): boot_ci(dduty_r0[L]) for L in census},
    }

    vgrid = {}
    for a in ALPHAS:
        vgrid[f"a{a}"] = {
            "dH": med(dH[a]), "dH_ci": boot_ci(dH[a]),
            "dH_rand": med(dH_rand[a]), "dH_rand_ci": boot_ci(dH_rand[a]),
            "dH_buckets": [med(dH_b[a][k]) for k in range(NBUCK)],
            "dH_buckets_ci": [boot_ci(dH_b[a][k]) for k in range(NBUCK)],
        }

    rec = {"manip_L": manip_L, "n_layers": n, "gate": gname,
           "has_bias": bool(has_bias), "census": census,
           "dose": dose, "vgrid": vgrid}
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s)")
    return rec


def decide_v(trained):
    """Pre-registered V verdict from the trained vgrid (mean arm)."""
    g = trained["vgrid"]
    a_keys = {float(k[1:]): k for k in g}
    if 1.0 not in a_keys:
        return {"verdict": "NO-ALPHA1"}
    at1 = g[a_keys[1.0]]["dH"]
    nonneg = all(g[a_keys[a]]["dH"] >= -1e-3 for a in a_keys)
    minat1 = all(g[a_keys[a]]["dH"] >= at1 - 1e-4 for a in a_keys)

    def beats(a):
        e = g.get(a_keys.get(a))
        if e is None or e["dH_ci"] is None or e["dH_rand_ci"] is None:
            return None
        return e["dH_ci"][0] > e["dH_rand_ci"][1]     # disjoint, ref above
    rel = beats(0.8)
    amp = beats(1.2)
    tercile_min = all(
        all(g[a_keys[a]]["dH_buckets"][k] >= g[a_keys[1.0]]["dH_buckets"][k]
            - 1e-3 for a in a_keys) for k in range(NBUCK))
    verdict = ("V-REPLICATES" if (nonneg and minat1 and rel and amp
                                  and tercile_min) else "PARTIAL/NOISY")
    return {"verdict": verdict, "nonneg": nonneg, "min_at_1": minat1,
            "release_beats_ctrl_0.8": rel, "amp_beats_ctrl_1.2": amp,
            "tercile_min_at_1": tercile_min, "dH_at_1": at1}


def decide_dose(trained):
    """Pre-registered dose verdict at the manip layer."""
    L = str(trained["manip_L"])
    dz = trained["dose"]
    a_keys = {float(k[1:]): k for k in dz if k.startswith("a")}

    def dd(a):
        return dz[a_keys[a]]["dduty"][L] if a in a_keys else None
    d0, d2 = dd(0.0), dd(2.0)
    sign_ok = (d0 is not None and d2 is not None and d0 > 0 and d2 < 0)
    # monotone non-increasing duty shift across alpha
    aa = sorted(a_keys)
    ser = [dz[a_keys[a]]["dduty"][L] for a in aa]
    mono = all(ser[i] >= ser[i + 1] - 1e-3 for i in range(len(ser) - 1))
    # alpha=0 removal beats norm-matched rand0 removal, disjoint CIs
    e0 = dz.get(a_keys.get(0.0))
    r0 = dz["rand0"]
    beats0 = None
    if e0 and e0["dduty_ci"][L] and r0["dduty_ci"][L]:
        beats0 = e0["dduty_ci"][L][0] > r0["dduty_ci"][L][1]
    verdict = ("DOSE-REPLICATES" if (sign_ok and mono and beats0)
               else "PARTIAL/NOISY")
    return {"verdict": verdict, "sign_ok": sign_ok, "monotone": mono,
            "a0_beats_rand0": beats0, "dduty_a0": d0, "dduty_a2": d2}


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  "
          f"alphas={len(ALPHAS)}  models={[mm[0] for mm in MODELS]}")
    rec = {}
    if os.path.exists(OUT):
        try:
            rec = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            rec = {}
    rec.setdefault("config", {})
    rec["config"].update({"smoke": SMOKE, "n_docs": N_DOCS, "skip": SKIP,
                          "alphas": ALPHAS, "manip_depth": MANIP_DEPTH,
                          "census_depths": CENSUS_DEPTHS, "seed": SEED,
                          "ci_nboot": CI_NBOOT})
    rec.setdefault("models", {})

    for hf_id, _ in MODELS:
        print(f"\n==== {hf_id} ====")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
            docs_ids = []
            for t in docs:
                ids = tok(t, truncation=True, max_length=MAXLEN,
                          return_tensors="pt")["input_ids"].to(DEV)
                if ids.shape[1] >= SKIP + 48:
                    docs_ids.append(ids)
            print(f"  {len(docs_ids)} docs usable")
            trained = run_model(hf_id, "trained", docs_ids, tok)
            twin = None
            if not SMOKE:
                torch.manual_seed(SEED)
                twin = run_model(hf_id, "twin", docs_ids, tok)
            ent = {"trained": trained, "twin": twin,
                   "v_decision": decide_v(trained),
                   "dose_decision": decide_dose(trained)}
            rec["models"][hf_id] = ent

            L = str(trained["manip_L"])
            vd, dd = ent["v_decision"], ent["dose_decision"]
            print(f"  V: {vd['verdict']}  dH@1={vd['dH_at_1']}  "
                  f"minat1={vd['min_at_1']} rel={vd['release_beats_ctrl_0.8']}"
                  f" amp={vd['amp_beats_ctrl_1.2']}")
            print(f"  DOSE: {dd['verdict']}  d_duty(a0)={dd['dduty_a0']} "
                  f"d_duty(a2)={dd['dduty_a2']} mono={dd['monotone']} "
                  f"beats_rand0={dd['a0_beats_rand0']}")
            if twin is not None:
                twk = twin["vgrid"]
                akeys = {float(k[1:]): k for k in twk}
                tw_amp = max(abs(twk[akeys[a]]["dH"]) for a in akeys)
                print(f"  twin: max|dH|={round(tw_amp, 4)}  "
                      f"d_duty(a0)={twin['dose']['a0.0']['dduty'][L]}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rec["models"][hf_id] = {"error": repr(ex)}
            print(f"  !! {hf_id} failed: {ex!r}")
            gc.collect()
            torch.cuda.empty_cache()

        json.dump(rec, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"  [saved {OUT}]")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
