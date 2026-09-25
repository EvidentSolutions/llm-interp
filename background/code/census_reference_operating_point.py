"""Is the reference a constant MAGNITUDE or a constant FRACTION? (background
paper v2, item #3.)

The paper describes b_L as a constant the network "carries and maintains", but
never states its magnitude schedule across depth. That leaves an obvious
objection open: the residual stream grows several-fold across the stack, so a
fixed coupling w.b_L cannot be a fixed operating point unless something
normalises it.

E228 (8 prompts) answered this on the raw-residual background. This re-measures
it on the paper's OWN census corpus and instruments, across ALL layers, so the
numbers are comparable with the rest of the paper rather than imported.

Per layer L, over far positions (>= SKIP), trained and random-init twin:
  resid_norm  mean ||x|| of the raw residual at the layer input (PRE-LN)
  m_norm      ||m_L||, norm of the corpus-mean raw residual   (PRE-LN)
  m_share     m_norm / resid_norm  -- the claim: ~constant across depth
  b_norm      ||b_L||, norm of the corpus-mean LN'd fc1 input (POST-LN;
              the object gates actually read) -- the claim: flat
  resting     median over neurons of (w_n . b_L + b_n) -- the operating point
  ref_over_b  median |w.b_L| / median |b_n|

Reading: if m_share is ~flat while resid_norm and m_norm both grow, then the
pre-LN background grows only to track the growing stream, LayerNorm divides the
growth out, and the post-LN reference (hence the resting term every gate sees)
is depth-stable. "A carried constant" is then more precisely a maintained
constant FRACTION.

CONSTRUCTION-VALIDITY GATE: `resting` at L6/L10/L14/L22 must reproduce the
paper's Table 2 medians (-1.535 / -1.226 / -1.091 / -1.130, computed there on 19
documents) to within ~0.15.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_operating_point.py
       SMOKE=1 fast pass
"""
import sys
import os
import gc
import json
import time
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
N_DOCS = int(os.environ.get("NDOCS", "6" if SMOKE else "40"))
MAXLEN = 256
SKIP = 16
WHICH = ["trained"] if SMOKE else ["random", "trained"]

# paper Table 2 (19 docs) -- construction-validity gate
TABLE2 = {6: -1.535, 10: -1.226, 14: -1.091, 22: -1.130}
GATE_TOL = 0.15

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-reference-operating-point.json")


@torch.no_grad()
def load_model(which):
    if which == "random":
        return AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False).to(DEV).eval()
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()


class Capture:
    def __init__(self, model, n_layers):
        self.model, self.n = model, n_layers
        self.resid, self.lnin, self.handles = {}, {}, []

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


@torch.no_grad()
def run_model(which, docs_ids):
    t0 = time.time()
    print(f"\n=== {which} ===", flush=True)
    m = load_model(which)
    d, n_layers = m.config.hidden_size, len(m.model.layers)

    sum_res = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_ln = {L: torch.zeros(d, device=DEV) for L in range(n_layers)}
    sum_resnorm = {L: 0.0 for L in range(n_layers)}
    cnt = 0

    for ids in docs_ids:
        with Capture(m, n_layers) as cap:
            m(input_ids=ids)
        T = cap.resid[0].shape[0]
        for L in range(n_layers):
            far = cap.resid[L][SKIP:]
            sum_res[L] += far.sum(0)
            sum_resnorm[L] += float(far.norm(dim=-1).sum())
            sum_ln[L] += cap.lnin[L][SKIP:].sum(0)
        cnt += T - SKIP

    rows = []
    for L in range(n_layers):
        m_L = sum_res[L] / cnt
        b_L = sum_ln[L] / cnt
        resid_norm = sum_resnorm[L] / cnt
        w = m.model.layers[L].mlp.fc1.weight.detach().float()      # (dff, d)
        bn = m.model.layers[L].mlp.fc1.bias.detach().float()
        wb = w @ b_L
        resting = wb + bn
        rows.append({
            "L": L,
            "resid_norm": round(resid_norm, 2),
            "m_norm": round(float(m_L.norm()), 2),
            "m_share": round(float(m_L.norm()) / max(resid_norm, 1e-9), 4),
            "b_norm": round(float(b_L.norm()), 3),
            "resting_median": round(float(resting.median()), 4),
            "wb_median": round(float(wb.median()), 4),
            "bn_absmedian": round(float(bn.abs().median()), 4),
            "ref_over_bias": round(
                float(wb.abs().median() / bn.abs().median().clamp(min=1e-9)), 1),
        })

    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{'L':>3} {'||resid||':>10} {'||m_L||':>9} {'m_share':>8} "
          f"{'||b_L||':>8} {'resting':>9} {'|wb|/|bn|':>10}")
    for r in rows:
        print(f"{r['L']:>3} {r['resid_norm']:>10.1f} {r['m_norm']:>9.1f} "
              f"{r['m_share']:>8.3f} {r['b_norm']:>8.2f} "
              f"{r['resting_median']:>+9.3f} {r['ref_over_bias']:>10.1f}")
    print(f"  {which} done ({time.time()-t0:.0f}s)")
    return rows


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV} SMOKE={SMOKE} docs={N_DOCS} skip={SKIP}")
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
                      "seed": SEED, "model": MODEL}}
    for which in WHICH:
        rec[which] = run_model(which, docs_ids)
    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT} ({time.time()-t0:.0f}s)")

    print("\n===== GATE: resting median vs paper Table 2 (19 docs) =====")
    tr = {r["L"]: r for r in rec.get("trained", [])}
    ok = True
    for L, want in TABLE2.items():
        got = tr.get(L, {}).get("resting_median")
        hit = got is not None and abs(got - want) < GATE_TOL
        ok &= hit
        print(f"  L{L}: paper {want:+.3f} vs new {got:+.3f}  "
              f"{'OK' if hit else 'MISMATCH'}")
    print(f"\nGATE {'PASSED' if ok else 'FAILED'}")

    if "trained" in rec:
        sh = [r["m_share"] for r in rec["trained"] if 2 <= r["L"] <= 28]
        bn = [r["b_norm"] for r in rec["trained"] if 2 <= r["L"] <= 28]
        rn = [r["resid_norm"] for r in rec["trained"]]
        print(f"\nHEADLINE (trained, L2-L28): m_share {min(sh):.3f}-{max(sh):.3f} "
              f"(spread {max(sh)-min(sh):.3f}); ||b_L|| {min(bn):.2f}-{max(bn):.2f}; "
              f"||resid|| {rn[0]:.0f} -> {rn[-1]:.0f} ({rn[-1]/max(rn[0],1e-9):.1f}x)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
