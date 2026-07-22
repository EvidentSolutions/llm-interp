"""Is the boilerplate INHIBITORY-suppressor axis (07-12) the same object as the
carried reference b_L, seen from the read side? (background paper, PLAN item 2;
was F.7 of the 07-17 triage.)

07-12 (census_c2_inhibitory_reader.py): 92% of Phi-2 CONJUNCTION/SELF neurons
carry an inhibitory reader -- a token whose PRESENCE suppresses firing --
and these concentrate on a SHARED low-dim boilerplate axis: 32 distinct
suppressor tokens dominated by function words ('s' x12, ' to' x5, ' of' x5,
' in', ',', '.', '\\n'). A neuron is suppressed there because its fc1 weight
w has NEGATIVE projection on the suppressor direction u ((x.u)(w.u) < 0 with
x.u > 0 -> w.u < 0).

The paper's reference coupling is the mirror statement: resting inhibition
w.b_L < 0, i.e. w points AWAY from the carried mean b_L. If the boilerplate
suppressor axis IS b_L (boilerplate = the frequent/mean content, so the
frequency-weighted mean gate input b_L points at boilerplate), then the
suppressor token directions align with b_hat_L (positive cos) and w.u < 0
is the same fact as w.b_L < 0. Two results unify. If orthogonal, the
reference is distinct from the known suppressor.

Both objects live in the SAME space: the empirical dictionary D[L] is the
post-input_layernorm gate input (line 12 of the 07-12 script), which is
exactly where b_L is the corpus mean. So cos is well-posed.

Per dict layer L in {2,6,10,14,18,22,26,30}:
  - b_hat_L : unit corpus-mean fc1 input (the reference), one forward pass
  - m_hat_L : unit corpus-mean raw residual (Sec 2 direction)
  - suppressor token dirs (from c2-inhibitory-reader.json, freq-weighted),
    excitatory token dirs (content contrast), all-dict tokens (isotropic null)
  - centered dirs Uc (what the 07-12 causal work used) AND raw unit dirs Uraw
    (centering removes the unweighted token mean, which partially overlaps
    b_L -- report cos(b_hat_L, mean-token) so the attenuation is explicit; the
    suppressor-vs-excitatory-vs-isotropic CONTRAST is centering-robust either
    way).
Floors: from_config twin b_hat_L (trained-dict dirs vs twin reference should
fall to isotropic -> the alignment is a trained property); isotropic all-token
median |cos|.

Weights + two forward passes (trained + twin). Phi-2 built-in arch, no
trust_remote_code. Reuses c2-empirical-dict.pt + c2-inhibitory-reader.json.
Usage: .venv/Scripts/python.exe superposition/code/census_suppressor_reference.py
       SMOKE=1 -> 3 layers, few docs, no twin.
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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
DICT = os.path.join(DATA, "c2-empirical-dict.pt")
INHIB = os.path.join(DATA, "c2-inhibitory-reader.json")
OUT = os.path.join(DATA, "census-suppressor-reference.json")


def cos_to(dirs, ref):
    """signed cos of each row of dirs (n,d) against unit ref (d,)."""
    dn = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (dn @ ref).numpy()


def collect_tokens():
    """per-layer suppressor token rows (freq-weighted) and excitatory rows."""
    j = json.load(open(INHIB, encoding="utf-8"))
    supp, exc = {}, {}
    for gk, rows in j["results"].items():
        for r in rows:
            if not r.get("has_inhib"):
                continue
            L = r["layer"]
            supp.setdefault(L, []).append(int(r["inhib_tok"]))
            if "exc_tok" in r and r["exc_tok"] is not None:
                exc.setdefault(L, []).append(int(r["exc_tok"]))
    return supp, exc


@torch.no_grad()
def reference_dirs(which, docs_text, tok, layers, hidden):
    """b_L (mean fc1 input) and m_raw (mean raw residual) per layer, unit."""
    cfg = AutoConfig.from_pretrained(MODEL)
    if which == "twin":
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    m = m.to(DEV).eval()

    gin, resid = {}, {}
    handles = []
    for L in layers:
        def mk_in(L=L):
            def f(mod, inp):
                gin[L] = inp[0][0, SKIP:].detach().float()
            return f

        def mk_res(L=L):
            def f(mod, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                resid[L] = hs[0, SKIP:].detach().float()
            return f
        handles.append(
            m.model.layers[L].mlp.fc1.register_forward_pre_hook(mk_in()))
        handles.append(m.model.layers[L].register_forward_pre_hook(
            mk_res(), with_kwargs=True))

    s_g = {L: torch.zeros(hidden, device=DEV) for L in layers}
    s_r = {L: torch.zeros(hidden, device=DEV) for L in layers}
    npos = 0
    for txt in docs_text:
        ids = tok(txt, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] < SKIP + 48:
            continue
        m(input_ids=ids)
        npos += gin[layers[0]].shape[0]
        for L in layers:
            s_g[L] += gin[L].sum(0)
            s_r[L] += resid[L].sum(0)
    for h in handles:
        h.remove()
    bL = {L: (s_g[L] / npos).cpu() for L in layers}
    mR = {L: (s_r[L] / npos).cpu() for L in layers}
    del m
    gc.collect()
    torch.cuda.empty_cache()
    bhat = {L: bL[L] / bL[L].norm().clamp_min(1e-8) for L in layers}
    mhat = {L: mR[L] / mR[L].norm().clamp_min(1e-8) for L in layers}
    return bhat, mhat


def summ(v):
    return {"med": round(float(np.median(v)), 3),
            "mean": round(float(np.mean(v)), 3),
            "n": int(len(v))}


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}")
    docs_text = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    tok = AutoTokenizer.from_pretrained(MODEL)
    dct = torch.load(DICT, map_location="cpu")
    keep = dct["keep"]
    tok2row = {t: i for i, t in enumerate(keep)}
    layers = list(dct["layers"])
    if SMOKE:
        layers = [10, 14, 18]
    D = {L: dct["D"][L].float() for L in layers}
    hidden = D[layers[0]].shape[1]

    supp_tok, exc_tok = collect_tokens()

    print("reference dirs: trained ...")
    bhat, mhat = reference_dirs("trained", docs_text, tok, layers, hidden)
    tbhat = None
    if not SMOKE:
        print("reference dirs: twin ...")
        torch.manual_seed(SEED)
        tbhat, _ = reference_dirs("twin", docs_text, tok, layers, hidden)

    rec = {"config": {"model": MODEL, "smoke": SMOKE, "n_docs": len(docs_text),
                      "skip": SKIP, "layers": layers}, "per_layer": {}}

    print(f"\n{'L':>3} {'meantok.b':>9} | {'supp.b':>7} {'exc.b':>7} "
          f"{'iso.b':>7} {'supp.m':>7} | {'supp.twin':>9}")
    for L in layers:
        Draw = D[L]
        mean_tok = Draw.mean(0)
        Dc = Draw - mean_tok                       # centered (07-12 basis)
        mean_tok_hat = mean_tok / mean_tok.norm().clamp_min(1e-8)
        # how much does the centering direction overlap the reference?
        cos_bL_meantok = float(mean_tok_hat @ bhat[L])

        s_rows = sorted(tok2row[t] for t in supp_tok.get(L, [])
                        if t in tok2row)
        e_rows = sorted(tok2row[t] for t in exc_tok.get(L, [])
                        if t in tok2row)
        rng = np.random.RandomState(SEED + L)
        iso_rows = rng.choice(Draw.shape[0],
                              min(500, Draw.shape[0]), replace=False).tolist()

        def block(rows):
            if not rows:
                return None
            idx = torch.tensor(rows)
            out = {
                "cos_bL_centered": summ(cos_to(Dc[idx], bhat[L])),
                "cos_bL_raw": summ(cos_to(Draw[idx], bhat[L])),
                "cos_mhat_centered": summ(cos_to(Dc[idx], mhat[L])),
            }
            if tbhat is not None:
                out["cos_twin_bL_centered"] = summ(cos_to(Dc[idx], tbhat[L]))
            return out

        # boilerplate axis as ONE direction: freq-weighted mean suppressor dir
        axis_cos = None
        if s_rows:
            idx = torch.tensor(s_rows)
            sub = Dc[idx]
            sub = sub / sub.norm(dim=1, keepdim=True).clamp_min(1e-8)
            axis = sub.mean(0)
            axis = axis / axis.norm().clamp_min(1e-8)
            axis_cos = round(float(axis @ bhat[L]), 3)

        rec["per_layer"][str(L)] = {
            "cos_bL_meantok": round(cos_bL_meantok, 3),
            "n_supp": len(s_rows), "n_exc": len(e_rows),
            "supp": block(s_rows), "exc": block(e_rows),
            "iso": block(iso_rows),
            "boilerplate_axis_cos_bL": axis_cos,
        }
        e = rec["per_layer"][str(L)]
        sb = e["supp"]["cos_bL_centered"]["med"] if e["supp"] else None
        eb = e["exc"]["cos_bL_centered"]["med"] if e["exc"] else None
        ib = e["iso"]["cos_bL_centered"]["med"] if e["iso"] else None
        sm = e["supp"]["cos_mhat_centered"]["med"] if e["supp"] else None
        st = (e["supp"]["cos_twin_bL_centered"]["med"]
              if e["supp"] and "cos_twin_bL_centered" in e["supp"] else None)
        print(f"{L:>3} {cos_bL_meantok:>9.3f} | {str(sb):>7} {str(eb):>7} "
              f"{str(ib):>7} {str(sm):>7} | {str(st):>9}   axis.b={axis_cos}")

    # aggregate across layers (centered cos to b_L)
    def agg(kind):
        vals = [rec["per_layer"][str(L)][kind]["cos_bL_centered"]["med"]
                for L in layers if rec["per_layer"][str(L)][kind]]
        return round(float(np.median(vals)), 3) if vals else None
    rec["aggregate"] = {"supp_cos_bL_med": agg("supp"),
                        "exc_cos_bL_med": agg("exc"),
                        "iso_cos_bL_med": agg("iso")}
    print(f"\nAGG centered cos(.,b_L): supp={rec['aggregate']['supp_cos_bL_med']} "
          f"exc={rec['aggregate']['exc_cos_bL_med']} "
          f"iso={rec['aggregate']['iso_cos_bL_med']}")

    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
