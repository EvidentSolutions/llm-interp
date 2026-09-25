# -*- coding: utf-8 -*-
"""Does misaligning the reference matter BEYOND the next token? (Olli, 2026-08-19.)

THE GAP THIS CLOSES. Every causal number in the b_L battery is scored on the
IMMEDIATE next-token prediction: the paper's dose-response scores duty and KL at the
perturbed positions; 15bo/15bq scored logP of one answer token at the final position;
15bs scored duty at the task position. Olli's point: many late writes are
FORWARD-FACING -- they set up content for positions beyond the next one -- so a
next-token-only instrument systematically under-measures them. The project's own
record supports the premise independently: type-level anticipation at bridge
positions (P8d), held selections and binary products (P9d/P9e), next_* registers
peaking ~30 layers later than cur_* (P13), quote-close anticipation (E143/E174).
Content is demonstrably carried forward, and we scored the reference as if it were
not.

Note also that "misaligning b_L barely affects prediction" is NOT an established
claim of ours -- see the status section of the accompanying report. The nearest thing
was WITHDRAWN (it compared a whole-degenerate-input duty figure against a task run
with a clean prompt). What IS established is the opposite for the immediate
prediction: removal costs KL 1.43/0.52/0.27 at L6/L10/L14 against a 0.020-0.025
control. The horizon question is simply unasked.

DESIGN. Perturb a WINDOW and measure CLEAN positions after it, so forward carry is
isolated from "this position is perturbed":
    window   positions [p, p+w) get the paper's manipulation at layer L,
             x' = x + (alpha-1)(x . m_hat_L) m_hat_L with alpha = 0
    horizon  positions p+w .. p+w+H are UNTOUCHED; any change there arrived
             through attention reading the perturbed window's keys/values
Reported per offset: KL(baseline || perturbed) of the full next-token distribution
(the primary -- it asks whether the prediction changed at all, independent of what
the true token happens to be) and delta logP of the true token.

Controls: a matched-norm fixed random direction, same window, same layer (the
paper's control). Identity check at alpha=1 must give KL = 0 exactly.

Perturbation layer is varied because it changes how much stack remains for the
perturbation to reach later positions through attention: a late perturbation can only
propagate through the layers above it.

Also measured: GREEDY DIVERGENCE. Teacher forcing hides compounding, so we greedily
decode H tokens from the end of the window with and without the perturbation and
report the first offset at which the two decodes differ.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_horizon.py
"""
import os
import sys
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
MAXLEN = 256
N_BL, N_EVAL = 20, 24
LAYERS = [6, 14, 24]
P0, WIN, HOR = 64, 8, 32          # window start, window width, horizon
GEN = 16
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-reference-horizon.json")


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz, d = m.model.layers, m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))
    bl_docs, ev_docs = docs[:N_BL], docs[N_BL:N_BL + N_EVAL]
    need = P0 + WIN + HOR + 2

    cap = {}
    caps = [Lz[L].register_forward_pre_hook(
        (lambda L: (lambda mod_, a, kw: cap.__setitem__(
            f"h{L}", (a[0] if a else kw["hidden_states"])[0].detach())))(L),
        with_kwargs=True) for L in LAYERS]

    # ---- m_hat_L on a disjoint set -----------------------------------------
    acc = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    n = 0
    for text in bl_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < 32:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in LAYERS:
            acc[L] += cap[f"h{L}"][16:].double().sum(0)
        n += len(ids) - 16
    mhat = {}
    for L in LAYERS:
        v = (acc[L] / n).float()
        mhat[L] = v / v.norm()
    for c in caps:
        c.remove()
    print(f"  m_hat from {n} positions of {len(bl_docs)} docs")

    g = torch.Generator(device=DEV).manual_seed(5)
    rdir = torch.randn(d, generator=g, device=DEV)
    rdir = rdir / rdir.norm()

    def perturb_hook(L, lo, hi, mode, alpha=0.0):
        u = mhat[L] if mode != "rand" else rdir
        ref = mhat[L]

        def h(mod_, a, kw):
            x = (a[0] if a else kw["hidden_states"])
            y = x.clone()
            seg = y[0, lo:hi].float()
            if mode == "rand":
                # matched magnitude: subtract the same per-position amount
                # along a fixed random direction (the paper's control)
                amt = (seg @ ref)[:, None]
                seg = seg - amt * u[None, :]
            else:
                seg = seg + (alpha - 1.0) * (seg @ u)[:, None] * u[None, :]
            y[0, lo:hi] = seg.to(y.dtype)
            if a:
                return (y,) + tuple(a[1:]), kw
            kw = dict(kw); kw["hidden_states"] = y
            return a, kw
        return Lz[L].register_forward_pre_hook(h, with_kwargs=True)

    def logits_of(ids, hook=None):
        hh = hook() if hook else None
        with torch.no_grad():
            lg = m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()
        if hh:
            hh.remove()
        return lg

    res = {}
    for L in LAYERS:
        lo, hi = P0, P0 + WIN
        rows = {"window_kl": [], "window_dlp": [],
                "kl": {k: [] for k in range(HOR)},
                "dlp": {k: [] for k in range(HOR)},
                "kl_rand": {k: [] for k in range(HOR)},
                "window_kl_rand": [], "div": []}
        idcheck = []
        for text in ev_docs:
            ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
            if len(ids) < need:
                continue
            base = logits_of(ids)
            bl = torch.log_softmax(base, -1)
            # identity gate: alpha=1 must be an exact no-op
            one = logits_of(ids, lambda: perturb_hook(L, lo, hi, "ref", 1.0))
            idcheck.append(float((one - base).abs().max()))
            pert = logits_of(ids, lambda: perturb_hook(L, lo, hi, "ref", 0.0))
            ctrl = logits_of(ids, lambda: perturb_hook(L, lo, hi, "rand"))
            pl = torch.log_softmax(pert, -1)
            cl = torch.log_softmax(ctrl, -1)

            def kl_at(t, q):
                p = bl[t].exp()
                return float((p * (bl[t] - q[t])).sum())

            for t in range(lo, hi):
                rows["window_kl"].append(kl_at(t, pl))
                rows["window_kl_rand"].append(kl_at(t, cl))
                nxt = ids[t + 1]
                rows["window_dlp"].append(float(pl[t, nxt] - bl[t, nxt]))
            for k in range(HOR):
                t = hi + k
                if t + 1 >= len(ids):
                    break
                nxt = ids[t + 1]
                rows["kl"][k].append(kl_at(t, pl))
                rows["kl_rand"][k].append(kl_at(t, cl))
                rows["dlp"][k].append(float(pl[t, nxt] - bl[t, nxt]))

            # greedy divergence from the end of the window
            a_ids, b_ids = list(ids[:hi]), list(ids[:hi])
            first = None
            for step in range(GEN):
                la = logits_of(a_ids)
                lb = logits_of(b_ids, lambda: perturb_hook(L, lo, hi, "ref", 0.0))
                ta, tb = int(la[-1].argmax()), int(lb[-1].argmax())
                if ta != tb and first is None:
                    first = step
                a_ids.append(ta); b_ids.append(tb)
            rows["div"].append(GEN if first is None else first)

        gate = float(np.max(idcheck))
        wk = float(np.mean(rows["window_kl"]))
        wkr = float(np.mean(rows["window_kl_rand"]))
        wd = float(np.mean(rows["window_dlp"]))
        print(f"\n=== L{L} === identity gate (alpha=1) max |dlogit| {gate:.2e} "
              f"{'PASS' if gate < 1e-2 else '**FAIL**'}")
        print(f"  INSIDE the perturbed window: KL {wk:.4f} "
              f"(random-dir control {wkr:.4f}, ratio {wk/max(wkr,1e-9):.1f}x), "
              f"dlogP {wd:+.4f}")
        print(f"  AFTER the window (untouched positions):")
        print(f"    {'offset':>7} {'KL':>9} {'KL rand':>9} {'ratio':>7} "
              f"{'dlogP':>9} {'n':>5}")
        prof = {}
        for k in list(range(0, 8)) + [11, 15, 23, 31]:
            if k >= HOR or not rows["kl"][k]:
                continue
            kk = float(np.mean(rows["kl"][k]))
            kr = float(np.mean(rows["kl_rand"][k]))
            dd = float(np.mean(rows["dlp"][k]))
            prof[k] = dict(kl=round(kk, 5), kl_rand=round(kr, 5),
                           ratio=round(kk / max(kr, 1e-9), 2),
                           dlogp=round(dd, 5), n=len(rows["kl"][k]))
            print(f"    {k:>7} {kk:>9.5f} {kr:>9.5f} "
                  f"{kk/max(kr,1e-9):>7.2f} {dd:>+9.5f} "
                  f"{len(rows['kl'][k]):>5}")
        div = np.array(rows["div"])
        print(f"  GREEDY divergence within {GEN} steps: "
              f"{100*float((div < GEN).mean()):.0f}% of runs diverge, "
              f"median first-diff step "
              f"{float(np.median(div)):.1f}")
        # decay: KL at offset 0 vs the tail
        tail = [float(np.mean(rows["kl"][k])) for k in range(HOR)
                if rows["kl"][k]][-8:]
        res[f"L{L}"] = dict(identity_gate=gate, window_kl=round(wk, 5),
                            window_kl_rand=round(wkr, 5),
                            window_dlogp=round(wd, 5), horizon=prof,
                            tail_kl_mean=round(float(np.mean(tail)), 5),
                            frac_diverge=round(float((div < GEN).mean()), 3),
                            median_first_diff=float(np.median(div)))

    json.dump({"model": MODEL, "layers": LAYERS, "p0": P0, "win": WIN,
               "hor": HOR, "gen": GEN, "results": res},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: if KL after the window is at the random-dir control, the")
    print("reference's function is LOCAL and next-token scoring was adequate.")
    print("If it stays well above control across offsets, the b_L battery has")
    print("been under-measuring a forward-facing role.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
