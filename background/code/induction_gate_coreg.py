"""INDUCTION vs GATE FORMATION -- co-registered on the dense 160M ladder grid.

Question (Olli 2026-09-11). The background paper's Formation section co-times induction,
gate sparsification (duty drop) and b_L reference-coupling formation, but Pythia has no
checkpoint between 512 and 1000, so it cannot resolve the onset finer. The local 160M
matched ladder DOES have a dense early grid (steps 50..12000 straddle the transition), so
we can (a) get a finer timing picture and (b) test lead/lag between induction and gate
formation.

The causal handle without a training run is the ARM CONTRAST. Per Olli's correction and
background_glu: b_L FORMS in every arm; what is gate-specific is the coupling's RELOCATION
into a gate branch. The one arm whose gate cannot build a real operating point is
BILINEAR -- an even function (product of two linear maps), no off-state, no threshold to
rest below. It is therefore the natural "gate-formation-suppressed" arm. So:
  - "does induction drive gate suppression?"  -> within-arm lead/lag (does duty drop AFTER
    induction rises?) + whether duty-drop timing is locked across arms or arm-dependent.
  - "does suppressing gate formation stop induction?" -> does bilinear (impaired gate) still
    form induction on the same schedule as gelu/relu2/swiglu?

Prior (session_report_2026-09-08_induction_bump_vs_bL): induction is architecture-agnostic
(same across arms); b_L frame relocation is gate-specific; verdict "co-timed but
mechanistically independent -- do not upgrade to linked without a manipulation." That report
flagged this finer-grid run as unrun-and-free.

Two signals per (arm, step), each one forward pass:
  induction   loss(first copy) - loss(exact repeat) on RANDOM tokens (phase_boundary measure);
              high => the model copies from context. Off-manifold by design; read TIMING only.
  gate        on the FIRST (control) branch of each arm's MLP, over mid-stack layers:
                duty      = mean(pre-activation > 0)      sparsification (LOW = gate formed)
                z         = -mu_g / sd(g)                 operating-point depth, in sd of drive
                frac_rest = |mu_x . w| / |mu_g|           share of the operating point carried
                                                          by the mean-input (b_L) coupling, not bias

CPU-only (DEVICE=cpu default; GPU busy). Resumable (skips done arm/step). Writes
minout/data/induction-gate-coreg.json, then prints per-arm onset steps and lead/lag.

Usage:
  ./.venv/Scripts/python.exe minout/code/induction_gate_coreg.py
  SUMMARY=1  -> only recompute the onset/lead-lag summary from the existing JSON
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DEVICE", "cpu")            # GPU is busy; force CPU unless overridden
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_probe import load_snapshot, eval_batch, ARMS, DEV, empty_cache   # noqa: E402

# dense grid where all four arms have checkpoints; straddles the induction transition (~800-12000)
STEPS = [int(s) for s in os.environ.get(
    "STEPS", "0,50,100,200,400,800,1500,3000,5000,8000,12000,20000,32000,50000").split(",")]
CORE_ARMS = os.environ.get("ARMS", "gelu,relu2,swiglu,bilinear").split(",")
OUT = os.environ.get("OUT", "minout/data/induction-gate-coreg.json")
NIND = int(os.environ.get("NIND", "32"))          # random sequences for the induction measure
NGATE = int(os.environ.get("NGATE", "12"))        # natural sequences for the gate capture
MID = [int(s) for s in os.environ.get("MID", "4,5,6,7,8").split(",")]   # mid-stack layers (of 12)
HALF, VOCAB = 128, 50000

# first (control) branch of each arm kind -> weight attribute on layer.mlp
GATE_ATTR = {"standard": "dense_h_to_4h", "relu2": "w_in", "swiglu": "w_gate", "bilinear": "w_a"}


@torch.no_grad()
def induction(model, gen):
    """loss on a block of random tokens, then on an exact repeat of that block."""
    ids = torch.randint(0, VOCAB, (NIND, HALF), generator=gen)
    seq = torch.cat([ids, ids], 1).to(DEV)
    first, second = [], []
    for i in range(0, NIND, 8):
        b = seq[i:i + 8]
        lg = model(b[:, :-1]).logits.float()
        l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1),
                            reduction="none").reshape(b.shape[0], -1)
        first.append(l[:, 8:HALF - 1].mean(1).cpu())     # skip the first few, no context yet
        second.append(l[:, HALF + 8:].mean(1).cpu())
    f, s = float(torch.cat(first).mean()), float(torch.cat(second).mean())
    return round(f, 4), round(s, 4), round(f - s, 4)


@torch.no_grad()
def gate_profile(model, kind, ids):
    """Per layer, on the first branch: median duty, operating-point depth z, and the share of
    the operating point carried by the mean-input (b_L) coupling rather than the bias."""
    attr = GATE_ATTR[kind]
    layers = model.gpt_neox.layers
    X = {}

    def cap(L):
        def fn(mod, inp):
            X.setdefault(L, []).append(
                inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
        return fn

    hks = [lay.mlp.register_forward_pre_hook(cap(L)) for L, lay in enumerate(layers)]
    for i in range(0, ids.shape[0], 2):
        model(ids[i:i + 2])
    for h in hks:
        h.remove()

    per_layer = {}
    for L, chunks in X.items():
        x = torch.cat(chunks).to(DEV)                    # (T, d)
        mu = x.mean(0)
        lin = getattr(layers[L].mlp, attr)
        W = lin.weight.detach().float()                  # (n_units, d)
        b = (lin.bias.detach().float() if lin.bias is not None
             else torch.zeros(W.shape[0], device=DEV))
        g = x @ W.T + b                                  # (T, n)
        mu_g = mu @ W.T + b                              # (n,)
        sd = g.std(0).clamp_min(1e-9)
        duty = (g > 0).float().mean(0)
        z = -mu_g / sd
        off_rest = mu @ W.T                              # mean-input projection = the b_L coupling
        frac_rest = off_rest.abs() / mu_g.abs().clamp_min(1e-9)
        per_layer[str(L)] = {
            "duty": round(float(duty.median()), 4),
            "z": round(float(z.median()), 4),
            "mu_g": round(float(mu_g.median()), 4),
            "frac_rest": round(float(frac_rest.median()), 4)}
        del x
        empty_cache()
    # mid-stack aggregate (median over MID layers)
    def agg(key):
        vals = [per_layer[str(L)][key] for L in MID if str(L) in per_layer]
        return round(float(np.median(vals)), 4) if vals else None
    mid = {k: agg(k) for k in ("duty", "z", "mu_g", "frac_rest")}
    return {"mid": mid, "per_layer": per_layer}


def run():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    ids = eval_batch(NGATE)
    for step in STEPS:
        for arm in CORE_ARMS:
            p = "minout/data/ladder/{}_step{}.pt".format(arm, step)
            if not os.path.exists(p):
                continue
            if str(step) in res.get(arm, {}):
                continue
            model, a, st = load_snapshot(p)
            gen = torch.Generator().manual_seed(7)       # same random block for every arm/step
            f, s, ind = induction(model, gen)
            gp = gate_profile(model, ARMS[arm]["kind"], ids)
            m = gp["mid"]
            res.setdefault(arm, {})[str(st)] = {
                "induction": ind, "ind_first": f, "ind_repeat": s,
                "duty": m["duty"], "z": m["z"], "mu_g": m["mu_g"], "frac_rest": m["frac_rest"],
                "per_layer": gp["per_layer"]}
            print("[{:>8s} {:>6d}]  induction {:+.4f}   mid: duty {:.3f}  z {:+.3f}  "
                  "frac_rest {:.3f}".format(arm, st, ind, m["duty"], m["z"], m["frac_rest"]),
                  flush=True)
            json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
            del model
            empty_cache()
    return res


def onset_step(steps, vals, rising):
    """Step (log-interpolated) at which a normalised curve first crosses 0.5.

    rising=True for induction (climbs); rising=False for duty (drops). Baseline = value at the
    smallest step, plateau = the extreme reached over the series; returns None if the swing is
    too small to have a defined onset (< 0.15 of a nat / of the duty range)."""
    steps = np.asarray(steps, dtype=float)
    vals = np.asarray(vals, dtype=float)
    ok = steps > 0
    steps, vals = steps[ok], vals[ok]
    if steps.size < 3:
        return None
    base = vals[0]
    ext = vals.max() if rising else vals.min()
    swing = ext - base if rising else base - ext
    if abs(swing) < 0.15:
        return None
    norm = (vals - base) / (ext - base) if rising else (base - vals) / (base - ext)
    lx = np.log10(steps)
    # first index where norm >= 0.5, then linear-interpolate in log-step
    idx = np.argmax(norm >= 0.5)
    if norm[idx] < 0.5:
        return None
    if idx == 0:
        return float(10 ** lx[0])
    x0, x1, y0, y1 = lx[idx - 1], lx[idx], norm[idx - 1], norm[idx]
    xc = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    return float(10 ** xc)


def summary(res):
    print("\n=== onset timing & lead/lag (mid-stack) ===", flush=True)
    print("  induction50 = step where induction reaches 50% of its rise")
    print("  duty50/z50  = step where duty-drop / operating-point-deepening reaches 50%")
    print("  lag = duty50 - induction50  (>0: sparsification LAGS induction = induction-first)\n")
    hdr = "{:>9s} {:>12s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "arm", "induction50", "duty50", "z50", "duty_lag", "z_lag")
    print(hdr)
    print("-" * len(hdr))
    for arm in CORE_ARMS:
        d = res.get(arm, {})
        steps = sorted(int(s) for s in d)
        if len(steps) < 3:
            continue
        ind = [d[str(s)]["induction"] for s in steps]
        duty = [d[str(s)]["duty"] for s in steps]
        z = [d[str(s)]["z"] for s in steps]
        i50 = onset_step(steps, ind, rising=True)
        d50 = onset_step(steps, duty, rising=False)
        z50 = onset_step(steps, z, rising=True)
        dl = (d50 - i50) if (d50 and i50) else None
        zl = (z50 - i50) if (z50 and i50) else None
        def fmt(v):
            return "{:>10.0f}".format(v) if v is not None else "{:>10s}".format("--")
        print("{:>9s} {:>12s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            arm, fmt(i50).strip().rjust(12), fmt(d50).strip().rjust(10),
            fmt(z50).strip().rjust(10),
            ("{:+.0f}".format(dl) if dl is not None else "--").rjust(10),
            ("{:+.0f}".format(zl) if zl is not None else "--").rjust(10)))
    # bilinear = the natural gate-suppressed arm: compare its induction curve to the others
    print("\n=== bilinear as the gate-suppressed control ===", flush=True)
    steps_all = sorted({int(s) for arm in CORE_ARMS for s in res.get(arm, {})})
    print("{:>7s}".format("step") + "".join("{:>10s}".format(a) for a in CORE_ARMS)
          + "   <- induction")
    for s in steps_all:
        row = "{:>7d}".format(s)
        for a in CORE_ARMS:
            v = res.get(a, {}).get(str(s), {}).get("induction")
            row += "{:>10}".format("{:+.3f}".format(v) if v is not None else "--")
        print(row)
    print("\n{:>7s}".format("step") + "".join("{:>10s}".format(a) for a in CORE_ARMS)
          + "   <- mid duty (low = gate formed)")
    for s in steps_all:
        row = "{:>7d}".format(s)
        for a in CORE_ARMS:
            v = res.get(a, {}).get(str(s), {}).get("duty")
            row += "{:>10}".format("{:.3f}".format(v) if v is not None else "--")
        print(row)


def main():
    if bool(int(os.environ.get("SUMMARY", "0"))):
        res = json.load(open(OUT, encoding="utf-8"))
    else:
        print("device={}  steps={}  arms={}".format(DEV, STEPS, CORE_ARMS), flush=True)
        res = run()
    summary(res)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
