"""The visibility threshold across scale (plan_mid_stack_empirical_basis.md
3ag, 3af's untested corollary made decisive, pre-registered 2026-07-17).

3af fixed f* ~ 0.9% at one architecture (Phi-2, d=2560) and left the
d-dependence untested. The closed form f* = 2 ln(2V) / (d + 2 ln(2V)) is
LINEAR in 1/d for d >> ln V (the 07-17 correction: sqrt(2 ln 2V)/sqrt(d)
is the AMPLITUDE bar, energy is amplitude squared). Across Pythia, V is
fixed and d spans 5x, so f* moves 4.8x -- a sharp, weights-only test.

Per model: plant a single W_U row at aligned-energy fraction f into an
isotropic bath, ask whether it lands in the top-15. f_50 = the f at which
hit-rate crosses 0.5 (log-linear interpolation). Then regress log(f_50)
on log(d).

  slope ~ -1.0  -> LINEAR: extreme-value account confirmed across scale,
                   the 07-17 exponent correction validated.
  slope ~ -0.5  -> SQRT: the correction AND the closed form are wrong.
  slope ~  0    -> FLAT: the max-statistic model does not govern; 3af's
                   Phi-2 agreement was a one-architecture coincidence.

Gates: (1) the d=2560 point is Phi-2 and must reproduce 3af's published
curve (0.010/0.140/0.900 at f=0.001/0.003/0.010) or the harness is broken;
(2) the f=0 bath max must sit at ~4.4-4.8 sigma at EVERY d (it depends on
V, not d) -- per the pre-registration, a model whose bath drifts above
the prediction violates the isotropy assumption and is reported but
EXCLUDED from the regression, with the exclusion named; (3) at f=0.3
hit-rate must be ~1.0 for every model.

Only W_U is needed -- the model is loaded on CPU, the unembedding is
lifted out, and the rest is freed immediately.

Usage: .venv/Scripts/python.exe \
           superposition/code/census_visibility_threshold_scale.py
       SMOKE=1 fast pass (2 models, 40 samples).
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
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

# (name, expected d) -- d recorded for the pre-registered table, verified
# against the loaded weights at runtime.
MODELS = [
    ("EleutherAI/pythia-70m-deduped", 512),
    ("EleutherAI/pythia-160m-deduped", 768),
    ("EleutherAI/pythia-410m-deduped", 1024),
    ("EleutherAI/pythia-1b-deduped", 2048),
    ("microsoft/phi-2", 2560),
]
if SMOKE:
    MODELS = [MODELS[0], MODELS[-1]]

# Pre-registered f grid: log-spaced, brackets f_50 for every model from
# 4.31% (d=512) down to 0.89% (d=2560) without extrapolation at either end.
FS = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3]
N_SAMP = 40 if SMOKE else 200
TOPK = 15
N_BATH = 100

# 3af's published Phi-2 hit-rates -- the reproduction gate.
AF_PUBLISHED = {0.001: 0.010, 0.003: 0.140, 0.01: 0.900}
AF_TOL = 0.12

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "census-visibility-threshold-scale.json")


def get_unembed(name):
    """Load on CPU, lift out W_U and the effective vocab, free the rest."""
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, low_cpu_mem_usage=True)
    head = getattr(m, "embed_out", None) or getattr(m, "lm_head", None)
    if head is None:
        raise RuntimeError(f"no unembedding found on {name}")
    WU = head.weight.detach().float().clone()
    # Pythia's embed_out is padded past the tokenizer (50304 vs 50277); the
    # padding rows never receive gradient. Pre-registered: slice them off.
    v_full = WU.shape[0]
    v_eff = min(len(tok), v_full)
    WU = WU[:v_eff]
    del m, head, tok
    gc.collect()
    return WU.to(DEV), v_full, v_eff


def f50_interp(fs, hits):
    """f at which the hit-rate curve crosses 0.5, log-linear interpolation.

    Returns None if the curve never crosses (which is itself a result --
    it means the grid failed to bracket, and the pre-registration says
    that must not be papered over).
    """
    for i in range(len(fs) - 1):
        h0, h1 = hits[i], hits[i + 1]
        if h0 < 0.5 <= h1:
            l0, l1 = np.log(fs[i]), np.log(fs[i + 1])
            t = (0.5 - h0) / (h1 - h0) if h1 != h0 else 0.0
            return float(np.exp(l0 + t * (l1 - l0)))
    return None


@torch.no_grad()
def sweep_model(name, d_expect, rng):
    t0 = time.time()
    WU, v_full, v_eff = get_unembed(name)
    V, d = WU.shape
    WUn = WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
    f_star = float(2 * np.log(2 * V) / (d + 2 * np.log(2 * V)))
    sig_pred = float(np.sqrt(2 * np.log(2 * V)))
    print(f"\n=== {name}  d={d} (expect {d_expect})  "
          f"V={v_eff} (padded {v_full})")
    print(f"    analytic f* = {f_star*100:.2f}%   "
          f"bath ceiling pred = {sig_pred:.2f} sigma")
    if d != d_expect:
        print(f"    !! d mismatch: got {d}, pre-registered {d_expect}")

    def rand_unit():
        v = torch.from_numpy(rng.randn(d).astype(np.float32)).to(DEV)
        return v / v.norm()

    # Gate 2: bath max at f=0 (depends on V, not d)
    bath = []
    for _ in range(N_BATH):
        lg = WUn @ rand_unit()
        bath.append(float(lg.abs().max() / lg.std()))
    bath_med = float(np.median(bath))
    print(f"    gate bath max = {bath_med:.2f} sigma "
          f"(pred {sig_pred:.2f})")

    # content-word pool, same construction as 3af
    pool = []
    for t in range(0, V, 7):
        if len(pool) >= 4000:
            break
        pool.append(t)
    pool = np.array(pool)

    curve = {}
    for f in FS:
        hits = []
        for _ in range(N_SAMP):
            tid = int(rng.choice(pool))
            delta = float(np.sqrt(1 - f)) * rand_unit() \
                + float(np.sqrt(f)) * WUn[tid]
            lg = (WUn @ delta).abs()
            top = torch.topk(lg, TOPK).indices
            hits.append(1.0 if tid in top else 0.0)
        curve[f] = float(np.mean(hits))
        print(f"      f={f:<7g} hit={curve[f]:.3f}")

    f50 = f50_interp(FS, [curve[f] for f in FS])
    print(f"    f_50 = {f50*100:.2f}%" if f50 else "    f_50 = NO CROSSING")
    print(f"    ratio f_50/f* = {f50/f_star:.2f}" if f50 else "")

    del WU, WUn
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "model": name, "d": d, "V_eff": v_eff, "V_padded": v_full,
        "f_star_analytic": round(f_star, 5),
        "bath_max_sigma_pred": round(sig_pred, 3),
        "bath_max_sigma_measured": round(bath_med, 3),
        "curve": {str(k): round(v, 4) for k, v in curve.items()},
        "f_50": round(f50, 5) if f50 else None,
        "f50_over_fstar": round(f50 / f_star, 3) if f50 else None,
        "secs": round(time.time() - t0, 1),
    }


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  n/f={N_SAMP}  models={len(MODELS)}")
    rng = np.random.RandomState(SEED)
    rec = {"config": {"smoke": SMOKE, "fs": FS, "n_samp": N_SAMP,
                      "topk": TOPK, "seed": SEED,
                      "models": [m for m, _ in MODELS]},
           "per_model": []}

    for name, d_exp in MODELS:
        try:
            rec["per_model"].append(sweep_model(name, d_exp, rng))
        except Exception as e:
            print(f"    !! FAILED {name}: {e}")
            rec["per_model"].append({"model": name, "error": str(e)})

    ok = [r for r in rec["per_model"] if r.get("f_50")]

    # Pre-registered bath-ceiling exclusion: the ceiling depends on V, not
    # d, so a model whose f=0 bath max drifts far above the analytic
    # prediction violates the isotropy assumption -- its point is reported
    # but EXCLUDED from the regression, with the exclusion named.
    BATH_EXCESS_TOL = 0.5
    for r in rec["per_model"]:
        if "bath_max_sigma_measured" in r:
            r["bath_gate_pass"] = bool(
                r["bath_max_sigma_measured"]
                <= r["bath_max_sigma_pred"] + BATH_EXCESS_TOL)
    iso = [r for r in ok if r.get("bath_gate_pass")]
    excluded = [r["model"] for r in rec["per_model"]
                if "bath_max_sigma_measured" in r
                and not r.get("bath_gate_pass")]

    # ---- Gates ----
    gates = {}
    phi = next((r for r in rec["per_model"]
                if r["model"] == "microsoft/phi-2"), None)
    if phi and "curve" in phi:
        dev = {str(f): round(phi["curve"][str(f)] - h, 3)
               for f, h in AF_PUBLISHED.items() if str(f) in phi["curve"]}
        gates["reproduction_3af"] = {
            "published": {str(k): v for k, v in AF_PUBLISHED.items()},
            "measured": {str(f): phi["curve"][str(f)]
                         for f in AF_PUBLISHED if str(f) in phi["curve"]},
            "deviation": dev,
            "tol": AF_TOL,
            "pass": all(abs(v) <= AF_TOL for v in dev.values())}
    gates["bath_ceiling_constant"] = {
        "measured": {r["model"]: r["bath_max_sigma_measured"]
                     for r in rec["per_model"]
                     if "bath_max_sigma_measured" in r},
        "tol_excess": BATH_EXCESS_TOL,
        "excluded_by_gate": excluded,
        "pass_all": not excluded,
        "spread_iso": round(max(r["bath_max_sigma_measured"] for r in iso)
                            - min(r["bath_max_sigma_measured"] for r in iso),
                            3) if iso else None}
    gates["easy_end"] = {
        "measured": {r["model"]: r["curve"]["0.3"] for r in ok
                     if "0.3" in r["curve"]},
        "pass_iso": all(r["curve"].get("0.3", 0) >= 0.95 for r in iso),
        "pass_all": all(r["curve"].get("0.3", 0) >= 0.95 for r in ok)}
    rec["gates"] = gates

    # ---- The test: regress log(f_50) on log(d), isotropic subset ----
    if len(iso) >= 3:
        ld = np.log([r["d"] for r in iso])
        lf = np.log([r["f_50"] for r in iso])
        slope, icept = np.polyfit(ld, lf, 1)
        pred = icept + slope * ld
        ss_res = float(((lf - pred) ** 2).sum())
        ss_tot = float(((lf - lf.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # also: does f_50 track the analytic f* itself?
        ratios = [r["f50_over_fstar"] for r in iso]
        if abs(slope - (-1.0)) < 0.25:
            verdict = "LINEAR"
        elif abs(slope - (-0.5)) < 0.15:
            verdict = "SQRT"
        elif abs(slope) < 0.2:
            verdict = "FLAT"
        else:
            verdict = "NEITHER"
        rec["regression"] = {
            "subset": "bath-gate-passing (isotropic) models only",
            "excluded_by_bath_gate": excluded,
            "slope_log_f50_vs_log_d": round(float(slope), 3),
            "r2": round(float(r2), 4),
            "n_points": len(iso),
            "f50_over_fstar": {r["model"]: r["f50_over_fstar"] for r in iso},
            "f50_over_fstar_spread": round(max(ratios) - min(ratios), 3),
            "verdict": verdict,
            "branches": {"LINEAR": "slope ~ -1: 07-17 correction validated",
                         "SQRT": "slope ~ -0.5: correction AND closed form wrong",
                         "FLAT": "slope ~ 0: max-statistic model does not govern"}}

        print("\n" + "=" * 64)
        print(f"{'model':<32} {'d':>5} {'f* an.':>8} {'f_50':>8} {'ratio':>6}")
        for r in ok:
            tag = "" if r.get("bath_gate_pass") else "  EXCLUDED (bath)"
            print(f"{r['model']:<32} {r['d']:>5} "
                  f"{r['f_star_analytic']*100:>7.2f}% "
                  f"{r['f_50']*100:>7.2f}% {r['f50_over_fstar']:>6.2f}{tag}")
        print("=" * 64)
        if excluded:
            print(f"excluded from regression by bath gate: "
                  f"{', '.join(excluded)}")
        print(f"slope log(f_50) ~ log(d) [n={len(iso)}] = {slope:.3f}   "
              f"R2 = {r2:.4f}")
        print(f"  predicted: -1.0 (LINEAR) | -0.5 (SQRT) | 0.0 (FLAT)")
        print(f"VERDICT: {verdict}")
        print(f"f_50/f* spread across {len(iso)} isotropic models: "
              f"{max(ratios)-min(ratios):.3f}")
        print("\nGATES:")
        for g, v in gates.items():
            status = v.get("pass", v.get("pass_iso", v.get("pass_all")))
            print(f"  {g}: {'PASS' if status else 'FAIL'} "
                  f"({ {k: w for k, w in v.items() if k.startswith('pass')} })")

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
