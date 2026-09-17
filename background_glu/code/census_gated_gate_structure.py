# -*- coding: utf-8 -*-
"""Is the GATE branch of a modern SwiGLU MLP the amplitude-selective mechanism
that a plain GELU lacks? (Olli, 2026-09-09.)

BACKGROUND. In GELU/ReLU MLPs (Phi-2, Pythia) the paper's picture is: one
elementwise threshold, content rides it small-signal (2-6% of the distance to the
knee), and the "small semantic amplitude band" (15a: successor amplitudes span
only 2.6-3.6x across 3-4 orders of P(s|t)) is maintained PASSIVELY -- b_L sets the
zero, LayerNorm sets the scale, writes are small. The earlier two-rails probe
found the sigma/amplitude rail is NOT dynamically recruited in GELU (no explicit
multiplicative knob at the gate).

A SwiGLU MLP is different: out = down( SiLU(W_g x_ln) * (W_v x_ln) ). The product
is an EXPLICIT multiplicative channel. Hypothesis: the GATE branch (W_g) conditions
the value's content on an AMPLITUDE/register feature read from a direction distinct
from the content (W_v), actively passing an amplitude band -- the "second reference
level" recruited dynamically, which the elementwise GELU could only do statically
through LN.

DISCRIMINATORS (null = "the gate is just a self-threshold, band maintained the
passive GELU way"):
  LEG A  the band: does the 15a compression (2.6-3.6x) exist in a modern SwiGLU
         model at all? (context-free trigger residual, decoded through W_U, binned
         by the model's own P(s|t)).
  LEG B  gate vs value read GEOMETRY (weights + b_L): per neuron cos(w_g,w_v)
         (self-gating -> high; amplitude-conditioning -> low); do gate rows align
         with the amplitude directions (b_L, high-variance x_ln dims) MORE than
         value rows; the gate's resting openness SiLU(w_g.b_L).
  LEG C  gate ACTIVATION vs amplitude (forward pass): per-neuron corr(SiLU(gate),
         ||x_ln||) vs corr(value, ||x_ln||). If the gate opens/closes with
         amplitude MORE than the value does, that is the active band-maintenance.

Qwen2.5-3B (SwiGLU, RMSNorm), on-manifold natural text. RMSNorm has no mean
subtraction, so b_L is the mean of the RMS-normed MLP input (still well defined).
Usage: .venv/Scripts/python.exe superposition/code/census_gated_gate_structure.py
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
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-3B"
MAXLEN, SKIP = 256, 16
N_CAL, N_EVAL = 24, 40
LAYERS = [9, 18, 27]           # mid-stack of 36
HIGHVAR_K = 64                 # top-variance x_ln dims (the amplitude rail)
N_TRIG = 32                    # trigger tokens for the band leg
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gated-gate-structure.json")


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    d = m.config.hidden_size
    dff = m.config.intermediate_size
    WU = m.lm_head.weight.detach()
    WUn = (WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)).float()
    docs = json.load(open(PILE, encoding="utf-8"))
    cal_docs, ev_docs = docs[:N_CAL], docs[N_CAL:N_CAL + N_EVAL]
    print(f"  {MODEL}: {len(Lz)} layers, d={d}, d_ff={dff}, "
          f"vocab={WU.shape[0]}")

    # capture x_ln (= gate_proj input), gate pre-activation, value -----------
    cap = {}
    hooks = []
    for L in LAYERS:
        mlp = Lz[L].mlp
        hooks.append(mlp.gate_proj.register_forward_hook(
            (lambda L: (lambda mod, inp, out: cap.__setitem__(
                f"x{L}", inp[0][0].detach()) or cap.__setitem__(
                f"g{L}", out[0].detach())))(L)))
        hooks.append(mlp.up_proj.register_forward_hook(
            (lambda L: (lambda mod, inp, out: cap.__setitem__(
                f"v{L}", out[0].detach())))(L)))

    def run(ids):
        with torch.no_grad():
            return m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()

    # ---- calibration: b_L and per-coord variance of x_ln -------------------
    s1 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    s2 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    n = 0
    for text in cal_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        run(ids)
        for L in LAYERS:
            x = cap[f"x{L}"][SKIP:].double()
            s1[L] += x.sum(0)
            s2[L] += (x * x).sum(0)
        n += len(ids) - SKIP
    bL = {L: (s1[L] / n).float() for L in LAYERS}
    bLn = {L: (bL[L] / bL[L].norm().clamp(min=1e-9)) for L in LAYERS}
    hivar = {L: torch.topk((s2[L] / n - (s1[L] / n) ** 2).float(),
                           HIGHVAR_K).indices for L in LAYERS}
    print(f"  b_L + variance from {n} positions\n")

    res = {}

    # ===== LEG B: gate vs value read geometry (weights + b_L) ================
    print("=== LEG B: gate (W_g) vs value (W_v) read geometry ===")
    print(f"  {'L':>3} {'cos(wg,wv)med':>13} {'|cos(wg,bL)|':>12} "
          f"{'|cos(wv,bL)|':>12} {'g hivar-egy':>12} {'v hivar-egy':>12} "
          f"{'rest openSiLU':>13}")
    for L in LAYERS:
        Wg = Lz[L].mlp.gate_proj.weight.detach().float()      # [dff, d]
        Wv = Lz[L].mlp.up_proj.weight.detach().float()
        # per-neuron cos(w_g, w_v)
        cg = F.cosine_similarity(Wg, Wv, dim=1)
        # alignment with b_L
        cgb = (Wg @ bLn[L]) / Wg.norm(dim=1).clamp(min=1e-9)
        cvb = (Wv @ bLn[L]) / Wv.norm(dim=1).clamp(min=1e-9)
        # read energy on the high-variance (amplitude) coords of x_ln
        mask = torch.zeros(d, dtype=torch.bool, device=DEV)
        mask[hivar[L]] = True
        g_hi = ((Wg[:, mask] ** 2).sum(1) / (Wg ** 2).sum(1).clamp(min=1e-9))
        v_hi = ((Wv[:, mask] ** 2).sum(1) / (Wv ** 2).sum(1).clamp(min=1e-9))
        # resting gate openness: SiLU(w_g . b_L)
        rest_g = Wg @ bL[L]
        open_rest = F.silu(rest_g)
        e = {"cos_wg_wv_med": float(cg.median()),
             "cos_wg_wv_p90": float(cg.float().quantile(0.90)),
             "abs_cos_wg_bL_med": float(cgb.abs().median()),
             "abscos_wv_bL_med": float(cvb.abs().median()),
             "g_hivar_energy_med": float(g_hi.median()),
             "v_hivar_energy_med": float(v_hi.median()),
             "hivar_iso": HIGHVAR_K / d,
             "rest_gpre_med": float(rest_g.median()),
             "rest_open_silu_med": float(open_rest.median()),
             "frac_gate_mostly_open": float((rest_g > 0).float().mean())}
        res[f"L{L}"] = {"legB": e}
        print(f"  {L:>3} {e['cos_wg_wv_med']:>13.3f} {e['abs_cos_wg_bL_med']:>12.3f} "
              f"{e['abscos_wv_bL_med']:>12.3f} {e['g_hivar_energy_med']:>12.4f} "
              f"{e['v_hivar_energy_med']:>12.4f} {e['rest_open_silu_med']:>13.4f}")
    print(f"  (hivar isotropic baseline = {HIGHVAR_K/d:.4f}; gate>>value hivar "
          f"=> gate reads the amplitude rail. low cos(wg,wv) => not self-gating.)")

    # ===== LEG C: gate activation vs per-position amplitude ==================
    print("\n=== LEG C: corr(activation, ||x_ln||) over positions ===")
    acc = {L: {k: torch.zeros(dff, device=DEV, dtype=torch.float64)
               for k in ("sg", "sg2", "sgA", "sv", "sv2", "svA",
                         "so", "so2", "soA")} for L in LAYERS}
    sA = {L: 0.0 for L in LAYERS}
    sA2 = {L: 0.0 for L in LAYERS}
    NA = {L: 0 for L in LAYERS}
    for text in ev_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        run(ids)
        for L in LAYERS:
            x = cap[f"x{L}"][SKIP:].float()
            A = x.norm(dim=-1)                                # [P]
            g = F.silu(cap[f"g{L}"][SKIP:].float())           # gate activation
            v = cap[f"v{L}"][SKIP:].float()                   # value
            o = (g * v).abs()                                 # gate-output magnitude
            a = acc[L]
            a["sg"] += g.sum(0).double(); a["sg2"] += (g * g).sum(0).double()
            a["sgA"] += (g * A[:, None]).sum(0).double()
            a["sv"] += v.sum(0).double(); a["sv2"] += (v * v).sum(0).double()
            a["svA"] += (v * A[:, None]).sum(0).double()
            a["so"] += o.sum(0).double(); a["so2"] += (o * o).sum(0).double()
            a["soA"] += (o * A[:, None]).sum(0).double()
            sA[L] += float(A.sum()); sA2[L] += float((A * A).sum())
            NA[L] += A.shape[0]

    def corr_vec(sxy, sx, sy, sx2, sy2, N):
        num = N * sxy - sx * sy
        den = ((N * sx2 - sx * sx).clamp(min=1e-9)
               * (N * sy2 - sy * sy)).clamp(min=1e-9).sqrt()
        return num / den

    print(f"  {'L':>3} {'mean|corr(gate,A)|':>17} {'mean|corr(val,A)|':>17} "
          f"{'mean corr(gate,A)':>17} {'mean corr(out,A)':>16} {'ratio g/v':>10}")
    for L in LAYERS:
        a = acc[L]; N = NA[L]
        cg = corr_vec(a["sgA"], a["sg"], sA[L], a["sg2"], sA2[L], N).cpu().numpy()
        cv = corr_vec(a["svA"], a["sv"], sA[L], a["sv2"], sA2[L], N).cpu().numpy()
        co = corr_vec(a["soA"], a["so"], sA[L], a["so2"], sA2[L], N).cpu().numpy()
        cg = cg[np.isfinite(cg)]; cv = cv[np.isfinite(cv)]; co = co[np.isfinite(co)]
        e = {"mean_abscorr_gate_A": float(np.mean(np.abs(cg))),
             "mean_abscorr_val_A": float(np.mean(np.abs(cv))),
             "mean_corr_gate_A": float(np.mean(cg)),
             "mean_corr_out_A": float(np.mean(co)),
             "gate_over_val_abscorr": float(np.mean(np.abs(cg))
                                            / max(np.mean(np.abs(cv)), 1e-9))}
        res[f"L{L}"]["legC"] = e
        print(f"  {L:>3} {e['mean_abscorr_gate_A']:>17.3f} "
              f"{e['mean_abscorr_val_A']:>17.3f} {e['mean_corr_gate_A']:>17.3f} "
              f"{e['mean_corr_out_A']:>16.3f} {e['gate_over_val_abscorr']:>10.2f}")
    print("  gate/v ratio > 1 => the GATE is more amplitude-coupled than the "
          "value (active band-maintenance); ~1 => amplitude-blind gate.")

    # ===== LEG A: the semantic amplitude band (15a, context-free) ===========
    # trigger tokens = most frequent word-initial tokens in the eval docs
    freq = torch.zeros(WU.shape[0], dtype=torch.long)
    for text in ev_docs:
        for i in tok(text, truncation=True, max_length=MAXLEN)["input_ids"]:
            freq[i] += 1
    order = torch.argsort(freq, descending=True)
    trig = []
    for i in order.tolist():
        s = tok.decode([i])
        if s[:1] == " " and s[1:2].isalpha() and len(s) >= 3:
            trig.append(i)
        if len(trig) >= N_TRIG:
            break
    print(f"\n=== LEG A: semantic amplitude band (15a), {len(trig)} triggers ===")
    print(f"  {'L':>3} {'d10/d1':>8} {'spearman(amp,logP)':>19} "
          f"{'d1..d10 (mean|a|)':>22}")
    for L in LAYERS:
        d10d1, sps, decile_prof = [], [], None
        prof_acc = np.zeros(10)
        prof_n = 0
        for t in trig:
            lg = run([t])                                     # context-free [t]
            P = F.softmax(lg[0], -1)                          # P(s|t)
            x = cap[f"x{L}"][0].float()                       # residual at t
            amp = ((x - bL[L]) @ WUn.T)                       # a_s over vocab
            # restrict to the support the model actually predicts
            topP, topI = torch.topk(P, 2000)
            a_s = amp[topI].abs().cpu().numpy()
            p_s = topP.cpu().numpy()
            # decile by P(s|t)
            oq = np.argsort(p_s)
            dec = np.array_split(oq, 10)
            means = np.array([a_s[dch].mean() for dch in dec])  # d1(rare)..d10
            prof_acc += means; prof_n += 1
            d10d1.append(means[-1] / max(means[0], 1e-9))
            lp = np.log10(p_s + 1e-12)
            sps.append(float(np.corrcoef(
                np.argsort(np.argsort(a_s)),
                np.argsort(np.argsort(lp)))[0, 1]))
        prof = prof_acc / prof_n
        e = {"d10_over_d1_med": float(np.median(d10d1)),
             "spearman_amp_logP_mean": float(np.mean(sps)),
             "decile_profile": prof.tolist()}
        res[f"L{L}"]["legA"] = e
        print(f"  {L:>3} {e['d10_over_d1_med']:>8.2f} "
              f"{e['spearman_amp_logP_mean']:>19.3f}   "
              f"{prof[0]:.4f}..{prof[-1]:.4f}")
    print("  Pythia/Phi-2 reference: d10/d1 ~ 2.6-3.6x (compressed band). "
          "Similar here => the band exists in a modern SwiGLU model.")

    for h in hooks:
        h.remove()
    json.dump({"model": MODEL, "layers": LAYERS, "skip": SKIP,
               "n_cal": n, "n_triggers": len(trig), "hivar_k": HIGHVAR_K,
               "results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: Leg A = is the band there. Legs B+C = is the GATE the "
          "amplitude-\nselective mechanism (low cos(wg,wv) + gate reads/tracks "
          "amplitude more than\nthe value) or just a passive self-threshold.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
