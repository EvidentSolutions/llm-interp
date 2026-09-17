# -*- coding: utf-8 -*-
"""CONTROL B for background_glu (2026-09-10): is the VALUE (up_proj) branch the
CONTENT carrier and the GATE the operating-point / threshold setter?

The geometry (census_gated_gate_structure) shows value rows are orthogonal to
b_L. But "orthogonal to the reference" is the ABSENCE of the coupling, not
positive evidence the value carries content. This control freezes each branch at
its b_L-resting constant (input-independent) at a single site and asks WHICH
KIND of damage each does:
    - freeze GATE  at SiLU-resting (out: const_gate * (W_v x))  -> value content
      still flows; only the gate's conditioning is removed.
    - freeze VALUE at resting       (out: SiLU(W_g x) * const_val) -> the gate's
      conditioning still flows; the value's token-specific content is removed.

Metrics per position (clean vs frozen final logits):
    KL(clean||frozen)     total divergence
    top1_flip             fraction where argmax changes (content scramble)
    dH                    entropy(frozen) - entropy(clean) (operating-point/
                          confidence shift pushes this up)
    dNLL_true             NLL(frozen) - NLL(clean) on the actual next token

PREDICTION if value=content / gate=threshold: freezing the VALUE flips top-1 and
raises dNLL_true MORE (it scrambles which token), while freezing the GATE behaves
more like a confidence/operating-point change (dH up, fewer flips). If the two
are COMPARABLE, the clean dichotomy is too strong and the note must say so.

CPU, float32. Qwen2.5-3B (SwiGLU/RMSNorm). Single-site per intervention.
Usage: .venv/Scripts/python.exe superposition/code/census_gated_value_content.py
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

DEV = "cpu"
DTYPE = torch.float32
MODEL = "Qwen/Qwen2.5-3B"
MAXLEN, SKIP = 160, 16
N_CAL, N_EVAL = 20, 12
LAYERS = [9, 18, 27]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gated-value-content.json")


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    Lz, d = m.model.layers, m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))
    cal_docs, ev_docs = docs[:N_CAL], docs[N_CAL:N_CAL + N_EVAL]
    print(f"  {MODEL} on {DEV}: {len(Lz)} layers, d={d}", flush=True)

    cap = {}
    xhooks = [Lz[L].mlp.gate_proj.register_forward_hook(
        (lambda L: (lambda mod, inp, out: cap.__setitem__(
            L, inp[0][0].detach())))(L)) for L in LAYERS]

    def run(ids):
        with torch.no_grad():
            return m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()

    # ---- calibration: b_L and resting constants ----------------------------
    s1 = {L: torch.zeros(d, dtype=torch.float64) for L in LAYERS}
    n = 0
    for k, text in enumerate(cal_docs):
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        run(ids)
        for L in LAYERS:
            s1[L] += cap[L][SKIP:].double().sum(0)
        n += len(ids) - SKIP
    bL = {L: (s1[L] / n).float() for L in LAYERS}
    gate_rest = {L: (Lz[L].mlp.gate_proj.weight.detach().float() @ bL[L])
                 for L in LAYERS}
    val_rest = {L: (Lz[L].mlp.up_proj.weight.detach().float() @ bL[L])
                for L in LAYERS}
    print(f"  b_L + resting constants from {n} positions "
          f"({time.time()-t0:.0f}s)\n", flush=True)

    def freeze_hook(const):
        def hook(mod, inp, out):
            o = out.clone()
            o[0, SKIP:] = const.to(o.dtype)
            return o
        return hook

    CONDS = ["gate", "value"]
    agg = {L: {c: {"kl": [], "flip": [], "dH": [], "dnll": []}
               for c in CONDS} for L in LAYERS}
    for di, text in enumerate(ev_docs):
        td = time.time()
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        lg0 = run(ids)
        lp0 = torch.log_softmax(lg0, -1)
        H0 = -(lp0.exp() * lp0).sum(-1)                      # [T]
        nxt = torch.tensor(ids[1:] + [ids[-1]])              # next-token ids
        for L in LAYERS:
            for c in CONDS:
                proj = Lz[L].mlp.gate_proj if c == "gate" else Lz[L].mlp.up_proj
                const = gate_rest[L] if c == "gate" else val_rest[L]
                h = proj.register_forward_hook(freeze_hook(const))
                lp1 = torch.log_softmax(run(ids), -1)
                h.remove()
                sl = slice(SKIP, lp0.shape[0] - 1)           # scorable positions
                p0 = lp0[sl].exp()
                kl = (p0 * (lp0[sl] - lp1[sl])).sum(-1)
                flip = (lp0[sl].argmax(-1) != lp1[sl].argmax(-1)).float()
                H1 = -(lp1[sl].exp() * lp1[sl]).sum(-1)
                dH = H1 - H0[sl]
                tn = nxt[sl]
                dnll = -lp1[sl].gather(-1, tn[:, None]).squeeze(-1) \
                    + lp0[sl].gather(-1, tn[:, None]).squeeze(-1)
                a = agg[L][c]
                a["kl"].extend(kl.tolist())
                a["flip"].extend(flip.tolist())
                a["dH"].extend(dH.tolist())
                a["dnll"].extend(dnll.tolist())
        print(f"    eval {di+1}/{len(ev_docs)}  {len(ids)} tok  "
              f"{time.time()-td:.0f}s/doc  (tot {time.time()-t0:.0f}s)", flush=True)
    for h in xhooks:
        h.remove()

    res = {}
    print("\n=== single-site freeze: content vs operating-point damage ===")
    print(f"  {'L':>3} {'cond':>6} {'KL':>8} {'top1_flip':>10} {'dH':>8} "
          f"{'dNLL_true':>10}")
    for L in LAYERS:
        res[f"L{L}"] = {}
        for c in CONDS:
            a = agg[L][c]
            e = {"kl_med": float(np.median(a["kl"])),
                 "flip_rate": float(np.mean(a["flip"])),
                 "dH_med": float(np.median(a["dH"])),
                 "dnll_med": float(np.median(a["dnll"])),
                 "dnll_mean": float(np.mean(a["dnll"]))}
            res[f"L{L}"][c] = e
            print(f"  {L:>3} {c:>6} {e['kl_med']:>8.4f} {e['flip_rate']:>10.3f} "
                  f"{e['dH_med']:>8.4f} {e['dnll_med']:>10.4f}")
    # ratios value/gate -- >1 on flip/dnll => value carries more content
    print("\n  value/gate ratios (flip, dNLL_true mean): >1 => value is the "
          "content branch")
    for L in LAYERS:
        g, v = res[f"L{L}"]["gate"], res[f"L{L}"]["value"]
        fr = v["flip_rate"] / max(g["flip_rate"], 1e-9)
        nr = v["dnll_mean"] / max(abs(g["dnll_mean"]), 1e-9)
        res[f"L{L}"]["value_over_gate_flip"] = fr
        res[f"L{L}"]["value_over_gate_dnll"] = nr
        print(f"  L{L}: flip x{fr:.2f}   dNLL x{nr:.2f}")

    json.dump({"model": MODEL, "device": DEV, "layers": LAYERS, "skip": SKIP,
               "n_cal": n, "n_eval": N_EVAL, "results": res},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: value-freeze >> gate-freeze on flip/dNLL => value is the "
          "content branch\n(gate sets the operating point). Comparable => the "
          "dichotomy is too strong; say so.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
