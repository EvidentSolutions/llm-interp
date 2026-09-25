# -*- coding: utf-8 -*-
"""Is the gate operating point actually disturbed AT THE TASK POSITION?
(2026-08-19; closing a conflation I introduced in 15bq.)

THE DEFECT BEING FIXED. 15bo measured duty explosion (L24: 0.087 -> 0.791) with
the ENTIRE input degenerate. 15bq then measured task damage with a CLEAN task
prompt appended to a degenerate preamble. I compared the two and concluded "a 9x
duty explosion does not proportionally break computation" -- but those are
DIFFERENT CONDITIONS. In 15bq the answer is read at a position whose recent
context is clean prose (the task prompt), so duty there was never measured and may
be nowhere near 0.791.

So the deflation claim was not supported by the two experiments as run. This
measures the missing quantity: gate duty and the reference component AT THE FINAL
TASK POSITION, under each preamble condition, on the same prompts 15bq used.

Three outcomes, each meaning something different:
  duty at the task position is ~NORMAL under a degenerate preamble
      -> the clean prompt RESTORES the operating point locally, 15bq's tasks were
         never run with disturbed gates, and the deflation claim must be WITHDRAWN
         (it compared incomparable conditions). The interesting positive becomes:
         the reference recovers within a few clean tokens.
  duty at the task position is EXPLODED and arithmetic still works
      -> the deflation claim stands as originally stated.
  intermediate -> report the gradient.

Also measured: how many clean tokens it takes to recover, by reading duty at each
position of the task prompt rather than only the last.

Usage: .venv/Scripts/python.exe superposition/code/census_duty_at_task_position.py
"""
import os
import sys
import json
import time
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
MAXLEN, SKIP = 256, 16
LAYERS = [8, 16, 24]
PRE_LEN = 128
CONDS = ["prose", "shuffle", "randid", "repeat"]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-duty-at-task-position.json")
RNG = np.random.default_rng(20260819)

PROMPTS = [("capital", "The capital of France is", " Paris"),
           ("capital", "The capital of Japan is", " Tokyo"),
           ("capital", "The capital of Italy is", " Rome"),
           ("arith", "5 + 3 =", " 8"),
           ("arith", "7 + 2 =", " 9"),
           ("arith", "4 + 4 =", " 8"),
           ("opposite", "The opposite of hot is", " cold"),
           ("opposite", "The opposite of big is", " small")]


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz, d = m.model.layers, m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))
    bl_docs, pre_docs = docs[:20], docs[20:36]

    cap, hooks = {}, []
    for L in LAYERS:
        hooks.append(Lz[L].register_forward_pre_hook(
            (lambda L: (lambda mod_, a, kw: cap.__setitem__(
                f"h{L}", (a[0] if a else kw["hidden_states"])[0].detach())))(L),
            with_kwargs=True))
        hooks.append(Lz[L].mlp.fc1.register_forward_hook(
            (lambda L: (lambda mod_, i, o: cap.__setitem__(
                f"z{L}", o[0].detach())))(L)))

    def run(ids):
        with torch.no_grad():
            return m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()

    # b_L^prose
    acc = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    n = 0
    for text in bl_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        run(ids)
        for L in LAYERS:
            with torch.no_grad():
                acc[L] += Lz[L].input_layernorm(
                    cap[f"h{L}"])[SKIP:].double().sum(0)
        n += len(ids) - SKIP
    bhat = {L: (acc[L] / n).float() / (acc[L] / n).float().norm()
            for L in LAYERS}

    pres = []
    for t in pre_docs:
        v = tok(t, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(v) >= PRE_LEN + 8:
            pres.append(v[:PRE_LEN])
    print(f"  {len(pres)} preambles of {PRE_LEN} tokens")

    def corrupt(ids, c):
        if c == "prose":
            return list(ids)
        if c == "shuffle":
            v = list(ids); RNG.shuffle(v); return v
        if c == "randid":
            return [int(x) for x in RNG.integers(0, 50000, len(ids))]
        return [int(Counter(ids).most_common(1)[0][0])] * len(ids)

    # duty in the DEGENERATE PREAMBLE ITSELF, for reference
    print("\n=== reference point: duty measured INSIDE the preamble "
          "(positions >=16, no task prompt) ===")
    pre_only = {}
    print(f"  {'cond':>8} " + " ".join(f"{'L%d duty'%L:>9}" for L in LAYERS)
          + " " + " ".join(f"{'L%d ref'%L:>8}" for L in LAYERS))
    for c in CONDS:
        du = {L: [] for L in LAYERS}
        rf = {L: [] for L in LAYERS}
        for p in pres:
            run(corrupt(p, c))
            for L in LAYERS:
                z = cap[f"z{L}"][SKIP:].float()
                du[L].append(float((z > 0).float().mean()))
                with torch.no_grad():
                    x = Lz[L].input_layernorm(cap[f"h{L}"]).float()[SKIP:]
                rf[L].append(float((x @ bhat[L]).mean()))
        pre_only[c] = {f"L{L}": dict(duty=round(float(np.mean(du[L])), 4),
                                     ref=round(float(np.mean(rf[L])), 3))
                       for L in LAYERS}
        print(f"  {c:>8} " + " ".join(
            f"{pre_only[c][f'L{L}']['duty']:>9.4f}" for L in LAYERS) + " "
            + " ".join(f"{pre_only[c][f'L{L}']['ref']:>8.3f}" for L in LAYERS))

    # ---- the measurement: AT the task positions ---------------------------
    print(f"\n=== AT THE TASK POSITIONS (clean prompt after a {PRE_LEN}-token "
          f"preamble) ===")
    res = {}
    for c in CONDS:
        per_off = {}
        last = {L: [] for L in LAYERS}
        lp_ok = []
        for fam, prompt, ans in PROMPTS:
            tids = tok(prompt, add_special_tokens=False)["input_ids"]
            aid = tok(ans, add_special_tokens=False)["input_ids"][0]
            for p in pres[:8]:
                ids = corrupt(p, c) + tids
                lg = run(ids)
                lp_ok.append(float(torch.log_softmax(lg[-1], -1)[aid]))
                for L in LAYERS:
                    z = cap[f"z{L}"].float()
                    for off in range(len(tids)):
                        pos = PRE_LEN + off
                        per_off.setdefault(L, {}).setdefault(off, []).append(
                            float((z[pos] > 0).float().mean()))
                    last[L].append(float((z[-1] > 0).float().mean()))
        res[c] = {"logp_answer": round(float(np.mean(lp_ok)), 4),
                  "final_pos_duty": {f"L{L}": round(float(np.mean(last[L])), 4)
                                     for L in LAYERS},
                  "by_offset": {f"L{L}": {str(o): round(float(np.mean(v)), 4)
                                          for o, v in sorted(per_off[L].items())}
                                for L in LAYERS}}
        print(f"  {c:>8}  logP(ans) {res[c]['logp_answer']:>7.3f}   "
              f"final-position duty " + ", ".join(
                  f"L{L} {res[c]['final_pos_duty'][f'L{L}']:.4f}"
                  for L in LAYERS))

    print(f"\n=== RECOVERY: duty by offset into the clean prompt "
          f"(offset 0 = first clean token) ===")
    for L in LAYERS:
        print(f"  --- L{L} ---")
        offs = sorted(int(o) for o in res["prose"]["by_offset"][f"L{L}"])[:8]
        print(f"    {'cond':>8} " + " ".join(f"{'off%d'%o:>7}" for o in offs))
        for c in CONDS:
            row = res[c]["by_offset"][f"L{L}"]
            print(f"    {c:>8} " + " ".join(
                f"{row[str(o)]:>7.4f}" for o in offs))

    for h_ in hooks:
        h_.remove()
    json.dump({"model": MODEL, "pre_len": PRE_LEN, "layers": LAYERS,
               "preamble_only": pre_only, "at_task": res},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: if final-position duty under `repeat` is close to `prose`,")
    print("then 15bq's tasks ran with a RESTORED operating point and the")
    print("'duty explosion does not break computation' claim must be withdrawn.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
