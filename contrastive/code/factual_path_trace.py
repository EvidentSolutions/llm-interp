"""
Factual-recall path trace via position-resolved denoising (path patching).

Same method as ioi_path_trace.py, applied to factual recall. Use a frame
where clean and corrupt differ in a SINGLE input token — the subject entity —
so all causal divergence originates at the subject position and we can trace
its flow to END.

  CLEAN:   The capital of France is → Paris
  CORRUPT: The capital of Japan  is → Tokyo   (single-token swap: France/Japan)

Prior art (Meng et al. 2022; Geva et al. 2023) localizes factual recall to the
SUBJECT token's MLP at mid layers, then movement to END. We test whether the
contrastive-trace machinery recovers that structure.

PART A — position × layer denoising map (recovery of P(Paris)).
PART B — subject vs END handoff (fine layer sweep).
PART C — position-resolved attention from END (does END read the subject?).
PART D — generality: subject/END crossover layer across several entity pairs.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads


def tid(name):
    return tok.encode(f" {name}", add_special_tokens=False)[0]


def first_tok_count(word):
    return len(tok.encode(f" {word}", add_special_tokens=False))


def trace_pair(clean_text, corrupt_text, answer, label, verbose=True):
    """Position-resolved denoising trace for a single-token-diff entity pair.
    Returns (clean_p, corrupt_p, subject_pos, crossover_layer)."""
    ans_id = tid(answer)
    ids_clean = tok.encode(clean_text, return_tensors="pt").to(DEV)
    ids_corr = tok.encode(corrupt_text, return_tensors="pt").to(DEV)

    ct = [tok.decode([t]).strip() for t in ids_clean[0].tolist()]
    rt = [tok.decode([t]).strip() for t in ids_corr[0].tolist()]
    if len(ct) != len(rt):
        if verbose:
            print(f"  [{label}] SKIP — length mismatch "
                  f"({len(ct)} vs {len(rt)})")
        return None
    diffs = [i for i, (a, b) in enumerate(zip(ct, rt)) if a != b]
    if len(diffs) != 1:
        if verbose:
            print(f"  [{label}] SKIP — {len(diffs)} differing positions "
                  f"{diffs}: {[ (i,ct[i],rt[i]) for i in diffs]}")
        return None
    subj_pos = diffs[0]
    end_pos = ids_clean.shape[1] - 1

    def pp(ids):
        with torch.no_grad():
            out = model(ids)
        return torch.softmax(out.logits[0, -1].float(), -1)

    p_clean = float(pp(ids_clean)[ans_id])
    p_corr = float(pp(ids_corr)[ans_id])

    with torch.no_grad():
        out_clean = model(ids_clean, output_hidden_states=True)
    clean_hs = [h.detach() for h in out_clean.hidden_states]
    del out_clean
    torch.cuda.empty_cache()

    def recov(p):
        d = p_clean - p_corr
        return 100.0 * (p - p_corr) / d if abs(d) > 1e-6 else float('nan')

    def patch(layer, positions):
        clean_val = clean_hs[layer + 1][0]

        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0].clone()
                for p in positions:
                    h[0, p, :] = clean_val[p]
                return (h,) + out[1:]
            h = out.clone()
            for p in positions:
                h[0, p, :] = clean_val[p]
            return h

        handle = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(ids_corr)
        handle.remove()
        return float(torch.softmax(out.logits[0, -1].float(), -1)[ans_id])

    if verbose:
        print(f"\n  [{label}] subject@{subj_pos}={ct[subj_pos]!r}/"
              f"{rt[subj_pos]!r}  END@{end_pos}")
        print(f"    CLEAN P({answer})={p_clean:.3f}  CORRUPT={p_corr:.3f}")

    # PART A map
    if verbose:
        layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
        named = {f"SUBJ({subj_pos})": [subj_pos], "END": [end_pos],
                 "SUBJ+END": [subj_pos, end_pos],
                 "ALL": list(range(ids_clean.shape[1]))}
        print("    " + "layer ".rjust(8) +
              "".join(f"{k:>11}" for k in named))
        for L in layers:
            row = f"    L{L:>3}  "
            for k, ps in named.items():
                row += f"{recov(patch(L, ps)):>10.0f}%"
            print(row)
            gc.collect(); torch.cuda.empty_cache()

    # PART B/D handoff crossover (fine sweep)
    crossover = None
    prev = None
    for L in range(0, NL, 2):
        r_s = recov(patch(L, [subj_pos]))
        r_e = recov(patch(L, [end_pos]))
        if crossover is None and r_e > r_s:
            crossover = L
        if verbose:
            print(f"      L{L:>2}: SUBJ={r_s:>6.0f}%   END={r_e:>6.0f}%")
        prev = (r_s, r_e)
        gc.collect(); torch.cuda.empty_cache()

    del clean_hs
    torch.cuda.empty_cache()
    return p_clean, p_corr, subj_pos, crossover, ids_clean, end_pos


# ══════════════════════════════════════════════════════════════
# PART A/B — detailed trace, primary pair
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("FACTUAL RECALL PATH TRACE — primary pair")
print("=" * 70)

res = trace_pair("The capital of France is", "The capital of Japan is",
                 "Paris", "France/Japan", verbose=True)


# ══════════════════════════════════════════════════════════════
# PART C — position-resolved attention from END
# ══════════════════════════════════════════════════════════════
if res is not None:
    _, _, subj_pos, crossover, ids_clean, end_pos = res
    print("\n" + "=" * 70)
    print("PART C: attention from END — SUBJ vs sink, CLEAN prompt")
    print("=" * 70)
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out_att = model(ids_clean, output_attentions=True)
    for L in [12, 16, 20, 24, 28]:
        att = out_att.attentions[L][0]
        end = att[:, -1, :].float()
        scored = [(h, float(end[h, subj_pos]), float(end[h, 0]))
                  for h in range(n_heads)]
        scored.sort(key=lambda x: -x[1])
        print(f"\n  L{L} (top heads by attention to SUBJ@{subj_pos})")
        for h, s, sk in scored[:4]:
            print(f"    H{h:>2}: SUBJ={s:.2f}  sink={sk:.2f}")
    model.set_attn_implementation("sdpa")


# ══════════════════════════════════════════════════════════════
# PART D — generality across entity pairs
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART D: SUBJ→END crossover layer across entity pairs")
print("=" * 70)

pairs = [
    ("The capital of France is", "The capital of Japan is", "Paris"),
    ("The capital of Japan is", "The capital of France is", "Tokyo"),
    ("The capital of Italy is", "The capital of Spain is", "Rome"),
    ("The capital of Spain is", "The capital of Italy is", "Madrid"),
    ("The capital of Egypt is", "The capital of Japan is", "Cairo"),
    ("The capital of Russia is", "The capital of France is", "Moscow"),
]

print(f"\n  {'pair':<28}{'clean P':>9}{'corr P':>9}{'crossover':>11}")
for clean, corr, ans in pairs:
    r = trace_pair(clean, corr, ans, ans, verbose=False)
    if r is None:
        print(f"  {ans:<28}{'--- skipped (not single-token diff) ---':>0}")
        continue
    p_clean, p_corr, subj_pos, crossover = r[0], r[1], r[2], r[3]
    cl = f"L{crossover}" if crossover is not None else "none"
    label = f"{ans} (subj@{subj_pos})"
    print(f"  {label:<28}{p_clean:>9.3f}{p_corr:>9.3f}{cl:>11}")
    gc.collect(); torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
