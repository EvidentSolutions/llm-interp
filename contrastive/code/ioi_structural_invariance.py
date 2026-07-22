"""
Structural vs token-superposed: differentiate by PROMPT DESIGN, not a probe.

The §5.1 claim is that the no-duplicate contrast surfaces a signal that is
"structural rather than token-shaped." The only evidence so far was that the
delta does not project to clean tokens through W_U — but W_U-illegibility is a
weak proxy (the logit lens is one linear readout; token content can be
superposed in a basis it cannot decode).

A cleaner differentiator needs no trained probe — better PROMPTS do it:

  Build a FAMILY of name-sets that share the IOI STRUCTURE but vary the token
  IDENTITY. For each set form two contrasts at END:

    name-swap : h(CLEAN, giver=A→B) - h(SWAPPED, giver=B→A)   [token content]
    no-dupe   : h(CLEAN, giver=A→B) - h(NODUPE,  giver=C, ambiguous)

  Then:
    (1) cross-set cosine of the delta direction.
        structural -> stable direction across name-sets (high cosine).
        token      -> rotates with the names (low cosine).
    (2) survival ratio  ||mean_i d_i|| / mean_i ||d_i||.
        averaging over the family CANCELS token-specific parts and KEEPS the
        shared structural part. High survival = invariant = structural.
        This IS desuperposition by prompt averaging (no probe).
    (3) W_U read of the surviving mean direction.

PART B — attention invariance (W_U-free structural signature):
  Do the S2-reader heads attend to the GIVER/duplicate POSITION regardless of
  which names fill it? Position-keying that ignores identity is structural.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import gc
import itertools
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
W_U = model.lm_head.weight.detach().float()


def tid(name):
    return tok.encode(f" {name}", add_special_tokens=False)


def single_token(name):
    return len(tid(name)) == 1


def topk_tok(vec, k=6):
    logits = vec.float().to(W_U.device) @ W_U.T
    idx = torch.topk(logits, k).indices
    return [tok.decode([int(i)]).strip()[:10] for i in idx]


# ── build name-sets that are all single-token ───────────────────────────
CANDIDATES = ["John", "Mary", "Sam", "Alice", "Bob", "Carl", "Tom", "Anna",
              "Mike", "Sara", "Paul", "Lucy", "Mark", "Jane", "Dave", "Kate",
              "Steve", "Emma", "Jack", "Rose", "Sean", "Adam", "Nina", "Paul"]
names = [n for n in dict.fromkeys(CANDIDATES) if single_token(n)]
print(f"  single-token names available: {names}")

# form 8 (A,B,C) triples, all distinct within a triple
triples = []
it = itertools.cycle(names)
buf = []
for _ in range(len(names) * 3):
    buf.append(next(it))
i = 0
while len(triples) < 8 and i + 2 < len(names):
    A, B, C = names[i], names[i + 1], names[i + 2]
    if len({A, B, C}) == 3:
        triples.append((A, B, C))
    i += 1


def prompts(A, B, C):
    base = "When {x} and {y} went to the store, {g} gave a drink to"
    return (base.format(x=A, y=B, g=A),   # CLEAN  giver=A (dup)  -> B
            base.format(x=A, y=B, g=B),   # SWAPPED giver=B (dup) -> A
            base.format(x=A, y=B, g=C))   # NODUPE giver=C        -> ambig


def end_hidden(text):
    """Residual stream at END for every layer: tensor (NL+1, d_model)."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = torch.stack([h[0, -1].float() for h in out.hidden_states])  # (NL+1,d)
    return hs, ids.shape[1]


# verify alignment + collect END deltas per layer
LAYERS = [16, 20, 22, 24, 26, 28]
swap_d = {L: [] for L in LAYERS}   # token-content contrast
nod_d = {L: [] for L in LAYERS}    # structural contrast
lengths = set()

print("\n  name-sets (A,B,C):")
for A, B, C in triples:
    cl, sw, nd = prompts(A, B, C)
    hs_cl, n = end_hidden(cl)
    hs_sw, _ = end_hidden(sw)
    hs_nd, _ = end_hidden(nd)
    lengths.add(n)
    print(f"    {A:>5}/{B:<5} giver_dup={A:<5} nodupe={C}")
    for L in LAYERS:
        swap_d[L].append(hs_cl[L + 1] - hs_sw[L + 1])
        nod_d[L].append(hs_cl[L + 1] - hs_nd[L + 1])
    del hs_cl, hs_sw, hs_nd
    gc.collect(); torch.cuda.empty_cache()
print(f"  token-length(s) across sets: {lengths}  (must be one value)")


def cross_cos(vlist):
    V = torch.stack(vlist)
    Vn = V / V.norm(dim=1, keepdim=True)
    G = Vn @ Vn.T
    n = G.shape[0]
    off = G[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean()), float(off.std())


def survival(vlist):
    V = torch.stack(vlist)
    mean_vec = V.mean(0)
    return float(mean_vec.norm()) / float(V.norm(dim=1).mean()), mean_vec


# ══════════════════════════════════════════════════════════════
# PART A — direction invariance across the name family
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART A: is the END delta direction NAME-INVARIANT? (prompt-only test)")
print("  high cosine / high survival = structural; low = token-shaped")
print("=" * 72)
print(f"\n  {'layer':>5} | {'name-swap (token)':>28} | {'no-dupe (structural?)':>28}")
print(f"  {'':>5} | {'cos(±sd)':>14}{'survival':>14} | {'cos(±sd)':>14}{'survival':>14}")
for L in LAYERS:
    sc, ss = cross_cos(swap_d[L])
    nc, ns = cross_cos(nod_d[L])
    s_surv, s_mean = survival(swap_d[L])
    n_surv, n_mean = survival(nod_d[L])
    print(f"  L{L:>3}  | {sc:>+7.2f}±{ss:.2f}{s_surv:>13.2f} |"
          f" {nc:>+7.2f}±{ns:.2f}{n_surv:>13.2f}")

# W_U read of the surviving (invariant) mean direction at the handoff layer
print("\n  W_U read of the surviving mean direction:")
for L in [20, 24]:
    _, s_mean = survival(swap_d[L])
    _, n_mean = survival(nod_d[L])
    print(f"    L{L} name-swap mean -> {topk_tok(s_mean)}")
    print(f"    L{L} no-dupe   mean -> {topk_tok(n_mean)}")
    # also a couple of individual sets for contrast
    print(f"    L{L} name-swap set0 -> {topk_tok(swap_d[L][0])}   "
          f"set1 -> {topk_tok(swap_d[L][1])}")
    print(f"    L{L} no-dupe   set0 -> {topk_tok(nod_d[L][0])}   "
          f"set1 -> {topk_tok(nod_d[L][1])}")


# ══════════════════════════════════════════════════════════════
# PART B — attention invariance: do S2-readers key on POSITION not name?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART B: attention from END to GIVER/duplicate position, across names")
print("  S2-reader heads keying on position (not identity) = structural")
print("=" * 72)

# giver/duplicate position: locate by tokenizing one prompt
A0, B0, C0 = triples[0]
cl0, _, _ = prompts(A0, B0, C0)
toks0 = [tok.decode([t]).strip() for t in tok.encode(cl0)]
# giver = second occurrence of A0
occ = [i for i, t in enumerate(toks0) if t == A0]
GIVER_POS = occ[1] if len(occ) > 1 else 9
print(f"  positions: {[f'{i}:{t}' for i, t in enumerate(toks0)]}")
print(f"  giver/duplicate position = {GIVER_POS}")

S2_HEADS = [10, 12, 20, 31]   # S2-readers found earlier
model.set_attn_implementation("eager")
print(f"\n  L20 attention from END -> giver-pos (dup), per name-set:")
header = "    " + "set".ljust(14) + "".join(f"   H{h:>2}" for h in S2_HEADS)
print(header)
for A, B, C in triples:
    cl, _, _ = prompts(A, B, C)
    ids = tok.encode(cl, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    att = out.attentions[20][0]  # (n_heads, seq, seq)
    row = f"    {A+'/'+B:<14}"
    for h in S2_HEADS:
        row += f"{float(att[h, -1, GIVER_POS]):>6.2f}"
    print(row)
    del out
    gc.collect(); torch.cuda.empty_cache()
model.set_attn_implementation("sdpa")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
