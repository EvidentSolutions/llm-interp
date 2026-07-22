"""
IOI path trace via position-resolved denoising (path patching).

Key fact: CLEAN and SWAPPED differ in exactly ONE input token — position 9,
the second subject / giver. So all causal divergence originates at S2 (pos 9)
and propagates forward. Position-resolved patching reveals the path.

  CLEAN:   When John and Mary went to the store, John  gave a drink to → Mary
  SWAPPED: When John and Mary went to the store, Mary  gave a drink to → John
  positions: 0:When 1:John 2:and 3:Mary 4:went 5:to 6:the 7:store 8:, 9:GIVER
             10:gave 11:a 12:drink 13:to(END)
  IO in clean = Mary (pos 3); S2/giver = pos 9; END = pos 13.

PART A — position × layer denoising map.
   Patch the full residual stream at (position p, layer L) from CLEAN into
   the corrupted (SWAPPED) run; measure how much P(Mary) is restored.
   Reveals where the causal information lives and when it moves.

PART B — S2→END handoff.
   Patch ONLY pos 9 (S2) at increasing layers: recovery should fall once the
   S-inhibition heads have read S2. Patch ONLY END at increasing layers:
   recovery should rise once name-movers have written. The crossover layer is
   the handoff.

PART C — position-resolved attention.
   For candidate heads, split attention from END into IO (pos 3) vs S2 (pos 9)
   vs sink (pos 0). Name-movers read IO; S-inhibition heads read S2.
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
).to(DEV).eval()  # default SDPA; switch to eager only for attention (PART C)
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads


def tid(name):
    return tok.encode(f" {name}", add_special_tokens=False)[0]


def predict_probs(ids):
    with torch.no_grad():
        out = model(ids)
    return torch.softmax(out.logits[0, -1].float(), -1)


CLEAN   = "When John and Mary went to the store, John gave a drink to"
SWAPPED = "When John and Mary went to the store, Mary gave a drink to"
mary_id, john_id = tid("Mary"), tid("John")

ids_clean = tok.encode(CLEAN, return_tensors="pt").to(DEV)
ids_corr = tok.encode(SWAPPED, return_tensors="pt").to(DEV)
clean_toks = [tok.decode([t]).strip() for t in ids_clean[0].tolist()]
corr_toks = [tok.decode([t]).strip() for t in ids_corr[0].tolist()]

# confirm single-token difference
print("\n  pos : CLEAN / SWAPPED")
diff_positions = []
for i, (a, b) in enumerate(zip(clean_toks, corr_toks)):
    mark = "  <-- DIFFERS" if a != b else ""
    if a != b:
        diff_positions.append(i)
    print(f"   {i:>2}: {a!r} / {b!r}{mark}")
print(f"  differing positions: {diff_positions}")

IO_POS, S2_POS, END_POS, SINK_POS = 3, 9, ids_clean.shape[1] - 1, 0

# baselines + clean hidden states
probs_clean = predict_probs(ids_clean)
probs_corr = predict_probs(ids_corr)
p_clean = float(probs_clean[mary_id])
p_corr = float(probs_corr[mary_id])
print(f"\n  CLEAN  P(Mary)={p_clean:.3f}  P(John)={float(probs_clean[john_id]):.3f}")
print(f"  CORRUPT P(Mary)={p_corr:.3f}  P(John)={float(probs_corr[john_id]):.3f}")
print(f"  recovery% = (patched - {p_corr:.3f}) / ({p_clean:.3f} - {p_corr:.3f}) * 100")

with torch.no_grad():
    out_clean = model(ids_clean, output_hidden_states=True)
clean_hs = [h.detach() for h in out_clean.hidden_states]  # len NL+1
del out_clean
torch.cuda.empty_cache()


def recov(p):
    d = p_clean - p_corr
    return 100.0 * (p - p_corr) / d if abs(d) > 1e-6 else float('nan')


def patch_resid(layer, positions):
    """Patch corrupted residual at (layer output, positions) with clean."""
    clean_val = clean_hs[layer + 1][0]  # residual after `layer`

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
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    return float(probs[mary_id])


# ══════════════════════════════════════════════════════════════
# PART A — POSITION × LAYER DENOISING MAP
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART A: position × layer denoising map (recovery% of P(Mary))")
print("=" * 70)

layers = [0, 2, 4, 8, 12, 16, 20, 24, 28, 31]
named = {"IO(3)": [IO_POS], "S2(9)": [S2_POS], "END": [END_POS],
         "S2+END": [S2_POS, END_POS], "ALL": list(range(ids_clean.shape[1]))}

header = "  layer  " + "".join(f"{k:>9}" for k in named)
print(header)
for L in layers:
    row = f"  L{L:>3}   "
    for k, ps in named.items():
        row += f"{recov(patch_resid(L, ps)):>8.0f}%"
    print(row)
    gc.collect()
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART B — S2 → END HANDOFF (fine layer sweep)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B: S2(9)-only vs END-only recovery across layers (handoff)")
print("=" * 70)
print(f"\n  {'layer':>6}  {'S2(9) only':>11}  {'END only':>10}")
for L in range(0, NL, 2):
    r_s2 = recov(patch_resid(L, [S2_POS]))
    r_end = recov(patch_resid(L, [END_POS]))
    print(f"  {L:>6}  {r_s2:>10.0f}%  {r_end:>9.0f}%")
    gc.collect()
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART C — POSITION-RESOLVED ATTENTION FROM END
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART C: attention from END — IO(3) vs S2(9) vs sink(0), CLEAN prompt")
print("=" * 70)

model.set_attn_implementation("eager")
with torch.no_grad():
    out_att = model(ids_clean, output_attentions=True)

for L in [18, 20, 22, 24, 26]:
    att = out_att.attentions[L][0]  # (n_heads, seq, seq)
    end = att[:, -1, :].float()
    # rank heads by max attention to IO or S2
    scored = []
    for h in range(n_heads):
        scored.append((h, float(end[h, IO_POS]), float(end[h, S2_POS]),
                       float(end[h, SINK_POS])))
    scored.sort(key=lambda x: -(x[1] + x[2]))
    print(f"\n  L{L}  (top heads by IO+S2 attention)")
    for h, io, s2, sk in scored[:5]:
        tag = "IO-reader" if io > s2 and io > 0.15 else (
              "S2-reader" if s2 > io and s2 > 0.15 else "sink/other")
        print(f"    H{h:>2}: IO(Mary)={io:.2f}  S2(giver)={s2:.2f}  "
              f"sink={sk:.2f}   {tag}")
model.set_attn_implementation("sdpa")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
