"""
Register / format tokens as global variables (the (b) probe).

The forward-pass confirmation showed function words are read by many downstream
HEADS, but the top realized reads are LOCAL (a 'the' read by its own noun). Olli's
hypothesis: register/format is "like a global variable" -- a value broadcast to
ALL later positions, not a local binding. If so, the discriminating axis is not
head-count but BREADTH: how large a fraction of later positions read the token.

For each position j we measure, in late layers, averaged over heads:
    recv(j)   = mean weight later positions place on j          (intensity)
    breadth(j)= fraction of later positions i that place > FLOOR (e.g. 2x uniform)
                attention on j                                  (how global)
A local binding target (function word read by its neighbour) has high recv but
LOW breadth. A true global variable (attention sink, format/newline boundary)
has high breadth: most of the document reads it.

Scenes here contain newlines and format markers so register tokens are present.
Position 0 (the canonical sink / global slot) is reported explicitly, not hidden.

Usage: .venv/Scripts/python.exe contrastive/code/attention_global_variable.py
"""
import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager").to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
NH = model.config.num_attention_heads
LATE = list(range(22, NL))
FLOOR = 2.0    # a later position "reads" j if it gives j > FLOOR x uniform

SCENES = [
    "Chapter 1.\nThe doctor told her she had caught a cold.\nShe should rest in "
    "bed, because the flu was going around.\nNobody wanted to walk to the shop.",
    "Abstract.\nEinstein was born in Germany.\nHe later moved to Paris, and then "
    "travelled on to Rome.\nThe train was slow, but the view was good.",
    "Note:\nShe put mustard on the hot dog.\nThen she poured some coffee.\nShe "
    "sat at the table by the river to eat her lunch.",
]

GROUPS = {
    "format":   set(),   # filled below: newline, and position 0
    "punct":    set(".,:;?!()\"'-"),
    "function": {"the", "of", "and", "is", "to", "a", "that", "but", "because",
                 "if", "than", "which", "so", "then", "on", "in", "by", "was",
                 "had", "would", "could", "should", "he", "she", "her", "his"},
    "content":  {"doctor", "cold", "bed", "flu", "einstein", "germany", "paris",
                 "rome", "train", "view", "mustard", "dog", "coffee", "table",
                 "river", "lunch", "shop", "chapter", "abstract", "note"},
}


def classify(s, is_pos0):
    if is_pos0:
        return "format"       # position 0 = the sink / global slot
    if "\n" in s:
        return "format"
    st_ = s.strip().lower()
    for g, vocab in GROUPS.items():
        if g == "format":
            continue
        if st_ in vocab:
            return g
    return None


import collections, statistics as st
acc_recv = collections.defaultdict(list)
acc_breadth = collections.defaultdict(list)

for scene in SCENES:
    ids = tok(scene, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_attentions=True)
    A = torch.stack([out.attentions[L][0] for L in LATE])   # (nLate,NH,T,T)
    Ah = A.mean(1)                                          # (nLate,T,T) avg heads
    T = A.shape[-1]
    for j in range(0, T - 1):
        g = classify(tok.decode([ids[j]]), is_pos0=(j == 0))
        if g is None:
            continue
        later = torch.arange(j + 1, T, device=DEV).float()
        uni = 1.0 / later                                  # (#later,) per-query uniform
        col = Ah[:, j + 1:, j]                              # (nLate, #later)
        recv = float((col / uni).mean())
        # breadth: fraction of later positions reading j above FLOOR x uniform,
        # averaged over late layers
        reads = (col / uni > FLOOR).float().mean(dim=-1)    # (nLate,) frac positions
        breadth = float(reads.mean())
        acc_recv[g].append(recv)
        acc_breadth[g].append(breadth)

print(f"\nLate layers {LATE[0]}..{NL-1}; heads averaged.\n"
      f"recv = mean attention (x uniform) later positions give the token "
      f"(intensity).\nbreadth = fraction of later positions that read it > "
      f"{FLOOR}x uniform (how GLOBAL).\n")
print(f"{'group':>10} {'n':>4} {'recv':>8} {'breadth':>9}")
for g in ["format", "punct", "function", "content"]:
    r = acc_recv[g]; b = acc_breadth[g]
    if not r:
        continue
    print(f"{g:>10} {len(r):>4} {st.mean(r):>8.2f} {st.mean(b):>9.3f}")

print("\nMost GLOBAL positions (highest breadth):")
rows = []
for scene in SCENES:
    ids = tok(scene, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_attentions=True)
    A = torch.stack([out.attentions[L][0] for L in LATE])
    Ah = A.mean(1)
    T = A.shape[-1]
    for j in range(0, T - 1):
        later = torch.arange(j + 1, T, device=DEV).float()
        uni = 1.0 / later
        col = Ah[:, j + 1:, j]
        breadth = float((col / uni > FLOOR).float().mean(dim=-1).mean())
        recv = float((col / uni).mean())
        g = classify(tok.decode([ids[j]]), is_pos0=(j == 0))
        rows.append((breadth, recv, repr(tok.decode([ids[j]])), g, j))
rows.sort(reverse=True)
for breadth, recv, t, g, j in rows[:18]:
    print(f"  breadth={breadth:>5.2f}  recv={recv:>6.2f}  pos={j:>3}  {t:>14}  [{g}]")
print("\nDONE")
