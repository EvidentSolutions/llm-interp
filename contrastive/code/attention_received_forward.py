"""
Forward-pass confirmation of the attention-read signature.

The weights-only probe (token_attention_read_signature.py) said function words
and punctuation are READABLE by many downstream heads (high MOVE/KEY enrichment),
while content and register-words sit at random. That is an affordance, not a
measurement. Here we run real forward passes on natural preamble scenes and ask:
do later positions actually ATTEND TO these tokens more?

For each token position j we measure, in late layers, the average attention
weight that strictly-later query positions place on j:
    recv(j) = mean_{L in late} mean_h mean_{i>j} A[L,h,i,j]
This is "how much position j is read by the rest of the sentence". We then group
positions by token category and compare.

Position 0 is the attention sink and is excluded. We also report recv relative to
the uniform share (1/j) so early vs late positions are comparable.

Usage: .venv/Scripts/python.exe contrastive/code/attention_received_forward.py
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

# natural preamble scenes (stable regime); a mix of function/punct/content
SCENES = [
    "Consider the following short scene. The doctor told her that she had caught "
    "a cold, and so she should rest in bed because the flu was going around.",
    "Consider the following short scene. Einstein was born in Germany, but he "
    "later moved to Paris, and then he travelled on to Rome by train.",
    "Consider the following short scene. She put mustard on the hot dog, poured "
    "some coffee, and sat at the table by the river to eat her lunch.",
    "Consider the following short scene. The report said that the weather was "
    "cold, so the dog stayed inside, and nobody wanted to walk to the shop.",
    "Consider the following short scene. Tesla built a machine that could not "
    "run without power, and if the current stopped then the motor would fail.",
]

GROUPS = {
    "punct":    set(".,:;?!()\"'-"),
    "negation": {"not", "no", "never", "none", "nor", "cannot", "nobody",
                 "without"},
    "function": {"the", "of", "and", "is", "to", "a", "that", "but", "because",
                 "if", "than", "which", "so", "then", "on", "in", "by", "was",
                 "had", "would", "could", "should", "he", "she", "her", "his"},
    "content":  {"doctor", "cold", "bed", "flu", "einstein", "germany", "paris",
                 "rome", "train", "mustard", "dog", "coffee", "table", "river",
                 "lunch", "weather", "shop", "tesla", "machine", "power", "motor",
                 "cold", "current", "scene"},
}


def classify(s):
    s = s.strip().lower()
    for g, vocab in GROUPS.items():
        if s in vocab:
            return g
    return None


import collections
SHARP = 5.0                              # a head "sharply reads" j if per-head recv > SHARP x uniform
acc = collections.defaultdict(list)      # group -> [recv_ratio ...]
acc_sharp = collections.defaultdict(list)  # group -> [n_sharp heads ...]

for scene in SCENES:
    ids = tok(scene, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_attentions=True)
    # attentions: tuple(NL) of (1, NH, T, T)
    A = torch.stack([out.attentions[L][0] for L in LATE])   # (nLate, NH, T, T)
    T = A.shape[-1]
    for j in range(1, T - 1):                # skip sink pos 0 and last token
        g = classify(tok.decode([ids[j]]))
        if g is None:
            continue
        col = A[:, :, j + 1:, j]             # (nLate, NH, #later)
        recv = float(col.mean())             # avg weight later positions give j
        # uniform baseline: a later query at pos i spreads over i+1 keys ~ 1/(i)
        later = torch.arange(j + 1, T, device=DEV).float()
        uni = float((1.0 / later).mean())
        acc[g].append(recv / uni)
        # per-head realized read: for each (late layer, head), mean weight it gives j
        perhead = col.mean(dim=-1)           # (nLate, NH) avg over later queries
        n_sharp = int((perhead / uni > SHARP).sum())
        acc_sharp[g].append(n_sharp)

nslot = len(LATE) * NH
print(f"\nLate layers {LATE[0]}..{NL-1} ({nslot} head slots); recv = avg attn "
      f"weight later positions place on the token.\nrecv_ratio = mean recv / "
      f"uniform-share.  sharp = # head slots reading j at > {SHARP}x uniform "
      f"(the realized analog of the weights-probe selectivity).\n")
print(f"{'group':>10} {'n':>4} {'recv_ratio':>11} {'sharp_heads':>12}")
import statistics as st
for g in ["punct", "function", "negation", "content"]:
    v = acc[g]; s = acc_sharp[g]
    if not v:
        continue
    print(f"{g:>10} {len(v):>4} {st.mean(v):>11.2f} {st.mean(s):>12.2f}")

# also show the strongest individually-attended tokens across all scenes
print("\nTop attended positions (recv_ratio) by token:")
allrows = []
for scene in SCENES:
    ids = tok(scene, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_attentions=True)
    A = torch.stack([out.attentions[L][0] for L in LATE])
    T = A.shape[-1]
    for j in range(1, T - 1):
        col = A[:, :, j + 1:, j]
        recv = float(col.mean())
        later = torch.arange(j + 1, T, device=DEV).float()
        uni = float((1.0 / later).mean())
        allrows.append((recv / uni, tok.decode([ids[j]]), classify(tok.decode([ids[j]]))))
allrows.sort(reverse=True)
for ratio, t, g in allrows[:20]:
    print(f"  {ratio:>7.2f}  {repr(t):>12}  [{g}]")
print("\nDONE")
