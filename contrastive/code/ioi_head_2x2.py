"""
2x2 per-head IOI analysis: {mixed-gender, same-gender} x {multiple pairs}
Which heads are active? Do they write the same direction?

Usage: .venv/Scripts/python.exe contrastive/code/ioi_head_2x2.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
NH = model.config.num_attention_heads
HD = model.config.hidden_size // NH
W_U = model.lm_head.weight.detach().float().cpu()


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def get_head_contributions(pa, pb, L=24):
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    captured = [{}, {}]

    for ci, ids in enumerate([ids_a, ids_b]):
        dense = model.model.layers[L].self_attn.dense

        def make_hook(store):
            def hook_fn(module, inp, out):
                store[L] = inp[0].detach().float().cpu()
            return hook_fn

        handle = dense.register_forward_hook(make_hook(captured[ci]))
        with torch.no_grad():
            model(torch.tensor([ids], device=DEV))
        handle.remove()

    O_w = model.model.layers[L].self_attn.dense.weight.float().cpu()
    da = captured[0][L][0, -1, :]
    db = captured[1][L][0, -1, :]

    heads = []
    for h in range(NH):
        dh = da[h * HD : (h + 1) * HD] - db[h * HD : (h + 1) * HD]
        O_h = O_w[:, h * HD : (h + 1) * HD]
        contribution = dh @ O_h.T
        logits = contribution @ W_U.T
        heads.append((float(contribution.norm()), h, logits, contribution))

    return heads


cases = [
    # Mixed gender
    ("John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to",
     "John/Mary (MF)"),
    ("Alice and Bob went to the store. Alice gave a book to",
     "Bob and Alice went to the store. Bob gave a book to",
     "Alice/Bob (MF)"),
    ("Dan and Eve went to the store. Dan gave a book to",
     "Eve and Dan went to the store. Eve gave a book to",
     "Dan/Eve (MF)"),
    # Same gender male
    ("John and Tom went to the store. John gave a book to",
     "Tom and John went to the store. Tom gave a book to",
     "John/Tom (MM)"),
    ("Mike and Dan went to the store. Mike gave a book to",
     "Dan and Mike went to the store. Dan gave a book to",
     "Mike/Dan (MM)"),
    ("Bob and Steve went to the store. Bob gave a book to",
     "Steve and Bob went to the store. Steve gave a book to",
     "Bob/Steve (MM)"),
    # Same gender female
    ("Mary and Sarah went to the store. Mary gave a book to",
     "Sarah and Mary went to the store. Sarah gave a book to",
     "Mary/Sarah (FF)"),
    ("Alice and Eve went to the store. Alice gave a book to",
     "Eve and Alice went to the store. Eve gave a book to",
     "Alice/Eve (FF)"),
    ("Lisa and Carol went to the store. Lisa gave a book to",
     "Carol and Lisa went to the store. Carol gave a book to",
     "Lisa/Carol (FF)"),
]

# ============================================================
print("=" * 100)
print("1. PER-HEAD CONTENT AT L24 — what does each head write?")
print("=" * 100)

all_norms = {}
all_vectors = {}

for pa, pb, label in cases:
    heads = get_head_contributions(pa, pb, L=24)
    heads_sorted = sorted(heads, reverse=True)
    norm_by_head = {h: n for n, h, _, _ in heads}
    vec_by_head = {h: v for _, h, _, v in heads}
    all_norms[label] = norm_by_head
    all_vectors[label] = vec_by_head

    print(f"\n  {label}:")
    for norm, h, logits, _ in heads_sorted[:6]:
        print(f"    H{h:>2} (norm={norm:>5.1f}): [{tk(logits)}]")
    torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("2. HEAD NORM TABLE — which heads are consistently active?")
print("=" * 100)

mean_norms = {}
for h in range(NH):
    norms = [all_norms[label][h] for label in all_norms]
    mean_norms[h] = sum(norms) / len(norms)

top_heads = sorted(mean_norms, key=lambda h: mean_norms[h], reverse=True)[:12]

print(f"\n  {'Case':<20}", end="")
for h in top_heads:
    print(f"  H{h:>2}", end="")
print()

for label in all_norms:
    print(f"  {label:<20}", end="")
    for h in top_heads:
        print(f" {all_norms[label][h]:>4.0f}", end="")
    print()

# Means by gender type
for gtype, suffix in [("Mean MF", "(MF)"), ("Mean MM", "(MM)"),
                       ("Mean FF", "(FF)")]:
    labels = [l for l in all_norms if suffix in l]
    print(f"  {gtype:<20}", end="")
    for h in top_heads:
        mn = sum(all_norms[l][h] for l in labels) / len(labels)
        print(f" {mn:>4.0f}", end="")
    print()

# ============================================================
print(f"\n{'='*100}")
print("3. PAIRWISE COSINE — do heads write the same direction across cases?")
print("=" * 100)

labels = list(all_vectors.keys())
for h in top_heads[:5]:
    print(f"\n  H{h} pairwise cosine:")
    print(f"  {'':>20}", end="")
    for l in labels:
        print(f" {l[:7]:>7}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:<20}", end="")
        for l2 in labels:
            v1 = all_vectors[l1][h]
            v2 = all_vectors[l2][h]
            if v1.norm() < 0.1 or v2.norm() < 0.1:
                print(f"    ---", end="")
            else:
                cos = float(torch.nn.functional.cosine_similarity(
                    v1.unsqueeze(0), v2.unsqueeze(0)))
                print(f" {cos:>+.3f}", end="")
        print()

# ============================================================
print(f"\n{'='*100}")
print("4. DO THE SAME HEADS WORK AT L28?")
print("=" * 100)

all_norms_28 = {}
for pa, pb, label in cases:
    heads = get_head_contributions(pa, pb, L=28)
    norm_by_head = {h: n for n, h, _, _ in heads}
    all_norms_28[label] = norm_by_head
    torch.cuda.empty_cache()

mean_norms_28 = {}
for h in range(NH):
    norms = [all_norms_28[label][h] for label in all_norms_28]
    mean_norms_28[h] = sum(norms) / len(norms)

top_28 = sorted(mean_norms_28, key=lambda h: mean_norms_28[h],
                reverse=True)[:12]

print(f"\n  {'Case':<20}", end="")
for h in top_28:
    print(f"  H{h:>2}", end="")
print()

for label in all_norms_28:
    print(f"  {label:<20}", end="")
    for h in top_28:
        print(f" {all_norms_28[label][h]:>4.0f}", end="")
    print()

for gtype, suffix in [("Mean MF", "(MF)"), ("Mean MM", "(MM)"),
                       ("Mean FF", "(FF)")]:
    labels = [l for l in all_norms_28 if suffix in l]
    print(f"  {gtype:<20}", end="")
    for h in top_28:
        mn = sum(all_norms_28[l][h] for l in labels) / len(labels)
        print(f" {mn:>4.0f}", end="")
    print()

print(f"\n{'='*100}")
print("DONE")
