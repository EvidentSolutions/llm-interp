"""
Deep dive into some/all and quantifier contrasts.

1. More contrasts for "some" to triangulate the shared signal
2. Try "nobody" and other negatives for the not-everyone pole
3. Cross-content: same quantifier contrasts with different nouns
4. Check if the garbage-reading positive pole has structure
   (e.g., is it consistent across noun phrases?)
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
import torch.nn.functional as F
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
d_model = model.config.hidden_size
W_U = model.lm_head.weight.detach().float()
cos = F.cosine_similarity


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_str(logits, k=6):
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{vals[j]:.1f}")
            for j in range(k)]


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [tok.decode([int(idxs[j])]).strip()[:14] for j in range(k)]


def get_h(text, layer):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float(), out


def inject_and_measure(text_b, delta, layer):
    ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
    injected = [False]

    def hook_fn(module, input, output):
        if injected[0]:
            return output
        injected[0] = True
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, -1, :] += delta.half().to(DEV)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, -1, :] += delta.half().to(DEV)
            return h

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(torch.tensor([ids_b], device=DEV))
    handle.remove()
    return torch.softmax(out.logits[0, -1].float(), -1)


def recov(probs_inj, tid, pa, pb):
    gap = pa - pb
    if abs(gap) < 0.001:
        return float('nan')
    return (float(probs_inj[tid]) - pb) / gap * 100


ref_L = _sl(28)[0]
layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)

# ══════════════════════════════════════════════════════════════
# PART 1: Expanded quantifier contrasts for "some students"
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: EXPANDED CONTRASTS — 'some students passed'")
print("=" * 70)

target = "Some of the students passed the exam, so"
contrasts = [
    ("all",      "All of the students passed the exam, so"),
    ("none",     "None of the students passed the exam, so"),
    ("most",     "Most of the students passed the exam, so"),
    ("few",      "Few of the students passed the exam, so"),
    ("many",     "Many of the students passed the exam, so"),
    ("nobody",   "Nobody passed the exam, so"),
    ("everybody","Everybody passed the exam, so"),
    ("half",     "Half of the students passed the exam, so"),
    ("three",    "Three of the students passed the exam, so"),
    ("only_one", "Only one of the students passed the exam, so"),
]

h_t, out_t = get_h(target, ref_L)
probs_t = torch.softmax(out_t.logits[0, -1].float(), -1)
tid = torch.argmax(probs_t).item()
ttok = tok.decode([tid]).strip()
pt = float(probs_t[tid])

print(f"  Target top-1: '{ttok}' (p={pt:.3f})")
top5_t = topk_tok(out_t.logits[0, -1].float(), 5)
print(f"  Target top-5: {top5_t}")
del out_t
torch.cuda.empty_cache()

deltas = {}
p_cs = {}
for label, text in contrasts:
    h_c, out_c = get_h(text, ref_L)
    deltas[label] = (h_t - h_c).cpu()
    probs_c = torch.softmax(out_c.logits[0, -1].float(), -1)
    p_cs[label] = float(probs_c[tid])
    top5_c = topk_tok(out_c.logits[0, -1].float(), 5)
    print(f"  {label:>10} top-5: {top5_c}  P({ttok})={p_cs[label]:.3f}")
    del out_c
    torch.cuda.empty_cache()

# Pairwise cosine
print(f"\n  Pairwise cosine (selected):")
lbls = list(deltas.keys())
for i, li in enumerate(lbls):
    for j, lj in enumerate(lbls):
        if j > i:
            c = float(cos(deltas[li].unsqueeze(0), deltas[lj].unsqueeze(0)))
            if abs(c) > 0.5 or li in ["nobody", "everybody"] or lj in ["nobody", "everybody"]:
                print(f"    {li:>10} × {lj:<10}: {c:+.3f}")

# W_U readout for each Δh
print(f"\n  W_U readout per contrast at L{ref_L}:")
for label in lbls:
    dh = deltas[label].to(DEV)
    logits = dh @ W_U.T
    pos = topk_tok(logits, 6)
    neg = topk_tok(-logits, 6)
    print(f"    some-{label:>10}: +[{', '.join(pos)}]  -[{', '.join(neg)}]")

# Mean direction across ALL contrasts
all_dh = torch.stack([deltas[l].to(DEV) for l in lbls])
mean_dir = all_dh.mean(dim=0)
mean_norm = mean_dir / mean_dir.norm()

logits_m = mean_dir @ W_U.T
print(f"\n  Mean direction (all {len(contrasts)} contrasts):")
print(f"    +[{', '.join(topk_tok(logits_m, 10))}]")
print(f"    -[{', '.join(topk_tok(-logits_m, 10))}]")

# Mean of just the "universal" quantifiers (all, none, everybody, nobody)
universal_dh = torch.stack([deltas[l].to(DEV) for l in ["all", "none", "everybody", "nobody"]])
mean_univ = universal_dh.mean(dim=0)
logits_u = mean_univ @ W_U.T
print(f"\n  Mean direction (universal quantifiers: all, none, everybody, nobody):")
print(f"    +[{', '.join(topk_tok(logits_u, 10))}]")
print(f"    -[{', '.join(topk_tok(-logits_u, 10))}]")

# Mean of partitive quantifiers (most, few, many, half, three)
part_dh = torch.stack([deltas[l].to(DEV) for l in ["most", "few", "many", "half", "three"]])
mean_part = part_dh.mean(dim=0)
logits_p = mean_part @ W_U.T
print(f"\n  Mean direction (partitives: most, few, many, half, three):")
print(f"    +[{', '.join(topk_tok(logits_p, 10))}]")
print(f"    -[{', '.join(topk_tok(-logits_p, 10))}]")

# Causality of each mean direction
print(f"\n  Causal recovery of mean directions (injecting into 'all' baseline):")
text_all = contrasts[0][1]
gap_all = pt - p_cs["all"]
if abs(gap_all) > 0.001:
    for dir_label, direction in [("full_mean", mean_dir),
                                  ("universal_mean", mean_univ),
                                  ("partitive_mean", mean_part)]:
        p_inj = inject_and_measure(text_all, direction, ref_L)
        r = recov(p_inj, tid, pt, p_cs["all"])
        logits_d = direction @ W_U.T
        dtok = topk_tok(logits_d, 5)
        print(f"    {dir_label:>16}: rec={r:>+6.1f}%  reads=[{', '.join(dtok)}]")


# ══════════════════════════════════════════════════════════════
# PART 2: Cross-content — same some/all with different nouns
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: CROSS-CONTENT — some/all across noun phrases")
print("=" * 70)

noun_pairs = [
    ("students", "Some of the students passed the exam, so",
                 "All of the students passed the exam, so"),
    ("cookies",  "Some of the cookies were eaten at the party, and",
                 "All of the cookies were eaten at the party, and"),
    ("houses",   "Some of the houses were damaged in the storm, and",
                 "All of the houses were damaged in the storm, and"),
    ("employees","Some of the employees were promoted this year, which",
                 "All of the employees were promoted this year, which"),
    ("books",    "Some of the books were translated into French, so",
                 "All of the books were translated into French, so"),
    ("countries","Some of the countries signed the agreement, meaning",
                 "All of the countries signed the agreement, meaning"),
]

cross_deltas = {}
for label, text_some, text_all in noun_pairs:
    h_s, _ = get_h(text_some, ref_L)
    h_a, _ = get_h(text_all, ref_L)
    dh = h_s - h_a
    cross_deltas[label] = dh.cpu()
    logits = dh @ W_U.T
    pos = topk_tok(logits, 6)
    neg = topk_tok(-logits, 6)
    print(f"  {label:>10}: +[{', '.join(pos)}]  -[{', '.join(neg)}]")
    torch.cuda.empty_cache()

# Cross-content cosine
print(f"\n  Cross-content cosine (some/all axis consistency):")
cnlbls = list(cross_deltas.keys())
for i, li in enumerate(cnlbls):
    for j, lj in enumerate(cnlbls):
        if j > i:
            c = float(cos(cross_deltas[li].unsqueeze(0),
                           cross_deltas[lj].unsqueeze(0)))
            print(f"    {li:>10} × {lj:<10}: {c:+.3f}")

# Mean across noun phrases
cross_all = torch.stack([cross_deltas[l].to(DEV) for l in cnlbls])
cross_mean = cross_all.mean(dim=0)
cross_mean_norm = cross_mean / cross_mean.norm()

logits_cm = cross_mean @ W_U.T
print(f"\n  Cross-content mean direction (some/all, 6 noun phrases):")
print(f"    +[{', '.join(topk_tok(logits_cm, 10))}]")
print(f"    -[{', '.join(topk_tok(-logits_cm, 10))}]")

# Per-noun decompose into shared + unique, with causality
print(f"\n  Shared vs unique for each noun phrase:")
for label, text_some, text_all in noun_pairs:
    dh = cross_deltas[label].to(DEV)
    shared = (dh @ cross_mean_norm) * cross_mean_norm
    unique = dh - shared

    h_a, out_a = get_h(text_all, ref_L)
    h_s, out_s = get_h(text_some, ref_L)
    probs_s = torch.softmax(out_s.logits[0, -1].float(), -1)
    probs_a = torch.softmax(out_a.logits[0, -1].float(), -1)
    # Use the some-prompt's top-1 as target
    local_tid = torch.argmax(probs_s).item()
    local_tok = tok.decode([local_tid]).strip()
    pa = float(probs_s[local_tid])
    pb = float(probs_a[local_tid])
    gap = pa - pb
    del out_a, out_s
    torch.cuda.empty_cache()

    if abs(gap) > 0.001:
        p_f = inject_and_measure(text_all, dh, ref_L)
        p_s = inject_and_measure(text_all, shared, ref_L)
        p_u = inject_and_measure(text_all, unique, ref_L)

        s_logits = shared @ W_U.T
        u_logits = unique @ W_U.T
        print(f"    {label:>10} (target='{local_tok}'): "
              f"shared=[{', '.join(topk_tok(s_logits, 3))}] rec={recov(p_s, local_tid, pa, pb):>+6.1f}%  "
              f"unique=[{', '.join(topk_tok(u_logits, 3))}] rec={recov(p_u, local_tid, pa, pb):>+6.1f}%  "
              f"full={recov(p_f, local_tid, pa, pb):>+6.1f}%")
    else:
        print(f"    {label:>10}: gap too small")


# ══════════════════════════════════════════════════════════════
# PART 3: Layer trajectory of the shared direction
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: LAYER TRAJECTORY of cross-content some/all mean")
print("=" * 70)

text_some = "Some of the students passed the exam, so"
text_all = "All of the students passed the exam, so"

ids_s = tok(text_some, add_special_tokens=False)["input_ids"]
ids_a = tok(text_all, add_special_tokens=False)["input_ids"]
with torch.no_grad():
    out_s = model(torch.tensor([ids_s], device=DEV), output_hidden_states=True)
    out_a = model(torch.tensor([ids_a], device=DEV), output_hidden_states=True)

for L in layers:
    h_s = out_s.hidden_states[L][0, -1, :].float()
    h_a = out_a.hidden_states[L][0, -1, :].float()
    dh = h_s - h_a

    # Project onto cross-content mean
    proj = float(dh @ cross_mean_norm.to(DEV))

    # Also read W_U
    logits = dh @ W_U.T
    pos = topk_tok(logits, 5)
    neg = topk_tok(-logits, 5)
    norm = float(dh.norm() / h_s.norm())

    print(f"  L{L:>2} ({norm:.3f}) proj={proj:>+7.1f}  "
          f"+[{', '.join(pos)}]  -[{', '.join(neg)}]")

del out_s, out_a
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART 4: Hotdog salty/sweet axis check
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: HOT DOG vs HOT CHOCOLATE — salty/sweet or meat/sweet?")
print("=" * 70)

# Get the unique component from hotdog vs hot_choc
h_hd, _ = get_h("The hot dog was", ref_L)
h_hc, _ = get_h("The hot chocolate was", ref_L)
dh_hd_hc = h_hd - h_hc

logits_hdhc = dh_hd_hc @ W_U.T
print(f"  hot_dog - hot_chocolate:")
print(f"    +[{', '.join(t[0] for t in topk_str(logits_hdhc, 10))}]")
print(f"    -[{', '.join(t[0] for t in topk_str(-logits_hdhc, 10))}]")

# Also check related taste axes
taste_pairs = [
    ("salty_sweet", "The food tasted very salty and", "The food tasted very sweet and"),
    ("savory_sweet", "The dish was savory and", "The dish was sweet and"),
    ("meat_veg", "He ate the grilled steak and", "He ate the fresh salad and"),
    ("fried_baked", "The fried chicken was", "The baked chicken was"),
    ("hot_cold_food", "The hot soup was", "The cold soup was"),
]

taste_dirs = {}
for label, text_a, text_b in taste_pairs:
    ha, _ = get_h(text_a, ref_L)
    hb, _ = get_h(text_b, ref_L)
    d = ha - hb
    taste_dirs[label] = d
    logits = d @ W_U.T
    print(f"\n  {label}:")
    print(f"    +[{', '.join(t[0] for t in topk_str(logits, 8))}]")
    print(f"    -[{', '.join(t[0] for t in topk_str(-logits, 8))}]")
    torch.cuda.empty_cache()

# Cosine of hotdog-hotchoc with each taste axis
print(f"\n  Cosine of (hot_dog - hot_chocolate) with taste axes:")
for label, d in taste_dirs.items():
    c = float(cos(dh_hd_hc.unsqueeze(0), d.unsqueeze(0)))
    print(f"    {label:>16}: {c:+.3f}")

torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
