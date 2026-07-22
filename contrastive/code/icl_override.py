"""
ICL override experiment.

Question: When in-context information contradicts parametric knowledge,
where does the model shift from parametric recall to contextual override?

Design:
  Prompt K (parametric): "The capital of France is Paris. The capital of France is"
  Prompt C (counterfactual): "The capital of France is Rome. The capital of France is"

The model knows Paris. The counterfactual prompt asserts Rome.
At each layer, the contrastive projection should show where "Rome"
begins to appear and "Paris" begins to be suppressed.

We test:
1. Basic trajectory: where does Rome emerge / Paris get suppressed?
2. Multiple facts: same design for several country/capital pairs
3. Distractor length: does inserting distractor text between the
   assertion and the query delay the override?
4. Multi-contrast: multiple counterfactual targets to isolate the
   "override" component from the specific city
5. Framing strength: "Some say" vs "It is well known that" vs bare assertion

Full prompts shown with all results.
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


def topk_tok(logits, k=5):
    return [tok.decode([int(i)]).strip()[:14]
            for i in torch.topk(logits.float(), k).indices]


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out


def predict(text, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)]


layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)
# Remove 32 if it exceeds NL
layers = [L for L in layers if L <= NL]


# ══════════════════════════════════════════════════════════════
# PART 1: BASIC TRAJECTORY — Paris vs Rome override
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: BASIC ICL OVERRIDE TRAJECTORY")
print("=" * 70)

pairs = [
    ("France/Rome",
     "The capital of France is Rome. The capital of France is",
     "The capital of France is Paris. The capital of France is",
     "Paris", "Rome"),
    ("Japan/London",
     "The capital of Japan is London. The capital of Japan is",
     "The capital of Japan is Tokyo. The capital of Japan is",
     "Tokyo", "London"),
    ("Germany/Madrid",
     "The capital of Germany is Madrid. The capital of Germany is",
     "The capital of Germany is Berlin. The capital of Germany is",
     "Berlin", "Madrid"),
    ("Italy/Vienna",
     "The capital of Italy is Vienna. The capital of Italy is",
     "The capital of Italy is Rome. The capital of Italy is",
     "Rome", "Vienna"),
]

for name, text_c, text_k, correct, override in pairs:
    print(f"\n  {name}:")
    print(f"    Counterfactual: \"{text_c}\"")
    print(f"    Parametric:     \"{text_k}\"")

    preds_c = predict(text_c)
    preds_k = predict(text_k)
    print(f"    Predictions counterfactual: {preds_c[:3]}")
    print(f"    Predictions parametric:     {preds_k[:3]}")

    out_c = get_hidden_states(text_c)
    out_k = get_hidden_states(text_k)

    print(f"\n    Trajectory (counterfactual − parametric):")
    print(f"    {'L':>4}  {'norm':>6}  +side (counterfactual)          -side (parametric)")
    for L in layers:
        hc = out_c.hidden_states[L][0, -1, :].float()
        hk = out_k.hidden_states[L][0, -1, :].float()
        dh = hc - hk
        norm = float(dh.norm() / hc.norm())
        logits = dh @ W_U.T
        pos = topk_tok(logits, 5)
        neg = topk_tok(-logits, 5)
        print(f"    L{L:>2}  {norm:.3f}  +[{', '.join(pos)}]  -[{', '.join(neg)}]")

    del out_c, out_k
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART 2: DOES THE MODEL ACTUALLY OVERRIDE?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: DOES THE MODEL OVERRIDE? (behavioral check)")
print("=" * 70)

override_tests = [
    # Bare assertion
    ("bare", "The capital of France is Rome. The capital of France is"),
    # Repeated assertion
    ("repeated", "The capital of France is Rome. Remember, the capital of France is Rome. The capital of France is"),
    # With distractor
    ("distractor", "The capital of France is Rome. The weather has been unusually warm this summer, with temperatures reaching record highs across Europe. The capital of France is"),
    # Weak framing
    ("weak", "Some say the capital of France is Rome. The capital of France is"),
    # Strong framing
    ("strong", "It is well established that the capital of France is Rome. The capital of France is"),
    # Fictional world
    ("fictional", "In the wizarding world, the capital of France is Rome. In the wizarding world, the capital of France is"),
    # No override (baseline)
    ("baseline", "The capital of France is"),
]

print(f"\n  France/capital override with different framings:")
for label, prompt in override_tests:
    preds = predict(prompt)
    print(f"    {label:>12}: {prompt[-50:]}")
    print(f"                 → {preds[:4]}")


# ══════════════════════════════════════════════════════════════
# PART 3: TRAJECTORY WITH DISTRACTOR LENGTH SWEEP
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: DISTRACTOR LENGTH — does distance delay override?")
print("=" * 70)

distractors = [
    ("0 words", ""),
    ("short", " The weather is nice."),
    ("medium", " The weather has been warm this summer, with temperatures reaching highs across the continent."),
    ("long", " The weather has been unusually warm this summer, with temperatures reaching record highs across Europe. Scientists attribute this to changing climate patterns and increased greenhouse gas emissions in the atmosphere."),
]

ref_L = _sl(28)[0]

for d_label, distractor in distractors:
    text_c = f"The capital of France is Rome.{distractor} The capital of France is"
    text_k = f"The capital of France is Paris.{distractor} The capital of France is"

    preds_c = predict(text_c)

    out_c = get_hidden_states(text_c)
    out_k = get_hidden_states(text_k)
    hc = out_c.hidden_states[ref_L][0, -1, :].float()
    hk = out_k.hidden_states[ref_L][0, -1, :].float()
    dh = hc - hk
    logits = dh @ W_U.T
    pos = topk_tok(logits, 5)
    neg = topk_tok(-logits, 5)

    print(f"\n  {d_label}:")
    print(f"    Prediction: {preds_c[:3]}")
    print(f"    L{ref_L} +[{', '.join(pos)}]  -[{', '.join(neg)}]")

    del out_c, out_k
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART 4: MULTI-CONTRAST — isolate the "override" component
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: MULTI-CONTRAST — what is the override signal?")
print("=" * 70)

# Same country, different counterfactual cities
target = "The capital of France is Rome. The capital of France is"
baselines = [
    ("vs_London", "The capital of France is London. The capital of France is"),
    ("vs_Madrid", "The capital of France is Madrid. The capital of France is"),
    ("vs_Berlin", "The capital of France is Berlin. The capital of France is"),
    ("vs_Tokyo", "The capital of France is Tokyo. The capital of France is"),
    ("vs_Paris", "The capital of France is Paris. The capital of France is"),
]

h_target, out_target = None, None
out_t = get_hidden_states(target)
h_t = out_t.hidden_states[ref_L][0, -1, :].float()

deltas = {}
for label, baseline in baselines:
    out_b = get_hidden_states(baseline)
    h_b = out_b.hidden_states[ref_L][0, -1, :].float()
    dh = h_t - h_b
    deltas[label] = dh.cpu()

    logits = dh @ W_U.T
    pos = topk_tok(logits, 5)
    neg = topk_tok(-logits, 5)
    print(f"  {label}: +[{', '.join(pos)}]  -[{', '.join(neg)}]")

    del out_b
    torch.cuda.empty_cache()

del out_t
torch.cuda.empty_cache()

# Pairwise cosine
print(f"\n  Pairwise cosine between override Δh vectors:")
lbls = list(deltas.keys())
for i, li in enumerate(lbls):
    for j, lj in enumerate(lbls):
        if j > i:
            c = float(cos(deltas[li].unsqueeze(0), deltas[lj].unsqueeze(0)))
            print(f"    {li:>12} × {lj:<12}: {c:+.3f}")

# Mean of non-Paris contrasts (Rome vs other-wrong-cities)
non_paris = ["vs_London", "vs_Madrid", "vs_Berlin", "vs_Tokyo"]
mean_dir = torch.stack([deltas[l] for l in non_paris]).mean(dim=0)
logits_m = mean_dir.to(DEV) @ W_U.T
print(f"\n  Mean (Rome vs other wrong cities):")
print(f"    +[{', '.join(topk_tok(logits_m, 8))}]")
print(f"    -[{', '.join(topk_tok(-logits_m, 8))}]")

# Rome vs Paris specifically
logits_rp = deltas["vs_Paris"].to(DEV) @ W_U.T
print(f"\n  Rome vs Paris (the override contrast):")
print(f"    +[{', '.join(topk_tok(logits_rp, 8))}]")
print(f"    -[{', '.join(topk_tok(-logits_rp, 8))}]")


# ══════════════════════════════════════════════════════════════
# PART 5: LAYER-RESOLVED — where does override happen?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: LAYER-RESOLVED OVERRIDE TRAJECTORY (Rome vs Paris)")
print("=" * 70)

text_rome = "The capital of France is Rome. The capital of France is"
text_paris = "The capital of France is Paris. The capital of France is"

out_rome = get_hidden_states(text_rome)
out_paris = get_hidden_states(text_paris)

# Track P(Rome) and P(Paris) token IDs
rome_id = tok.encode(" Rome", add_special_tokens=False)[0]
paris_id = tok.encode(" Paris", add_special_tokens=False)[0]

print(f"\n  Layer-by-layer contrastive (Rome assertion − Paris assertion):")
print(f"  Also tracking specific token projections for Rome and Paris.\n")

for L in layers:
    hr = out_rome.hidden_states[L][0, -1, :].float()
    hp = out_paris.hidden_states[L][0, -1, :].float()
    dh = hr - hp
    norm = float(dh.norm() / hr.norm())
    logits = dh @ W_U.T

    # Where do Rome and Paris rank?
    rome_logit = float(logits[rome_id])
    paris_logit = float(logits[paris_id])

    pos = topk_tok(logits, 5)
    neg = topk_tok(-logits, 5)

    print(f"  L{L:>2} ({norm:.3f})  Rome={rome_logit:>+7.1f}  Paris={paris_logit:>+7.1f}  "
          f"+[{', '.join(pos)}]  -[{', '.join(neg)}]")

del out_rome, out_paris
torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
