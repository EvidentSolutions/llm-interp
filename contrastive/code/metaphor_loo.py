"""
Leave-one-out cross-validation for metaphor domain routing.

For each word (cold, sharp, bright, heavy):
  - 4 literal-metaphorical pairs available
  - Extract routing direction from 3 pairs
  - Test injection on the held-out 4th pair
  - Report: does the held-out pair still flip?
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
W_U = model.lm_head.weight.detach().float()


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_tok(logits, k=5):
    return [tok.decode([int(i)]).strip()[:14]
            for i in torch.topk(logits.float(), k).indices]


def get_h(text, layer):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float()


def predict(text, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)]


def inject_and_predict(text, delta, layer, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
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
        out = model(torch.tensor([ids], device=DEV))
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)]


ref_L = _sl(24)[0]

domains = {
    "cold": [
        ("The ice in the bucket was extremely cold. The temperature was",
         "The reception at the party was extremely cold. The atmosphere was"),
        ("The water in the lake was extremely cold. The temperature was",
         "The welcome from the host was extremely cold. The atmosphere was"),
        ("The metal railing was extremely cold. The temperature was",
         "The response from the audience was extremely cold. The atmosphere was"),
        ("The wind outside was extremely cold. The temperature was",
         "The tone of the email was extremely cold. The atmosphere was"),
    ],
    "sharp": [
        ("The knife on the counter was extremely sharp. The blade was",
         "The criticism in the review was extremely sharp. The tone was"),
        ("The scissors were extremely sharp. The blade was",
         "The rebuke from the manager was extremely sharp. The tone was"),
        ("The razor was extremely sharp. The blade was",
         "The wit of the comedian was extremely sharp. The tone was"),
        ("The needle was extremely sharp. The blade was",
         "The sarcasm in her voice was extremely sharp. The tone was"),
    ],
    "bright": [
        ("The lamp in the corner was extremely bright. The light was",
         "The student in the class was extremely bright. The child was"),
        ("The spotlight was extremely bright. The light was",
         "The idea she proposed was extremely bright. The child was"),
        ("The headlights were extremely bright. The light was",
         "The future of the company was extremely bright. The outlook was"),
        ("The screen was extremely bright. The light was",
         "The young scientist was extremely bright. The researcher was"),
    ],
    "heavy": [
        ("The boulder on the trail was extremely heavy. The weight was",
         "The news from the hospital was extremely heavy. The mood was"),
        ("The barbell was extremely heavy. The weight was",
         "The silence in the room was extremely heavy. The mood was"),
        ("The suitcase was extremely heavy. The weight was",
         "The responsibility on her shoulders was extremely heavy. The burden was"),
        ("The anchor was extremely heavy. The weight was",
         "The atmosphere after the argument was extremely heavy. The mood was"),
    ],
}

print("=" * 70)
print("LEAVE-ONE-OUT CROSS-VALIDATION FOR METAPHOR DOMAIN ROUTING")
print("=" * 70)

for word, pairs in domains.items():
    print(f"\n{'─'*70}")
    print(f"  {word.upper()} ({len(pairs)} pairs)")
    print(f"{'─'*70}")

    for held_out_idx in range(len(pairs)):
        # Extract direction from all pairs EXCEPT held_out
        train_pairs = [p for i, p in enumerate(pairs) if i != held_out_idx]
        test_lit, test_met = pairs[held_out_idx]

        train_dirs = []
        for lit, met in train_pairs:
            h_lit = get_h(lit, ref_L)
            h_met = get_h(met, ref_L)
            train_dirs.append((h_lit - h_met).cpu())
            torch.cuda.empty_cache()

        mean_dir = torch.stack(train_dirs).mean(dim=0).to(DEV)

        # Test on held-out pair
        base_lit = predict(test_lit, 3)
        base_met = predict(test_met, 3)

        # Inject literal direction into metaphorical context
        inj_met = inject_and_predict(test_met, mean_dir, ref_L, 3)
        # Inject metaphorical direction into literal context
        inj_lit = inject_and_predict(test_lit, -mean_dir, ref_L, 3)

        print(f"\n    Hold out pair {held_out_idx}:")
        print(f"      Literal:  \"{test_lit[-45:]}\"")
        print(f"        baseline: {base_lit}")
        print(f"        −1× (→metaph): {inj_lit}")
        print(f"      Metaph:   \"{test_met[-45:]}\"")
        print(f"        baseline: {base_met}")
        print(f"        +1× (→literal): {inj_met}")

        torch.cuda.empty_cache()


print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
