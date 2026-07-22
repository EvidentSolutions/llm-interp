"""
Causal test: is the "edible" signal under H19's output causally efficacious?

1. Extract minimum eatability direction from multiple contrasts
2. Inject it into non-food dog prompts at the dog position
3. Check if prediction shifts toward food tokens
4. Subtract it from hot-dog and check if prediction shifts away
5. Vary injection magnitude to find threshold

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_causal_edible.py
"""
import sys
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    for p in model.parameters(): p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float()
    else:
        W_U = model.embed_out.weight.detach().float()

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=5):
        probs = torch.softmax(logits.float(), -1)
        v, i = torch.topk(probs, k)
        return [(tok.decode([int(i[j])]).strip(), round(float(v[j]), 4))
                for j in range(k)]

    def predict(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV))
        return tk(out.logits[0, -1])

    # The injection layer: L4 post = hidden_states[5] = L5 input
    # We hook into layer 5's input and add a perturbation at dog_pos
    inject_L = 5  # we intervene at the INPUT to this layer

    # ================================================================
    # 1. Extract eatability directions from multiple contrasts
    # ================================================================
    print(f"{'='*100}")
    print(f"1. EXTRACT EATABILITY DIRECTIONS")
    print(f"{'='*100}")

    target = "The hot dog was"
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    dog_pos = 2

    food_contrasts = [
        ("angry dog",  "The angry dog was"),
        ("pet dog",    "The pet dog was"),
        ("old dog",    "The old dog was"),
        ("stray dog",  "The stray dog was"),
        ("cold dog",   "The cold dog was"),
    ]

    # Get target hidden state at L5 input (= hidden_states[inject_L])
    with torch.no_grad():
        out_t = model(torch.tensor([target_ids], device=DEV),
                      output_hidden_states=True)
    h_target = out_t.hidden_states[inject_L][0, dog_pos, :].float().cpu()
    del out_t; torch.cuda.empty_cache()

    directions = {}
    for label, ctext in food_contrasts:
        c_ids = tok(ctext, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out_c = model(torch.tensor([c_ids], device=DEV),
                          output_hidden_states=True)
        h_c = out_c.hidden_states[inject_L][0, dog_pos, :].float().cpu()
        del out_c; torch.cuda.empty_cache()

        d = h_target - h_c
        directions[label] = d
        ld = d @ W_U.cpu().T
        print(f"  hot dog - {label:>12}: ||={float(d.norm()):>5.1f}  [{', '.join(t[0] for t in tk(ld, 5))}]")

    # Mean direction
    mean_dir = torch.stack(list(directions.values())).mean(dim=0)
    mean_dir_norm = mean_dir / mean_dir.norm()
    ld_mean = mean_dir @ W_U.cpu().T
    print(f"\n  Mean eatability direction:")
    print(f"    ||mean||={float(mean_dir.norm()):.1f}")
    print(f"    W_U reads: [{', '.join(t[0] for t in tk(ld_mean, 8))}]")

    # Pairwise cosine to check consistency
    labels = list(directions.keys())
    print(f"\n  Pairwise cosine:")
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if j > i:
                c = float(F.cosine_similarity(
                    directions[l1].unsqueeze(0),
                    directions[l2].unsqueeze(0)))
                print(f"    {l1} vs {l2}: {c:+.3f}")

    # Project each individual direction onto the mean to find
    # the shared component
    print(f"\n  Projection onto mean direction:")
    for label, d in directions.items():
        proj = float(torch.dot(d, mean_dir_norm))
        cos = float(F.cosine_similarity(d.unsqueeze(0), mean_dir_norm.unsqueeze(0)))
        print(f"    {label:>12}: proj={proj:>+6.1f}  cos={cos:+.3f}")

    # ================================================================
    # 2. Baseline predictions
    # ================================================================
    print(f"\n{'='*100}")
    print(f"2. BASELINE PREDICTIONS")
    print(f"{'='*100}")

    test_prompts = [
        "The hot dog was",
        "The cold dog was",
        "The angry dog was",
        "The pet dog was",
        "The old dog was",
        "The big dog was",
    ]
    for text in test_prompts:
        preds = predict(text)
        print(f"  {text:>25} → {preds}")

    # ================================================================
    # 3. CAUSAL INJECTION: add eatability to non-food dogs
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. CAUSAL INJECTION — add eatability direction at dog pos")
    print(f"{'='*100}")

    def predict_with_injection(text, direction, scale, pos):
        """Run model with a hook that adds direction*scale at pos in layer inject_L."""
        ids = tok(text, add_special_tokens=False)["input_ids"]

        perturbation = (direction * scale).half().to(DEV)

        def hook_fn(module, args):
            # args[0] is the hidden states input to this layer
            h = args[0]
            h_new = h.clone()
            h_new[0, pos, :] += perturbation
            return (h_new,) + args[1:]

        # Hook the layer's forward (pre-hook to modify input)
        layer = model.model.layers[inject_L]
        handle = layer.register_forward_pre_hook(hook_fn)

        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV))

        handle.remove()
        return tk(out.logits[0, -1])

    # Inject mean eatability direction at various scales
    inject_dir = mean_dir_norm.cpu()
    base_scale = float(mean_dir.norm())  # natural scale

    for text in ["The cold dog was", "The angry dog was", "The pet dog was"]:
        print(f"\n  {text}:")
        baseline = predict(text)
        print(f"    baseline:     {baseline}")
        for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
            scale = base_scale * frac
            preds = predict_with_injection(text, inject_dir, scale, dog_pos)
            print(f"    +{frac:.2f}× eat:   {preds}")

    # ================================================================
    # 4. CAUSAL SUBTRACTION: remove eatability from hot dog
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. CAUSAL SUBTRACTION — remove eatability from hot dog")
    print(f"{'='*100}")

    text = "The hot dog was"
    baseline = predict(text)
    print(f"  {text}:")
    print(f"    baseline:     {baseline}")
    for frac in [-0.25, -0.5, -1.0, -1.5, -2.0]:
        scale = base_scale * frac
        preds = predict_with_injection(text, inject_dir, scale, dog_pos)
        print(f"    {frac:+.2f}× eat:   {preds}")

    # ================================================================
    # 5. INDIVIDUAL CONTRAST DIRECTIONS — which is most causally potent?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"5. INDIVIDUAL DIRECTIONS — which contrast is most causally potent?")
    print(f"{'='*100}")

    target_text = "The cold dog was"
    baseline = predict(target_text)
    print(f"  Injecting into: {target_text}")
    print(f"  Baseline: {baseline}")

    for label, d in directions.items():
        d_norm = d / d.norm()
        nat_scale = float(d.norm())
        preds = predict_with_injection(target_text, d_norm, nat_scale, dog_pos)
        print(f"  +1× {label:>12}: {preds}")

    # ================================================================
    # 6. MINIMUM DOSE — what's the smallest injection that changes top-1?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"6. MINIMUM DOSE — smallest injection that changes top-1 prediction")
    print(f"{'='*100}")

    for target_text in ["The cold dog was", "The angry dog was"]:
        baseline = predict(target_text)
        baseline_top1 = baseline[0][0]
        print(f"\n  {target_text}  baseline top-1: {baseline_top1}")

        for frac_pct in range(5, 205, 5):
            frac = frac_pct / 100.0
            scale = base_scale * frac
            preds = predict_with_injection(target_text, inject_dir, scale, dog_pos)
            top1 = preds[0][0]
            if top1 != baseline_top1:
                print(f"    FLIPS at {frac:.2f}× ({scale:.1f} norm): "
                      f"{baseline_top1} → {top1}  {preds[:3]}")
                break
        else:
            print(f"    No flip up to 2.0×")

    # ================================================================
    # 7. DOES IT WORK AT THE WAS POSITION TOO?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"7. INJECTION AT WAS POSITION (pos 3) — later is less direct")
    print(f"{'='*100}")

    text = "The cold dog was"
    baseline = predict(text)
    print(f"  {text}  baseline: {baseline[:3]}")

    for pos_label, pos in [("dog (pos 2)", 2), ("was (pos 3)", 3)]:
        for frac in [0.5, 1.0, 2.0]:
            scale = base_scale * frac
            preds = predict_with_injection(text, inject_dir, scale, pos)
            print(f"  +{frac:.1f}× at {pos_label}: {preds[:3]}")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
