"""
Data-driven decomposition of what the model stores at the noun
position of "The [adj] [noun] was" compounds.

Instead of imposing semantic axes, collect many variants at the
noun position and let SVD find the model's own axes. Then read
each principal component through W_U to see what it means.

Variants:
  adjectives: hot, cold, old, young, angry, happy, big, small, ...
  nouns: dog, cat, rod, car, chocolate, sauce, water, ...
  compounds: hot dog, hot rod, hot chocolate, hot sauce, cold dog, ...

The SVD of centered hidden states at the noun position tells us
what dimensions the model uses to distinguish these, and W_U
tells us what each dimension means in token space.

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_pca_decomp.py
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
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu()
    else:
        W_U = model.embed_out.weight.detach().float().cpu()

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=8):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    ref_L = _sl(5)[0]  # L4 post = where compound is recognized

    # ================================================================
    # Collect hidden states at noun position for many "The [adj] [noun] was"
    # ================================================================
    adjectives = ["hot", "cold", "old", "young", "angry", "happy",
                  "big", "small", "red", "blue", "new", "broken",
                  "fresh", "raw", "fried", "grilled", "cheap", "fancy"]
    nouns = ["dog", "cat", "rod", "car", "chocolate", "sauce",
             "water", "air", "chicken", "fish", "potato", "cake",
             "coffee", "tea", "burger", "pizza", "soup", "plate"]

    print(f"Collecting hidden states at noun pos (L{ref_L-1} post)...")
    print(f"  {len(adjectives)} adjectives × {len(nouns)} nouns = {len(adjectives)*len(nouns)} prompts")

    states = {}
    labels = []
    vecs = []

    for adj in adjectives:
        for noun in nouns:
            text = f"The {adj} {noun} was"
            ids = tok(text, add_special_tokens=False)["input_ids"]
            tokens = [tok.decode([t]) for t in ids]

            # Find noun position (should be pos 2 for "The adj noun was")
            noun_pos = None
            for i, t in enumerate(tokens):
                if noun.lower() in t.lower().strip():
                    noun_pos = i
                    break
            if noun_pos is None:
                continue

            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            h = out.hidden_states[ref_L][0, noun_pos, :].float().cpu()
            del out
            torch.cuda.empty_cache()

            key = f"{adj}_{noun}"
            states[key] = h
            labels.append(key)
            vecs.append(h)

    print(f"  Collected {len(vecs)} states")

    mat = torch.stack(vecs)  # (N, hidden)
    center = mat.mean(dim=0)
    centered = mat - center

    # ================================================================
    # SVD
    # ================================================================
    print(f"\n{'='*100}")
    print(f"1. SVD of centered noun-position states")
    print(f"{'='*100}")

    U, S, V = torch.svd(centered)
    total_var = S.pow(2).sum()

    print(f"\n  Top 20 singular values:")
    for i in range(min(20, len(S))):
        pct = S[i]**2 / total_var
        cum = S[:i+1].pow(2).sum() / total_var
        print(f"    PC{i+1:>2}: S={S[i]:.1f}  var={pct:.1%}  cumulative={cum:.1%}")

    # ================================================================
    # What does each PC read as through W_U?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"2. WHAT EACH PC READS AS through W_U")
    print(f"{'='*100}")

    for i in range(min(12, len(S))):
        pc = V[:, i]  # unit vector in hidden space
        ld = pc @ W_U.T
        print(f"\n  PC{i+1} (var={S[i]**2/total_var:.1%}):")
        print(f"    + pole: [{tk(ld)}]")
        print(f"    - pole: [{tk(-ld)}]")

    # ================================================================
    # Project specific compounds onto PCs
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. PROJECTION OF KEY COMPOUNDS onto top PCs")
    print(f"{'='*100}")

    key_compounds = [
        "hot_dog", "cold_dog", "hot_cat", "hot_rod",
        "angry_dog", "old_dog", "hot_chocolate", "hot_sauce",
        "fried_chicken", "grilled_chicken", "fresh_fish",
        "hot_coffee", "cold_water", "big_burger", "cheap_pizza",
    ]

    # Header
    print(f"\n  {'compound':>20}", end="")
    for i in range(8):
        print(f" {'PC'+str(i+1):>7}", end="")
    print()

    for key in key_compounds:
        if key not in states:
            continue
        h = states[key] - center
        print(f"  {key:>20}", end="")
        for i in range(8):
            pc = V[:, i]
            proj = float(torch.dot(h, pc))
            print(f" {proj:>+7.1f}", end="")
        print()

    # ================================================================
    # Reconstruct hot dog - cold dog using PCs
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. RECONSTRUCT 'hot dog - cold dog' from PCs")
    print(f"{'='*100}")

    delta = states["hot_dog"] - states["cold_dog"]
    ld_full = delta @ W_U.T
    print(f"\n  Full Δ: [{tk(ld_full)}]  ||Δ||={float(delta.norm()):.1f}")

    cumulative = torch.zeros_like(delta)
    print(f"\n  {'PCs':>6} {'cos':>8} {'var%':>7}  reads as")

    for n_pcs in [1, 2, 3, 4, 5, 8, 12, 20, 40, 80, len(S)]:
        n_pcs = min(n_pcs, len(S))
        # Project delta onto top-n PCs
        V_n = V[:, :n_pcs]
        recon = V_n @ (V_n.T @ delta)
        c = float(F.cosine_similarity(delta.unsqueeze(0), recon.unsqueeze(0)))
        v = float(recon.norm()**2 / delta.norm()**2)
        ld = recon @ W_U.T
        print(f"  {n_pcs:>4}PC {c:>+8.4f} {v:>6.1%}  [{tk(ld, 6)}]")

    # ================================================================
    # What do the top PCs separate? Group by adjective and noun
    # ================================================================
    print(f"\n{'='*100}")
    print(f"5. WHAT DO PCs SEPARATE? Mean projection by adjective and noun")
    print(f"{'='*100}")

    # By adjective
    print(f"\n  Mean PC projection by adjective:")
    print(f"  {'adj':>10}", end="")
    for i in range(6):
        print(f" {'PC'+str(i+1):>7}", end="")
    print()

    for adj in adjectives:
        projs = []
        for noun in nouns:
            key = f"{adj}_{noun}"
            if key in states:
                h = states[key] - center
                projs.append([float(torch.dot(h, V[:, i])) for i in range(6)])
        if projs:
            means = [sum(p[i] for p in projs)/len(projs) for i in range(6)]
            print(f"  {adj:>10}", end="")
            for m in means:
                print(f" {m:>+7.1f}", end="")
            print()

    # By noun
    print(f"\n  Mean PC projection by noun:")
    print(f"  {'noun':>12}", end="")
    for i in range(6):
        print(f" {'PC'+str(i+1):>7}", end="")
    print()

    for noun in nouns:
        projs = []
        for adj in adjectives:
            key = f"{adj}_{noun}"
            if key in states:
                h = states[key] - center
                projs.append([float(torch.dot(h, V[:, i])) for i in range(6)])
        if projs:
            means = [sum(p[i] for p in projs)/len(projs) for i in range(6)]
            print(f"  {noun:>12}", end="")
            for m in means:
                print(f" {m:>+7.1f}", end="")
            print()

    # ================================================================
    # Can we identify what each PC corresponds to?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"6. PC INTERPRETATION — which adjective/noun contrasts align with each PC?")
    print(f"{'='*100}")

    for i in range(6):
        pc = V[:, i]
        # Find which compounds project most positively and negatively
        projs = [(key, float(torch.dot(states[key] - center, pc)))
                 for key in labels]
        projs.sort(key=lambda x: -x[1])
        top5 = projs[:5]
        bot5 = projs[-5:]
        print(f"\n  PC{i+1} (var={S[i]**2/total_var:.1%}):")
        print(f"    W_U +: [{tk(pc @ W_U.T)}]")
        print(f"    W_U -: [{tk(-pc @ W_U.T)}]")
        print(f"    Top-5: {', '.join(f'{k}({v:+.1f})' for k,v in top5)}")
        print(f"    Bot-5: {', '.join(f'{k}({v:+.1f})' for k,v in bot5)}")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
