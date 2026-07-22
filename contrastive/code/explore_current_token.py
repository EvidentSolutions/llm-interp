"""
Current-token integration: what happens when the differing word IS the
last token? The contrast at L0 is pure embedding difference; deeper
layers show how the token integrates with context.

Three experiments:
1. Same context, different last token (pure token integration)
2. Different context, same last token (pure context effect — baseline)
3. 2x2: {context} × {last token}

Usage: .venv/Scripts/python.exe contrastive/code/explore_current_token.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")


# 2x2 grids: (label, prefix_template, last_tokens, contexts)
# prefix_template has {ctx} slot, last token appended separately
GRIDS = [
    ("cat_dog × table_floor",
     "The {ctx} sat on the",
     (" table", " floor"),
     ("cat", "dog")),

    ("he_she × ran_walked",
     "When the bell rang, {ctx}",
     (" ran", " walked"),
     ("he", "she")),

    ("knife_spoon × dropped_found",
     "{ctx} a {ctx2} and",
     (" screamed", " smiled"),
     # special: need different handling
     None),

    ("hot_cold × water_soup",
     "The {ctx} was very",
     (" hot", " cold"),
     ("water", "soup")),

    ("big_small × house_car",
     "The {ctx} was very",
     (" big", " small"),
     ("house", "car")),

    ("old_new × man_building",
     "The {ctx} was very",
     (" old", " new"),
     ("man", "building")),

    ("happy_sad × boy_girl",
     "The {ctx} was very",
     (" happy", " sad"),
     ("boy", "girl")),

    ("clean_dirty × kitchen_street",
     "The {ctx} was very",
     (" clean", " dirty"),
     ("kitchen", "street")),

    ("full_empty × glass_room",
     "The {ctx} was",
     (" full", " empty"),
     ("glass", "room")),

    ("alive_dead × fish_bird",
     "The {ctx} was",
     (" alive", " dead"),
     ("fish", "bird")),

    ("open_closed × door_book",
     "The {ctx} was",
     (" open", " closed"),
     ("door", "book")),

    ("broken_fixed × car_watch",
     "The {ctx} was finally",
     (" broken", " fixed"),
     ("car", "watch")),
]


def get_top5(logits, tok):
    probs = torch.softmax(logits, -1)
    vals, idxs = torch.topk(probs, 5)
    return [(tok.decode([int(idxs[i])]).strip(), round(float(vals[i]), 4))
            for i in range(5)]


def get_topk_str(logits, tok, k=5, largest=True):
    vals, idxs = torch.topk(logits, k, largest=largest)
    return ", ".join(tok.decode([int(idxs[j])]).strip()[:12] for j in range(k))


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    W_U = model.lm_head.weight.detach()

    for label, prefix_template, (tok_a, tok_b), contexts in GRIDS:
        if contexts is None:
            continue

        ctx_a, ctx_b = contexts
        prompts = {}
        # 4 prompts: {ctx} × {last_token}
        for ctx in [ctx_a, ctx_b]:
            for last_tok in [tok_a, tok_b]:
                prefix = prefix_template.format(ctx=ctx)
                full = prefix + last_tok
                key = f"{ctx}+{last_tok.strip()}"
                prompts[key] = full

        keys = list(prompts.keys())

        # Run all 4
        outs = {}
        for key, text in prompts.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            outs[key] = out

        # Predictions
        preds = {}
        for key in keys:
            top5 = get_top5(outs[key].logits[0, -1], tok)
            preds[key] = top5

        print(f"\n{'='*120}")
        print(f"  {label}")
        for key in keys:
            print(f"  {key:>20}: \"{prompts[key]}\" → {preds[key][0][0]} "
                  f"({preds[key][0][1]:.3f})")

        # Key contrasts:
        # 1. Same context, diff last token (token integration)
        # 2. Diff context, same last token (context effect)
        # 3. Both differ

        contrast_pairs = [
            (f"{ctx_a}+{tok_a.strip()}", f"{ctx_a}+{tok_b.strip()}",
             f"same ctx ({ctx_a}), Δ last token"),
            (f"{ctx_b}+{tok_a.strip()}", f"{ctx_b}+{tok_b.strip()}",
             f"same ctx ({ctx_b}), Δ last token"),
            (f"{ctx_a}+{tok_a.strip()}", f"{ctx_b}+{tok_a.strip()}",
             f"same last ({tok_a.strip()}), Δ context"),
            (f"{ctx_a}+{tok_b.strip()}", f"{ctx_b}+{tok_b.strip()}",
             f"same last ({tok_b.strip()}), Δ context"),
        ]

        for k1, k2, desc in contrast_pairs:
            print(f"\n  --- {desc} ---")
            print(f"    \"{prompts[k1]}\"")
            print(f"    \"{prompts[k2]}\"")

            for L in range(0, NL + 1, 4):
                h_x = outs[k1].hidden_states[L][0, -1, :]
                h_y = outs[k2].hidden_states[L][0, -1, :]
                dh = h_x - h_y
                norm = float(dh.norm() / h_x.norm())
                logits_d = dh @ W_U.T
                pos = get_topk_str(logits_d, tok, 5, largest=True)
                neg = get_topk_str(logits_d, tok, 5, largest=False)
                print(f"    L{L:>2} ({norm:.3f}) [{pos}] vs [{neg}]")

        for key in keys:
            del outs[key]
        torch.cuda.empty_cache()

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
