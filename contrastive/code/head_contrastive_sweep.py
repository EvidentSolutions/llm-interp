"""
What does each head carry, measured by contrastive reading across
diverse prompt structures?

For each minimal pair, compute per-head V[pos]@O_h contrastive
and read through W_U. This tells us which heads carry which
distinctions, and whether they generalize across syntactic frames.

Usage: .venv/Scripts/python.exe contrastive/code/head_contrastive_sweep.py
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
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
    HIDDEN = model.config.hidden_size

    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu()
    else:
        W_U = model.embed_out.weight.detach().float().cpu()

    DENSE_ATTR = "dense" if hasattr(model.model.layers[0].self_attn, "dense") else "o_proj"

    def tk(logits, k=4):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def get_head_contrastive(text_a, text_b, pos_a, pos_b, L):
        """Get per-head V[pos]@O_h contrastive for a pair."""
        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)
        O_w = dense.weight.float().cpu()
        ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None

        if hasattr(layer_mod.self_attn, 'qkv_proj'):
            W_qkv = layer_mod.self_attn.qkv_proj.weight.float().cpu()
            b_qkv = layer_mod.self_attn.qkv_proj.bias
            b_qkv = b_qkv.float().cpu() if b_qkv is not None else None
            W_V = W_qkv[2*HIDDEN:3*HIDDEN, :]
            b_V = b_qkv[2*HIDDEN:3*HIDDEN] if b_qkv is not None else None
        elif hasattr(layer_mod.self_attn, 'v_proj'):
            W_V = layer_mod.self_attn.v_proj.weight.float().cpu()
            b_V = getattr(layer_mod.self_attn.v_proj, 'bias', None)
            if b_V is not None: b_V = b_V.float().cpu()

        ids_a = tok(text_a, add_special_tokens=False)["input_ids"]
        ids_b = tok(text_b, add_special_tokens=False)["input_ids"]

        with torch.no_grad():
            out_a = model(torch.tensor([ids_a], device=DEV),
                          output_hidden_states=True)
            out_b = model(torch.tensor([ids_b], device=DEV),
                          output_hidden_states=True)

        h_a = out_a.hidden_states[L][0, pos_a, :].float()
        h_b = out_b.hidden_states[L][0, pos_b, :].float()
        del out_a, out_b; torch.cuda.empty_cache()

        if ln is not None:
            with torch.no_grad():
                n_a = ln(h_a.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
                n_b = ln(h_b.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
        else:
            n_a = h_a.cpu(); n_b = h_b.cpu()

        results = []
        for h in range(NH):
            W_V_h = W_V[h*HD:(h+1)*HD, :]
            O_h = O_w[:, h*HD:(h+1)*HD]
            va = n_a @ W_V_h.T
            vb = n_b @ W_V_h.T
            if b_V is not None:
                bv = b_V[h*HD:(h+1)*HD]
                va = va + bv; vb = vb + bv
            d = (va - vb) @ O_h.T
            results.append(d)

        return results  # list of NH tensors, each (HIDDEN,)

    # ================================================================
    # Define diverse contrastive pairs
    # ================================================================
    # (label, category, text_a, text_b, pos_a, pos_b)
    # pos = -1 means last token

    pairs = [
        # --- NOUN IDENTITY across frames ---
        ("det-adj-noun", "noun",
         "The hot dog was", "The hot cat was", 2, 2),
        ("possessive", "noun",
         "I have a dog at home. It", "I have a cat at home. It", 3, 3),
        ("plural-temporal", "noun",
         "Earlier, there were many dogs in the", "Earlier, there were many cats in the", 5, 5),
        ("prepositional", "noun",
         "She looked at the dog and", "She looked at the cat and", 4, 4),
        ("subject", "noun",
         "The dog ran across the yard and", "The cat ran across the yard and", 1, 1),
        ("compound-subj", "noun",
         "A large dog appeared at the", "A large cat appeared at the", 2, 2),

        # --- ADJECTIVE/PROPERTY across frames ---
        ("det-adj-noun2", "property",
         "The hot dog was", "The cold dog was", 2, 2),
        ("comparison", "property",
         "Simon is taller than", "Simon is shorter than", 2, 2),
        ("adverb", "property",
         "She ran quickly through the", "She ran slowly through the", 2, 2),
        ("predicate", "property",
         "The water was hot and", "The water was cold and", 3, 3),
        ("embedded", "property",
         "I think the big one is", "I think the small one is", 3, 3),

        # --- LOCATION across frames ---
        ("in-location", "location",
         "Bird populations in Florida are", "Bird populations in Alaska are", 3, 3),
        ("from-location", "location",
         "The wine from France was", "The wine from Japan was", 3, 3),
        ("lives-in", "location",
         "She lives in Paris and", "She lives in Tokyo and", 3, 3),

        # --- TENSE/TIME across frames ---
        ("temporal-adv", "time",
         "Yesterday I walked to the", "Tomorrow I will walk to the", 0, 0),
        ("verb-tense", "time",
         "The bird sang a beautiful", "The bird sings a beautiful", 2, 2),

        # --- SUBJECT/AGENT across frames ---
        ("agent", "agent",
         "The boy kicked the ball and", "The girl kicked the ball and", 1, 1),
        ("pronoun", "agent",
         "He decided to go to the", "She decided to go to the", 0, 0),
        ("named", "agent",
         "Einstein wrote a paper about", "Shakespeare wrote a paper about", 0, 0),

        # --- VERB/ACTION across frames ---
        ("verb-object", "action",
         "I wonder if I can go swimming tomorrow", "I wonder if I can go skiing tomorrow", 7, 7),
        ("main-verb", "action",
         "The man walked into the store and", "The man ran into the store and", 2, 2),
        ("inf-verb", "action",
         "She wanted to cook dinner but", "She wanted to paint dinner but", 3, 3),

        # --- QUANTITY ---
        ("quantity", "quantity",
         "There were three cats on the", "There were seven cats on the", 2, 2),
        ("many-few", "quantity",
         "Earlier, there were many birds in the", "Earlier, there were few birds in the", 4, 4),

        # --- MODALITY/CERTAINTY ---
        ("modal", "modality",
         "I know that the answer is", "I doubt that the answer is", 1, 1),
        ("will-might", "modality",
         "He will finish the project by", "He might finish the project by", 1, 1),

        # --- SENTIMENT ---
        ("sentiment", "sentiment",
         "The movie was absolutely wonderful and", "The movie was absolutely terrible and", 4, 4),
        ("food-eval", "sentiment",
         "The cake tasted delicious and", "The cake tasted disgusting and", 3, 3),
    ]

    # ================================================================
    # Run all pairs at key layers, collect per-head contrastive
    # ================================================================
    test_layers = _sl(5, 12, 20, 28)

    print(f"Running {len(pairs)} contrastive pairs × {len(test_layers)} layers × {NH} heads")
    print(f"{'='*120}")

    # For each layer: store results as (pair_label, category, head_idx, norm, wu_readout)
    all_results = {L: [] for L in test_layers}

    for label, category, text_a, text_b, pos_a, pos_b in pairs:
        # Handle -1 as last pos
        if pos_a == -1:
            pos_a = len(tok(text_a, add_special_tokens=False)["input_ids"]) - 1
        if pos_b == -1:
            pos_b = len(tok(text_b, add_special_tokens=False)["input_ids"]) - 1

        for L in test_layers:
            head_contrastives = get_head_contrastive(
                text_a, text_b, pos_a, pos_b, L)

            for h, d in enumerate(head_contrastives):
                n = float(d.norm())
                if n > 1.0:  # only track heads with meaningful signal
                    ld = d @ W_U.T
                    readout = tk(ld)
                    all_results[L].append({
                        "pair": label, "cat": category,
                        "h": h, "norm": n, "wu": readout, "vec": d,
                    })

    # ================================================================
    # For each layer: which heads are active across categories?
    # ================================================================
    for L in test_layers:
        print(f"\n{'='*120}")
        print(f"Layer {L}")
        print(f"{'='*120}")

        results = all_results[L]

        # Group by head — which categories does each head participate in?
        head_cats = {}
        for r in results:
            h = r["h"]
            if h not in head_cats:
                head_cats[h] = {}
            cat = r["cat"]
            if cat not in head_cats[h]:
                head_cats[h][cat] = []
            head_cats[h][cat].append(r)

        # Find heads active in 3+ categories (generalist heads)
        print(f"\n  GENERALIST HEADS (active in 3+ categories):")
        print(f"  {'H':>4} {'cats':>4} {'total':>5} {'categories + top readouts':>70}")

        generalists = [(h, data) for h, data in head_cats.items()
                       if len(data) >= 3]
        generalists.sort(key=lambda x: -sum(
            max(r["norm"] for r in cat_data)
            for cat_data in x[1].values()))

        for h, cat_data in generalists[:15]:
            n_cats = len(cat_data)
            total = sum(len(v) for v in cat_data.values())
            parts = []
            for cat, entries in sorted(cat_data.items()):
                best = max(entries, key=lambda x: x["norm"])
                parts.append(f"{cat}({best['norm']:.0f}→{best['wu'][:20]})")
            print(f"  H{h:>2} {n_cats:>4} {total:>5}  {'; '.join(parts)}")

        # Find heads active in exactly 1 category (specialist heads)
        print(f"\n  SPECIALIST HEADS (active in exactly 1 category, norm>3):")
        specialists = [(h, data) for h, data in head_cats.items()
                       if len(data) == 1]
        specialists.sort(key=lambda x: -max(
            r["norm"] for r in list(x[1].values())[0]))

        for h, cat_data in specialists[:10]:
            cat = list(cat_data.keys())[0]
            entries = cat_data[cat]
            best = max(entries, key=lambda x: x["norm"])
            if best["norm"] > 3:
                pair_names = ", ".join(set(e["pair"] for e in entries))
                print(f"  H{h:>2}  {cat:>10} (norm={best['norm']:>5.1f}): "
                      f"[{best['wu']}]  pairs: {pair_names}")

        # What does each category look like across heads?
        print(f"\n  TOP HEADS PER CATEGORY:")
        categories = sorted(set(r["cat"] for r in results))
        for cat in categories:
            cat_results = [r for r in results if r["cat"] == cat]
            # Group by head, take max norm per head
            by_head = {}
            for r in cat_results:
                if r["h"] not in by_head or r["norm"] > by_head[r["h"]]["norm"]:
                    by_head[r["h"]] = r
            top = sorted(by_head.values(), key=lambda x: -x["norm"])[:5]
            heads_str = "  ".join(
                f"H{r['h']}({r['norm']:.0f}→{r['wu'][:15]})"
                for r in top)
            print(f"    {cat:>10}: {heads_str}")

    # ================================================================
    # Cross-frame consistency: does the same head read the same thing
    # for the same contrast type in different frames?
    # ================================================================
    print(f"\n{'='*120}")
    print(f"CROSS-FRAME CONSISTENCY at L{test_layers[-1]}")
    print(f"Does the same head read the same thing in different frames?")
    print(f"{'='*120}")

    L = test_layers[-1]
    results = all_results[L]

    # For "noun" category: do the noun-identity heads produce
    # consistent directions across different syntactic frames?
    for cat in ["noun", "property", "agent"]:
        cat_results = [r for r in results if r["cat"] == cat]
        if len(cat_results) < 4:
            continue

        # Find heads active in 2+ pairs of this category
        by_head = {}
        for r in cat_results:
            by_head.setdefault(r["h"], []).append(r)

        print(f"\n  {cat} — cross-frame cosine per head:")
        multi_heads = [(h, entries) for h, entries in by_head.items()
                       if len(entries) >= 2 and max(e["norm"] for e in entries) > 3]
        multi_heads.sort(key=lambda x: -max(e["norm"] for e in x[1]))

        for h, entries in multi_heads[:8]:
            print(f"    H{h:>2}:", end="")
            # Pairwise cosine
            cosines = []
            for i in range(len(entries)):
                for j in range(i+1, len(entries)):
                    c = float(F.cosine_similarity(
                        entries[i]["vec"].unsqueeze(0),
                        entries[j]["vec"].unsqueeze(0)))
                    cosines.append(c)
            mean_cos = sum(cosines) / len(cosines)
            pair_names = [e["pair"][:12] for e in entries]
            norms = [f"{e['norm']:.0f}" for e in entries]
            print(f" mean_cos={mean_cos:+.3f}  "
                  f"pairs=[{', '.join(pair_names)}]  "
                  f"norms=[{', '.join(norms)}]")

    torch.cuda.empty_cache()
    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
