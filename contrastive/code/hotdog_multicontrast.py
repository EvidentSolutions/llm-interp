"""
Multi-contrast reading of "The hot dog was" at the dog position.

Each contrast removes a different component, revealing a different
facet of what the residual stream encodes at dog-L5:
  - vs cold dog → temperature axis (expect: fried/food)
  - vs hot cat  → noun identity (expect: dog-ness)
  - vs hot rod  → compound meaning (expect: food vs vehicle)
  - vs old dog  → adjective identity (expect: hot/temperature)
  - vs hot dog is → tense (past markers?)
  - vs angry dog → animate vs food reading
  - vs hot chocolate → food-compound identity
  - vs the dog was → modifier removed (expect: hot/compound)

Then: per-head V->O logit lens at the dog position across layers,
to see what attention actually broadcasts FROM that position.
Compare head broadcasts to each contrastive reading.

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_multicontrast.py
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


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
    HIDDEN = model.config.hidden_size

    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float()
    else:
        W_U = model.embed_out.weight.detach().float()

    _sample_attn = model.model.layers[0].self_attn
    DENSE_ATTR = "dense" if hasattr(_sample_attn, "dense") else "o_proj"

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def get_hidden(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        return out, ids

    # ================================================================
    # Setup: verify tokenization
    # ================================================================
    target = "The hot dog was"
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    target_tokens = [tok.decode([t]) for t in target_ids]
    print(f"Target: \"{target}\"")
    print(f"Tokens: {target_tokens} (ids: {target_ids})")

    # Find the "dog" position
    dog_pos = None
    for i, t in enumerate(target_tokens):
        if "dog" in t.lower():
            dog_pos = i
            break
    assert dog_pos is not None, "dog token not found"
    print(f"Dog position: {dog_pos}")
    was_pos = len(target_ids) - 1
    print(f"Was position: {was_pos}")

    # ================================================================
    # 1. MULTI-CONTRAST at dog position
    # ================================================================
    print(f"\n{'='*100}")
    print(f"1. MULTI-CONTRAST at dog position (pos {dog_pos})")
    print(f"{'='*100}")

    contrasts = [
        ("cold dog",     "The cold dog was",     "temperature"),
        ("hot cat",      "The hot cat was",       "noun identity"),
        ("hot rod",      "The hot rod was",       "compound: food vs vehicle"),
        ("old dog",      "The old dog was",       "adjective: hot vs old"),
        ("angry dog",    "The angry dog was",     "animate dog vs food"),
        ("hot chocolate", "The hot chocolate was", "food compound identity"),
        ("the dog was",  "The dog was",           "modifier removed"),
        ("a hot dog was", "A hot dog was",        "determiner"),
    ]

    target_out, _ = get_hidden(target)

    # Store contrastive vectors at dog pos for later comparison
    contrast_vecs = {}
    layers_to_show = _sl(0, 3, 5, 8, 12, 16, 20, 24, 28, 32)

    for label, contrast_text, axis_desc in contrasts:
        c_ids = tok(contrast_text, add_special_tokens=False)["input_ids"]
        c_tokens = [tok.decode([t]) for t in c_ids]

        # Find matching position for "dog" in the contrast
        # For "The dog was" (no modifier), positions shift
        # We need to read at the last shared-meaning position
        # For most contrasts, dog is at the same position
        # For "The dog was", dog is at pos 1 (no "hot")
        # For "The hot chocolate was", chocolate maps to dog's role

        # Find the noun position in the contrast
        c_noun_pos = None
        if "dog" in contrast_text.lower():
            for i, t in enumerate(c_tokens):
                if "dog" in t.lower():
                    c_noun_pos = i
                    break
        elif "cat" in contrast_text.lower():
            for i, t in enumerate(c_tokens):
                if "cat" in t.lower():
                    c_noun_pos = i
                    break
        elif "rod" in contrast_text.lower():
            for i, t in enumerate(c_tokens):
                if "rod" in t.lower():
                    c_noun_pos = i
                    break
        elif "chocolate" in contrast_text.lower():
            for i, t in enumerate(c_tokens):
                if "choc" in t.lower():
                    c_noun_pos = i
                    break

        if c_noun_pos is None:
            print(f"\n  {label}: SKIP (can't find noun position)")
            continue

        c_out, _ = get_hidden(contrast_text)

        print(f"\n  --- vs \"{contrast_text}\" [{axis_desc}] ---")
        print(f"  Target pos {dog_pos} (\"{target_tokens[dog_pos]}\") "
              f"vs contrast pos {c_noun_pos} (\"{c_tokens[c_noun_pos]}\")")

        for L in layers_to_show:
            h_t = target_out.hidden_states[L][0, dog_pos, :].float()
            h_c = c_out.hidden_states[L][0, c_noun_pos, :].float()
            dh = h_t - h_c
            norm = float(dh.norm() / h_t.norm())
            ld = dh @ W_U.T
            pos_toks = tk(ld)
            neg_toks = tk(-ld)
            print(f"    L{L:>2} ({norm:.3f}) +[{pos_toks}]  -[{neg_toks}]")

            # Store at key layers
            if L in _sl(5, 16, 24):
                key = f"{label}_L{L}"
                contrast_vecs[key] = dh.cpu()

        del c_out
        torch.cuda.empty_cache()

    # ================================================================
    # 2. CROSS-CONTRAST COSINE at key layers
    # ================================================================
    print(f"\n{'='*100}")
    print(f"2. CROSS-CONTRAST COSINE — do different contrasts see the same thing?")
    print(f"{'='*100}")

    cos = torch.nn.functional.cosine_similarity

    for L in _sl(5, 16, 24):
        print(f"\n  Layer {L}:")
        keys_at_L = [k for k in contrast_vecs if k.endswith(f"_L{L}")]
        labels_at_L = [k.replace(f"_L{L}", "") for k in keys_at_L]

        # Print header
        print(f"  {'':>16}", end="")
        for lab in labels_at_L:
            print(f" {lab[:8]:>9}", end="")
        print()

        for i, k1 in enumerate(keys_at_L):
            print(f"  {labels_at_L[i]:>16}", end="")
            for j, k2 in enumerate(keys_at_L):
                c = float(cos(contrast_vecs[k1].unsqueeze(0),
                              contrast_vecs[k2].unsqueeze(0)))
                print(f" {c:>+9.3f}", end="")
            print()

    # ================================================================
    # 3. LOGIT LENS at dog position (non-contrastive)
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. LOGIT LENS at dog position (non-contrastive) — what does the model see?")
    print(f"{'='*100}")

    for L in layers_to_show:
        h = target_out.hidden_states[L][0, dog_pos, :].float()
        ld = h @ W_U.T
        print(f"  L{L:>2}: [{tk(ld, 8)}]")

    # ================================================================
    # 4. PER-HEAD V->O at dog position — what does each head broadcast?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. PER-HEAD V->O at dog position — what does attention broadcast FROM here?")
    print(f"{'='*100}")

    for L in _sl(3, 5, 8, 12, 16, 24):
        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)

        captured = {}
        def hook_fn(module, inp, out):
            captured["data"] = inp[0].detach().float().cpu()
        handle = dense.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(torch.tensor([target_ids], device=DEV))
        handle.remove()

        O_w = dense.weight.float().cpu()
        dense_in = captured["data"][0, dog_pos, :]  # (hidden,)

        print(f"\n  Layer {L} — top heads by norm at dog pos:")
        head_data = []
        for h_idx in range(NH):
            dh = dense_in[h_idx * HD : (h_idx + 1) * HD]
            O_h = O_w[:, h_idx * HD : (h_idx + 1) * HD]
            contribution = dh @ O_h.T
            n = float(contribution.norm())
            logits = contribution @ W_U.cpu().T
            head_data.append((h_idx, n, logits, contribution))

        # Sort by norm, show top 8
        head_data.sort(key=lambda x: -x[1])
        for h_idx, n, logits, _ in head_data[:8]:
            print(f"    H{h_idx:>2} (norm={n:>5.1f}): [{tk(logits)}]")

        # Compare top heads to each contrast direction
        if L in _sl(5, 16, 24):
            print(f"\n    Cosine of top-4 heads vs each contrast direction:")
            top4 = head_data[:4]
            keys_at_L = [k for k in contrast_vecs if k.endswith(f"_L{L}")]
            labels_at_L = [k.replace(f"_L{L}", "") for k in keys_at_L]

            print(f"    {'Head':>6}", end="")
            for lab in labels_at_L:
                print(f" {lab[:10]:>11}", end="")
            print()

            for h_idx, n, _, contribution in top4:
                print(f"    H{h_idx:>2}   ", end="")
                for k in keys_at_L:
                    c = float(cos(contribution.unsqueeze(0),
                                  contrast_vecs[k].unsqueeze(0)))
                    print(f" {c:>+11.3f}", end="")
                print()

    # ================================================================
    # 5. SAME ANALYSIS at was position (prediction site)
    # ================================================================
    print(f"\n{'='*100}")
    print(f"5. MULTI-CONTRAST at was position (pos {was_pos}) — prediction site")
    print(f"{'='*100}")

    # Just the most informative contrasts
    was_contrasts = [
        ("cold dog",     "The cold dog was",     "temperature"),
        ("hot cat",      "The hot cat was",       "noun identity"),
        ("hot rod",      "The hot rod was",       "compound meaning"),
        ("angry dog",    "The angry dog was",     "animate vs food"),
    ]

    for label, contrast_text, axis_desc in was_contrasts:
        c_out, c_ids_list = get_hidden(contrast_text)
        c_was_pos = len(c_ids_list) - 1

        print(f"\n  --- vs \"{contrast_text}\" [{axis_desc}] at last pos ---")
        for L in _sl(0, 5, 8, 12, 16, 20, 24, 28, 32):
            h_t = target_out.hidden_states[L][0, was_pos, :].float()
            h_c = c_out.hidden_states[L][0, c_was_pos, :].float()
            dh = h_t - h_c
            norm = float(dh.norm() / h_t.norm())
            ld = dh @ W_U.T
            print(f"    L{L:>2} ({norm:.3f}) +[{tk(ld)}]  -[{tk(-ld)}]")

        del c_out
        torch.cuda.empty_cache()

    # ================================================================
    # 6. BONUS: "The hot dog was" vs "The hot dog walked"
    #    — does the verb choice change what dog-pos encodes?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"6. VERB EFFECT: does 'was' vs 'walked' change dog-position encoding?")
    print(f"{'='*100}")

    walked_out, walked_ids = get_hidden("The hot dog walked")
    print(f"  Cosine between dog-pos hidden states (was vs walked):")
    for L in _sl(0, 3, 5, 8, 12, 16, 24, 32):
        h_was = target_out.hidden_states[L][0, dog_pos, :].float()
        h_walked = walked_out.hidden_states[L][0, dog_pos, :].float()
        c = float(cos(h_was.unsqueeze(0), h_walked.unsqueeze(0)))
        # Also show what the difference reads as
        dh = h_was - h_walked
        ld = dh @ W_U.T
        norm = float(dh.norm() / h_was.norm())
        print(f"    L{L:>2}: cos={c:+.4f}  (Δnorm={norm:.3f})  "
              f"Δ→[{tk(ld, 4)}]")

    del walked_out, target_out
    torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
