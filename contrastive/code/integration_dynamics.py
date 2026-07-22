"""
Current-token integration dynamics.

Core experiment: same last token, different context. At L0 the
representations are identical (same embedding). Track layer by layer:
- When does context first differentiate the representation?
- How fast does differentiation grow?
- Is it attention or MLP that does the integration?

Design: "The {X} was very {Y}" with multiple X and Y.
Read at position Y (last token). Measure:
1. cos(Δh[L], Δh[L0]) — how much the difference has rotated from
   the pure embedding difference
2. cos(Δh_ctxA[L], Δh_ctxB[L]) — for same Y, different X: when do
   contexts start producing different representations of the same token?
3. Attention vs MLP attribution: which component drives the change?

Usage: .venv/Scripts/python.exe contrastive/code/integration_dynamics.py
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

# {context} × {last_token} grid
# All prompts: "The {ctx} was very {adj}"
CONTEXTS = ["man", "woman", "boy", "dog", "building", "car", "tree", "soup"]
ADJECTIVES = ["old", "hot", "big", "clean"]


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

    # Verify all adjectives are single tokens
    for adj in ADJECTIVES:
        ids = tok(f" {adj}", add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"'{adj}' is multi-token: {ids}"

    # ================================================================
    # EXPERIMENT 1: Context divergence
    # For each adjective, how quickly do different contexts produce
    # different representations of the same token?
    # ================================================================
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: Context divergence")
    print("For each adjective, cos(Δh_ctxA, Δh_ctxB) across layers")
    print("where Δh = h(ctx+adj) - h(ctx+adj_baseline)")
    print("At L0 this is 1.0 (same embedding). Drops as context integrates.")
    print("=" * 100)

    for adj in ADJECTIVES:
        print(f"\n--- Adjective: '{adj}' ---")

        # Run all contexts with this adjective
        states = {}
        for ctx in CONTEXTS:
            text = f"The {ctx} was very {adj}"
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(
                    torch.tensor([ids], device=DEV),
                    output_hidden_states=True,
                )
            states[ctx] = [
                out.hidden_states[L][0, -1, :].float().detach().clone()
                for L in range(NL + 1)
            ]
            del out

        # For each pair of contexts, compute cos of their last-token
        # hidden state at each layer
        # (At L0 they should be identical — same token embedding)
        ctx_pairs = []
        for i, c1 in enumerate(CONTEXTS):
            for c2 in CONTEXTS[i + 1 :]:
                ctx_pairs.append((c1, c2))

        print(f"\n  Pairwise cos(h_ctxA[L], h_ctxB[L]) at last token position:")
        print(f"  (1.0 = identical, lower = more differentiated by context)")
        print(f"  {'L':>3} |", end="")

        # Show a selection of pairs
        show_pairs = [
            ("man", "building"),
            ("man", "woman"),
            ("man", "dog"),
            ("man", "car"),
            ("building", "car"),
            ("dog", "soup"),
        ]
        show_pairs = [(a, b) for a, b in show_pairs
                      if a in CONTEXTS and b in CONTEXTS]

        for c1, c2 in show_pairs:
            print(f" {c1[:4]}-{c2[:4]:>4} |", end="")
        print(f" {'mean':>6}")

        for L in range(0, NL + 1, 2):
            print(f"  {L:>3} |", end="")
            layer_cos = []
            for c1, c2 in show_pairs:
                h1 = states[c1][L]
                h2 = states[c2][L]
                cos = float(torch.nn.functional.cosine_similarity(
                    h1.unsqueeze(0), h2.unsqueeze(0)))
                layer_cos.append(cos)
                print(f"    {cos:.3f} |", end="")

            # Mean across ALL pairs, not just shown
            all_cos = []
            for c1, c2 in ctx_pairs:
                h1 = states[c1][L]
                h2 = states[c2][L]
                cos = float(torch.nn.functional.cosine_similarity(
                    h1.unsqueeze(0), h2.unsqueeze(0)))
                all_cos.append(cos)
            print(f" {sum(all_cos)/len(all_cos):.3f}")

        # Cleanup
        del states
        torch.cuda.empty_cache()

    # ================================================================
    # EXPERIMENT 2: Rotation from embedding
    # For each (context, adjective) pair, how far has the contrastive
    # direction rotated from the raw embedding difference?
    # Compare "The X was very old" vs "The X was very young" for each X.
    # ================================================================
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: Rotation from embedding difference")
    print("cos(Δh[L], emb_diff) — starts at 1.0, drops as model adds content")
    print("Δh = h(ctx+old) - h(ctx+young)")
    print("=" * 100)

    adj_pairs = [("old", "young"), ("hot", "cold"), ("big", "small"),
                 ("clean", "dirty")]

    for adj_a, adj_b in adj_pairs:
        # Check single token
        id_a = tok(f" {adj_a}", add_special_tokens=False)["input_ids"]
        id_b = tok(f" {adj_b}", add_special_tokens=False)["input_ids"]
        if len(id_a) != 1 or len(id_b) != 1:
            print(f"\n  SKIP {adj_a}/{adj_b}: multi-token")
            continue

        emb_a = model.model.embed_tokens.weight[id_a[0]].float()
        emb_b = model.model.embed_tokens.weight[id_b[0]].float()
        emb_diff = emb_a - emb_b

        print(f"\n--- {adj_a} vs {adj_b} ---")
        print(f"  {'L':>3} |", end="")

        use_contexts = ["man", "woman", "building", "car", "dog", "soup"]
        for ctx in use_contexts:
            print(f" {ctx:>8} |", end="")
        print(f" {'mean':>6}")

        for L in range(0, NL + 1, 4):
            print(f"  {L:>3} |", end="")
            cos_vals = []
            for ctx in use_contexts:
                text_a = f"The {ctx} was very {adj_a}"
                text_b = f"The {ctx} was very {adj_b}"
                ids_a = tok(text_a, add_special_tokens=False)["input_ids"]
                ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
                with torch.no_grad():
                    out_a = model(torch.tensor([ids_a], device=DEV),
                                  output_hidden_states=True)
                    out_b = model(torch.tensor([ids_b], device=DEV),
                                  output_hidden_states=True)
                h_a = out_a.hidden_states[L][0, -1, :].float()
                h_b = out_b.hidden_states[L][0, -1, :].float()
                dh = h_a - h_b

                cos = float(torch.nn.functional.cosine_similarity(
                    dh.unsqueeze(0), emb_diff.unsqueeze(0)))
                cos_vals.append(cos)
                print(f"   {cos:>+.3f} |", end="")

                del out_a, out_b
            print(f" {sum(cos_vals)/len(cos_vals):>+.3f}")

        torch.cuda.empty_cache()

    # ================================================================
    # EXPERIMENT 3: Cross-context contrastive similarity
    # For "old vs young": is Δh(man) similar to Δh(building)?
    # Measures whether the model applies the SAME old/young
    # transformation regardless of context, or context-specific ones.
    # ================================================================
    print("\n" + "=" * 100)
    print("EXPERIMENT 3: Cross-context contrastive similarity")
    print("cos(Δh_ctxA, Δh_ctxB) where Δh = h(ctx+old) - h(ctx+young)")
    print("High = same old/young axis regardless of context")
    print("Low = context-specific old/young representations")
    print("=" * 100)

    for adj_a, adj_b in [("old", "young"), ("hot", "cold")]:
        id_a = tok(f" {adj_a}", add_special_tokens=False)["input_ids"]
        id_b = tok(f" {adj_b}", add_special_tokens=False)["input_ids"]
        if len(id_a) != 1 or len(id_b) != 1:
            continue

        print(f"\n--- {adj_a} vs {adj_b} ---")

        use_contexts = ["man", "woman", "building", "car", "dog", "soup"]
        # Compute Δh for each context
        dh_per_ctx = {}
        for ctx in use_contexts:
            text_a = f"The {ctx} was very {adj_a}"
            text_b = f"The {ctx} was very {adj_b}"
            ids_a = tok(text_a, add_special_tokens=False)["input_ids"]
            ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out_a = model(torch.tensor([ids_a], device=DEV),
                              output_hidden_states=True)
                out_b = model(torch.tensor([ids_b], device=DEV),
                              output_hidden_states=True)
            dh_per_ctx[ctx] = [
                (out_a.hidden_states[L][0, -1, :].float() -
                 out_b.hidden_states[L][0, -1, :].float()).detach().clone()
                for L in range(NL + 1)
            ]
            del out_a, out_b

        # Pairwise cos of Δh across contexts at each layer
        cross_pairs = [
            ("man", "building"), ("man", "car"), ("man", "dog"),
            ("man", "soup"), ("building", "car"), ("building", "dog"),
        ]

        print(f"  {'L':>3} |", end="")
        for c1, c2 in cross_pairs:
            print(f" {c1[:4]}-{c2[:4]:>4} |", end="")
        print(f" {'mean':>6}")

        for L in range(0, NL + 1, 4):
            print(f"  {L:>3} |", end="")
            all_cos = []
            for c1, c2 in cross_pairs:
                cos = float(torch.nn.functional.cosine_similarity(
                    dh_per_ctx[c1][L].unsqueeze(0),
                    dh_per_ctx[c2][L].unsqueeze(0)))
                all_cos.append(cos)
                print(f"    {cos:>+.3f} |", end="")
            print(f" {sum(all_cos)/len(all_cos):>+.3f}")

        del dh_per_ctx
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("DONE")


if __name__ == "__main__":
    main()
