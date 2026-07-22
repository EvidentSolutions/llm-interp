"""
Is W_U readability a marker for causal relevance?

For many contrastive pairs across diverse prompts:
1. Get the full residual-stream contrastive (known causal)
2. Get each head's contrastive V→O output
3. Measure "readability" of each head's W_U projection
   (do top tokens form real words, or junk?)
4. Measure causal alignment = cos(head output, full contrastive)
5. Check: does readability correlate with causal alignment?

Also: for a subset, do actual causal injection per-head to validate.

Usage: .venv/Scripts/python.exe contrastive/code/readability_vs_causality.py
"""
import sys
import torch
import torch.nn.functional as F
import re

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

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def is_word(s):
        """Is this a readable English word (not a subword fragment)?"""
        s = s.strip()
        if len(s) < 2: return False
        if not s[0].isalpha(): return False
        if s[0].islower() and len(s) < 3: return False
        # Check if it looks like a real word (all alpha, no weird chars)
        return bool(re.match(r'^[A-Za-z]{2,}$', s))

    def readability_score(logits, top_k=5):
        """How readable are the top-k tokens? 0-1 scale."""
        v, idx = torch.topk(logits.float(), top_k)
        words = [tok.decode([int(idx[j])]).strip() for j in range(top_k)]
        n_readable = sum(1 for w in words if is_word(w))
        return n_readable / top_k, words

    def get_head_outputs_and_full(text_a, text_b, pos_a, pos_b, L):
        """Get per-head V[pos]@O_h contrastive + full residual contrastive."""
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
            out_a = model(torch.tensor([ids_a], device=DEV), output_hidden_states=True)
            out_b = model(torch.tensor([ids_b], device=DEV), output_hidden_states=True)

        h_a = out_a.hidden_states[L][0, pos_a, :].float()
        h_b = out_b.hidden_states[L][0, pos_b, :].float()

        # Full residual contrastive at the LAST layer (prediction)
        h_a_last = out_a.hidden_states[NL][0, -1, :].float().cpu()
        h_b_last = out_b.hidden_states[NL][0, -1, :].float().cpu()
        full_delta = h_a_last - h_b_last

        del out_a, out_b; torch.cuda.empty_cache()

        if ln is not None:
            with torch.no_grad():
                n_a = ln(h_a.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
                n_b = ln(h_b.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
        else:
            n_a = h_a.cpu(); n_b = h_b.cpu()

        head_outputs = []
        for h in range(NH):
            W_V_h = W_V[h*HD:(h+1)*HD, :]
            O_h = O_w[:, h*HD:(h+1)*HD]
            va = n_a @ W_V_h.T
            vb = n_b @ W_V_h.T
            if b_V is not None:
                bv = b_V[h*HD:(h+1)*HD]
                va = va + bv; vb = vb + bv
            d = (va - vb) @ O_h.T
            head_outputs.append(d)

        return head_outputs, full_delta

    # ================================================================
    # Define diverse pairs
    # ================================================================
    pairs = [
        ("hot/cold dog", "The hot dog was", "The cold dog was", 2, 2),
        ("dog/cat", "The big dog was", "The big cat was", 2, 2),
        ("John/Mary IOI", "John and Mary went to the store. John gave a book to",
         "Mary and John went to the store. Mary gave a book to", -1, -1),
        ("France/Japan", "The capital of France is", "The capital of Japan is", -1, -1),
        ("Mon/Tue successor", "After Monday comes", "After Tuesday comes", -1, -1),
        ("true/false", "Paris is the capital of France. This is",
         "Paris is not the capital of France. This is", -1, -1),
        ("boy/girl agent", "The boy kicked the ball and", "The girl kicked the ball and", 1, 1),
        ("tall/short", "Simon is taller than", "Simon is shorter than", 2, 2),
        ("swim/ski", "I wonder if I can go swimming tomorrow",
         "I wonder if I can go skiing tomorrow", 7, 7),
        ("Florida/Alaska", "Bird populations in Florida are",
         "Bird populations in Alaska are", 3, 3),
        ("happy/sad", "She woke up feeling incredibly happy and",
         "She woke up feeling incredibly sad and", 4, 4),
        ("know/doubt", "I know that the answer is", "I doubt that the answer is", 1, 1),
        ("expensive/cheap", "The extremely expensive car was",
         "The extremely cheap car was", 2, 2),
        ("stolen/found", "The stolen painting was", "The returned painting was", 1, 1),
        ("ancient/modern", "The ancient temple was", "The modern temple was", 1, 1),
        ("burning/standing", "The burning building was", "The standing building was", 1, 1),
    ]

    # Fix -1 positions
    fixed_pairs = []
    for label, ta, tb, pa, pb in pairs:
        if pa == -1:
            pa = len(tok(ta, add_special_tokens=False)["input_ids"]) - 1
        if pb == -1:
            pb = len(tok(tb, add_special_tokens=False)["input_ids"]) - 1
        fixed_pairs.append((label, ta, tb, pa, pb))

    # ================================================================
    # Collect readability and causal alignment for all heads
    # ================================================================
    test_layers = _sl(5, 12, 20, 28)

    # Accumulate data points: (readability, causal_alignment, norm)
    all_points = []

    for label, ta, tb, pa, pb in fixed_pairs:
        for L in test_layers:
            head_outputs, full_delta = get_head_outputs_and_full(
                ta, tb, pa, pb, L)

            for h, d in enumerate(head_outputs):
                norm = float(d.norm())
                if norm < 1.0:
                    continue

                # Readability
                ld = d @ W_U.T
                read_score, words = readability_score(ld)

                # Causal alignment = cos with full prediction-site contrastive
                causal_cos = float(F.cosine_similarity(
                    d.unsqueeze(0), full_delta.unsqueeze(0)))

                all_points.append({
                    "pair": label, "L": L, "H": h,
                    "norm": norm,
                    "readability": read_score,
                    "causal_cos": abs(causal_cos),
                    "words": words,
                })

    print(f"Collected {len(all_points)} data points")

    # ================================================================
    # Bin by readability and compute mean causal alignment
    # ================================================================
    print(f"\n{'='*100}")
    print(f"READABILITY vs CAUSAL ALIGNMENT")
    print(f"readability = fraction of top-5 W_U tokens that are real words")
    print(f"causal_cos = |cos| with full prediction-site contrastive")
    print(f"{'='*100}")

    bins = [(0.0, 0.0, "0/5 readable"),
            (0.2, 0.2, "1/5 readable"),
            (0.4, 0.4, "2/5 readable"),
            (0.6, 0.6, "3/5 readable"),
            (0.8, 0.8, "4/5 readable"),
            (1.0, 1.0, "5/5 readable")]

    print(f"\n  {'Bin':>15} {'N':>6} {'mean |cos|':>10} {'std':>6} "
          f"{'mean norm':>10} {'median |cos|':>12}")

    for threshold, _, label in bins:
        matching = [p for p in all_points
                    if abs(p["readability"] - threshold) < 0.01]
        if not matching:
            print(f"  {label:>15} {0:>6}")
            continue
        cos_vals = [p["causal_cos"] for p in matching]
        norm_vals = [p["norm"] for p in matching]
        mean_cos = sum(cos_vals) / len(cos_vals)
        std_cos = (sum((c - mean_cos)**2 for c in cos_vals) / len(cos_vals)) ** 0.5
        mean_norm = sum(norm_vals) / len(norm_vals)
        sorted_cos = sorted(cos_vals)
        median_cos = sorted_cos[len(sorted_cos)//2]
        print(f"  {label:>15} {len(matching):>6} {mean_cos:>10.4f} "
              f"{std_cos:>6.4f} {mean_norm:>10.1f} {median_cos:>12.4f}")

    # ================================================================
    # Correlation
    # ================================================================
    reads = torch.tensor([p["readability"] for p in all_points])
    causals = torch.tensor([p["causal_cos"] for p in all_points])
    norms = torch.tensor([p["norm"] for p in all_points])

    # Pearson correlation
    r_read_causal = float(torch.corrcoef(torch.stack([reads, causals]))[0, 1])
    r_norm_causal = float(torch.corrcoef(torch.stack([norms, causals]))[0, 1])
    r_read_norm = float(torch.corrcoef(torch.stack([reads, norms]))[0, 1])

    print(f"\n  Pearson correlations:")
    print(f"    readability vs |causal_cos|: r = {r_read_causal:+.4f}")
    print(f"    norm vs |causal_cos|:        r = {r_norm_causal:+.4f}")
    print(f"    readability vs norm:         r = {r_read_norm:+.4f}")

    # ================================================================
    # Per-layer breakdown
    # ================================================================
    print(f"\n{'='*100}")
    print(f"PER-LAYER BREAKDOWN")
    print(f"{'='*100}")

    for L in test_layers:
        layer_pts = [p for p in all_points if p["L"] == L]
        if not layer_pts:
            continue
        reads_L = torch.tensor([p["readability"] for p in layer_pts])
        causals_L = torch.tensor([p["causal_cos"] for p in layer_pts])
        r = float(torch.corrcoef(torch.stack([reads_L, causals_L]))[0, 1])

        # Readable vs unreadable split
        readable = [p for p in layer_pts if p["readability"] >= 0.6]
        unreadable = [p for p in layer_pts if p["readability"] <= 0.2]

        mean_r = sum(p["causal_cos"] for p in readable) / max(len(readable), 1)
        mean_u = sum(p["causal_cos"] for p in unreadable) / max(len(unreadable), 1)

        print(f"\n  Layer {L}: r(readability, |causal_cos|) = {r:+.4f}  "
              f"N={len(layer_pts)}")
        print(f"    readable(≥3/5):   N={len(readable):>4}  mean|cos|={mean_r:.4f}")
        print(f"    unreadable(≤1/5): N={len(unreadable):>4}  mean|cos|={mean_u:.4f}")

    # ================================================================
    # Show examples: highest causal cos with readable vs unreadable
    # ================================================================
    print(f"\n{'='*100}")
    print(f"EXAMPLES: most causal + readable vs most causal + unreadable")
    print(f"{'='*100}")

    readable_pts = sorted([p for p in all_points if p["readability"] >= 0.8],
                           key=lambda x: -x["causal_cos"])
    unreadable_pts = sorted([p for p in all_points if p["readability"] == 0],
                             key=lambda x: -x["causal_cos"])

    print(f"\n  Top 10 CAUSAL + READABLE:")
    for p in readable_pts[:10]:
        print(f"    L{p['L']:>2}.H{p['H']:<2} {p['pair']:>20}  "
              f"|cos|={p['causal_cos']:.3f}  read={p['readability']:.1f}  "
              f"norm={p['norm']:.0f}  [{', '.join(p['words'])}]")

    print(f"\n  Top 10 CAUSAL + UNREADABLE:")
    for p in unreadable_pts[:10]:
        print(f"    L{p['L']:>2}.H{p['H']:<2} {p['pair']:>20}  "
              f"|cos|={p['causal_cos']:.3f}  read={p['readability']:.0f}  "
              f"norm={p['norm']:.0f}  [{', '.join(p['words'])}]")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
