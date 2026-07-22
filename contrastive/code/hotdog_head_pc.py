"""
Do head V broadcasts decompose into the same PCs as the residual stream?

1. Recompute the noun-position PCs (from hotdog_pca_decomp)
2. Get per-head V[dog]@O_h for hot dog and cold dog at L4 and L5
3. Project each head's contrastive contribution onto the PCs
4. Check: do heads write into the same subspace? Different PCs?
5. Does the sum of head PC projections = the full contrastive PC projection?

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_head_pc.py
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
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
    HIDDEN = model.config.hidden_size

    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu()
    else:
        W_U = model.embed_out.weight.detach().float().cpu()

    _sample_attn = model.model.layers[0].self_attn
    DENSE_ATTR = "dense" if hasattr(_sample_attn, "dense") else "o_proj"

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    ref_L = _sl(5)[0]
    dog_pos = 2

    # ================================================================
    # 1. Build PCs from noun-position data (same as pca_decomp)
    # ================================================================
    print("Building PCs from noun-position data...")

    adjectives = ["hot", "cold", "old", "young", "angry", "happy",
                  "big", "small", "red", "blue", "new", "broken",
                  "fresh", "raw", "fried", "grilled", "cheap", "fancy"]
    nouns = ["dog", "cat", "rod", "car", "chocolate", "sauce",
             "water", "air", "chicken", "fish", "potato", "cake",
             "coffee", "tea", "burger", "pizza", "soup", "plate"]

    vecs = []
    for adj in adjectives:
        for noun in nouns:
            text = f"The {adj} {noun} was"
            ids = tok(text, add_special_tokens=False)["input_ids"]
            tokens = [tok.decode([t]) for t in ids]
            noun_pos = None
            for i, t in enumerate(tokens):
                if noun.lower() in t.lower().strip():
                    noun_pos = i; break
            if noun_pos is None:
                continue
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            h = out.hidden_states[ref_L][0, noun_pos, :].float().cpu()
            del out; torch.cuda.empty_cache()
            vecs.append(h)

    mat = torch.stack(vecs)
    center = mat.mean(dim=0)
    centered = mat - center
    U, S, V = torch.svd(centered)
    total_var = S.pow(2).sum()
    print(f"  {len(vecs)} states, top-5 PCs explain "
          f"{S[:5].pow(2).sum()/total_var:.1%}")

    # ================================================================
    # 2. Get the full contrastive and its PC decomposition
    # ================================================================
    target_ids = tok("The hot dog was", add_special_tokens=False)["input_ids"]
    cold_ids = tok("The cold dog was", add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out_t = model(torch.tensor([target_ids], device=DEV),
                      output_hidden_states=True)
        out_c = model(torch.tensor([cold_ids], device=DEV),
                      output_hidden_states=True)

    h_hot = out_t.hidden_states[ref_L][0, dog_pos, :].float().cpu()
    h_cold = out_c.hidden_states[ref_L][0, dog_pos, :].float().cpu()
    delta_full = h_hot - h_cold

    del out_t, out_c; torch.cuda.empty_cache()

    print(f"\n  Full contrastive (hot-cold) at dog pos:")
    print(f"    W_U: [{tk(delta_full @ W_U.T)}]")

    # Project onto PCs
    full_pc_projs = []
    for i in range(min(40, len(S))):
        full_pc_projs.append(float(torch.dot(delta_full, V[:, i])))

    print(f"    Top-8 PC projections: ", end="")
    for i in range(8):
        print(f"PC{i+1}={full_pc_projs[i]:+.1f} ", end="")
    print()

    # ================================================================
    # 3. Get per-head V[dog]@O_h contributions (contrastive)
    # ================================================================
    for L in [4, 5]:
        print(f"\n{'='*100}")
        print(f"Layer {L} — per-head V[dog]@O_h projected onto noun-position PCs")
        print(f"{'='*100}")

        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)
        O_w = dense.weight.float().cpu()

        # Get V projections at dog pos
        ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None

        # Get layer input (hidden_states[L] = output of L-1)
        with torch.no_grad():
            out_t = model(torch.tensor([target_ids], device=DEV),
                          output_hidden_states=True)
            out_c = model(torch.tensor([cold_ids], device=DEV),
                          output_hidden_states=True)

        input_t = out_t.hidden_states[L][0, dog_pos, :].float()
        input_c = out_c.hidden_states[L][0, dog_pos, :].float()

        del out_t, out_c; torch.cuda.empty_cache()

        # Apply layernorm
        if ln is not None:
            with torch.no_grad():
                normed_t = ln(input_t.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
                normed_c = ln(input_c.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
        else:
            normed_t = input_t.cpu()
            normed_c = input_c.cpu()

        # Get V projection
        if hasattr(layer_mod.self_attn, 'v_proj'):
            W_V = layer_mod.self_attn.v_proj.weight.float().cpu()
            b_V = layer_mod.self_attn.v_proj.bias
            if b_V is not None: b_V = b_V.float().cpu()
        elif hasattr(layer_mod.self_attn, 'qkv_proj'):
            W_qkv = layer_mod.self_attn.qkv_proj.weight.float().cpu()
            b_qkv = layer_mod.self_attn.qkv_proj.bias
            if b_qkv is not None: b_qkv = b_qkv.float().cpu()
            W_V = W_qkv[2*HIDDEN:3*HIDDEN, :]
            b_V = b_qkv[2*HIDDEN:3*HIDDEN] if b_qkv is not None else None
        else:
            print(f"  Can't find V projection"); continue

        V_t = normed_t @ W_V.T
        V_c = normed_c @ W_V.T
        if b_V is not None:
            V_t = V_t + b_V
            V_c = V_c + b_V

        # Per-head: ΔV[dog] @ O_h → project onto PCs
        print(f"\n  Per-head contrastive V[dog]@O_h:")
        print(f"  {'H':>4} {'||ΔVO||':>8} {'W_U reads as':>50}  "
              f"{'PC1':>6} {'PC2':>6} {'PC3':>6} {'PC4':>6} {'PC5':>6} {'PC6':>6}")

        head_contributions = []
        for h_idx in range(NH):
            dv = V_t[h_idx*HD:(h_idx+1)*HD] - V_c[h_idx*HD:(h_idx+1)*HD]
            O_h = O_w[:, h_idx*HD:(h_idx+1)*HD]
            contrib = dv @ O_h.T  # (hidden,)
            n = float(contrib.norm())
            ld = contrib @ W_U.T
            # Project onto PCs
            pc_projs = [float(torch.dot(contrib, V[:, i])) for i in range(6)]
            head_contributions.append((h_idx, n, ld, pc_projs, contrib))

        # Sort by norm
        head_contributions.sort(key=lambda x: -x[1])
        for h_idx, n, ld, pc_projs, _ in head_contributions[:12]:
            tok_str = tk(ld, 4)
            print(f"  H{h_idx:>2} {n:>8.1f} {tok_str:>50}  "
                  f"{pc_projs[0]:>+6.1f} {pc_projs[1]:>+6.1f} "
                  f"{pc_projs[2]:>+6.1f} {pc_projs[3]:>+6.1f} "
                  f"{pc_projs[4]:>+6.1f} {pc_projs[5]:>+6.1f}")

        # ================================================================
        # Sum of head PC projections vs full delta PC projections
        # ================================================================
        # The full delta at this layer is hidden_states[L+1] - hidden_states[L]
        # decomposed into attn + MLP. The V@O part is the attention contribution.
        # Sum of per-head V@O = total attention output (pre-residual)

        sum_heads = torch.zeros(HIDDEN)
        for _, _, _, _, contrib in head_contributions:
            sum_heads += contrib

        print(f"\n  Sum of all heads V@O (contrastive):")
        ld_sum = sum_heads @ W_U.T
        print(f"    W_U: [{tk(ld_sum)}]")
        sum_pc_projs = [float(torch.dot(sum_heads, V[:, i])) for i in range(8)]
        print(f"    PCs: ", end="")
        for i in range(8):
            print(f"PC{i+1}={sum_pc_projs[i]:+.1f} ", end="")
        print()

        # Compare to full contrastive
        # The full contrastive = pre-layer delta + attn delta + MLP delta
        # We have the attn delta (sum_heads). How much of the full delta's
        # PC structure comes from attention vs MLP?
        pre_delta = (input_t - input_c).cpu()
        attn_delta = sum_heads
        mlp_delta = delta_full - pre_delta.cpu() - attn_delta  # only valid if this IS the right layer

        print(f"\n  Comparison of PC projections:")
        print(f"  {'':>12} {'PC1':>7} {'PC2':>7} {'PC3':>7} {'PC4':>7} "
              f"{'PC5':>7} {'PC6':>7} {'PC7':>7} {'PC8':>7}")
        for label, vec in [("full delta", delta_full),
                           ("pre-layer", pre_delta.cpu()),
                           ("attn Σheads", attn_delta),
                           ]:
            projs = [float(torch.dot(vec, V[:, i])) for i in range(8)]
            print(f"  {label:>12}", end="")
            for p in projs:
                print(f" {p:>+7.1f}", end="")
            print()

        # ================================================================
        # Which PCs does each head write to most?
        # ================================================================
        print(f"\n  Per-head PC affinity (|projection| / head norm):")
        print(f"  {'H':>4} {'norm':>6}  dominant PCs")

        for h_idx, n, _, pc_projs, _ in head_contributions[:12]:
            if n < 0.5: continue
            # Fraction of head's norm in each PC direction
            fracs = [(abs(pc_projs[i]) / max(n, 1e-6), i, pc_projs[i])
                     for i in range(6)]
            fracs.sort(key=lambda x: -x[0])
            top_pcs = [f"PC{f[1]+1}({f[2]:+.1f}, {f[0]:.0%})"
                       for f in fracs[:3] if f[0] > 0.05]
            print(f"  H{h_idx:>2} {n:>6.1f}  {', '.join(top_pcs)}")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
