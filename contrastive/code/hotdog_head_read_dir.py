"""
What direction in residual stream does each head read from?

H19@L5 routes "eaten, cooked" from dog→was. It reads via V projection.
The head's "read direction" in residual stream is the direction that
maximally activates its V→O→W_U output.

For head h at layer L:
  read_dir = W_V[h]^T @ O_h^T @ W_U^T @ target_logit_direction

But simpler: the contrastive V[dog] for head h is:
  ΔV_h = (Δh @ W_V^T)[h*HD:(h+1)*HD]
  contribution = ΔV_h @ O_h^T
So the "read direction" is the row of (O_h @ W_U^T) that the head
projects V onto, pulled back through W_V.

Empirically: try many contrasts, find which one's Δh at dog pos
best aligns with what H19 reads. Then try linear combinations.

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_head_read_dir.py
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
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
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

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    dog_pos = 2
    target_ids = tok("The hot dog was", add_special_tokens=False)["input_ids"]

    # ================================================================
    # 1. Compute each head's "read direction" at L5 dog position
    # ================================================================
    L = 5
    layer_mod = model.model.layers[L]
    attn_mod = layer_mod.self_attn
    DENSE_ATTR = "dense" if hasattr(attn_mod, "dense") else "o_proj"
    dense = getattr(attn_mod, DENSE_ATTR)
    O_w = dense.weight.float().cpu()  # (HIDDEN, HIDDEN)

    # Get W_V
    if hasattr(attn_mod, 'qkv_proj'):
        W_qkv = attn_mod.qkv_proj.weight.float().cpu()
        b_qkv = attn_mod.qkv_proj.bias
        b_qkv = b_qkv.float().cpu() if b_qkv is not None else None
        W_V = W_qkv[2*HIDDEN:3*HIDDEN, :]  # (HIDDEN, HIDDEN)
    elif hasattr(attn_mod, 'v_proj'):
        W_V = attn_mod.v_proj.weight.float().cpu()
    else:
        print("Can't find W_V"); return

    # For head h, the full path from residual stream to token logits is:
    #   h_input → LN → W_V → [select h's slice] → O_h → W_U → logits
    # The effective "what does head h read" direction in PRE-LN space
    # is complex (LN is input-dependent). But we can compute the
    # effective linear map from post-LN input to token logits:
    #   M_h = W_V[h_slice, :].T @ O_h.T @ W_U.T
    # This is (HIDDEN, vocab) — the columns are token read directions.

    print(f"{'='*100}")
    print(f"1. HEAD READ DIRECTIONS at L{L}")
    print(f"   What does each head's V projection read from the residual stream?")
    print(f"{'='*100}")

    # For each head: compute the effective read→token map
    # and find what the hot-dog state projects to
    ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None

    # Get the actual layer input at dog pos
    with torch.no_grad():
        out = model(torch.tensor([target_ids], device=DEV),
                    output_hidden_states=True)
    h_input = out.hidden_states[L][0, dog_pos, :].float()
    if ln is not None:
        with torch.no_grad():
            h_normed = ln(h_input.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
    else:
        h_normed = h_input.cpu()
    del out; torch.cuda.empty_cache()

    print(f"\n  Per-head V[dog]@O_h@W_U (single input, what hot-dog broadcasts):")
    for h in range(NH):
        W_V_h = W_V[h*HD:(h+1)*HD, :]  # (HD, HIDDEN)
        O_h = O_w[:, h*HD:(h+1)*HD]     # (HIDDEN, HD)
        v_h = h_normed @ W_V_h.T         # (HD,)
        contrib = v_h @ O_h.T            # (HIDDEN,)
        n = float(contrib.norm())
        if n > 3.0:
            ld = contrib @ W_U.T
            print(f"    H{h:>2} (norm={n:>5.1f}): [{tk(ld)}]")

    # ================================================================
    # 2. Compute head read directions in residual stream space
    #    For each head h, the direction d such that (d @ W_V^T)[h_slice]
    #    @ O_h^T @ W_U^T maximally activates some token.
    #    Simpler: head h's effective read matrix is
    #    W_V[h_slice, :]^T @ O_h^T  →  (HIDDEN, HIDDEN)
    #    This maps residual stream → residual stream (via V→O path)
    # ================================================================

    print(f"\n{'='*100}")
    print(f"2. HEAD READ DIRECTION — what residual-stream direction does each head amplify?")
    print(f"{'='*100}")

    head_read_dirs = {}
    for h in [17, 19, 11, 5, 0, 9]:  # heads of interest
        W_V_h = W_V[h*HD:(h+1)*HD, :]  # (HD, HIDDEN)
        O_h = O_w[:, h*HD:(h+1)*HD]     # (HIDDEN, HD)

        # The V→O map: input (HIDDEN) → W_V_h → (HD) → O_h^T → (HIDDEN)
        # Combined: M_h = O_h @ W_V_h  shape (HIDDEN, HIDDEN)
        # Actually: v = input @ W_V_h.T → (HD,), then output = v @ O_h.T → (HIDDEN,)
        # So output = input @ W_V_h.T @ O_h.T = input @ (O_h @ W_V_h).T
        # Effective map: M_h = (O_h @ W_V_h).T  or  M_h = W_V_h.T @ O_h.T

        M_h = W_V_h.T @ O_h.T  # (HIDDEN, HIDDEN): residual → residual via head h

        # What does this map do to the hot-dog input?
        output = h_normed @ M_h  # (HIDDEN,)
        n = float(output.norm())

        # SVD of M_h to find its dominant read/write directions
        U_h, S_h, Vh_h = torch.svd(M_h)
        # Top singular direction in input space = Vh_h[:, 0]
        # Top singular direction in output space = U_h[:, 0]
        top_read = Vh_h[:, 0]  # direction in residual stream that h reads most
        top_write = U_h[:, 0]  # direction in residual stream that h writes to

        head_read_dirs[h] = top_read

        ld_output = output @ W_U.T
        ld_read = top_read @ W_U.T
        ld_write = top_write @ W_U.T

        print(f"\n  H{h:>2}:")
        print(f"    Hot-dog output (V@O):  [{tk(ld_output)}] (norm={n:.1f})")
        print(f"    Top read dir (SVD):    [{tk(ld_read)}]  (σ={S_h[0]:.2f})")
        print(f"    Top write dir (SVD):   [{tk(ld_write)}]")
        print(f"    Top-3 σ: {S_h[0]:.2f}, {S_h[1]:.2f}, {S_h[2]:.2f}")

    # ================================================================
    # 3. Try many contrasts — which best aligns with each head's read dir?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. WHICH CONTRAST ALIGNS WITH EACH HEAD'S READ DIRECTION?")
    print(f"{'='*100}")

    contrasts = [
        ("cold dog",      "The cold dog was"),
        ("hot cat",       "The hot cat was"),
        ("hot rod",       "The hot rod was"),
        ("old dog",       "The old dog was"),
        ("angry dog",     "The angry dog was"),
        ("hot chocolate",  "The hot chocolate was"),
        ("the dog was",   "The dog was"),
        ("big dog",       "The big dog was"),
        ("small dog",     "The small dog was"),
        ("pet dog",       "The pet dog was"),
        ("stray dog",     "The stray dog was"),
        ("dead dog",      "The dead dog was"),
        ("raw dog",       "The raw dog was"),
        ("fried dog",     "The fried dog was"),
        ("grilled dog",   "The grilled dog was"),
        ("cooked dog",    "The cooked dog was"),
        ("hot burger",    "The hot burger was"),
        ("hot pizza",     "The hot pizza was"),
        ("hot chicken",   "The hot chicken was"),
        ("hot fish",      "The hot fish was"),
        ("hot soup",      "The hot soup was"),
        ("hot water",     "The hot water was"),
    ]

    # Get Δh at dog pos (L5 input = hidden_states[5]) for each contrast
    ref_L_idx = _sl(5)[0]
    delta_vecs = {}

    with torch.no_grad():
        out_t = model(torch.tensor([target_ids], device=DEV),
                      output_hidden_states=True)
    h_target = out_t.hidden_states[ref_L_idx][0, dog_pos, :].float().cpu()
    del out_t; torch.cuda.empty_cache()

    for label, ctext in contrasts:
        c_ids = tok(ctext, add_special_tokens=False)["input_ids"]
        c_tokens = [tok.decode([t]) for t in c_ids]
        # Find noun pos
        c_pos = dog_pos
        for noun_word in ["dog", "cat", "rod", "chocolate", "burger",
                          "pizza", "chicken", "fish", "soup", "water"]:
            for i, t in enumerate(c_tokens):
                if noun_word in t.lower().strip():
                    c_pos = i; break
            else:
                continue
            break

        with torch.no_grad():
            out_c = model(torch.tensor([c_ids], device=DEV),
                          output_hidden_states=True)
        h_c = out_c.hidden_states[ref_L_idx][0, c_pos, :].float().cpu()
        del out_c; torch.cuda.empty_cache()

        delta = h_target - h_c
        delta_vecs[label] = delta

    # For each head of interest, find which contrast aligns best
    # with the head's read direction AND with the head's actual output
    for h in [17, 19, 11]:
        W_V_h = W_V[h*HD:(h+1)*HD, :]
        O_h = O_w[:, h*HD:(h+1)*HD]
        read_dir = head_read_dirs[h]

        # What does this head actually output for hot dog (from section 1)?
        v_h = h_normed @ W_V_h.T
        actual_output = v_h @ O_h.T
        ld_actual = actual_output @ W_U.T

        print(f"\n  H{h:>2} actual output: [{tk(ld_actual)}]")
        print(f"  {'contrast':>16} {'cos(Δh,read)':>13} {'cos(Δh@VO,actual)':>18} {'Δh W_U':>40} {'Δh@VO W_U':>40}")

        results = []
        for label in delta_vecs:
            delta = delta_vecs[label]
            # How well does Δh align with the head's read direction?
            cos_read = float(F.cosine_similarity(
                delta.unsqueeze(0), read_dir.unsqueeze(0)))

            # What does this head output for the contrastive?
            if ln is not None:
                with torch.no_grad():
                    # We need Δ of the LN output, not Δ of the input
                    # For now, approximate: pass delta through the V→O path
                    # This is inexact because LN is nonlinear
                    pass
            dv = delta @ W_V_h.T  # (HD,)  — approximate (ignores LN)
            d_output = dv @ O_h.T  # (HIDDEN,)
            cos_output = float(F.cosine_similarity(
                d_output.unsqueeze(0), actual_output.unsqueeze(0)))

            ld_delta = delta @ W_U.T
            ld_d_output = d_output @ W_U.T

            results.append((label, cos_read, cos_output,
                            tk(ld_delta, 3), tk(ld_d_output, 3)))

        # Sort by cos with actual output
        results.sort(key=lambda x: -x[2])
        for label, cos_r, cos_o, delta_toks, output_toks in results:
            print(f"  {label:>16} {cos_r:>+13.3f} {cos_o:>+18.3f} "
                  f"{delta_toks:>40} {output_toks:>40}")

    # ================================================================
    # 4. Can we find a LINEAR COMBINATION of contrasts that matches?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. LINEAR COMBINATION — can we mix contrasts to match H19's output?")
    print(f"{'='*100}")

    for h in [17, 19]:
        W_V_h = W_V[h*HD:(h+1)*HD, :]
        O_h = O_w[:, h*HD:(h+1)*HD]
        v_h = h_normed @ W_V_h.T
        actual_output = v_h @ O_h.T
        ld_actual = actual_output @ W_U.T

        print(f"\n  H{h} target: [{tk(ld_actual)}]")

        # Build matrix of Δh vectors and their V→O outputs
        clabels = list(delta_vecs.keys())
        D = torch.stack([delta_vecs[l] for l in clabels])  # (N, HIDDEN)
        # Each row through V→O
        D_vo = D @ W_V_h.T @ O_h.T  # (N, HIDDEN)

        # Least squares: find weights w such that w @ D_vo ≈ actual_output
        # D_vo^T @ w = actual_output → solve with lstsq
        result = torch.linalg.lstsq(D_vo.T, actual_output)
        weights = result.solution  # (N,)

        # Reconstruct
        recon = weights @ D_vo
        cos_recon = float(F.cosine_similarity(
            recon.unsqueeze(0), actual_output.unsqueeze(0)))
        ld_recon = recon @ W_U.T

        print(f"  Reconstructed: [{tk(ld_recon)}]  cos={cos_recon:.4f}")
        print(f"  Top weights:")
        w_sorted = sorted(zip(clabels, weights.tolist()),
                          key=lambda x: -abs(x[1]))
        for label, w in w_sorted[:8]:
            if abs(w) > 0.01:
                print(f"    {w:>+.3f} × {label}")

        # What does the weighted Δh look like through W_U directly?
        weighted_delta = weights @ D  # (HIDDEN,)
        ld_wd = weighted_delta @ W_U.T
        print(f"  Weighted Δh through W_U: [{tk(ld_wd)}]")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
