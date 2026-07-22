"""
Follow the routing: what arrives at was FROM dog via attention?

At dog position, MLP builds the compound signal in noun-identity PCs.
Attention writes orthogonally at dog. But attention ROUTES from dog→was.
What does that routed signal contain? Does it arrive in the PC subspace
at was, or in a different subspace?

For each layer L (4-12):
  1. Per-head: attn_weight[was→dog] × V[dog] @ O_h (what gets routed)
  2. Project routed signal onto noun-position PCs
  3. Track cumulative signal at was: what has arrived and from where?
  4. Compare: was-position contrastive at each layer vs what attention routed

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_routing.py
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

    DENSE_ATTR = "dense" if hasattr(model.model.layers[0].self_attn, "dense") else "o_proj"

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    ref_L = _sl(5)[0]
    dog_pos = 2
    was_pos = 3

    # ================================================================
    # Build noun-position PCs (reuse from pca_decomp)
    # ================================================================
    print("Building noun-position PCs...")
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
            noun_pos = next((i for i, t in enumerate(tokens)
                             if noun.lower() in t.lower().strip()), None)
            if noun_pos is None: continue
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            vecs.append(out.hidden_states[ref_L][0, noun_pos, :].float().cpu())
            del out; torch.cuda.empty_cache()

    mat = torch.stack(vecs)
    center = mat.mean(dim=0)
    _, S_pc, V_pc = torch.svd(mat - center)
    total_var = S_pc.pow(2).sum()
    print(f"  {len(vecs)} states, top-5 PCs: {S_pc[:5].pow(2).sum()/total_var:.1%}")

    # ================================================================
    # Run both prompts with attention weights
    # ================================================================
    target_ids = tok("The hot dog was", add_special_tokens=False)["input_ids"]
    cold_ids = tok("The cold dog was", add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out_t = model(torch.tensor([target_ids], device=DEV),
                      output_hidden_states=True, output_attentions=True)
        out_c = model(torch.tensor([cold_ids], device=DEV),
                      output_hidden_states=True, output_attentions=True)

    # ================================================================
    # Layer-by-layer: what arrives at was via attention from dog?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"LAYER-BY-LAYER: what attention routes from dog to was (contrastive)")
    print(f"{'='*100}")

    print(f"\n  {'L':>3} {'||Δwas_pre||':>12} {'||Δwas_attn||':>13} "
          f"{'||Δwas_post||':>13} {'attn W_U reads':>45}  "
          f"{'top heads (was→dog attn weight, routed W_U)':>50}")

    for L in range(NL):
        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)
        O_w = dense.weight.float().cpu()

        # Layer input = hidden_states[L]
        h_t_pre = out_t.hidden_states[L][0].float().cpu()
        h_c_pre = out_c.hidden_states[L][0].float().cpu()
        h_t_post = out_t.hidden_states[L+1][0].float().cpu()
        h_c_post = out_c.hidden_states[L+1][0].float().cpu()

        delta_pre = h_t_pre[was_pos] - h_c_pre[was_pos]
        delta_post = h_t_post[was_pos] - h_c_post[was_pos]

        # Attention weights
        attn_t = out_t.attentions[L][0].float().cpu()  # (NH, seq, seq)
        attn_c = out_c.attentions[L][0].float().cpu()

        # Compute V at dog position
        ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None
        dog_input_t = h_t_pre[dog_pos]
        dog_input_c = h_c_pre[dog_pos]

        if ln is not None:
            with torch.no_grad():
                normed_t = ln(dog_input_t.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
                normed_c = ln(dog_input_c.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
        else:
            normed_t = dog_input_t
            normed_c = dog_input_c

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
        else:
            continue

        V_dog_t = normed_t @ W_V.T + (b_V if b_V is not None else 0)
        V_dog_c = normed_c @ W_V.T + (b_V if b_V is not None else 0)

        # Per-head effective routing: attn[was→dog] × V[dog] @ O_h
        total_routed_t = torch.zeros(HIDDEN)
        total_routed_c = torch.zeros(HIDDEN)
        head_info = []

        for h in range(NH):
            aw_t = float(attn_t[h, was_pos, dog_pos])
            aw_c = float(attn_c[h, was_pos, dog_pos])
            v_t = V_dog_t[h*HD:(h+1)*HD]
            v_c = V_dog_c[h*HD:(h+1)*HD]
            O_h = O_w[:, h*HD:(h+1)*HD]
            routed_t = aw_t * (v_t @ O_h.T)
            routed_c = aw_c * (v_c @ O_h.T)
            total_routed_t += routed_t
            total_routed_c += routed_c
            d_routed = routed_t - routed_c
            n = float(d_routed.norm())
            if n > 0.3:
                head_info.append((h, n, aw_t, aw_c, d_routed))

        delta_routed = total_routed_t - total_routed_c
        norm_routed = float(delta_routed.norm())

        # W_U readout of what was routed
        if norm_routed > 0.5:
            ld_routed = delta_routed @ W_U.T
            routed_str = tk(ld_routed, 4)
        else:
            routed_str = "(tiny)"

        # Top contributing heads
        head_info.sort(key=lambda x: -x[1])
        top_heads_str = ""
        for h, n, aw_t, aw_c, d_r in head_info[:2]:
            ld_h = d_r @ W_U.T
            top_heads_str += f"H{h}({aw_t:.2f}/{aw_c:.2f}→{tk(ld_h, 2)}) "

        print(f"  L{L:>2} {float(delta_pre.norm()):>12.1f} "
              f"{norm_routed:>13.1f} {float(delta_post.norm()):>13.1f} "
              f"{routed_str:>45}  {top_heads_str}")

    # ================================================================
    # Detailed view at key layers: project routed signal onto PCs
    # ================================================================
    print(f"\n{'='*100}")
    print(f"DETAILED: routed signal projected onto noun-position PCs")
    print(f"{'='*100}")

    for L in [4, 5, 6, 7, 8, 12, 16, 24]:
        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)
        O_w = dense.weight.float().cpu()

        h_t_pre = out_t.hidden_states[L][0].float().cpu()
        h_c_pre = out_c.hidden_states[L][0].float().cpu()
        attn_t = out_t.attentions[L][0].float().cpu()
        attn_c = out_c.attentions[L][0].float().cpu()

        ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None
        dog_input_t = h_t_pre[dog_pos]
        dog_input_c = h_c_pre[dog_pos]
        if ln is not None:
            with torch.no_grad():
                normed_t = ln(dog_input_t.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
                normed_c = ln(dog_input_c.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
        else:
            normed_t = dog_input_t; normed_c = dog_input_c

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
        else:
            continue

        V_dog_t = normed_t @ W_V.T + (b_V if b_V is not None else 0)
        V_dog_c = normed_c @ W_V.T + (b_V if b_V is not None else 0)

        print(f"\n  Layer {L}:")

        total_routed = torch.zeros(HIDDEN)
        head_details = []
        for h in range(NH):
            aw_t = float(attn_t[h, was_pos, dog_pos])
            aw_c = float(attn_c[h, was_pos, dog_pos])
            v_t = V_dog_t[h*HD:(h+1)*HD]
            v_c = V_dog_c[h*HD:(h+1)*HD]
            O_h = O_w[:, h*HD:(h+1)*HD]
            d_routed = aw_t * (v_t @ O_h.T) - aw_c * (v_c @ O_h.T)
            total_routed += d_routed
            n = float(d_routed.norm())
            # PC projections
            pcs = [float(torch.dot(d_routed, V_pc[:, i])) for i in range(6)]
            head_details.append((h, n, d_routed, pcs, aw_t, aw_c))

        head_details.sort(key=lambda x: -x[1])

        # Total routed signal
        ld_total = total_routed @ W_U.T
        total_pcs = [float(torch.dot(total_routed, V_pc[:, i])) for i in range(8)]
        print(f"    Total routed dog→was: [{tk(ld_total, 5)}]  ||={float(total_routed.norm()):.1f}")
        print(f"    PCs: " + " ".join(f"PC{i+1}={total_pcs[i]:+.1f}" for i in range(8)))

        # Compare to what was position actually gained
        delta_pre = h_t_pre[was_pos] - h_c_pre[was_pos]
        delta_post = out_t.hidden_states[L+1][0, was_pos, :].float().cpu() - \
                     out_c.hidden_states[L+1][0, was_pos, :].float().cpu()
        delta_gained = delta_post - delta_pre
        ld_gained = delta_gained @ W_U.T
        gained_pcs = [float(torch.dot(delta_gained, V_pc[:, i])) for i in range(8)]
        print(f"    Was actually gained:  [{tk(ld_gained, 5)}]  ||={float(delta_gained.norm()):.1f}")
        print(f"    PCs: " + " ".join(f"PC{i+1}={gained_pcs[i]:+.1f}" for i in range(8)))

        cos_routed_gained = float(F.cosine_similarity(
            total_routed.unsqueeze(0), delta_gained.unsqueeze(0)))
        print(f"    cos(routed, gained): {cos_routed_gained:+.3f}")

        # Top heads
        print(f"    Top heads routing dog→was:")
        for h, n, d_r, pcs, aw_t, aw_c in head_details[:5]:
            if n < 0.2: break
            ld_h = d_r @ W_U.T
            pc_str = " ".join(f"PC{i+1}={pcs[i]:+.1f}" for i in range(6) if abs(pcs[i]) > 0.3)
            print(f"      H{h:>2} ||={n:>5.1f} attn={aw_t:.2f}/{aw_c:.2f}  "
                  f"[{tk(ld_h, 4)}]  {pc_str}")

    del out_t, out_c
    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
