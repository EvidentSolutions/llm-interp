"""
Value decomposition at dog position for L5 attention.

For each head at L5:
  1. V[dog] @ O_h @ W_U  — what this head offers to broadcast from dog
  2. attn_out[was] per head @ W_U — what actually arrives at was
  3. Attention weight from was→dog — how much was reads from dog
  4. Contrastive (hot dog - cold dog) for each of the above

If a head's V[dog] reads as "food" and it attends from was→dog,
that head is the one carrying the compound signal.

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_value_decomp.py
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

    _sample_attn = model.model.layers[0].self_attn
    DENSE_ATTR = "dense" if hasattr(_sample_attn, "dense") else "o_proj"

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    target = "The hot dog was"
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    target_tokens = [tok.decode([t]) for t in target_ids]
    dog_pos = 2
    was_pos = 3
    print(f"Target: {target_tokens}, dog_pos={dog_pos}, was_pos={was_pos}")

    contrast_text = "The cold dog was"
    contrast_ids = tok(contrast_text, add_special_tokens=False)["input_ids"]

    # Do this for layers around the compound recognition point
    for L in [4, 5, 6, 7, 8]:
        layer_mod = model.model.layers[L]
        attn_mod = layer_mod.self_attn
        dense = getattr(attn_mod, DENSE_ATTR)
        O_w = dense.weight.float().cpu()  # (hidden, hidden)

        # We need: Q, K, V projections and attention weights
        # Hook the attention to capture attention weights and V
        attn_data = {}

        def make_hooks(store, label):
            hooks = []

            # Hook dense input (= concatenated head outputs after attn weighting)
            def dense_hook(module, inp, out):
                store[f"{label}_dense_in"] = inp[0].detach().float().cpu()
            hooks.append(dense.register_forward_hook(dense_hook))

            # Hook the attention output to get attention weights
            # In Phi-2, self_attn forward returns (attn_output, attn_weights, past_kv)
            # We need to pass output_attentions=True
            return hooks

        # Run with attention weights
        # For Phi-2 we need to access the attention weights
        # Simpler: compute V manually from the layer input

        # Get hidden states at layer input (= hidden_states[L])
        def run_and_capture(input_ids):
            captured = {}
            hooks = []

            def dense_hook(module, inp, out):
                captured["dense_in"] = inp[0].detach().float().cpu()
            hooks.append(dense.register_forward_hook(dense_hook))

            with torch.no_grad():
                out = model(
                    torch.tensor([input_ids], device=DEV),
                    output_hidden_states=True,
                    output_attentions=True,
                )

            for h in hooks:
                h.remove()

            # Attention weights for this layer
            # out.attentions[L] shape: (batch, num_heads, seq_len, seq_len)
            attn_weights = out.attentions[L][0].float().cpu()  # (NH, seq, seq)

            # Layer input = hidden_states[L]
            layer_input = out.hidden_states[L][0].float().cpu()  # (seq, hidden)

            # Compute V projection manually
            # In Phi-2: v_proj is attn_mod.v_proj
            if hasattr(attn_mod, 'v_proj'):
                W_V = attn_mod.v_proj.weight.float().cpu()  # (hidden, hidden)
                b_V = attn_mod.v_proj.bias
                if b_V is not None:
                    b_V = b_V.float().cpu()
            elif hasattr(attn_mod, 'qkv_proj'):
                # Phi-2 uses fused qkv
                W_qkv = attn_mod.qkv_proj.weight.float().cpu()
                b_qkv = attn_mod.qkv_proj.bias
                if b_qkv is not None:
                    b_qkv = b_qkv.float().cpu()
                # Split: Q, K, V each of size hidden
                W_V = W_qkv[2*HIDDEN:3*HIDDEN, :]
                b_V = b_qkv[2*HIDDEN:3*HIDDEN] if b_qkv is not None else None
            else:
                # Try packed QKV
                print(f"  WARNING: can't find V projection at L{L}")
                W_V = None
                b_V = None

            # Apply layernorm to get actual input to attention
            ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None
            if ln is not None:
                with torch.no_grad():
                    normed = ln(layer_input.half().to(DEV)).float().cpu()
            else:
                normed = layer_input

            # V = normed @ W_V.T + b_V
            if W_V is not None:
                V = normed @ W_V.T
                if b_V is not None:
                    V = V + b_V
            else:
                V = None

            return {
                "attn_weights": attn_weights,
                "layer_input": layer_input,
                "dense_in": captured.get("dense_in"),
                "V": V,  # (seq, hidden)
                "hidden_states": [h[0].float().cpu() for h in out.hidden_states],
            }

        print(f"\n{'='*100}")
        print(f"Layer {L}")
        print(f"{'='*100}")

        data_t = run_and_capture(target_ids)
        data_c = run_and_capture(contrast_ids)

        # ================================================
        # 1. Attention weights: was→dog per head
        # ================================================
        print(f"\n  Attention weights: was→dog and was→hot")
        print(f"  {'Head':>6} {'was→The':>8} {'was→hot':>8} {'was→dog':>8} {'was→was':>8}")

        for h in range(NH):
            w = data_t["attn_weights"][h, was_pos, :]  # (seq,)
            vals = [float(w[i]) for i in range(len(target_ids))]
            # Only print heads with meaningful was→dog attention
            if vals[dog_pos] > 0.02 or h < 5:
                print(f"  H{h:>4} {vals[0]:>8.3f} {vals[1]:>8.3f} "
                      f"{vals[2]:>8.3f} {vals[3]:>8.3f}")

        # ================================================
        # 2. V[dog] per head → what each head offers from dog
        # ================================================
        if data_t["V"] is not None:
            print(f"\n  V[dog] @ O_h @ W_U — what each head offers to broadcast from dog pos:")
            print(f"  (single input, not contrastive)")

            head_v_data = []
            V_dog = data_t["V"][dog_pos]  # (hidden,)
            for h in range(NH):
                v_h = V_dog[h*HD:(h+1)*HD]
                O_h = O_w[:, h*HD:(h+1)*HD]
                contribution = v_h @ O_h.T  # (hidden,)
                n = float(contribution.norm())
                logits = contribution @ W_U.T
                head_v_data.append((h, n, logits, contribution))

            head_v_data.sort(key=lambda x: -x[1])
            for h, n, logits, _ in head_v_data[:12]:
                # Also show attention weight was→dog for this head
                aw = float(data_t["attn_weights"][h, was_pos, dog_pos])
                print(f"    H{h:>2} (||V||={n:>5.1f}, attn was→dog={aw:.3f}): [{tk(logits)}]")

            # ================================================
            # 3. Contrastive V[dog]: hot dog - cold dog
            # ================================================
            print(f"\n  Contrastive V[dog] (hot-cold) @ O_h @ W_U:")

            V_dog_c = data_c["V"][dog_pos]
            head_dv_data = []
            for h in range(NH):
                dv_h = V_dog[h*HD:(h+1)*HD] - V_dog_c[h*HD:(h+1)*HD]
                O_h = O_w[:, h*HD:(h+1)*HD]
                contribution = dv_h @ O_h.T
                n = float(contribution.norm())
                logits = contribution @ W_U.T
                head_dv_data.append((h, n, logits, contribution))

            head_dv_data.sort(key=lambda x: -x[1])
            for h, n, logits, _ in head_dv_data[:12]:
                aw_t = float(data_t["attn_weights"][h, was_pos, dog_pos])
                aw_c = float(data_c["attn_weights"][h, was_pos, dog_pos])
                print(f"    H{h:>2} (||ΔV||={n:>5.1f}, attn: {aw_t:.3f}/{aw_c:.3f}): [{tk(logits)}]")

            # ================================================
            # 4. Effective broadcast: attn_weight * V[dog] per head
            # ================================================
            print(f"\n  Effective broadcast to was: attn_weight[was→dog] × V[dog] @ O_h @ W_U:")

            head_eff = []
            for h in range(NH):
                aw = data_t["attn_weights"][h, was_pos, dog_pos]
                v_h = V_dog[h*HD:(h+1)*HD]
                O_h = O_w[:, h*HD:(h+1)*HD]
                contribution = float(aw) * v_h @ O_h.T
                n = float(contribution.norm())
                logits = contribution @ W_U.T
                head_eff.append((h, n, logits, float(aw)))

            head_eff.sort(key=lambda x: -x[1])
            for h, n, logits, aw in head_eff[:8]:
                print(f"    H{h:>2} (eff={n:>5.1f}, attn={aw:.3f}): [{tk(logits)}]")

            # Contrastive effective broadcast
            print(f"\n  Contrastive effective broadcast (hot-cold):")

            head_eff_d = []
            for h in range(NH):
                aw_t = float(data_t["attn_weights"][h, was_pos, dog_pos])
                aw_c = float(data_c["attn_weights"][h, was_pos, dog_pos])
                vt_h = V_dog[h*HD:(h+1)*HD]
                vc_h = V_dog_c[h*HD:(h+1)*HD]
                O_h = O_w[:, h*HD:(h+1)*HD]
                eff_t = aw_t * vt_h @ O_h.T
                eff_c = aw_c * vc_h @ O_h.T
                d_eff = eff_t - eff_c
                n = float(d_eff.norm())
                logits = d_eff @ W_U.T
                head_eff_d.append((h, n, logits))

            head_eff_d.sort(key=lambda x: -x[1])
            for h, n, logits in head_eff_d[:8]:
                print(f"    H{h:>2} (||Δeff||={n:>5.1f}): [{tk(logits)}]")

        # ================================================
        # 5. What actually arrived at was (dense_in at was)
        # ================================================
        if data_t["dense_in"] is not None:
            print(f"\n  What actually arrived at was pos (dense_in contrastive):")
            dense_t = data_t["dense_in"][0, was_pos, :]
            dense_c = data_c["dense_in"][0, was_pos, :]

            head_arrived = []
            for h in range(NH):
                dt_h = dense_t[h*HD:(h+1)*HD]
                dc_h = dense_c[h*HD:(h+1)*HD]
                d_h = dt_h - dc_h
                O_h = O_w[:, h*HD:(h+1)*HD]
                contribution = d_h @ O_h.T
                n = float(contribution.norm())
                logits = contribution @ W_U.T
                head_arrived.append((h, n, logits))

            head_arrived.sort(key=lambda x: -x[1])
            for h, n, logits in head_arrived[:8]:
                print(f"    H{h:>2} (||Δ||={n:>5.1f}): [{tk(logits)}]")

        del data_t, data_c
        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
