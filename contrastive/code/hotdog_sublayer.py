"""
Sub-layer decomposition at the hot-dog compound recognition point.

hidden_states[L] is AFTER layer L's attention + MLP.
So hidden_states[4] is what L5 attention READS (its input).
hidden_states[5] is AFTER L5 has written.

This script decomposes:
  h[4]           — what L5 attn sees as input (pre-L5)
  h[4] + attn[5] — after L5 attn writes but before L5 MLP
  h[5]           — after L5 attn + MLP (what L6 attn reads)
  attn[5] alone  — what L5 attn specifically contributed
  mlp[5] alone   — what L5 MLP specifically contributed

For each, we take the contrastive (hot dog - contrast) and project
through W_U to see what's token-readable at each sub-step.

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_sublayer.py
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

    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float()
    else:
        W_U = model.embed_out.weight.detach().float()

    _sample_attn = model.model.layers[0].self_attn
    DENSE_ATTR = "dense" if hasattr(_sample_attn, "dense") else "o_proj"

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def get_sublayer_states(text, layers_of_interest):
        """Get pre-attn, post-attn (pre-MLP), and post-layer states,
        plus isolated attn and MLP contributions."""
        ids = tok(text, add_special_tokens=False)["input_ids"]

        attn_outs = {}
        hooks = []

        for L in layers_of_interest:
            dense = getattr(model.model.layers[L].self_attn, DENSE_ATTR)

            def make_attn_out_hook(layer_idx):
                def hook_fn(module, inp, out):
                    # out is the attention output (after O projection)
                    attn_outs[layer_idx] = out.detach().float()
                return hook_fn

            h = dense.register_forward_hook(make_attn_out_hook(L))
            hooks.append(h)

        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)

        for h in hooks:
            h.remove()

        result = {}
        for L in layers_of_interest:
            pre_layer = out.hidden_states[L][0].float()      # input to layer L
            post_layer = out.hidden_states[L + 1][0].float()  # output of layer L

            # attn output at this layer
            attn_out = attn_outs[L][0].float() if L in attn_outs else None

            # The residual connection means:
            #   post_attn = pre_layer + attn_out  (before MLP)
            #   post_layer = post_attn + mlp_out
            # But there may be layernorms. In Phi-2 architecture:
            #   attn_input = layernorm(pre_layer)
            #   attn_out = self_attn(attn_input)
            #   post_attn = pre_layer + attn_out   [residual]
            #   mlp_input = layernorm(post_attn)  OR layernorm(pre_layer) for parallel
            #   mlp_out = mlp(mlp_input)
            #   post_layer = post_attn + mlp_out

            # For Phi-2 specifically, attention and MLP run in PARALLEL:
            #   post_layer = pre_layer + attn_out + mlp_out
            # So: mlp_out = post_layer - pre_layer - attn_out

            if attn_out is not None:
                mlp_out = post_layer - pre_layer - attn_out
                post_attn = pre_layer + attn_out
            else:
                mlp_out = None
                post_attn = None

            result[L] = {
                "pre": pre_layer.cpu(),         # what attn reads
                "attn": attn_out.cpu() if attn_out is not None else None,
                "post_attn": post_attn.cpu() if post_attn is not None else None,
                "mlp": mlp_out.cpu() if mlp_out is not None else None,
                "post": post_layer.cpu(),       # what next layer reads
            }

        return result, ids

    # ================================================================
    # Setup
    # ================================================================
    target = "The hot dog was"
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    target_tokens = [tok.decode([t]) for t in target_ids]
    print(f"Target: \"{target}\" → {target_tokens}")

    dog_pos = 2
    was_pos = 3

    # Layers around the compound recognition point
    # L5 is where "fried" appears in the contrast — so we want L3-L8
    layers = list(range(NL))

    contrasts = [
        ("cold dog",  "The cold dog was"),
        ("hot cat",   "The hot cat was"),
        ("angry dog", "The angry dog was"),
        ("hot rod",   "The hot rod was"),
    ]

    # ================================================================
    # 1. Find the exact sub-layer where compound recognition happens
    # ================================================================
    print(f"\n{'='*100}")
    print(f"1. SUB-LAYER DECOMPOSITION: where does 'fried' first appear?")
    print(f"   (pre = L-1 output = L input; attn = L's attn write; mlp = L's MLP write; post = L output)")
    print(f"{'='*100}")

    target_states, _ = get_sublayer_states(target, layers)

    for clabel, ctext in contrasts:
        c_states, c_ids = get_sublayer_states(ctext, layers)
        c_tokens = [tok.decode([t]) for t in c_ids]

        # Find noun pos in contrast
        c_pos = dog_pos  # same for most
        if "cat" in ctext:
            for i, t in enumerate(c_tokens):
                if "cat" in t.lower():
                    c_pos = i
                    break
        elif "rod" in ctext:
            for i, t in enumerate(c_tokens):
                if "rod" in t.lower():
                    c_pos = i
                    break

        print(f"\n  --- vs \"{ctext}\" (target pos {dog_pos}, contrast pos {c_pos}) ---")
        print(f"  {'L':>3} {'stage':>10} {'norm':>6}  tokens")
        print(f"  {'-'*85}")

        for L in range(max(0, 2), min(NL, 10)):
            ts = target_states[L]
            cs = c_states[L]

            stages = [
                ("pre",  ts["pre"][dog_pos],  cs["pre"][c_pos]),
                ("attn", ts["attn"][dog_pos] if ts["attn"] is not None else None,
                         cs["attn"][c_pos] if cs["attn"] is not None else None),
                ("mlp",  ts["mlp"][dog_pos] if ts["mlp"] is not None else None,
                         cs["mlp"][c_pos] if cs["mlp"] is not None else None),
                ("post", ts["post"][dog_pos], cs["post"][c_pos]),
            ]

            for stage_name, t_vec, c_vec in stages:
                if t_vec is None or c_vec is None:
                    continue
                dh = t_vec - c_vec
                norm = float(dh.norm())
                ld = dh @ W_U.cpu().T
                toks = tk(ld)
                print(f"  L{L:>2} {stage_name:>10} {norm:>6.1f}  [{toks}]")

        del c_states
        torch.cuda.empty_cache()

    # ================================================================
    # 2. Which component (attn vs MLP) drives compound recognition?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"2. ATTN vs MLP contribution norms at dog pos (layer 3-8)")
    print(f"{'='*100}")

    c_states_cold, _ = get_sublayer_states("The cold dog was", layers)

    print(f"\n  vs cold dog:")
    print(f"  {'L':>3} {'||Δpre||':>9} {'||Δattn||':>9} {'||Δmlp||':>9} "
          f"{'||Δpost||':>9} {'attn%':>7}")

    for L in range(2, 10):
        ts = target_states[L]
        cs = c_states_cold[L]

        d_pre = (ts["pre"][dog_pos] - cs["pre"][dog_pos]).norm()
        d_post = (ts["post"][dog_pos] - cs["post"][dog_pos]).norm()

        if ts["attn"] is not None and cs["attn"] is not None:
            d_attn = (ts["attn"][dog_pos] - cs["attn"][dog_pos]).norm()
            d_mlp = (ts["mlp"][dog_pos] - cs["mlp"][dog_pos]).norm()
            attn_pct = float(d_attn / (d_attn + d_mlp)) * 100
            print(f"  L{L:>2} {float(d_pre):>9.1f} {float(d_attn):>9.1f} "
                  f"{float(d_mlp):>9.1f} {float(d_post):>9.1f} "
                  f"{attn_pct:>6.1f}%")
        else:
            print(f"  L{L:>2} {float(d_pre):>9.1f} {'N/A':>9} {'N/A':>9} "
                  f"{float(d_post):>9.1f}")

    # ================================================================
    # 3. What does L5 attention READ vs what it WRITES?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. L5 ATTENTION: what it reads (pre) vs what it writes (attn out)")
    print(f"   For each contrast: Δpre is input, Δattn is output")
    print(f"{'='*100}")

    for clabel, ctext in contrasts:
        c_states, c_ids = get_sublayer_states(ctext, [5])
        c_tokens = [tok.decode([t]) for t in c_ids]

        c_pos = dog_pos
        if "cat" in ctext:
            for i, t in enumerate(c_tokens):
                if "cat" in t.lower():
                    c_pos = i; break
        elif "rod" in ctext:
            for i, t in enumerate(c_tokens):
                if "rod" in t.lower():
                    c_pos = i; break

        ts5 = target_states[5]
        cs5 = c_states[5]

        d_pre = ts5["pre"][dog_pos] - cs5["pre"][c_pos]
        d_attn = ts5["attn"][dog_pos] - cs5["attn"][c_pos]
        d_mlp = ts5["mlp"][dog_pos] - cs5["mlp"][c_pos]
        d_post = ts5["post"][dog_pos] - cs5["post"][c_pos]

        cos = torch.nn.functional.cosine_similarity

        print(f"\n  vs {clabel}:")
        print(f"    Δpre  (L5 input):  [{tk(d_pre @ W_U.cpu().T)}]")
        print(f"    Δattn (L5 wrote):  [{tk(d_attn @ W_U.cpu().T)}]")
        print(f"    Δmlp  (L5 wrote):  [{tk(d_mlp @ W_U.cpu().T)}]")
        print(f"    Δpost (L5 output): [{tk(d_post @ W_U.cpu().T)}]")
        print(f"    cos(Δpre, Δattn): {float(cos(d_pre.unsqueeze(0), d_attn.unsqueeze(0))):+.3f}")
        print(f"    cos(Δpre, Δmlp):  {float(cos(d_pre.unsqueeze(0), d_mlp.unsqueeze(0))):+.3f}")
        print(f"    cos(Δattn, Δmlp): {float(cos(d_attn.unsqueeze(0), d_mlp.unsqueeze(0))):+.3f}")
        print(f"    ||Δpre||={float(d_pre.norm()):.1f}  ||Δattn||={float(d_attn.norm()):.1f}  "
              f"||Δmlp||={float(d_mlp.norm()):.1f}  ||Δpost||={float(d_post.norm()):.1f}")

        del c_states

    # ================================================================
    # 4. Per-head decomposition of L5 attention at dog pos
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. PER-HEAD L5 ATTENTION: which heads write the compound signal?")
    print(f"{'='*100}")

    L = 5
    dense = getattr(model.model.layers[L].self_attn, DENSE_ATTR)
    O_w = dense.weight.float().cpu()

    # Get dense inputs for target and cold dog
    for clabel, ctext in [("cold dog", "The cold dog was"),
                           ("angry dog", "The angry dog was")]:
        captured_t = {}
        captured_c = {}

        def make_hook(store):
            def hook_fn(module, inp, out):
                store["data"] = inp[0].detach().float().cpu()
            return hook_fn

        handle = dense.register_forward_hook(make_hook(captured_t))
        with torch.no_grad():
            model(torch.tensor([target_ids], device=DEV))
        handle.remove()

        c_ids = tok(ctext, add_special_tokens=False)["input_ids"]
        handle = dense.register_forward_hook(make_hook(captured_c))
        with torch.no_grad():
            model(torch.tensor([c_ids], device=DEV))
        handle.remove()

        print(f"\n  vs {clabel} at L5 dog pos — per-head contrastive:")
        head_data = []
        for h_idx in range(NH):
            t_h = captured_t["data"][0, dog_pos, h_idx*HD:(h_idx+1)*HD]
            c_h = captured_c["data"][0, dog_pos, h_idx*HD:(h_idx+1)*HD]
            d_h = t_h - c_h
            O_h = O_w[:, h_idx*HD:(h_idx+1)*HD]
            contribution = d_h @ O_h.T
            n = float(contribution.norm())
            logits = contribution @ W_U.cpu().T
            head_data.append((h_idx, n, logits))

        head_data.sort(key=lambda x: -x[1])
        for h_idx, n, logits in head_data[:10]:
            print(f"    H{h_idx:>2} (norm={n:>5.1f}): [{tk(logits)}]")

    # ================================================================
    # 5. Same analysis at was position — what does L8 attn read?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"5. WAS POSITION: sub-layer decomposition (layers 5-12)")
    print(f"   What information has arrived at 'was' and when?")
    print(f"{'='*100}")

    c_states_cold, _ = get_sublayer_states("The cold dog was", layers)

    print(f"\n  vs cold dog at was pos:")
    print(f"  {'L':>3} {'stage':>10} {'norm':>6}  tokens")
    print(f"  {'-'*85}")

    for L in range(4, 14):
        ts = target_states[L]
        cs = c_states_cold[L]

        for stage_name, t_key in [("pre", "pre"), ("attn", "attn"),
                                   ("mlp", "mlp"), ("post", "post")]:
            t_vec = ts[t_key]
            c_vec = cs[t_key]
            if t_vec is None or c_vec is None:
                continue
            dh = t_vec[was_pos] - c_vec[was_pos]
            norm = float(dh.norm())
            ld = dh @ W_U.cpu().T
            toks = tk(ld)
            print(f"  L{L:>2} {stage_name:>10} {norm:>6.1f}  [{toks}]")

    del target_states, c_states_cold
    torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
