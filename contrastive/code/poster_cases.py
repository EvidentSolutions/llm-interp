"""
Full mechanistic trace for poster cases.

For each case:
1. Per-position contrastive reading (where does disambiguation first appear?)
2. Attn/MLP decomposition at the disambiguation point
3. Attention weights (which position reads from which?)

Usage: .venv/Scripts/python.exe contrastive/code/poster_cases.py
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


def tk(logits, k=6):
    v, i = torch.topk(logits, k)
    return ", ".join(
        tok.decode([int(i[j])]).strip()[:12] for j in range(k)
    )


def get_components(model, input_ids):
    attn_outs = {}
    mlp_outs = {}

    def make_attn_hook(idx):
        def hook(module, input, output):
            attn_outs[idx] = output[0].detach().clone()
        return hook

    def make_mlp_hook(idx):
        def hook(module, input, output):
            mlp_outs[idx] = output.detach().clone()
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(
            layer.self_attn.register_forward_hook(make_attn_hook(i))
        )
        handles.append(
            layer.mlp.register_forward_hook(make_mlp_hook(i))
        )

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    for h in handles:
        h.remove()

    return out, attn_outs, mlp_outs


print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()

CASES = [
    (
        "caught cold/fish",
        "He caught a cold and",
        "He caught a fish and",
    ),
    (
        "bank steep/closed",
        "The bank was steep and",
        "The bank was closed and",
    ),
]

for label, prompt_a, prompt_b in CASES:
    ids_a = tok(prompt_a, add_special_tokens=False)["input_ids"]
    ids_b = tok(prompt_b, add_special_tokens=False)["input_ids"]

    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"  A: \"{prompt_a}\"")
    print(f"  B: \"{prompt_b}\"")
    print(f"{'='*120}")

    # Show tokens
    print(f"\n  Tokens:")
    for i in range(max(len(ids_a), len(ids_b))):
        ta = tok.decode([ids_a[i]]) if i < len(ids_a) else ""
        tb = tok.decode([ids_b[i]]) if i < len(ids_b) else ""
        same = "=" if ta == tb else "!"
        print(f"    pos {i}: A=\"{ta.strip()[:10]}\"  B=\"{tb.strip()[:10]}\"  {same}")

    # Get components for both
    out_a, attn_a, mlp_a = get_components(
        model, torch.tensor([ids_a], device=DEV)
    )
    out_b, attn_b, mlp_b = get_components(
        model, torch.tensor([ids_b], device=DEV)
    )

    # Predictions
    probs_a = torch.softmax(out_a.logits[0, -1].float(), -1)
    probs_b = torch.softmax(out_b.logits[0, -1].float(), -1)
    va, ia = torch.topk(probs_a, 5)
    vb, ib = torch.topk(probs_b, 5)
    print(f"\n  A predicts: {[(tok.decode([int(ia[j])]).strip(), round(float(va[j]),3)) for j in range(5)]}")
    print(f"  B predicts: {[(tok.decode([int(ib[j])]).strip(), round(float(vb[j]),3)) for j in range(5)]}")

    # === 1. Per-position contrastive reading ===
    print(f"\n  --- PER-POSITION CONTRASTIVE (every layer at each position) ---")

    n_pos = min(len(ids_a), len(ids_b))
    for pos in range(n_pos):
        ta = tok.decode([ids_a[pos]]).strip()[:10]
        same = "=" if (pos < len(ids_a) and pos < len(ids_b)
                       and ids_a[pos] == ids_b[pos]) else "!"

        # Find first layer with readable content at this position
        for L in range(0, NL + 1, 4):
            h_a = out_a.hidden_states[L][0, pos, :].float()
            h_b = out_b.hidden_states[L][0, pos, :].float()
            dh = h_a - h_b
            norm = float(dh.norm() / max(h_a.norm(), 1e-8))

            if norm < 0.01:
                continue

            ld = dh @ W_U.float().T
            reading = tk(ld)
            print(
                f"    pos {pos} \"{ta}\" {same}"
                f" L{L:>2} ({norm:.3f}) [{reading}]"
            )

    # === 2. Focus on the disambiguation position ===
    # Find where the tokens first differ
    diff_pos = None
    for i in range(n_pos):
        if ids_a[i] != ids_b[i]:
            diff_pos = i
            break

    if diff_pos is None:
        print("  No token difference found!")
        continue

    ta = tok.decode([ids_a[diff_pos]]).strip()
    tb = tok.decode([ids_b[diff_pos]]).strip()
    print(f"\n  --- DISAMBIGUATION AT POS {diff_pos}"
          f" (\"{ta}\" vs \"{tb}\") ---")
    print(f"  Reading at the position AFTER the differing token"
          f" (pos {diff_pos + 1})")

    read_pos = diff_pos + 1
    if read_pos >= n_pos:
        read_pos = diff_pos

    print(f"\n  Layer-by-layer at pos {read_pos}"
          f" (\"{tok.decode([ids_a[read_pos]]).strip()}\"):")

    for L in range(NL):
        h_a = out_a.hidden_states[L + 1][0, read_pos, :].float()
        h_b = out_b.hidden_states[L + 1][0, read_pos, :].float()
        dh_full = h_a - h_b
        norm = float(dh_full.norm() / h_a.norm())

        if norm < 0.05:
            continue

        ld_full = dh_full @ W_U.float().T

        d_attn = (attn_a[L][0, read_pos, :].float()
                  - attn_b[L][0, read_pos, :].float())
        d_mlp = (mlp_a[L][0, read_pos, :].float()
                 - mlp_b[L][0, read_pos, :].float())

        ld_attn = d_attn @ W_U.float().T
        ld_mlp = d_mlp @ W_U.float().T

        an = float(d_attn.norm())
        mn = float(d_mlp.norm())

        print(f"    L{L:>2} ({norm:.3f}) attn={an:.1f} mlp={mn:.1f}")
        print(f"      full: [{tk(ld_full)}]")
        print(f"      attn: [{tk(ld_attn)}]")
        print(f"      mlp:  [{tk(ld_mlp)}]")

    # === 3. Also read at LAST position (prediction site) ===
    last_pos = n_pos - 1
    print(f"\n  --- PREDICTION SITE (pos {last_pos},"
          f" \"{tok.decode([ids_a[last_pos]]).strip()}\") ---")

    for L in range(0, NL + 1, 2):
        h_a = out_a.hidden_states[L][0, last_pos, :].float()
        h_b = out_b.hidden_states[L][0, last_pos, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())

        if norm < 0.05:
            continue

        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) A=[{tk(ld)}]")

    # === 4. Attention weights at key layers ===
    print(f"\n  --- ATTENTION WEIGHTS (eager mode needed) ---")
    print(f"  (Skipping — would need model reload with eager attention)")
    print(f"  Use the per-position trace above to infer the information flow.")

    del out_a, out_b, attn_a, attn_b, mlp_a, mlp_b
    torch.cuda.empty_cache()

print(f"\n{'='*120}")
print("DONE")
