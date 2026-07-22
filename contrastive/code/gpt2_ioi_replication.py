"""
GPT-2 IOI circuit replication via contrastive projection.

Wang et al. (2023) mapped the IOI circuit in GPT-2-small. This script
tests whether the contrastive method recovers the same heads:

Known IOI circuit (Wang et al.):
  Name movers (positive):  L9H9, L10H0, L9H6
  Name movers (negative):  L10H7, L11H10
  S-inhibition:            L7H3, L7H9, L8H6, L8H10
  Duplicate token:         L0H1, L3H0
  Induction:               L5H5, L6H9

Tests:
  1. Basic contrastive trajectory (name-swap)
  2. Per-head decomposition at every layer — find heads with largest
     contrastive norm and check which known-circuit heads they match
  3. Attention patterns — verify name-movers attend to IO token
  4. Position-resolved patching — trace S2 → END transfer
  5. Multiple name pairs for robustness

Usage: .venv/Scripts/python.exe contrastive/code/gpt2_ioi_replication.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.n_layer   # 12
NH = model.config.n_head    # 12
HD = model.config.n_embd // NH  # 64
d_model = model.config.n_embd   # 768

W_U = model.lm_head.weight.detach().float()  # (vocab, 768)

# Known circuit heads from Wang et al.
KNOWN_CIRCUIT = {
    'name_mover_pos': [(9, 9), (10, 0), (9, 6)],
    'name_mover_neg': [(10, 7), (11, 10)],
    's_inhibition':   [(7, 3), (7, 9), (8, 6), (8, 10)],
    'duplicate_token': [(0, 1), (3, 0)],
    'induction':      [(5, 5), (6, 9)],
}

ALL_KNOWN = set()
for heads in KNOWN_CIRCUIT.values():
    ALL_KNOWN.update(heads)


def tk(logits, k=5):
    v, i = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def bk(logits, k=5):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True,
                    output_attentions=True)
    return out, ids


def head_label(layer, head):
    """Return label with circuit role if known."""
    key = (layer, head)
    for role, heads in KNOWN_CIRCUIT.items():
        if key in heads:
            return f"L{layer}H{head} [{role}]"
    return f"L{layer}H{head}"


# ============================================================
# IOI prompts — multiple name pairs
# ============================================================
IOI_PAIRS = [
    ("When John and Mary went to the store, John gave a drink to",
     "When Mary and John went to the store, Mary gave a drink to",
     "Mary", "John"),
    ("When Alice and Bob went to the park, Alice gave a gift to",
     "When Bob and Alice went to the park, Bob gave a gift to",
     "Bob", "Alice"),
    ("When Dan and Eve ate dinner together, Dan passed the salt to",
     "When Eve and Dan ate dinner together, Eve passed the salt to",
     "Eve", "Dan"),
]


# ============================================================
# TEST 1: Basic contrastive trajectory
# ============================================================
print("=" * 100)
print("TEST 1: CONTRASTIVE TRAJECTORY — does the IO name appear?")
print("=" * 100)

for pa, pb, io_a, io_b in IOI_PAIRS:
    out_a, ids_a = get_hidden_states(pa)
    out_b, ids_b = get_hidden_states(pb)

    # What does the model predict?
    pred_a = tok.decode([out_a.logits[0, -1].argmax().item()])
    pred_b = tok.decode([out_b.logits[0, -1].argmax().item()])
    print(f'\n  A: "...{pa[-40:]}" → "{pred_a.strip()}"')
    print(f'  B: "...{pb[-40:]}" → "{pred_b.strip()}"')

    for L in range(NL + 1):
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        ld = dh @ W_U.T
        norm = float(dh.norm() / h_a.norm())
        print(f"    L{L:>2} ({norm:.3f}) A=[{tk(ld)}]  B=[{bk(ld)}]")

    del out_a, out_b
    torch.cuda.empty_cache()


# ============================================================
# TEST 2: Per-head decomposition — find name movers
# ============================================================
print(f"\n{'='*100}")
print("TEST 2: PER-HEAD DECOMPOSITION — which heads carry the IO name?")
print("=" * 100)

# For GPT-2, attn output goes through c_proj. We need to hook into
# the attention output BEFORE c_proj to get per-head contributions.
# Actually, we hook c_proj's input to get the concatenated head outputs.

for pa, pb, io_a, io_b in IOI_PAIRS:
    print(f"\n  Pair: {io_a}/{io_b}")

    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]

    # Capture c_proj inputs (concatenated head outputs) at every layer
    captured_a = {}
    captured_b = {}
    hooks = []

    for L in range(NL):
        def make_hook(store, layer_idx):
            def hook_fn(module, inp, out):
                # inp[0] is the input to c_proj: (batch, seq, d_model)
                # This is the concatenated per-head outputs
                store[layer_idx] = inp[0][0, -1, :].detach().float().cpu()
            return hook_fn

        h = model.transformer.h[L].attn.c_proj.register_forward_hook(
            make_hook(captured_a, L))
        hooks.append(h)

    with torch.no_grad():
        model(torch.tensor([ids_a], device=DEV))
    for h in hooks:
        h.remove()
    hooks.clear()

    for L in range(NL):
        def make_hook(store, layer_idx):
            def hook_fn(module, inp, out):
                store[layer_idx] = inp[0][0, -1, :].detach().float().cpu()
            return hook_fn

        h = model.transformer.h[L].attn.c_proj.register_forward_hook(
            make_hook(captured_b, L))
        hooks.append(h)

    with torch.no_grad():
        model(torch.tensor([ids_b], device=DEV))
    for h in hooks:
        h.remove()
    hooks.clear()

    # Per-head contrastive decomposition
    # c_proj weight: (d_model, d_model) — maps concatenated heads to residual
    # head h contributes: input[h*HD:(h+1)*HD] @ c_proj.weight[:, h*HD:(h+1)*HD].T
    # Wait — GPT-2 c_proj is a Conv1D, weight shape is (d_model, d_model)
    # where input @ weight gives output. So head h's contribution is:
    # input[h*HD:(h+1)*HD] @ weight[h*HD:(h+1)*HD, :]

    print(f"\n  Top heads by contrastive norm at END position:")
    print(f"  {'Head':>20} {'Norm':>6} {'A-pole tokens':>40} {'B-pole tokens':>40}")
    print(f"  {'─'*20} {'─'*6} {'─'*40} {'─'*40}")

    all_head_data = []

    for L in range(NL):
        c_proj_w = model.transformer.h[L].attn.c_proj.weight.float().cpu()
        # GPT-2 Conv1D: weight shape is (in_features, out_features)
        # output = input @ weight
        # So head h's contribution = input[h*HD:(h+1)*HD] @ weight[h*HD:(h+1)*HD, :]

        for h in range(NH):
            inp_a = captured_a[L][h*HD:(h+1)*HD]
            inp_b = captured_b[L][h*HD:(h+1)*HD]
            d_inp = inp_a - inp_b

            # Head's contribution to residual stream
            w_h = c_proj_w[h*HD:(h+1)*HD, :]  # (HD, d_model)
            contribution = d_inp @ w_h  # (d_model,)

            norm = float(contribution.norm())
            logits = contribution @ W_U.cpu().T
            a_toks = tk(logits)
            b_toks = bk(logits)

            role = ""
            for rname, rheads in KNOWN_CIRCUIT.items():
                if (L, h) in rheads:
                    role = f" ← {rname}"
                    break

            all_head_data.append((L, h, norm, a_toks, b_toks, role))

    # Sort by norm and show top 15
    all_head_data.sort(key=lambda x: -x[2])
    for L, h, norm, a_toks, b_toks, role in all_head_data[:20]:
        marker = " ***" if (L, h) in ALL_KNOWN else ""
        print(f"  L{L:>2}H{h:<2} {role:>20} {norm:>6.1f}  "
              f"A=[{a_toks[:35]:>35}]  B=[{b_toks[:35]:>35}]{marker}")

    # Count how many known circuit heads are in the top N
    for topn in [5, 10, 15, 20]:
        top_heads = set((d[0], d[1]) for d in all_head_data[:topn])
        found = top_heads & ALL_KNOWN
        found_names = [head_label(l, h) for l, h in sorted(found)]
        print(f"\n  Top-{topn}: {len(found)}/{len(ALL_KNOWN)} known circuit heads: "
              f"{', '.join(found_names) if found_names else 'none'}")

    torch.cuda.empty_cache()


# ============================================================
# TEST 3: Attention patterns — do name movers attend to IO?
# ============================================================
print(f"\n{'='*100}")
print("TEST 3: ATTENTION PATTERNS — do name movers attend to the IO token?")
print("=" * 100)

pa, pb, io_a, io_b = IOI_PAIRS[0]  # John/Mary
ids_a = tok(pa, add_special_tokens=False)["input_ids"]
ids_b = tok(pb, add_special_tokens=False)["input_ids"]
tokens_a = [tok.decode([t]) for t in ids_a]
tokens_b = [tok.decode([t]) for t in ids_b]

print(f"  Tokens A: {tokens_a}")
print(f"  Tokens B: {tokens_b}")

# Find IO and S2 positions
io_pos_a = None
s2_pos_a = None
for i, t in enumerate(tokens_a):
    if io_a.lower() in t.lower() and io_pos_a is None:
        io_pos_a = i
    if io_b.lower() in t.lower():
        # Second occurrence of the subject = S2
        s2_pos_a = i

io_pos_b = None
s2_pos_b = None
for i, t in enumerate(tokens_b):
    if io_b.lower() in t.lower() and io_pos_b is None:
        io_pos_b = i
    if io_a.lower() in t.lower():
        s2_pos_b = i

print(f"  IO position (A): {io_pos_a} ('{tokens_a[io_pos_a]}')")
print(f"  S2 position (A): {s2_pos_a} ('{tokens_a[s2_pos_a]}')")

out_a, _ = get_hidden_states(pa)
out_b, _ = get_hidden_states(pb)

# Check attention from END to IO and S2 for known name movers
print(f"\n  Attention from END position to IO and S2:")
print(f"  {'Head':>25} {'attn→IO (A)':>12} {'attn→IO (B)':>12} {'attn→S2 (A)':>12} {'attn→S2 (B)':>12}")
print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

heads_to_check = (
    KNOWN_CIRCUIT['name_mover_pos'] +
    KNOWN_CIRCUIT['name_mover_neg'] +
    KNOWN_CIRCUIT['s_inhibition']
)

for L, h in heads_to_check:
    attn_a = out_a.attentions[L][0, h, -1, :]  # from END
    attn_b = out_b.attentions[L][0, h, -1, :]

    a_io = float(attn_a[io_pos_a]) if io_pos_a is not None else 0
    b_io = float(attn_b[io_pos_b]) if io_pos_b is not None else 0
    a_s2 = float(attn_a[s2_pos_a]) if s2_pos_a is not None else 0
    b_s2 = float(attn_b[s2_pos_b]) if s2_pos_b is not None else 0

    label = head_label(L, h)
    print(f"  {label:>25} {a_io:>12.3f} {b_io:>12.3f} {a_s2:>12.3f} {b_s2:>12.3f}")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# TEST 4: Position-resolved patching (denoising)
# ============================================================
print(f"\n{'='*100}")
print("TEST 4: POSITION-RESOLVED PATCHING — trace S2 → END transfer")
print("=" * 100)

pa, pb, io_a, io_b = IOI_PAIRS[0]
ids_a = tok(pa, add_special_tokens=False)["input_ids"]
ids_b = tok(pb, add_special_tokens=False)["input_ids"]
tokens_a = [tok.decode([t]) for t in ids_a]

# Target token
target_tok = io_a  # "Mary"
target_id = tok(" " + target_tok, add_special_tokens=False)["input_ids"]
if len(target_id) == 1:
    target_id = target_id[0]
else:
    target_id = tok(target_tok, add_special_tokens=False)["input_ids"][0]

# Clean and corrupt baselines
with torch.no_grad():
    out_clean = model(torch.tensor([ids_a], device=DEV), output_hidden_states=True)
    out_corrupt = model(torch.tensor([ids_b], device=DEV), output_hidden_states=True)

p_clean = float(torch.softmax(out_clean.logits[0, -1].float(), -1)[target_id])
p_corrupt = float(torch.softmax(out_corrupt.logits[0, -1].float(), -1)[target_id])
gap = p_clean - p_corrupt

print(f"  Clean: P({target_tok}) = {p_clean:.4f}")
print(f"  Corrupt: P({target_tok}) = {p_corrupt:.4f}")
print(f"  Gap = {gap:.4f}")

# Find key positions
io_pos = io_pos_a
s2_pos = s2_pos_a
end_pos = len(ids_a) - 1

print(f"  IO pos: {io_pos}, S2 pos: {s2_pos}, END pos: {end_pos}")

positions_to_patch = [
    ("IO", io_pos),
    ("S2", s2_pos),
    ("END", end_pos),
    ("S2+END", [s2_pos, end_pos]),
]

print(f"\n  {'Position':>10}", end="")
for L in range(NL + 1):
    print(f"  L{L:>2}", end="")
print()

for pos_name, pos in positions_to_patch:
    print(f"  {pos_name:>10}", end="")
    for L in range(NL + 1):
        # Patch: replace corrupt hidden state at this position/layer with clean
        dh_at_pos = {}
        if isinstance(pos, list):
            for p in pos:
                dh_at_pos[p] = (out_clean.hidden_states[L][0, p, :] -
                                out_corrupt.hidden_states[L][0, p, :]).detach()
        else:
            dh_at_pos[pos] = (out_clean.hidden_states[L][0, pos, :] -
                              out_corrupt.hidden_states[L][0, pos, :]).detach()

        injected = [False]

        def make_hook(deltas):
            def hook_fn(module, input, output):
                if injected[0]:
                    return output
                if isinstance(output, tuple):
                    h = output[0].clone()
                    for p, d in deltas.items():
                        h[0, p, :] += d
                    injected[0] = True
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    for p, d in deltas.items():
                        h[0, p, :] += d
                    injected[0] = True
                    return h
            return hook_fn

        if L < NL:
            handle = model.transformer.h[L].register_forward_hook(make_hook(dh_at_pos))
        else:
            handle = model.transformer.ln_f.register_forward_hook(make_hook(dh_at_pos))

        injected[0] = False
        with torch.no_grad():
            out_patched = model(torch.tensor([ids_b], device=DEV))
        handle.remove()

        p_patched = float(torch.softmax(out_patched.logits[0, -1].float(), -1)[target_id])
        recovery = (p_patched - p_corrupt) / gap * 100 if gap != 0 else 0
        print(f"  {recovery:>4.0f}", end="")

    print("%")

del out_clean, out_corrupt
torch.cuda.empty_cache()


# ============================================================
# TEST 5: Accuracy across name pairs and templates
# ============================================================
print(f"\n{'='*100}")
print("TEST 5: IOI ACCURACY — across templates and name pairs")
print("=" * 100)

templates = [
    "When {A} and {B} went to the store, {A} gave a drink to",
    "When {A} and {B} had lunch, {A} passed the bill to",
    "{A} and {B} went to the park. {A} gave a ball to",
]
name_pairs = [
    ("John", "Mary"), ("Alice", "Bob"), ("Dan", "Eve"),
    ("Tom", "Sarah"), ("Mike", "Lisa"),
]

correct = 0
total = 0
for template in templates:
    for a_name, b_name in name_pairs:
        for swap in [False, True]:
            if swap:
                s1, s2 = b_name, a_name
                expected = a_name
            else:
                s1, s2 = a_name, b_name
                expected = b_name

            prompt = template.format(A=s1, B=s2)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV))
            pred = tok.decode([out.logits[0, -1].argmax().item()]).strip()

            is_correct = expected.lower() in pred.lower()
            if is_correct:
                correct += 1
            total += 1

    torch.cuda.empty_cache()

print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

print(f"\n{'='*100}")
print("DONE")
print("=" * 100)
