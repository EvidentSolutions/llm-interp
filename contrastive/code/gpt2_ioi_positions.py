"""
Per-position contrastive reading for IOI in GPT-2.

Read the contrastive projection at EVERY position and layer, not just END.
This shows where each signal originates and how it propagates.

Prompt A: "When John and Mary went to the store, John gave a drink to"
Prompt B: "When Mary and John went to the store, Mary gave a drink to"

Tokens:    When  John  and  Mary  went  to  the  store  ,  John  gave  a  drink  to
Position:    0     1    2    3     4     5   6    7      8   9    10   11   12    13

Positions that differ: 1 (John/Mary), 3 (Mary/John), 9 (John/Mary)
Prediction position: 13 (END)

The signal must flow from the differing positions to END.

Usage: .venv/Scripts/python.exe contrastive/code/gpt2_ioi_positions.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.n_layer
NH = model.config.n_head
HD = model.config.n_embd // NH
W_U = model.lm_head.weight.detach().float()


def tk(logits, k=4):
    v, i = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:10] for j in range(k))


def run(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out, ids


def run_capture(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    captured = {}
    hooks = []
    for L in range(NL):
        def make_hook(li):
            def hook_fn(m, inp, out):
                captured[li] = inp[0].detach().float().cpu()
            return hook_fn
        h = model.transformer.h[L].attn.c_proj.register_forward_hook(make_hook(L))
        hooks.append(h)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    for h in hooks:
        h.remove()
    return out, ids, captured


PA = "When John and Mary went to the store, John gave a drink to"
PB = "When Mary and John went to the store, Mary gave a drink to"

ids_a = tok(PA, add_special_tokens=False)["input_ids"]
ids_b = tok(PB, add_special_tokens=False)["input_ids"]
tokens_a = [tok.decode([t]) for t in ids_a]
tokens_b = [tok.decode([t]) for t in ids_b]
SEQ_LEN = len(ids_a)

print(f"A: {PA}")
print(f"B: {PB}")
print(f"Tokens A: {tokens_a}")
print(f"Tokens B: {tokens_b}")
print(f"Differ at: pos 1 ({tokens_a[1]}/{tokens_b[1]}), "
      f"pos 3 ({tokens_a[3]}/{tokens_b[3]}), "
      f"pos 9 ({tokens_a[9]}/{tokens_b[9]})")


# ============================================================
# 1. POSITION × LAYER MAP: contrastive norm and top tokens
# ============================================================
print(f"\n{'='*120}")
print("1. CONTRASTIVE READOUT AT EVERY POSITION AND LAYER")
print("   What does the residual stream difference read through W_U at each (position, layer)?")
print("=" * 120)

out_a, _ = run(PA)
out_b, _ = run(PB)

# First show norms as a heatmap-like table
print(f"\n  Contrastive norm ||Δh|| at each position × layer:")
print(f"  {'':>4}", end="")
for p in range(SEQ_LEN):
    label = tokens_a[p].strip()[:5]
    print(f" {label:>6}", end="")
print()
print(f"  {'':>4}", end="")
for p in range(SEQ_LEN):
    print(f"  p{p:<4}", end="")
print()

for L in range(NL + 1):
    print(f"  L{L:<2}", end="")
    for p in range(SEQ_LEN):
        h_a = out_a.hidden_states[L][0, p, :].float()
        h_b = out_b.hidden_states[L][0, p, :].float()
        norm = float((h_a - h_b).norm())
        print(f" {norm:>6.1f}", end="")
    print()

# Now show the W_U readout at key positions across all layers
KEY_POSITIONS = [
    (1, "S1/IO_B"),    # John in A, Mary in B (first name mention)
    (3, "IO_A/S1_B"),  # Mary in A, John in B (second name mention)
    (9, "S2"),          # John in A, Mary in B (repeated subject)
    (13, "END"),        # prediction position
]

for pos, pos_label in KEY_POSITIONS:
    print(f"\n  Position {pos} ({pos_label}): A='{tokens_a[pos]}', B='{tokens_b[pos]}'")
    print(f"  {'Layer':>5} {'Norm':>6}  {'A-pole':>40}  {'B-pole':>40}")
    for L in range(NL + 1):
        h_a = out_a.hidden_states[L][0, pos, :].float()
        h_b = out_b.hidden_states[L][0, pos, :].float()
        dh = h_a - h_b
        norm = float(dh.norm())
        ld = dh @ W_U.T
        a_pole = tk(ld)
        b_pole = tk(-ld)
        print(f"  L{L:>3} {norm:>6.1f}  {a_pole:>40}  {b_pole:>40}")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# 2. PER-HEAD DECOMPOSITION AT EACH KEY POSITION
# ============================================================
print(f"\n{'='*120}")
print("2. PER-HEAD DECOMPOSITION AT EACH KEY POSITION")
print("   Which heads contribute at which positions?")
print("=" * 120)

out_a, _, cap_a = run_capture(PA)
out_b, _, cap_b = run_capture(PB)

for pos, pos_label in KEY_POSITIONS:
    print(f"\n  ── Position {pos} ({pos_label}) ──")
    print(f"  {'#':>2} {'Head':>6} {'Norm':>6}  {'A-pole':>35}  {'B-pole':>35}")

    results = []
    for L in range(NL):
        W = model.transformer.h[L].attn.c_proj.weight.float().cpu()
        for h in range(NH):
            d = cap_a[L][0, pos, h*HD:(h+1)*HD] - cap_b[L][0, pos, h*HD:(h+1)*HD]
            contrib = d @ W[h*HD:(h+1)*HD, :]
            norm = float(contrib.norm())
            logits = contrib @ W_U.cpu().T
            results.append((L, h, norm, logits))

    results.sort(key=lambda x: -x[2])
    for rank, (L, h, norm, logits) in enumerate(results[:10], 1):
        a_pole = tk(logits)
        b_pole = tk(-logits)
        print(f"  {rank:>2} L{L}H{h:<2} {norm:>6.1f}  {a_pole:>35}  {b_pole:>35}")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# 3. SAME-POSITION READING ACROSS NAME PAIRS
#    Multi-contrast at each position to separate name from structure
# ============================================================
print(f"\n{'='*120}")
print("3. MULTI-CONTRAST AT EACH POSITION (10 name pairs)")
print("   Average across pairs to see what's shared vs pair-specific at each position.")
print("=" * 120)

TEMPLATE = "When {A} and {B} went to the store, {A} gave a drink to"
PAIRS = [
    ("John", "Mary"), ("Alice", "Bob"), ("Dan", "Eve"),
    ("Tom", "Sarah"), ("Mike", "Lisa"), ("James", "Anna"),
    ("Peter", "Jane"), ("David", "Emma"), ("Chris", "Laura"),
    ("Steve", "Karen"),
]

# Collect hidden states for each pair at key positions
# For each (position, layer): accumulate Δh vectors
from collections import defaultdict
dh_accum = defaultdict(list)  # (pos, layer) → list of Δh vectors

for a, b in PAIRS:
    pa = TEMPLATE.format(A=a, B=b)
    pb = TEMPLATE.format(A=b, B=a)
    out_a, _ = run(pa)
    out_b, _ = run(pb)

    for pos, _ in KEY_POSITIONS:
        for L in range(NL + 1):
            h_a = out_a.hidden_states[L][0, pos, :].float().cpu()
            h_b = out_b.hidden_states[L][0, pos, :].float().cpu()
            dh_accum[(pos, L)].append(h_a - h_b)

    del out_a, out_b

torch.cuda.empty_cache()

for pos, pos_label in KEY_POSITIONS:
    print(f"\n  Position {pos} ({pos_label}):")
    print(f"  {'Layer':>5} {'MeanNorm':>8} {'AvgNorm':>8} {'Retain':>7}  "
          f"{'Single (pair 1)':>35}  {'Averaged (10 pairs)':>35}")

    for L in range(NL + 1):
        vectors = dh_accum[(pos, L)]
        norms = [float(v.norm()) for v in vectors]
        mean_norm = sum(norms) / len(norms)
        avg_vec = torch.stack(vectors).mean(dim=0)
        avg_norm = float(avg_vec.norm())
        retention = avg_norm / mean_norm if mean_norm > 0 else 0

        single_logits = vectors[0] @ W_U.cpu().T
        avg_logits = avg_vec @ W_U.cpu().T

        s_read = tk(single_logits)
        a_read = tk(avg_logits)

        print(f"  L{L:>3} {mean_norm:>8.1f} {avg_norm:>8.1f} {retention:>6.0%}  "
              f"{s_read:>35}  {a_read:>35}")


# ============================================================
# 4. Per-head multi-contrast at S2 and at pos 1 (S1)
# ============================================================
print(f"\n{'='*120}")
print("4. PER-HEAD MULTI-CONTRAST AT S1 (pos 1) AND S2 (pos 9)")
print("   Which heads write shared signal at the name positions?")
print("=" * 120)

# Collect per-head contributions at S1 and S2 across pairs
head_contribs = defaultdict(lambda: defaultdict(list))  # pos → (L,H) → list

for a, b in PAIRS:
    pa = TEMPLATE.format(A=a, B=b)
    pb = TEMPLATE.format(A=b, B=a)
    _, _, cap_a = run_capture(pa)
    _, _, cap_b = run_capture(pb)

    for pos in [1, 9]:
        for L in range(NL):
            W = model.transformer.h[L].attn.c_proj.weight.float().cpu()
            for h in range(NH):
                d = cap_a[L][0, pos, h*HD:(h+1)*HD] - cap_b[L][0, pos, h*HD:(h+1)*HD]
                contrib = d @ W[h*HD:(h+1)*HD, :]
                head_contribs[pos][(L, h)].append(contrib)

    torch.cuda.empty_cache()

for pos, pos_label in [(1, "S1"), (9, "S2")]:
    print(f"\n  ── Position {pos} ({pos_label}) ──")

    # Compute averaged norms and readouts
    head_results = []
    for L in range(NL):
        for h in range(NH):
            contribs = head_contribs[pos][(L, h)]
            norms = [float(c.norm()) for c in contribs]
            mean_norm = sum(norms) / len(norms)
            avg = torch.stack(contribs).mean(dim=0)
            avg_norm = float(avg.norm())
            retention = avg_norm / mean_norm if mean_norm > 0 else 0
            avg_logits = avg @ W_U.cpu().T
            single_logits = contribs[0] @ W_U.cpu().T
            head_results.append((L, h, mean_norm, avg_norm, retention,
                                 single_logits, avg_logits))

    head_results.sort(key=lambda x: -x[2])

    print(f"  {'#':>2} {'Head':>6} {'MNorm':>6} {'ANorm':>6} {'Ret':>5}  "
          f"{'Single (pair 1)':>30}  {'Averaged':>30}")
    for rank, (L, h, mn, an, ret, slg, alg) in enumerate(head_results[:12], 1):
        print(f"  {rank:>2} L{L}H{h:<2} {mn:>6.1f} {an:>6.1f} {ret:>4.0%}  "
              f"{tk(slg):>30}  {tk(alg):>30}")


print(f"\n{'='*120}")
print("DONE")
print("=" * 120)
