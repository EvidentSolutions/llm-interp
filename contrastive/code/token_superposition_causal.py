"""
Token-shaped causality under superposition.

NOT an energy measurement. Instead:
1. Extract Δh at reference layer
2. Decompose into token-readable component (top-k W_U projection)
   and residual (everything else)
3. Inject EACH separately → measure prediction recovery
4. Iteratively peel: remove top-k, re-read remainder through W_U,
   check if NEW causal tokens emerge (the "cooked under becoming" effect)
5. Check compositionality: does injecting component 1 + component 2
   equal injecting the whole?

The question: is the causal content token-shaped, even if the energy isn't?
And: does peeling reveal hidden causal structure in superposition?
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
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
d_model = model.config.hidden_size
W_U = model.lm_head.weight.detach().float()  # (vocab, d_model)


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_str(logits, k=6):
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{vals[j]:.1f}")
            for j in range(k)]


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out, ids


def get_probs(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    return torch.softmax(out.logits[0, -1].float(), -1)


def inject_and_measure(text_b, delta, layer, pos=-1):
    """Inject delta into text_b at layer, return top-5 probs."""
    ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
    injected = [False]

    def hook_fn(module, input, output):
        if injected[0]:
            return output
        injected[0] = True
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, pos, :] += delta.half().to(DEV)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, pos, :] += delta.half().to(DEV)
            return h

    if layer < NL:
        handle = model.model.layers[layer].register_forward_hook(hook_fn)
    else:
        handle = model.model.final_layernorm.register_forward_hook(hook_fn)

    with torch.no_grad():
        out = model(torch.tensor([ids_b], device=DEV))
    handle.remove()

    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, 5)
    top5 = [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(5)]
    return probs, top5


def project_onto_token_subspace(vec, k=20):
    """
    Decompose vec into:
    - token_component: projection onto top-k + bottom-k W_U directions
    - residual: everything else
    Returns token_component, residual, token indices used
    """
    vec_f = vec.float().to(DEV)
    logits = vec_f @ W_U.T
    topk_vals, topk_idx = torch.topk(logits, k)
    botk_vals, botk_idx = torch.topk(-logits, k)
    all_idx = torch.cat([topk_idx, botk_idx])

    # Get W_U rows and build orthonormal basis
    wu_rows = W_U[all_idx]  # (2k, d_model)
    Q, R = torch.linalg.qr(wu_rows.T)  # Q: (d_model, 2k)

    # Project
    token_component = Q @ (Q.T @ vec_f)
    residual = vec_f - token_component

    return token_component, residual, all_idx


def iterative_peel(vec, n_rounds=4, k=10):
    """
    Iteratively remove top-k token directions, re-read through W_U.
    Reveals hidden tokens in superposition.
    """
    remainder = vec.float().to(DEV)
    rounds = []

    for r in range(n_rounds):
        logits = remainder @ W_U.T
        top_toks = topk_str(logits, k=6)
        bot_toks = topk_str(-logits, k=6)
        norm = float(remainder.norm())

        # Record this round
        rounds.append({
            "round": r,
            "norm": norm,
            "top": top_toks,
            "bot": bot_toks,
        })

        # Remove top-k + bottom-k directions
        topk_vals, topk_idx = torch.topk(logits.abs(), k)
        wu_rows = W_U[topk_idx]
        Q, R = torch.linalg.qr(wu_rows.T)
        projection = Q @ (Q.T @ remainder)
        remainder = remainder - projection

    return rounds


# ── TEST CASES ────────────────────────────────────────────────

cases = [
    {
        "name": "hotdog_food",
        "text_a": "The hot dog was",
        "text_b": "The cold dog was",
        "description": "Food compound vs literal animal",
    },
    {
        "name": "IOI_names",
        "text_a": "When Mary and John went to the store, John gave a drink to",
        "text_b": "When Mary and John went to the store, Mary gave a drink to",
        "description": "IOI name-mover circuit",
    },
    {
        "name": "grief_joy",
        "text_a": "She learned that her mother had passed away. She felt",
        "text_b": "She learned that her mother had won the lottery. She felt",
        "description": "Emotional valence",
    },
    {
        "name": "caught_cold_fish",
        "text_a": "She caught a cold and went to",
        "text_b": "She caught a fish and went to",
        "description": "Lexical disambiguation",
    },
    {
        "name": "metaphor_cold",
        "text_a": "The ice in the bucket was extremely cold. The temperature was",
        "text_b": "The reception at the party was extremely cold. The atmosphere was",
        "description": "Literal vs metaphorical",
    },
    {
        "name": "some_all",
        "text_a": "Some of the students passed the exam, so",
        "text_b": "All of the students passed the exam, so",
        "description": "Scalar quantifier",
    },
    {
        "name": "agent_swap",
        "text_a": "The dog chased the cat through the park. The animal that was exhausted was the",
        "text_b": "The cat chased the dog through the park. The animal that was exhausted was the",
        "description": "Agent-patient role swap (structural)",
    },
    {
        "name": "active_passive",
        "text_a": "The dog chased the cat. The one that was caught was the",
        "text_b": "The cat was chased by the dog. The one that was caught was the",
        "description": "Voice alternation (same meaning)",
    },
    {
        "name": "capital_france",
        "text_a": "The capital of France is",
        "text_b": "The capital of Germany is",
        "description": "Factual recall",
    },
    {
        "name": "theft_moral",
        "text_a": "He slipped a bottle under his coat and walked out without paying. He",
        "text_b": "He picked up a bottle, went to the register and paid. He",
        "description": "Moral valence",
    },
]

ref_L = _sl(28)[0]
K_PEEL = 10   # tokens per peel round
K_PROJ = 20   # tokens for projection decomposition

for case in cases:
    print("\n" + "="*70)
    print(f"CASE: {case['name']} — {case['description']}")
    print("="*70)

    text_a, text_b = case["text_a"], case["text_b"]

    # ── Get hidden states and Δh ──
    out_a, ids_a = get_hidden_states(text_a)
    out_b, ids_b = get_hidden_states(text_b)

    ha = out_a.hidden_states[ref_L][0, -1, :].float()
    hb = out_b.hidden_states[ref_L][0, -1, :].float()
    dh = ha - hb

    # Baseline predictions
    probs_a = torch.softmax(out_a.logits[0, -1].float(), -1)
    probs_b = torch.softmax(out_b.logits[0, -1].float(), -1)

    top_a = [(tok.decode([int(i)]).strip()[:14], float(v))
             for v, i in zip(*torch.topk(probs_a, 5))]
    top_b = [(tok.decode([int(i)]).strip()[:14], float(v))
             for v, i in zip(*torch.topk(probs_b, 5))]

    print(f"\n  Baseline A: {top_a}")
    print(f"  Baseline B: {top_b}")
    print(f"  ||Δh||: {dh.norm():.1f}")

    del out_a, out_b
    torch.cuda.empty_cache()

    # ── Decompose Δh ──
    token_comp, residual, tok_idx = project_onto_token_subspace(dh, k=K_PROJ)

    print(f"\n  Decomposition (k={K_PROJ}, {2*K_PROJ} directions):")
    print(f"    ||token_component||: {token_comp.norm():.1f}  "
          f"({token_comp.norm()**2 / dh.norm()**2 * 100:.1f}% energy)")
    print(f"    ||residual||:        {residual.norm():.1f}  "
          f"({residual.norm()**2 / dh.norm()**2 * 100:.1f}% energy)")

    # ── Inject full Δh ──
    print(f"\n  INJECTION AT L{ref_L}:")
    _, top_full = inject_and_measure(text_b, dh, ref_L)
    print(f"    Full Δh:            {top_full}")

    # ── Inject token component only ──
    _, top_tok = inject_and_measure(text_b, token_comp, ref_L)
    print(f"    Token component:    {top_tok}")

    # ── Inject residual only ──
    _, top_res = inject_and_measure(text_b, residual, ref_L)
    print(f"    Residual only:      {top_res}")

    # ── Compositionality: token + residual ≈ full? ──
    # (Should be identical if decomposition is orthogonal — this verifies)
    composed = token_comp + residual
    _, top_composed = inject_and_measure(text_b, composed, ref_L)
    print(f"    Composed (t+r):     {top_composed}")

    # ── Quantitative recovery ──
    # Pick target token: top-1 from text_a's prediction
    target_id = torch.argmax(probs_a).item()
    target_tok = tok.decode([target_id]).strip()
    p_a = float(probs_a[target_id])
    p_b = float(probs_b[target_id])
    gap = p_a - p_b

    if abs(gap) > 0.001:
        probs_full, _ = inject_and_measure(text_b, dh, ref_L)
        probs_tok, _ = inject_and_measure(text_b, token_comp, ref_L)
        probs_res, _ = inject_and_measure(text_b, residual, ref_L)

        rec_full = (float(probs_full[target_id]) - p_b) / gap * 100
        rec_tok = (float(probs_tok[target_id]) - p_b) / gap * 100
        rec_res = (float(probs_res[target_id]) - p_b) / gap * 100

        print(f"\n  Recovery of P({target_tok}) [gap={gap:.3f}]:")
        print(f"    Full Δh:         {rec_full:>+7.1f}%")
        print(f"    Token component: {rec_tok:>+7.1f}%")
        print(f"    Residual only:   {rec_res:>+7.1f}%")
        print(f"    Sum (t+r):       {rec_tok + rec_res:>+7.1f}%  "
              f"(should ≈ full if linear)")

    # ── ITERATIVE PEELING ──
    print(f"\n  ITERATIVE PEEL (removing top-{K_PEEL} abs-logit tokens per round):")
    rounds = iterative_peel(dh, n_rounds=5, k=K_PEEL)
    for r in rounds:
        print(f"    Round {r['round']} (||remainder||={r['norm']:.1f}):")
        print(f"      +[{', '.join(f'{t[0]}({t[1]})' for t in r['top'][:5])}]")
        print(f"      -[{', '.join(f'{t[0]}({t[1]})' for t in r['bot'][:5])}]")

    # ── CAUSAL PEELING: inject each peeled layer ──
    print(f"\n  CAUSAL PEEL — is each peeled layer independently causal?")
    remainder = dh.float().to(DEV)
    for r_idx in range(4):
        logits = remainder @ W_U.T
        topk_vals, topk_idx = torch.topk(logits.abs(), K_PEEL)
        wu_rows = W_U[topk_idx]
        Q, R = torch.linalg.qr(wu_rows.T)
        layer_component = Q @ (Q.T @ remainder)
        remainder = remainder - layer_component

        # What does this layer read as?
        layer_logits = layer_component @ W_U.T
        layer_top = topk_str(layer_logits, 4)
        layer_bot = topk_str(-layer_logits, 4)

        # Inject just this layer
        if abs(gap) > 0.001:
            probs_layer, _ = inject_and_measure(text_b, layer_component, ref_L)
            rec_layer = (float(probs_layer[target_id]) - p_b) / gap * 100
            print(f"    Peel {r_idx}: ||={layer_component.norm():.1f}  "
                  f"rec={rec_layer:>+6.1f}%  "
                  f"+[{', '.join(t[0] for t in layer_top)}]  "
                  f"-[{', '.join(t[0] for t in layer_bot)}]")
        else:
            _, top_layer = inject_and_measure(text_b, layer_component, ref_L)
            print(f"    Peel {r_idx}: ||={layer_component.norm():.1f}  "
                  f"+[{', '.join(t[0] for t in layer_top)}]  "
                  f"-[{', '.join(t[0] for t in layer_bot)}]  "
                  f"→ {top_layer[:3]}")

    # ── Final remainder after all peels ──
    if abs(gap) > 0.001:
        probs_rem, top_rem = inject_and_measure(text_b, remainder, ref_L)
        rec_rem = (float(probs_rem[target_id]) - p_b) / gap * 100
        rem_logits = remainder @ W_U.T
        rem_top = topk_str(rem_logits, 4)
        print(f"    Remainder: ||={remainder.norm():.1f}  "
              f"rec={rec_rem:>+6.1f}%  "
              f"+[{', '.join(t[0] for t in rem_top)}]")

    torch.cuda.empty_cache()


print("\n" + "="*70)
print("DONE")
print("="*70)
