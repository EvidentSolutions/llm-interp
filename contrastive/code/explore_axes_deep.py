"""
Deep follow-up on wild axis findings.

1. Code modality: does assignment/test extend to other constructs?
   (function def vs call, loop vs conditional, print vs return)
2. Metaphor probe: is "cold reception" processed differently from
   "cold ice" at each layer? When does the model decide it's figurative?
3. Epistemic gradient: bare → know → believe → suspect → doubt → deny
   Is there a linear epistemic axis, or discrete clusters?
4. Voice entanglement: try with longer contexts where voice is clearer
5. Negation layering: "not", "never", "no longer", double negation

Usage: MODEL=microsoft/phi-4 python3 contrastive/code/explore_axes_deep.py
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
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach()
    else:
        W_U = model.embed_out.weight.detach()  # Pythia

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    ref_L = _sl(28)[0]
    cos = torch.nn.functional.cosine_similarity

    def get_h(text, layer=None):
        if layer is None:
            layer = ref_L
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        return out.hidden_states[layer][0, -1, :].float(), out

    def topk_str(logits, k=6):
        vals, idxs = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(idxs[j])]).strip()[:12]
                         for j in range(k))

    def show_contrast(label, text_a, text_b, layers=None):
        if layers is None:
            layers = _sl(0, 8, 16, 20, 24, 28, 32)
        print(f"\n  {label}")
        print(f"  A: \"{text_a[-60:]}\"")
        print(f"  B: \"{text_b[-60:]}\"")
        ids_a = tok(text_a, add_special_tokens=False)["input_ids"]
        ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out_a = model(torch.tensor([ids_a], device=DEV),
                          output_hidden_states=True)
            out_b = model(torch.tensor([ids_b], device=DEV),
                          output_hidden_states=True)
        for L in layers:
            ha = out_a.hidden_states[L][0, -1, :].float()
            hb = out_b.hidden_states[L][0, -1, :].float()
            dh = ha - hb
            norm = float(dh.norm() / ha.norm())
            ld = dh @ W_U.float().T
            pos = topk_str(ld)
            neg = topk_str(-ld)
            print(f"    L{L:>2} ({norm:.3f}) +[{pos}]  -[{neg}]")
        del out_a, out_b
        torch.cuda.empty_cache()

    # ================================================================
    # 1. EPISTEMIC GRADIENT
    # ================================================================
    print("=" * 100)
    print("1. EPISTEMIC GRADIENT — is there a linear axis?")
    print("=" * 100)

    base = "the capital of France is"
    framings = [
        ("bare", f"The capital of France is"),
        ("know", f"I know that {base}"),
        ("believe", f"I believe that {base}"),
        ("think", f"I think that {base}"),
        ("suspect", f"I suspect that {base}"),
        ("doubt", f"I doubt that {base}"),
        ("deny", f"I deny that {base}"),
        ("false", f"It is false that {base}"),
    ]

    # Get hidden states at reference layer
    vecs = {}
    for label, text in framings:
        h, out = get_h(text)
        vecs[label] = h.cpu()
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        top3_v, top3_i = torch.topk(probs, 3)
        top3 = [(tok.decode([int(top3_i[j])]).strip(),
                  round(float(top3_v[j]), 3)) for j in range(3)]
        print(f"  {label:>10}: {top3}")
        del out
    torch.cuda.empty_cache()

    # Pairwise cosine matrix
    labels = [l for l, _ in framings]
    print(f"\n  Pairwise cosine at L{ref_L}:")
    print(f"  {'':>10}", end="")
    for l in labels:
        print(f" {l[:6]:>7}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:>10}", end="")
        for l2 in labels:
            c = float(cos(vecs[l1].unsqueeze(0), vecs[l2].unsqueeze(0)))
            print(f" {c:>+7.3f}", end="")
        print()

    # Contrastive differences: project each framing against "bare"
    print(f"\n  Each framing minus bare, projected through W_U at L{ref_L}:")
    bare_h = vecs["bare"]
    for label in labels[1:]:
        dh = vecs[label] - bare_h
        ld = dh.to(DEV) @ W_U.float().T
        print(f"  {label:>10}: +[{topk_str(ld)}]  -[{topk_str(-ld)}]")

    # Is there a linear gradient? Project onto know→doubt direction
    know_doubt = vecs["doubt"] - vecs["know"]
    print(f"\n  Projection onto know→doubt axis:")
    for label in labels:
        proj = float(torch.dot(vecs[label] - bare_h, know_doubt)
                      / know_doubt.norm())
        print(f"    {label:>10}: {proj:>+8.1f}")

    # ================================================================
    # 2. METAPHOR PROCESSING — layer-by-layer
    # ================================================================
    print(f"\n{'='*100}")
    print("2. METAPHOR — when does the model decide it's figurative?")
    print("=" * 100)

    show_contrast("literal cold vs metaphorical cold",
        "The ice in the bucket was extremely cold. The temperature was",
        "The reception at the party was extremely cold. The atmosphere was")

    show_contrast("literal sharp vs metaphorical sharp",
        "The knife on the counter was extremely sharp. The blade was",
        "The criticism in the review was extremely sharp. The tone was")

    show_contrast("literal bright vs metaphorical bright",
        "The lamp in the corner was extremely bright. The light was",
        "The student in the class was extremely bright. The child was")

    show_contrast("literal heavy vs metaphorical heavy",
        "The boulder on the trail was extremely heavy. The weight was",
        "The news from the hospital was extremely heavy. The mood was")

    # ================================================================
    # 3. CODE MODALITY — more constructs
    # ================================================================
    print(f"\n{'='*100}")
    print("3. CODE MODALITY — what code constructs are token-readable?")
    print("=" * 100)

    code_pairs = [
        ("def vs call",
         "def calculate_sum(a, b):\n    return a +",
         "result = calculate_sum(3,"),

        ("for vs while",
         "for i in range(10):\n    total +=",
         "while i < 10:\n    total +="),

        ("print vs return",
         "def foo(x):\n    print(x +",
         "def foo(x):\n    return x +"),

        ("if vs elif",
         "if x > 5:\n    result =",
         "elif x > 5:\n    result ="),

        ("list vs dict",
         "data = [1, 2, 3]\nfirst = data[",
         "data = {'a': 1, 'b': 2}\nfirst = data["),

        ("python vs javascript",
         "def greet(name):\n    return f'Hello {name}'\ngreet(",
         "function greet(name) {\n    return `Hello ${name}`;\n}\ngreet("),
    ]

    for label, a, b in code_pairs:
        show_contrast(label, a, b, layers=_sl(16, 24, 28, 32))

    # Cross-construct cosine: are code axes consistent?
    print(f"\n  Cross-construct cosine at L{ref_L}:")
    code_deltas = {}
    for label, a, b in code_pairs:
        ha, _ = get_h(a)
        hb, _ = get_h(b)
        code_deltas[label] = (ha - hb).cpu()
    torch.cuda.empty_cache()

    code_labels = [l for l, _, _ in code_pairs]
    print(f"  {'':>20}", end="")
    for l in code_labels:
        print(f" {l[:8]:>9}", end="")
    print()
    for l1 in code_labels:
        print(f"  {l1:>20}", end="")
        for l2 in code_labels:
            c = float(cos(code_deltas[l1].unsqueeze(0),
                          code_deltas[l2].unsqueeze(0)))
            print(f" {c:>+9.3f}", end="")
        print()

    # ================================================================
    # 4. NEGATION VARIANTS
    # ================================================================
    print(f"\n{'='*100}")
    print("4. NEGATION — different negation types, same content")
    print("=" * 100)

    neg_variants = [
        ("bare", "The dog is in the garden. The animal is"),
        ("not", "The dog is not in the garden. The animal is"),
        ("never", "The dog is never in the garden. The animal is"),
        ("no_longer", "The dog is no longer in the garden. The animal is"),
        ("rarely", "The dog is rarely in the garden. The animal is"),
        ("double_neg", "The dog is not never in the garden. The animal is"),
    ]

    neg_vecs = {}
    for label, text in neg_variants:
        h, out = get_h(text)
        neg_vecs[label] = h.cpu()
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        top3_v, top3_i = torch.topk(probs, 3)
        top3 = [(tok.decode([int(top3_i[j])]).strip(),
                  round(float(top3_v[j]), 3)) for j in range(3)]
        print(f"  {label:>12}: {top3}")
        del out

    print(f"\n  Pairwise cosine at L{ref_L}:")
    neg_labels = [l for l, _ in neg_variants]
    print(f"  {'':>12}", end="")
    for l in neg_labels:
        print(f" {l[:8]:>9}", end="")
    print()
    for l1 in neg_labels:
        print(f"  {l1:>12}", end="")
        for l2 in neg_labels:
            c = float(cos(neg_vecs[l1].unsqueeze(0),
                          neg_vecs[l2].unsqueeze(0)))
            print(f" {c:>+9.3f}", end="")
        print()

    # Contrastive readout: each negation minus bare
    print(f"\n  Each negation minus bare, W_U projection:")
    bare_h = neg_vecs["bare"]
    for label in neg_labels[1:]:
        dh = neg_vecs[label] - bare_h
        ld = dh.to(DEV) @ W_U.float().T
        norm = float(dh.norm())
        print(f"  {label:>12} (||dh||={norm:.0f}): "
              f"+[{topk_str(ld)}]  -[{topk_str(-ld)}]")

    # Are negation directions consistent? Project onto not-axis
    not_dir = neg_vecs["not"] - bare_h
    print(f"\n  Projection onto 'not' axis:")
    for label in neg_labels:
        proj = float(torch.dot(neg_vecs[label] - bare_h, not_dir)
                      / not_dir.norm())
        print(f"    {label:>12}: {proj:>+8.1f}")

    torch.cuda.empty_cache()

    # ================================================================
    # 5. THOUGHT vs SPEECH — what exactly is the direction?
    # ================================================================
    print(f"\n{'='*100}")
    print("5. THOUGHT vs SPEECH — multiple contents")
    print("=" * 100)

    ts_pairs = [
        ("food quality",
         "He thought that the food was",
         "He said that the food was"),
        ("weather",
         "She thought that the weather was",
         "She said that the weather was"),
        ("person",
         "They thought that the man was",
         "They said that the man was"),
        ("plan quality",
         "He thought that the plan was",
         "He said that the plan was"),
    ]

    ts_deltas = {}
    for label, a, b in ts_pairs:
        ha, _ = get_h(a)
        hb, _ = get_h(b)
        ts_deltas[label] = (ha - hb).cpu()
        ld = ts_deltas[label].to(DEV) @ W_U.float().T
        print(f"  {label:>15}: +[{topk_str(ld)}]  -[{topk_str(-ld)}]")
    torch.cuda.empty_cache()

    # Consistency
    print(f"\n  Pairwise cosine of thought-speech directions:")
    ts_labels = [l for l, _, _ in ts_pairs]
    for i, l1 in enumerate(ts_labels):
        for j, l2 in enumerate(ts_labels):
            if j > i:
                c = float(cos(ts_deltas[l1].unsqueeze(0),
                              ts_deltas[l2].unsqueeze(0)))
                print(f"    {l1} vs {l2}: cos={c:+.3f}")

    # Mean direction
    mean_ts = sum(ts_deltas.values()) / len(ts_deltas)
    ld = mean_ts.to(DEV) @ W_U.float().T
    print(f"\n  Mean thought-speech direction reads as:")
    print(f"    thought pole: [{topk_str(ld)}]")
    print(f"    speech pole:  [{topk_str(-ld)}]")

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
