"""
Wild axis exploration: does the model store linguistic dimensions
as token-readable directions in residual stream?

Each 2x2 crosses two orthogonal axes. We measure:
- Whether each axis produces a consistent direction (AC vs BD cosine)
- Whether the two axes are orthogonal (cross-cosine)
- Whether the diagonal decomposes linearly
- What tokens the contrastive direction projects onto (token-readable?)

Axes probed:
  1. claim vs question        (speech act)
  2. code vs natural language  (register/modality)
  3. ALL CAPS vs lowercase     (typography)
  4. doubt vs certainty        (epistemic stance)
  5. animate vs inanimate      (ontological category)
  6. thought vs speech         (mental vs verbal)
  7. formal vs informal        (register)
  8. active vs passive         (syntactic voice)
  9. literal vs metaphorical   (figurativity)
 10. first vs third person     (perspective)
 11. past vs future            (temporal direction)
 12. positive vs negated       (polarity)

Usage: MODEL=microsoft/phi-4 python3 contrastive/code/explore_axes_wild.py
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

# Each case: (label, axis1_name, axis2_name, A, B, C, D)
# Layout:  axis1=row, axis2=col
#   A = axis1_val1 + axis2_val1
#   B = axis1_val1 + axis2_val2
#   C = axis1_val2 + axis2_val1
#   D = axis1_val2 + axis2_val2
CASES = [
    # === 1. CLAIM vs QUESTION × content ===
    ("claim_vs_question",
     "claim/question", "france/japan",
     "The capital of France is",
     "The capital of Japan is",
     "What is the capital of France? The answer is",
     "What is the capital of Japan? The answer is"),

    # === 2. CODE vs NATURAL LANGUAGE × content ===
    ("code_vs_natural",
     "code/natural", "five/ten",
     "x = 5\nprint(x +",
     "x = 10\nprint(x +",
     "The number is five, so adding one gives",
     "The number is ten, so adding one gives"),

    # === 3. ASSIGNMENT vs EQUALITY TEST ===
    ("assign_vs_test",
     "assign/test", "five/ten",
     "a = 5\nb =",
     "a = 10\nb =",
     "if a == 5:\n    result =",
     "if a == 10:\n    result ="),

    # === 4. ALL CAPS vs LOWERCASE × content ===
    ("caps_vs_lower",
     "CAPS/lower", "dog_ran/cat_sat",
     "THE DOG RAN ACROSS THE YARD AND THEN IT",
     "THE CAT SAT ON THE MAT AND THEN IT",
     "the dog ran across the yard and then it",
     "the cat sat on the mat and then it"),

    # === 5. DOUBT vs CERTAINTY × content ===
    ("doubt_vs_certain",
     "doubt/certain", "france/germany",
     "I doubt that the capital of France is",
     "I doubt that the capital of Germany is",
     "I am certain that the capital of France is",
     "I am certain that the capital of Germany is"),

    # === 6. ANIMATE vs INANIMATE × action ===
    ("animate_vs_inanimate",
     "animate/inanimate", "broke/fell",
     "The boy accidentally broke the",
     "The boy accidentally fell down the",
     "The wind accidentally broke the",
     "The wind accidentally fell down the"),

    # === 7. THOUGHT vs SPEECH × sentiment ===
    ("thought_vs_speech",
     "thought/speech", "good/bad",
     "He thought that the food was",
     "He thought that the movie was",
     "He said that the food was",
     "He said that the movie was"),

    # === 8. FORMAL vs INFORMAL × content ===
    ("formal_vs_informal",
     "formal/informal", "heart/head",
     "The patient presented with acute chest pain radiating to the left arm. The diagnosis was",
     "The patient presented with severe headaches and visual disturbances. The diagnosis was",
     "The guy came in saying his chest really hurt and his arm felt weird. The diagnosis was",
     "The guy came in saying his head was killing him and he couldn't see straight. The diagnosis was"),

    # === 9. ACTIVE vs PASSIVE × who ===
    ("active_vs_passive",
     "active/passive", "dog_cat/cat_dog",
     "The dog chased the cat through the garden. The animal that was caught was the",
     "The cat chased the dog through the garden. The animal that was caught was the",
     "The cat was chased by the dog through the garden. The animal that was caught was the",
     "The dog was chased by the cat through the garden. The animal that was caught was the"),

    # === 10. LITERAL vs METAPHORICAL × domain ===
    ("literal_vs_metaphor",
     "literal/metaphor", "cold/sharp",
     "The ice in the bucket was extremely cold. The temperature was",
     "The knife on the counter was extremely sharp. The blade was",
     "The reception at the party was extremely cold. The atmosphere was",
     "The criticism in the review was extremely sharp. The tone was"),

    # === 11. FIRST vs THIRD PERSON × content ===
    ("first_vs_third",
     "1st/3rd person", "happy/sad",
     "I woke up this morning feeling incredibly happy. I decided to",
     "I woke up this morning feeling incredibly sad. I decided to",
     "She woke up this morning feeling incredibly happy. She decided to",
     "She woke up this morning feeling incredibly sad. She decided to"),

    # === 12. PAST vs FUTURE × content ===
    ("past_vs_future",
     "past/future", "rain/snow",
     "Yesterday it rained heavily and the streets were",
     "Yesterday it snowed heavily and the streets were",
     "Tomorrow it will rain heavily and the streets will be",
     "Tomorrow it will snow heavily and the streets will be"),

    # === 13. POSITIVE vs NEGATED × content ===
    ("positive_vs_negated",
     "positive/negated", "france/japan",
     "Paris is the capital of France. This statement is",
     "Tokyo is the capital of Japan. This statement is",
     "Paris is not the capital of France. This statement is",
     "Tokyo is not the capital of Japan. This statement is"),

    # === 14. SALIENT ENTITY vs GENERIC × action ===
    ("salient_vs_generic",
     "salient/generic", "wrote/built",
     "Einstein wrote a paper about",
     "Einstein built a device that",
     "A scientist wrote a paper about",
     "A scientist built a device that"),

    # === 15. CAUSE vs EFFECT × content ===
    ("cause_vs_effect",
     "cause/effect", "fire/flood",
     "The fire started because the",
     "The flood started because the",
     "The fire caused the building to",
     "The flood caused the building to"),

    # === 16. ENGLISH vs FRENCH × content ===
    ("english_vs_french",
     "english/french", "dog/cat",
     "The dog is in the garden. The animal is a",
     "The cat is in the garden. The animal is a",
     "Le chien est dans le jardin. L'animal est un",
     "Le chat est dans le jardin. L'animal est un"),
]


def get_top5(logits, tok_obj):
    probs = torch.softmax(logits.float(), -1)
    vals, idxs = torch.topk(probs, 5)
    return [(tok_obj.decode([int(idxs[i])]).strip(), round(float(vals[i]), 4))
            for i in range(5)]


def get_topk_str(logits, tok_obj, k=5, largest=True):
    vals, idxs = torch.topk(logits.float(), k, largest=largest)
    return ", ".join(tok_obj.decode([int(idxs[j])]).strip()[:12] for j in range(k))


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

    # Sample layers for trajectory display
    traj_layers = _sl(0, 8, 12, 16, 20, 24, 28, 32)
    ref_L = _sl(28)[0]  # reference layer for geometry

    cos = torch.nn.functional.cosine_similarity

    # Collect summary table
    summary = []

    for label, ax1_name, ax2_name, text_a, text_b, text_c, text_d in CASES:
        texts = {"A": text_a, "B": text_b, "C": text_c, "D": text_d}
        outs = {}
        preds = {}

        for key, text in texts.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            outs[key] = out
            top5 = get_top5(out.logits[0, -1], tok)
            preds[key] = top5

        print(f"\n{'='*120}")
        print(f"  {label}   [{ax1_name}] × [{ax2_name}]")
        for key in "ABCD":
            p = preds[key][0][0]
            print(f"  {key}: \"{texts[key][-65:]}\"  → {p} "
                  f"({preds[key][0][1]:.2f})")

        # === Axis 1 (rows): A-C and B-D ===
        # === Axis 2 (cols): A-B and C-D ===
        # Get Δh at reference layer
        deltas = {}
        for k1, k2 in [("A","C"), ("B","D"), ("A","B"), ("C","D"),
                        ("A","D"), ("B","C")]:
            h1 = outs[k1].hidden_states[ref_L][0, -1, :].float()
            h2 = outs[k2].hidden_states[ref_L][0, -1, :].float()
            deltas[f"{k1}{k2}"] = (h1 - h2).cpu()

        d_AC = deltas["AC"]
        d_BD = deltas["BD"]
        d_AB = deltas["AB"]
        d_CD = deltas["CD"]
        d_AD = deltas["AD"]
        d_BC = deltas["BC"]

        # Axis consistency
        ax1_cons = float(cos(d_AC.unsqueeze(0), d_BD.unsqueeze(0)))
        ax2_cons = float(cos(d_AB.unsqueeze(0), d_CD.unsqueeze(0)))

        # Orthogonality
        ortho_vals = []
        for n1, d1 in [("AC", d_AC), ("BD", d_BD)]:
            for n2, d2 in [("AB", d_AB), ("CD", d_CD)]:
                ortho_vals.append(
                    float(cos(d1.unsqueeze(0), d2.unsqueeze(0))))
        mean_ortho = sum(abs(v) for v in ortho_vals) / len(ortho_vals)

        # Diagonal compositionality
        ad_comp = float(cos(d_AD.unsqueeze(0),
                            (d_AC + d_AB).unsqueeze(0)))

        # Norms (are axes comparable in magnitude?)
        ax1_norm = (float(d_AC.norm()) + float(d_BD.norm())) / 2
        ax2_norm = (float(d_AB.norm()) + float(d_CD.norm())) / 2

        print(f"\n  --- Geometry at L{ref_L} ---")
        print(f"  {ax1_name} consistency (AC vs BD): cos={ax1_cons:+.3f}")
        print(f"  {ax2_name} consistency (AB vs CD): cos={ax2_cons:+.3f}")
        print(f"  Orthogonality (|cross-cos| mean):  {mean_ortho:.3f}")
        for n1, d1 in [("AC", d_AC), ("BD", d_BD)]:
            for n2, d2 in [("AB", d_AB), ("CD", d_CD)]:
                v = float(cos(d1.unsqueeze(0), d2.unsqueeze(0)))
                print(f"    {n1} ⊥ {n2}: {v:+.3f}")
        print(f"  Diagonal A-D ≈ (A-C)+(A-B): cos={ad_comp:+.3f}")
        print(f"  Axis norms: {ax1_name}={ax1_norm:.1f}  "
              f"{ax2_name}={ax2_norm:.1f}  ratio={ax1_norm/max(ax2_norm,1e-6):.2f}")

        # What does each axis read as in token space?
        mean_ax1 = (d_AC + d_BD) / 2
        mean_ax2 = (d_AB + d_CD) / 2
        ld1 = mean_ax1.to(DEV) @ W_U.float().T
        ld2 = mean_ax2.to(DEV) @ W_U.float().T

        print(f"\n  {ax1_name} axis reads as:")
        print(f"    + pole: [{get_topk_str(ld1, tok, 6, True)}]")
        print(f"    - pole: [{get_topk_str(ld1, tok, 6, False)}]")
        print(f"  {ax2_name} axis reads as:")
        print(f"    + pole: [{get_topk_str(ld2, tok, 6, True)}]")
        print(f"    - pole: [{get_topk_str(ld2, tok, 6, False)}]")

        # Layer-by-layer trajectory for BOTH axes
        for pair_label, k1, k2, ax_name in [
            ("axis1 (row)", "A", "C", ax1_name),
            ("axis2 (col)", "A", "B", ax2_name),
        ]:
            print(f"\n  --- {pair_label}: {k1}-{k2} [{ax_name}] ---")
            print(f"  {'L':>3} | {'norm':>6} | "
                  f"{'+ pole':^40} | {'- pole':^40}")
            print(f"  {'-'*90}")
            for L in traj_layers:
                h1 = outs[k1].hidden_states[L][0, -1, :]
                h2 = outs[k2].hidden_states[L][0, -1, :]
                dh = h1 - h2
                norm = float(dh.norm() / h1.norm())
                ld = dh @ W_U.T
                pos = get_topk_str(ld, tok, 5, True)
                neg = get_topk_str(ld, tok, 5, False)
                print(f"  {L:>3} | {norm:.3f} | {pos:>40} | {neg:>40}")

        summary.append((label, ax1_name, ax2_name,
                         ax1_cons, ax2_cons, mean_ortho, ad_comp,
                         ax1_norm, ax2_norm))

        for key in "ABCD":
            del outs[key]
        torch.cuda.empty_cache()

    # === SUMMARY TABLE ===
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    print(f"{'Case':<25} {'Axis1':<18} {'Axis2':<18} "
          f"{'Ax1 cons':>9} {'Ax2 cons':>9} {'|⊥|':>6} "
          f"{'Diag':>6} {'N1':>6} {'N2':>6} {'N1/N2':>6}")
    print("-" * 120)
    for row in summary:
        label, a1, a2, c1, c2, ort, diag, n1, n2 = row
        print(f"{label:<25} {a1:<18} {a2:<18} "
              f"{c1:>+9.3f} {c2:>+9.3f} {ort:>6.3f} "
              f"{diag:>+6.3f} {n1:>6.0f} {n2:>6.0f} "
              f"{n1/max(n2,1e-6):>6.2f}")

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
