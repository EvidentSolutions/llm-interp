"""
2x2 factorial v2: all six pairwise contrasts, no validity gate.

The 2x2 design crosses a content axis (e.g. chase vs pet) with a
prediction axis (e.g. cat vs dog).  Even when the model's top-1
predictions don't form a clean factorial, the *contrastive trajectories*
reveal how the model internally separates content from prediction.

Reports for each case:
- All four predictions (top-5)
- The 2x2 prediction structure (which predictions match/differ)
- All six pairwise contrastive trajectories
- Cosine between content-axis and prediction-axis Δh (are they orthogonal?)

Usage: MODEL=microsoft/phi-4 python3 contrastive/code/explore_2x2_v2.py
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

CASES = [
    # === Animal: what animal is it? ===
    # Content axis: chase scene vs pet description
    # Prediction axis: cat vs dog
    ("animal_identity",
     # A: chase + cat outcome
     "The dog chased the cat through the garden and cornered it near the "
     "fence. The animal that was trapped was the",
     # B: pet description + cat outcome
     "The small furry animal purred loudly and rubbed against her leg, "
     "then curled up on the warm blanket. The pet was a",
     # C: chase + dog outcome (swap agents)
     "The cat chased the dog through the garden and cornered it near the "
     "fence. The animal that was trapped was the",
     # D: pet description + dog outcome
     "The large animal wagged its tail excitedly and fetched the ball "
     "from across the yard. The pet was a"),

    # === Temperature: what does she feel? ===
    # Content axis: water vs air
    # Prediction axis: hot vs cold
    ("hot_cold",
     "The water had been boiling for ten minutes. She touched the surface "
     "and it was",
     "The desert sun had been beating down all day. She stepped outside "
     "and it was",
     "The water had been in the freezer for an hour. She touched the "
     "surface and it was",
     "The blizzard had been raging all night. She stepped outside "
     "and it was"),

    # === What did they eat? ===
    ("food",
     "For breakfast he fried some bacon and eggs in a pan and ate the",
     "For dinner she roasted a whole chicken with herbs and ate the",
     "For breakfast he poured some cereal into a bowl with milk and ate the",
     "For dinner she tossed some lettuce and tomatoes with dressing and ate the"),

    # === What happened to the team? ===
    ("sports_outcome",
     "The basketball team scored a three-pointer in the final second to "
     "take the lead. They",
     "The soccer team scored a goal in stoppage time to take the lead. "
     "They",
     "The basketball team missed a free throw in the final second and "
     "fell behind. They",
     "The soccer team missed a penalty kick in stoppage time and fell "
     "behind. They"),

    # === What does the doctor say? ===
    ("diagnosis",
     "The scan showed a large tumor that had spread to nearby organs. The "
     "doctor told the patient the diagnosis was",
     "The x-ray showed a compound fracture with bone fragments near the "
     "artery. The doctor told the patient the diagnosis was",
     "The scan showed a small cyst that had not changed in five years. The "
     "doctor told the patient the diagnosis was",
     "The x-ray showed a hairline crack that was already healing on its "
     "own. The doctor told the patient the diagnosis was"),

    # === Where did they go? ===
    ("destination",
     "The man collapsed on the sidewalk clutching his chest and could "
     "barely breathe. The ambulance rushed him to the",
     "The woman fell off her bicycle and broke her arm in two places. "
     "The ambulance rushed her to the",
     "The couple had just gotten engaged and wanted to celebrate with "
     "champagne. They drove straight to the",
     "The family had been driving for hours and the children were "
     "starving. They drove straight to the"),

    # === Guilty vs innocent ===
    ("verdict",
     "The defendant's DNA was found on the murder weapon and three "
     "witnesses saw him at the scene. The jury found him",
     "The defendant was caught on camera stealing the jewelry and the "
     "items were found in his car. The jury found him",
     "The defendant had a confirmed alibi with hotel records placing "
     "him in another city on the night of the murder. The jury found him",
     "The defendant had a receipt proving he purchased the jewelry "
     "and the camera footage was proven to be doctored. The jury found him"),

    # === What did they build? ===
    ("construction",
     "The carpenter spent months cutting timber and fitting wooden beams "
     "together. He built a",
     "The mason spent months cutting limestone blocks and laying them "
     "with mortar. He built a",
     "The carpenter spent an afternoon nailing wooden planks between "
     "the posts along the property line. He built a",
     "The mason spent an afternoon stacking flat stones along the edge "
     "of the garden. He built a"),

    # === What color? ===
    ("color",
     "She looked up at the clear sky on a perfect summer day. The sky "
     "above her was",
     "She looked down at the lawn that had been watered every day all "
     "summer. The grass beneath her feet was",
     "She looked up at the sky just as the sun was setting behind the "
     "mountains. The sky above her was",
     "She looked down at the lawn after three months without rain in "
     "the heat of late summer. The grass beneath her feet was"),
]


def get_top5(logits, tok_obj):
    probs = torch.softmax(logits, -1)
    vals, idxs = torch.topk(probs, 5)
    return [(tok_obj.decode([int(idxs[i])]).strip(), round(float(vals[i]), 4))
            for i in range(5)]


def get_topk_str(logits, tok_obj, k=5, largest=True):
    vals, idxs = torch.topk(logits, k, largest=largest)
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
    W_U = model.lm_head.weight.detach()

    def _sl(*layers):
        """Scale layer indices from 32-layer base to current NL."""
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    # Layers for trajectory display — every 4th, scaled
    traj_layers = list(range(0, NL + 1, max(1, NL // 8)))
    if NL not in traj_layers:
        traj_layers.append(NL)

    for label, text_a, text_b, text_c, text_d in CASES:
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

        a1 = preds["A"][0][0]
        b1 = preds["B"][0][0]
        c1 = preds["C"][0][0]
        d1 = preds["D"][0][0]

        ab = "=" if a1 == b1 else "≠"
        cd = "=" if c1 == d1 else "≠"
        ac = "=" if a1 == c1 else "≠"
        bd = "=" if b1 == d1 else "≠"
        valid = (a1 == b1) and (c1 == d1) and (a1 != c1)

        print(f"\n{'='*120}")
        status = "✓ clean 2x2" if valid else "≈ partial"
        print(f"  {label}  {status}")
        print(f"  A{ab}B: {a1},{b1}   C{cd}D: {c1},{d1}   "
              f"A{ac}C   B{bd}D")
        for key in "ABCD":
            print(f"  {key}: \"{texts[key][-70:]}\"")
            print(f"     → {preds[key]}")

        # === All six pairwise contrasts ===
        pair_labels = [
            ("A", "B", "same content-row, diff context"),
            ("C", "D", "same content-row, diff context"),
            ("A", "C", "same context-col, diff content"),
            ("B", "D", "same context-col, diff content"),
            ("A", "D", "diagonal"),
            ("B", "C", "diagonal"),
        ]

        # Collect Δh at a reference layer for axis-independence analysis
        ref_L = _sl(28)[0]
        delta_vecs = {}

        for k1, k2, desc in pair_labels:
            p1, p2 = preds[k1][0][0], preds[k2][0][0]
            pred_match = "same" if p1 == p2 else "diff"
            print(f"\n  --- {k1}-{k2} ({desc}, pred={pred_match}: "
                  f"{p1}/{p2}) ---")

            # Show truncated prompts
            t1 = texts[k1].split(". ")[-1] if ". " in texts[k1] else texts[k1][-50:]
            t2 = texts[k2].split(". ")[-1] if ". " in texts[k2] else texts[k2][-50:]
            print(f"  {k1}: \"{t1[-60:]}\"")
            print(f"  {k2}: \"{t2[-60:]}\"")

            print(f"  {'L':>3} | {'norm':>6} | "
                  f"{'→ ' + k1 + ' pole':^42} | {'→ ' + k2 + ' pole':^42}")
            print(f"  {'-'*100}")

            for L in traj_layers:
                h_x = outs[k1].hidden_states[L][0, -1, :]
                h_y = outs[k2].hidden_states[L][0, -1, :]
                dh = h_x - h_y
                norm = float(dh.norm() / h_x.norm())
                logits_d = dh @ W_U.T
                pos = get_topk_str(logits_d, tok, 5, largest=True)
                neg = get_topk_str(logits_d, tok, 5, largest=False)
                print(f"  {L:>3} | {norm:.3f} | {pos:>42} | {neg:>42}")

            # Save Δh at reference layer for axis analysis
            h_x = outs[k1].hidden_states[ref_L][0, -1, :].float()
            h_y = outs[k2].hidden_states[ref_L][0, -1, :].float()
            delta_vecs[f"{k1}{k2}"] = (h_x - h_y).cpu()

        # === Axis independence analysis at reference layer ===
        print(f"\n  --- Axis geometry at L{ref_L} ---")
        # Content axis: mean of A-C and B-D
        # Context axis: mean of A-B and C-D
        # Prediction axis: mean of A-C and B-D (same as content if design works)
        d_AC = delta_vecs.get("AC", torch.zeros(1))
        d_BD = delta_vecs.get("BD", torch.zeros(1))
        d_AB = delta_vecs.get("AB", torch.zeros(1))
        d_CD = delta_vecs.get("CD", torch.zeros(1))

        cos = torch.nn.functional.cosine_similarity

        print(f"  Content consistency (AC vs BD):  "
              f"cos={float(cos(d_AC.unsqueeze(0), d_BD.unsqueeze(0))):.3f}")
        print(f"  Context consistency (AB vs CD):  "
              f"cos={float(cos(d_AB.unsqueeze(0), d_CD.unsqueeze(0))):.3f}")
        print(f"  Content ⊥ Context   (AC vs AB):  "
              f"cos={float(cos(d_AC.unsqueeze(0), d_AB.unsqueeze(0))):.3f}")
        print(f"  Content ⊥ Context   (AC vs CD):  "
              f"cos={float(cos(d_AC.unsqueeze(0), d_CD.unsqueeze(0))):.3f}")
        print(f"  Content ⊥ Context   (BD vs AB):  "
              f"cos={float(cos(d_BD.unsqueeze(0), d_AB.unsqueeze(0))):.3f}")
        print(f"  Content ⊥ Context   (BD vs CD):  "
              f"cos={float(cos(d_BD.unsqueeze(0), d_CD.unsqueeze(0))):.3f}")

        # Diagonal check: A-D ≈ (A-C) + (A-B)?  B-C ≈ (B-D) + (C-D)?
        d_AD = delta_vecs.get("AD", torch.zeros(1))
        d_BC = delta_vecs.get("BC", torch.zeros(1))
        ad_composed = d_AC + d_AB
        bc_composed = d_BD + d_CD  # B-D + C-D ≈ B-C (if B-C = B-D - (C-D))
        # Actually B-C = (B-D) + (D-C) = (B-D) - (C-D)
        bc_composed2 = d_BD - d_CD

        print(f"  Diagonal compositionality:")
        print(f"    A-D vs (A-C)+(A-B): "
              f"cos={float(cos(d_AD.unsqueeze(0), ad_composed.unsqueeze(0))):.3f}  "
              f"ratio={float(d_AD.norm()/ad_composed.norm()):.2f}")
        print(f"    B-C vs (B-D)-(C-D): "
              f"cos={float(cos(d_BC.unsqueeze(0), bc_composed2.unsqueeze(0))):.3f}  "
              f"ratio={float(d_BC.norm()/bc_composed2.norm()):.2f}")

        for key in "ABCD":
            del outs[key]
        torch.cuda.empty_cache()

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
