"""
2x2 factorial: content axis × prediction axis.

Take pairs that predict the same token (A, B), add negation to get (C, D)
that predict differently. Six pairwise contrasts.

Usage: .venv/Scripts/python.exe contrastive/code/explore_2x2.py
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

# Each case: (label, A, B, C, D)
# A,B predict same token (content differs)
# C,D = negated versions of A,B (prediction flips)
CASES = [
    ("emotion",
     # A: mother died
     "She received the phone call at midnight and learned that her mother had "
     "passed away. She sat down on the bed and felt",
     # B: house burned
     "She received the phone call at midnight and learned that her house had "
     "burned down with everything inside. She sat down on the bed and felt",
     # C: mother did NOT die
     "She received the phone call at midnight and learned that her mother had "
     "not passed away after all. She sat down on the bed and felt",
     # D: house did NOT burn
     "She received the phone call at midnight and learned that her house had "
     "not burned down after all. She sat down on the bed and felt"),

    ("temperature",
     # A: boiling water
     "The water in the pot had been boiling for ten minutes and steam was "
     "rising from the surface. She dipped her hand in and the water felt",
     # B: hot metal
     "The kettle had been left on the stove for hours and the metal handle "
     "was glowing red. She dipped her hand in and the water felt",
     # C: water NOT boiling
     "The water in the pot had not been heated at all and was still at room "
     "temperature. She dipped her hand in and the water felt",
     # D: kettle NOT hot
     "The kettle had not been turned on and the metal handle was cool to the "
     "touch. She dipped her hand in and the water felt"),

    ("severity",
     # A: catastrophic earthquake
     "The earthquake measured 9.1 on the Richter scale and the tsunami that "
     "followed destroyed entire cities along the coast. The damage was",
     # B: catastrophic hurricane
     "The hurricane made landfall as a category five storm and the storm "
     "surge swept away entire neighborhoods along the coast. The damage was",
     # C: earthquake NOT strong
     "The earthquake measured only 2.1 on the Richter scale and was not "
     "felt by most people in the area. The damage was",
     # D: hurricane NOT strong
     "The hurricane was downgraded to a tropical storm and did not make "
     "landfall anywhere near the coast. The damage was"),

    ("bridge",
     # A: bridge damaged (cracks)
     "Engineers found severe cracks in three of the main support columns "
     "during the annual inspection. The bridge was",
     # B: bridge damaged (erosion)
     "Divers found that the underwater foundations had eroded significantly "
     "and several bolts were missing. The bridge was",
     # C: bridge NOT damaged (cracks)
     "Engineers found no cracks in any of the support columns during the "
     "annual inspection. The bridge was",
     # D: bridge NOT damaged (erosion)
     "Divers found that the underwater foundations were in perfect condition "
     "with no signs of erosion. The bridge was"),

    ("patient",
     # A: overdose
     "The patient took three times the recommended dose of the medication "
     "and collapsed on the kitchen floor. The paramedic said the patient was",
     # B: fell down stairs
     "The patient had fallen down two flights of stairs and was lying at "
     "the bottom with a visible head wound. The paramedic said the patient was",
     # C: patient NOT overdosed
     "The patient had not taken any medication at all and was sitting "
     "comfortably in the living room. The paramedic said the patient was",
     # D: patient NOT injured
     "The patient had not fallen and was standing at the top of the stairs "
     "feeling perfectly fine. The paramedic said the patient was"),

    ("crime",
     # A: shoplifting
     "The man walked into the store and slipped a bottle of wine under his "
     "coat without paying. When the security guard stopped him, he",
     # B: armed robbery
     "The man walked into the store and pulled out a knife, demanding the "
     "cash from the register. When the security guard stopped him, he",
     # C: NOT shoplifting (just browsing)
     "The man walked into the store and did not take anything, just browsed "
     "the shelves for a while. When the security guard stopped him, he",
     # D: NOT armed (just asking directions)
     "The man walked into the store and did not threaten anyone, just asked "
     "for directions politely. When the security guard stopped him, he"),

    ("wealth",
     # A: rich (mansion)
     "The family lived in a mansion with twelve bedrooms and a fleet of "
     "luxury cars in the driveway. The family was",
     # B: rich (jet)
     "The family owned a private jet and three vacation homes across "
     "Europe. The family was",
     # C: NOT rich (no house)
     "The family did not own a house and had no savings in the bank at "
     "all. The family was",
     # D: NOT rich (no car)
     "The family could not afford a car and relied on public transport "
     "to get around. The family was"),
]


def get_top5(logits, tok):
    probs = torch.softmax(logits, -1)
    vals, idxs = torch.topk(probs, 5)
    return [(tok.decode([int(idxs[i])]).strip(), round(float(vals[i]), 4))
            for i in range(5)]


def get_topk_str(logits, tok, k=5, largest=True):
    vals, idxs = torch.topk(logits, k, largest=largest)
    return ", ".join(tok.decode([int(idxs[j])]).strip()[:12] for j in range(k))


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

        print(f"\n{'='*120}")
        print(f"  {label}")
        for key in "ABCD":
            print(f"  {key}: \"{texts[key]}\"")
            print(f"     → {preds[key]}")

        # Check 2x2 structure
        a1 = preds["A"][0][0]
        b1 = preds["B"][0][0]
        c1 = preds["C"][0][0]
        d1 = preds["D"][0][0]
        print(f"\n  Predictions: A={a1}, B={b1}, C={c1}, D={d1}")
        print(f"  A=B: {a1==b1}, C=D: {c1==d1}, A≠C: {a1!=c1}, B≠D: {b1!=d1}")

        # Six pairwise contrasts
        pairs = [
            ("AB", "same pred, diff content"),
            ("CD", "same pred, diff content (negated)"),
            ("AC", "same content, diff pred"),
            ("BD", "same content, diff pred"),
            ("AD", "diff content, diff pred"),
            ("BC", "diff content, diff pred"),
        ]

        for pair, desc in pairs:
            k1, k2 = pair[0], pair[1]
            print(f"\n  --- {k1}-{k2} ({desc}) ---")
            print(f"  {k1}: \"{texts[k1][-70:]}\"")
            print(f"  {k2}: \"{texts[k2][-70:]}\"")
            print(f"  {'L':>3} | {'norm':>6} | "
                  f"{'→ ' + k1 + ' pole':^42} | {'→ ' + k2 + ' pole':^42}")
            print(f"  {'-'*100}")

            for L in range(12, NL + 1, 4):
                h_x = outs[k1].hidden_states[L][0, -1, :]
                h_y = outs[k2].hidden_states[L][0, -1, :]
                dh = h_x - h_y
                norm = float(dh.norm() / h_x.norm())
                logits_d = dh @ W_U.T
                pos = get_topk_str(logits_d, tok, 5, largest=True)
                neg = get_topk_str(logits_d, tok, 5, largest=False)
                print(f"  {L:>3} | {norm:.3f} | {pos:>42} | {neg:>42}")

        for key in "ABCD":
            del outs[key]
        torch.cuda.empty_cache()

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
