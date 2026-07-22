"""
Contrastive triplet explorer for Phi-2.

Three contexts per case. Reports all three pairwise contrasts.
Design: two contexts predict the same token (A, B), one predicts
differently (C). This gives:
  A-B: same prediction, different content → pure representation
  A-C: different prediction, different content → representation + prediction
  B-C: different prediction, different content → representation + prediction

Usage: .venv/Scripts/python.exe contrastive/code/explore_triplets.py
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

# Each case: (label, context_A, context_B, context_C)
# A and B should ideally predict the same token; C should predict differently.
# But we don't know in advance — we'll report what we find.
CASES = [
    # === Temperature: hot vs cold vs warm ===
    ("temperature",
     # A: boiling
     "The water in the pot had been boiling for ten minutes and steam was rising "
     "from the surface. She dipped her hand in and the water felt",
     # B: fire/hot but different source
     "The kettle had been left on the stove for hours and the metal handle was "
     "glowing red. She dipped her hand in and the water felt",
     # C: cold
     "The water in the pot had been sitting in the fridge overnight and ice was "
     "forming on the surface. She dipped her hand in and the water felt"),

    # === Grief vs joy vs calm ===
    ("emotion",
     # A: grief
     "She received the phone call at midnight and learned that her mother had "
     "passed away. She sat down on the bed and felt",
     # B: shock/grief different cause
     "She received the phone call at midnight and learned that her house had "
     "burned down with everything inside. She sat down on the bed and felt",
     # C: joy
     "She received the phone call at midnight and learned that her mother had "
     "won the lottery. She sat down on the bed and felt"),

    # === Sound: loud vs quiet vs moderate ===
    ("sound",
     # A: loud
     "The concert hall was packed and the orchestra was playing fortissimo with "
     "all brass blaring. The room was",
     # B: loud, different source
     "The construction site was at full operation with jackhammers drilling and "
     "dump trucks reversing with their alarms beeping. The room was",
     # C: quiet
     "The concert hall was empty and a single violin played a slow quiet melody "
     "in the dark. The room was"),

    # === Severity: catastrophic vs minor vs moderate ===
    ("severity",
     # A: catastrophic earthquake
     "The earthquake measured 9.1 on the Richter scale and the tsunami that "
     "followed destroyed entire cities along the coast. The damage was",
     # B: catastrophic hurricane
     "The hurricane made landfall as a category five storm and the storm surge "
     "swept away entire neighborhoods along the coast. The damage was",
     # C: minor
     "The earthquake measured 2.1 on the Richter scale and a few cups rattled "
     "on kitchen shelves around the town. The damage was"),

    # === Wealth: rich vs poor vs middle ===
    ("wealth",
     # A: rich
     "The family lived in a mansion with twelve bedrooms, a swimming pool, and "
     "a fleet of luxury cars in the driveway. The family was",
     # B: rich, different signals
     "The family owned three vacation homes, a private jet, and had a full-time "
     "staff of twelve at their estate. The family was",
     # C: poor
     "The family lived in a one-room apartment with no furniture, sharing a "
     "single mattress on the floor. The family was"),

    # === Medical: overdose vs healthy vs moderate illness ===
    ("medical",
     # A: overdose
     "The patient took three times the recommended dose of the medication and "
     "collapsed on the kitchen floor. The paramedic said the patient was",
     # B: injury
     "The patient had fallen down two flights of stairs and was lying at the "
     "bottom with a visible head wound. The paramedic said the patient was",
     # C: healthy
     "The patient took the recommended dose of the medication and sat down to "
     "read a book. The paramedic said the patient was"),

    # === Moral: theft vs honest vs ambiguous ===
    ("moral",
     # A: theft
     "The man walked into the store and slipped a bottle of wine under his "
     "coat. He walked out without paying. When the security guard stopped him, he",
     # B: robbery (different crime)
     "The man walked into the store and pulled out a knife, demanding all the "
     "cash from the register. When the security guard stopped him, he",
     # C: honest
     "The man walked into the store and picked up a bottle of wine from the "
     "shelf. He walked to the register and paid. When the security guard "
     "stopped him, he"),

    # === Building: ancient vs new vs damaged ===
    ("building",
     # A: ancient
     "The cathedral had been built in 1142 and its stone walls had weathered "
     "eight centuries of storms. The building was",
     # B: ancient, different building
     "The fortress had stood on the hilltop since the Roman era and its walls "
     "bore the scars of a dozen sieges. The building was",
     # C: new
     "The office tower had been completed last month and its glass walls still "
     "had protective film on them. The building was"),

    # === Register: formal vs informal vs neutral ===
    ("register",
     # A: formal medical
     "The patient presented with acute myocardial infarction and was "
     "administered intravenous thrombolytics. The prognosis was",
     # B: formal legal
     "The defendant was found liable for negligence pursuant to the applicable "
     "statute of limitations. The prognosis was",
     # C: informal
     "The guy came in having a heart attack and we gave him clot-busting drugs "
     "through an IV. The prognosis was"),

    # === Fire vs flood vs earthquake ===
    ("disaster",
     # A: fire
     "The building had been on fire for hours and the roof had collapsed. The "
     "firefighters arrived and found the interior was",
     # B: earthquake
     "The earthquake had shaken the building for thirty seconds and the walls "
     "had cracked from foundation to roof. The firefighters arrived and found "
     "the interior was",
     # C: flood
     "The river had been flooding for hours and the water had risen to the "
     "second floor. The rescue team arrived and found the interior was"),

    # === Animal identity ===
    ("animal",
     # A: dog chases cat
     "The dog chased the cat through the garden and cornered it near the fence. "
     "The animal that was trapped was the",
     # B: fox chases rabbit
     "The fox chased the rabbit through the garden and cornered it near the "
     "fence. The animal that was trapped was the",
     # C: cat chases dog
     "The cat chased the dog through the garden and cornered it near the fence. "
     "The animal that was trapped was the"),

    # === Student: cheater vs achiever vs average ===
    ("student",
     # A: cheater
     "The student who had cheated on every exam and plagiarized every paper "
     "somehow managed to graduate. The diploma was",
     # B: lazy but not dishonest
     "The student who had skipped most classes and barely passed every exam "
     "somehow managed to graduate. The diploma was",
     # C: achiever
     "The student who had earned perfect scores on every exam and published "
     "original research managed to graduate. The diploma was"),
]


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

    for label, text_a, text_b, text_c in CASES:
        ids_a = tok(text_a, add_special_tokens=False)["input_ids"]
        ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
        ids_c = tok(text_c, add_special_tokens=False)["input_ids"]

        with torch.no_grad():
            out_a = model(torch.tensor([ids_a], device=DEV),
                          output_hidden_states=True)
            out_b = model(torch.tensor([ids_b], device=DEV),
                          output_hidden_states=True)
            out_c = model(torch.tensor([ids_c], device=DEV),
                          output_hidden_states=True)

        # Predictions
        probs_a = torch.softmax(out_a.logits[0, -1], -1)
        probs_b = torch.softmax(out_b.logits[0, -1], -1)
        probs_c = torch.softmax(out_c.logits[0, -1], -1)
        top_a = [(tok.decode([int(i)]).strip(), round(float(v), 4))
                 for i, v in zip(*torch.topk(probs_a, 5))]
        top_b = [(tok.decode([int(i)]).strip(), round(float(v), 4))
                 for i, v in zip(*torch.topk(probs_b, 5))]
        top_c = [(tok.decode([int(i)]).strip(), round(float(v), 4))
                 for i, v in zip(*torch.topk(probs_c, 5))]

        a1 = tok.decode([int(probs_a.argmax())]).strip()
        b1 = tok.decode([int(probs_b.argmax())]).strip()
        c1 = tok.decode([int(probs_c.argmax())]).strip()

        print(f"\n{'='*120}")
        print(f"  {label}")
        print(f"  A: \"{text_a[-70:]}\"  → {a1} ({top_a[0][1]:.3f})")
        print(f"  B: \"{text_b[-70:]}\"  → {b1} ({top_b[0][1]:.3f})")
        print(f"  C: \"{text_c[-70:]}\"  → {c1} ({top_c[0][1]:.3f})")
        print(f"  A top-5: {top_a}")
        print(f"  B top-5: {top_b}")
        print(f"  C top-5: {top_c}")
        ab = "SAME" if a1 == b1 else "DIFF"
        ac = "SAME" if a1 == c1 else "DIFF"
        bc = "SAME" if b1 == c1 else "DIFF"
        print(f"  Predictions: A-B={ab}, A-C={ac}, B-C={bc}")

        # All three pairwise contrasts
        pairs = [
            ("A-B", out_a, out_b),
            ("A-C", out_a, out_c),
            ("B-C", out_b, out_c),
        ]

        for pair_label, out_x, out_y in pairs:
            print(f"\n  --- {pair_label} ---")
            print(f"  {'L':>3} | {'norm':>6} | "
                  f"{'first pole':^42} | {'second pole':^42}")
            print(f"  {'-'*100}")

            for L in range(12, NL + 1, 4):
                h_x = out_x.hidden_states[L][0, -1, :]
                h_y = out_y.hidden_states[L][0, -1, :]
                dh = h_x - h_y
                norm = float(dh.norm() / h_x.norm())
                logits_d = dh @ W_U.T
                pos = get_topk_str(logits_d, tok, 5, largest=True)
                neg = get_topk_str(logits_d, tok, 5, largest=False)
                print(f"  {L:>3} | {norm:.3f} | {pos:>42} | {neg:>42}")

        del out_a, out_b, out_c
        torch.cuda.empty_cache()

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
