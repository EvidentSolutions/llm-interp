"""
Contrastive trajectory explorer for Phi-2.

Explores a wide variety of contrastive pairs, reports:
- Both predictions (top-5)
- Whether top-1 matches
- Contrastive trajectory at every 3rd layer (both poles)
- Shared suffix length
- Signal norm ||Δh||/||h_c|| at key layers

Usage: .venv/Scripts/python.exe contrastive/code/explore_cases.py
"""
import sys
import json
import torch
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

# Wide variety of contrastive pairs — different types of contrast
CASES = [
    # === Moral/ethical contrasts ===
    ("theft_moral",
     "The man walked into the store and slipped a bottle of wine under his coat. He walked out without paying. When the security guard stopped him, he",
     "The man walked into the store and picked up a bottle of wine from the shelf. He walked to the register and paid. When the security guard stopped him, he"),

    ("lie_truth",
     "She looked him in the eye and told him she had been at work all day, even though she had spent the afternoon with someone else. When he asked again, she",
     "She looked him in the eye and told him she had been at the park all day, reading a book on the bench. When he asked again, she"),

    # === Physical/sensory contrasts ===
    ("temperature",
     "The water in the pot had been boiling for ten minutes and steam was rising from the surface. She dipped her hand in and the water felt",
     "The water in the pot had been sitting in the fridge overnight and ice was forming on the surface. She dipped her hand in and the water felt"),

    ("sound",
     "The concert hall was packed and the orchestra was playing fortissimo with all brass blaring. The room was",
     "The concert hall was empty and a single violin played a slow quiet melody in the dark. The room was"),

    # === Emotional contrasts ===
    ("grief_joy",
     "She received the phone call at midnight and learned that her mother had passed away. She sat down on the bed and felt",
     "She received the phone call at midnight and learned that her mother had won the lottery. She sat down on the bed and felt"),

    ("trust_betrayal",
     "He had shared his passwords and bank details with his business partner of twenty years. The partner transferred all the money to an offshore account. He felt",
     "He had shared his passwords and bank details with his business partner of twenty years. The partner used them to set up automatic bill payments. He felt"),

    # === Factual/knowledge contrasts ===
    ("animal_size",
     "The elephant walked slowly across the savanna, its massive body casting a long shadow on the ground. The animal weighed approximately",
     "The mouse scurried quickly across the kitchen floor, its tiny body barely visible in the dim light. The animal weighed approximately"),

    ("history",
     "The year was 1969 and the Apollo spacecraft had just touched down on the lunar surface. The astronaut stepped out and saw",
     "The year was 1969 and the Woodstock festival had just begun in a muddy field in New York. The musician stepped out and saw"),

    # === Causal/consequential contrasts ===
    ("fire_flood",
     "The building had been on fire for hours and the roof had collapsed. The firefighters arrived and found the interior was",
     "The river had been flooding for hours and the water had risen to the second floor. The rescue team arrived and found the interior was"),

    ("medicine",
     "The patient took three times the recommended dose of the medication and collapsed on the kitchen floor. The paramedic said the patient was",
     "The patient took the recommended dose of the medication and sat down to read a book. The paramedic said the patient was"),

    # === Social/status contrasts ===
    ("wealth",
     "The family lived in a mansion with twelve bedrooms, a swimming pool, and a fleet of luxury cars in the driveway. The family was",
     "The family lived in a one-room apartment with no furniture, sharing a single mattress on the floor. The family was"),

    ("power",
     "The general commanded an army of fifty thousand soldiers and controlled the entire northern border. The general was",
     "The private had just completed basic training last week and had never held a real weapon before. The private was"),

    # === Temporal contrasts ===
    ("age_building",
     "The cathedral had been built in 1142 and its stone walls had weathered eight centuries of storms. The building was",
     "The office tower had been completed last month and its glass walls still had protective film on them. The building was"),

    # === Syntactic/structural (same tokens, different meaning) ===
    ("agent_swap",
     "The dog chased the cat through the garden and cornered it near the fence. The animal that was trapped was the",
     "The cat chased the dog through the garden and cornered it near the fence. The animal that was trapped was the"),

    # === Ambiguity/resolution ===
    ("pronoun_resolution",
     "John told Mary that he had been promoted at work and she should be proud of him. Mary said she was happy for",
     "Mary told John that she had been promoted at work and he should be proud of her. Mary said she was happy for"),

    # === Register/formality (same content) ===
    ("register",
     "The patient presented with acute myocardial infarction and was administered intravenous thrombolytics. The prognosis was",
     "The guy came in having a heart attack and we gave him clot-busting drugs through an IV. The prognosis was"),

    # === Quantity/degree ===
    ("severity",
     "The earthquake measured 9.1 on the Richter scale and the tsunami that followed destroyed entire cities along the coast. The damage was",
     "The earthquake measured 2.1 on the Richter scale and a few cups rattled on kitchen shelves around the town. The damage was"),

    # === Counterfactual ===
    ("negation",
     "The bridge had been inspected last month and engineers found severe cracks in three of the main support columns. The bridge was",
     "The bridge had been inspected last month and engineers found no defects in any of the structural components. The bridge was"),

    # === Profession/identity ===
    ("profession",
     "She spent her days performing open heart surgery and transplanting organs in the operating theater. She was a",
     "She spent her days arranging flowers and creating bouquets for weddings and funerals. She was a"),

    # === Abstract/philosophical ===
    ("existence",
     "The old man sat alone in the empty house where his wife had lived for forty years. She had died last winter. The house felt",
     "The old man sat alone in the empty house where his wife had lived for forty years. She had gone to visit their daughter. The house felt"),
]


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

    results = []

    for label, ctx_text, ctrl_text in CASES:
        ctx_ids = tok(ctx_text, add_special_tokens=False)["input_ids"]
        ctrl_ids = tok(ctrl_text, add_special_tokens=False)["input_ids"]

        # Shared suffix
        min_len = min(len(ctx_ids), len(ctrl_ids))
        shared = 0
        for i in range(1, min_len + 1):
            if ctx_ids[-i] == ctrl_ids[-i]:
                shared = i
            else:
                break

        with torch.no_grad():
            ctx_out = model(
                torch.tensor([ctx_ids], device=DEV), output_hidden_states=True
            )
            ctrl_out = model(
                torch.tensor([ctrl_ids], device=DEV), output_hidden_states=True
            )

        # Predictions
        probs_c = torch.softmax(ctx_out.logits[0, -1], -1)
        probs_k = torch.softmax(ctrl_out.logits[0, -1], -1)
        vc, ic = torch.topk(probs_c, 5)
        vk, ik = torch.topk(probs_k, 5)
        c_top = [(tok.decode([int(ic[i])]).strip(), round(float(vc[i]), 4))
                 for i in range(5)]
        k_top = [(tok.decode([int(ik[i])]).strip(), round(float(vk[i]), 4))
                 for i in range(5)]
        same_top1 = int(ic[0]) == int(ik[0])

        print(f"\n{'='*100}")
        print(f"  {label} | shared={shared} | same_top1={same_top1}")
        print(f"  c: \"{ctx_text[-80:]}\"")
        print(f"  k: \"{ctrl_text[-80:]}\"")
        print(f"  c predicts: {c_top}")
        print(f"  k predicts: {k_top}")

        # Trajectory at every 2nd layer from 8 onward
        print(f"\n  {'L':>3} | {'||Δh||/||hc||':>12} | {'c-k pole':^45} | {'k-c pole':^45}")
        print(f"  {'-'*110}")

        for L in range(8, NL + 1, 2):
            h_c = ctx_out.hidden_states[L][0, -1, :]
            h_k = ctrl_out.hidden_states[L][0, -1, :]
            dh = h_c - h_k
            norm_ratio = float(dh.norm() / h_c.norm())
            logits_d = dh @ W_U.T
            tv, ti = torch.topk(logits_d, 5)
            bv, bi = torch.topk(logits_d, 5, largest=False)
            ck = ", ".join(tok.decode([int(ti[j])]).strip()[:10] for j in range(5))
            kc = ", ".join(tok.decode([int(bi[j])]).strip()[:10] for j in range(5))
            print(f"  {L:>3} |       {norm_ratio:.3f} | {ck:>45} | {kc:>45}")

        del ctx_out, ctrl_out
        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
