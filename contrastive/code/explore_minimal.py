"""
Minimal-pair contrastive explorer: one word changes.

Simple prompts, one word difference, 2x2 factorial where possible.

Usage: .venv/Scripts/python.exe contrastive/code/explore_minimal.py
"""
import sys
import torch
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

# 2x2 grids: (label, template, axis1_values, axis2_values)
# template has {a1} and {a2} slots
GRIDS = [
    ("food_cat × table_floor",
     "The {a1} was on the {a2}, so I",
     ("food", "cat"),
     ("table", "floor")),

    ("dog_baby × sleeping_crying",
     "The {a1} was {a2} in the room, so I",
     ("dog", "baby"),
     ("sleeping", "crying")),

    ("water_milk × hot_cold",
     "The {a1} was {a2}, so she",
     ("water", "milk"),
     ("hot", "cold")),

    ("door_window × open_locked",
     "The {a1} was {a2}, so he",
     ("door", "window"),
     ("open", "locked")),

    ("car_phone × broken_stolen",
     "His {a1} was {a2}, so he",
     ("car", "phone"),
     ("broken", "stolen")),

    ("man_woman × tall_short",
     "The {a1} was very {a2} and walked into the",
     ("man", "woman"),
     ("tall", "short")),

    ("knife_book × on_under",
     "The {a1} was {a2} the table, so she",
     ("knife", "book"),
     ("on", "under")),

    ("fire_noise × upstairs_outside",
     "There was a {a1} {a2}, so they",
     ("fire", "noise"),
     ("upstairs", "outside")),

    ("kid_dog × in_near",
     "The {a1} was {a2} the pool, so she",
     ("kid", "dog"),
     ("in", "near")),

    ("money_keys × missing_found",
     "The {a1} was {a2}, so he",
     ("money", "keys"),
     ("missing", "found")),

    ("light_alarm × on_off",
     "The {a1} was {a2}, so she",
     ("light", "alarm"),
     ("on", "off")),

    ("patient_prisoner × asleep_awake",
     "The {a1} was {a2}, so the nurse",
     ("patient", "prisoner"),
     ("asleep", "awake")),

    ("rain_snow × heavy_light",
     "The {a1} was {a2}, so they",
     ("rain", "snow"),
     ("heavy", "light")),

    ("soup_bath × too_hot_too_cold",
     "The {a1} was too {a2}, so she",
     ("soup", "bath"),
     ("hot", "cold")),

    ("letter_package × from_mom_from_work",
     "The {a1} was from {a2}, so he",
     ("letter", "package"),
     ("mom", "work")),
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

    def _sl(*layers):
        """Scale layer indices from 32-layer base to current NL."""
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    for label, template, (v1a, v1b), (v2a, v2b) in GRIDS:
        # Generate 4 prompts
        prompts = {
            f"{v1a}+{v2a}": template.format(a1=v1a, a2=v2a),
            f"{v1a}+{v2b}": template.format(a1=v1a, a2=v2b),
            f"{v1b}+{v2a}": template.format(a1=v1b, a2=v2a),
            f"{v1b}+{v2b}": template.format(a1=v1b, a2=v2b),
        }
        keys = list(prompts.keys())

        # Run all 4
        outs = {}
        preds = {}
        for key, text in prompts.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            outs[key] = out
            top5 = get_top5(out.logits[0, -1], tok)
            preds[key] = top5

        print(f"\n{'='*120}")
        print(f"  {label}")
        for key in keys:
            p1 = preds[key][0][0]
            print(f"  {key:>20}: \"{prompts[key]}\" → {p1} "
                  f"({preds[key][0][1]:.3f})  [{', '.join(t[0] for t in preds[key][1:])}]")

        # Check which pairs share predictions
        pred_tokens = {k: preds[k][0][0] for k in keys}

        # Show all 6 pairwise contrasts, but compact
        print(f"\n  Pairwise trajectories ({', '.join(f'L{l}' for l in _sl(16, 24, 32))}):")
        for (k1, k2) in combinations(keys, 2):
            same = "=" if pred_tokens[k1] == pred_tokens[k2] else "≠"
            # What differs between k1 and k2?
            parts1 = k1.split("+")
            parts2 = k2.split("+")
            if parts1[0] == parts2[0]:
                diff = f"Δ{parts1[1]}→{parts2[1]}"
            elif parts1[1] == parts2[1]:
                diff = f"Δ{parts1[0]}→{parts2[0]}"
            else:
                diff = "Δboth"

            print(f"\n  {k1} vs {k2} (pred {same}, {diff})")
            print(f"    \"{prompts[k1]}\"")
            print(f"    \"{prompts[k2]}\"")

            for L in _sl(16, 24, 32):
                h_x = outs[k1].hidden_states[L][0, -1, :]
                h_y = outs[k2].hidden_states[L][0, -1, :]
                dh = h_x - h_y
                norm = float(dh.norm() / h_x.norm())
                logits_d = dh @ W_U.T
                pos = get_topk_str(logits_d, tok, 5, largest=True)
                neg = get_topk_str(logits_d, tok, 5, largest=False)
                print(f"    L{L}: ({norm:.3f}) [{pos}] vs [{neg}]")

        for key in keys:
            del outs[key]
        torch.cuda.empty_cache()

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
