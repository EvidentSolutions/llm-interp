"""
Direct per-token logit readout: does the individual state encode content
independently of the contrastive subtraction?

For the car/phone × broken/stolen 2x2, read specific token logits
from each individual state at each layer. Check:
1. Does h("car broken") score "car" higher than h("phone broken")?
2. Does h("car broken") score "car" the same as h("car stolen")?
   (compositionality: object identity independent of event)
3. Does the contrastive Δh just amplify what's already in each state?

Also: show the FULL logit-lens top-10 for each state at key layers,
so we can see where the content tokens sit relative to everything else.

Usage: .venv/Scripts/python.exe contrastive/code/direct_readout.py
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

CASES = {
    "car+broken":  "His car was broken, so he",
    "car+stolen":  "His car was stolen, so he",
    "phone+broken": "His phone was broken, so he",
    "phone+stolen": "His phone was stolen, so he",
}

# Tokens to track — the content words and some controls
PROBE_WORDS = [
    " car", " cars", " vehicle",
    " phone", " phones", " device",
    " broken", " fix", " repair", " mechanic",
    " stolen", " police", " report", " security",
    " had", " decided", " called",  # prediction tokens
    " the", " a", " was",  # function words (baseline)
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

    def _sl(*layers):
        """Scale layer indices from 32-layer base to current NL."""
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    # Get token IDs for probe words
    probe_ids = {}
    for word in PROBE_WORDS:
        ids = tok(word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            probe_ids[word.strip()] = ids[0]
        else:
            print(f"  SKIP multi-token: '{word}' → {ids}")
    print(f"  Tracking {len(probe_ids)} tokens: {list(probe_ids.keys())}")

    # Run all 4 cases
    outs = {}
    for key, text in CASES.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        outs[key] = out
        # Show prediction
        probs = torch.softmax(out.logits[0, -1], -1)
        vals, idxs = torch.topk(probs, 3)
        top3 = [(tok.decode([int(idxs[i])]).strip(), round(float(vals[i]), 3))
                for i in range(3)]
        print(f"  {key:>15}: \"{text}\" → {top3}")

    # === Part 1: Track specific token logits across layers ===
    print(f"\n{'='*120}")
    print("PART 1: Specific token logits at each layer")
    print("  (logit = ⟨h[L], W_U[token]⟩ — higher = more aligned)")

    # Show a few key tokens across layers for all 4 states
    key_tokens = ["car", "phone", "broken", "stolen", "fix", "police", "had"]

    for token_name in key_tokens:
        tid = probe_ids[token_name]
        print(f"\n  Token: '{token_name}' (id={tid})")
        print(f"  {'L':>3} | {'car+broken':>12} {'car+stolen':>12} "
              f"{'phone+broken':>12} {'phone+stolen':>12} | notes")
        print(f"  {'-'*75}")

        for L in range(0, NL + 1, 4):
            vals = {}
            ranks = {}
            for key in CASES:
                h = outs[key].hidden_states[L][0, -1, :]
                logits = h @ W_U.T
                val = float(logits[tid])
                rank = int((logits > logits[tid]).sum().item()) + 1
                vals[key] = val
                ranks[key] = rank

            # Note if the pattern is compositional
            # car tokens should be same for car+broken and car+stolen
            # broken tokens should be same for car+broken and phone+broken
            notes = ""
            if token_name in ("car", "cars", "vehicle"):
                # should be higher for car+X than phone+X
                car_avg = (vals["car+broken"] + vals["car+stolen"]) / 2
                phone_avg = (vals["phone+broken"] + vals["phone+stolen"]) / 2
                if car_avg > phone_avg + 0.1:
                    notes = f"car>{phone_avg:.1f}"
            elif token_name in ("phone", "phones", "device"):
                car_avg = (vals["car+broken"] + vals["car+stolen"]) / 2
                phone_avg = (vals["phone+broken"] + vals["phone+stolen"]) / 2
                if phone_avg > car_avg + 0.1:
                    notes = f"phone>{car_avg:.1f}"
            elif token_name in ("broken", "fix", "repair", "mechanic"):
                broken_avg = (vals["car+broken"] + vals["phone+broken"]) / 2
                stolen_avg = (vals["car+stolen"] + vals["phone+stolen"]) / 2
                if broken_avg > stolen_avg + 0.1:
                    notes = f"broken>{stolen_avg:.1f}"
            elif token_name in ("stolen", "police", "report", "security"):
                broken_avg = (vals["car+broken"] + vals["phone+broken"]) / 2
                stolen_avg = (vals["car+stolen"] + vals["phone+stolen"]) / 2
                if stolen_avg > broken_avg + 0.1:
                    notes = f"stolen>{broken_avg:.1f}"

            print(f"  {L:>3} | {vals['car+broken']:>12.2f} {vals['car+stolen']:>12.2f} "
                  f"{vals['phone+broken']:>12.2f} {vals['phone+stolen']:>12.2f} | {notes}")

    # === Part 2: Compositionality check ===
    print(f"\n{'='*120}")
    print("PART 2: Compositionality — is object identity independent of event?")
    print("  Δ_object = logit(token, car+X) - logit(token, phone+X)")
    print("  If compositional, Δ_object should be same for X=broken and X=stolen")

    for L in _sl(16, 24, 32):
        print(f"\n  Layer {L}:")
        print(f"  {'token':>12} | {'Δ(broken)':>10} {'Δ(stolen)':>10} {'diff':>8} | "
              f"{'Δ_event(car)':>12} {'Δ_event(phn)':>12} {'diff':>8}")
        print(f"  {'-'*85}")

        for token_name in key_tokens:
            tid = probe_ids[token_name]
            logits = {}
            for key in CASES:
                h = outs[key].hidden_states[L][0, -1, :]
                logits[key] = float((h @ W_U.T)[tid])

            # Object axis: car - phone, for each event
            d_obj_broken = logits["car+broken"] - logits["phone+broken"]
            d_obj_stolen = logits["car+stolen"] - logits["phone+stolen"]
            obj_diff = abs(d_obj_broken - d_obj_stolen)

            # Event axis: broken - stolen, for each object
            d_evt_car = logits["car+broken"] - logits["car+stolen"]
            d_evt_phone = logits["phone+broken"] - logits["phone+stolen"]
            evt_diff = abs(d_evt_car - d_evt_phone)

            print(f"  {token_name:>12} | {d_obj_broken:>+10.2f} {d_obj_stolen:>+10.2f} "
                  f"{obj_diff:>8.2f} | {d_evt_car:>+12.2f} {d_evt_phone:>+12.2f} "
                  f"{evt_diff:>8.2f}")

    # === Part 3: Rank of content tokens in individual logit lens ===
    print(f"\n{'='*120}")
    print("PART 3: Where do content tokens rank in the individual logit lens?")
    print("  (rank out of 51200 — how far from the top)")

    for L in _sl(16, 24, 32):
        print(f"\n  Layer {L}:")
        print(f"  {'token':>12} | {'car+broken':>12} {'car+stolen':>12} "
              f"{'phone+broken':>12} {'phone+stolen':>12}")
        print(f"  {'-'*65}")

        for token_name in key_tokens:
            tid = probe_ids[token_name]
            ranks = {}
            for key in CASES:
                h = outs[key].hidden_states[L][0, -1, :]
                logits = h @ W_U.T
                rank = int((logits > logits[tid]).sum().item()) + 1
                ranks[key] = rank

            print(f"  {token_name:>12} | {ranks['car+broken']:>12} "
                  f"{ranks['car+stolen']:>12} {ranks['phone+broken']:>12} "
                  f"{ranks['phone+stolen']:>12}")

    # === Part 4: Logit lens top-10 for each state at L24 ===
    print(f"\n{'='*120}")
    print(f"PART 4: Full logit-lens top-10 for each state at L{_sl(24)[0]}")
    print("  (what the logit lens sees — the shared state)")

    for key in CASES:
        h = outs[key].hidden_states[_sl(24)[0]][0, -1, :]
        logits = h @ W_U.T
        vals, idxs = torch.topk(logits, 10)
        top10 = [(tok.decode([int(idxs[i])]).strip()[:15],
                  round(float(vals[i]), 2))
                 for i in range(10)]
        print(f"  {key:>15}: {top10}")

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
