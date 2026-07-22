"""
Decompose the hot-dog compound signal into semantic axes.

Extract directions from contrastive pairs that isolate specific
food dimensions, then project the hot-dog state onto each axis.
If the representation is a linear combination of semantic features,
the sum of axis-projected W_U readouts should reconstruct
"eaten, cooked, fried, bun, vendor, served."

Axes to try:
  food/not-food:    "hot dog" vs "hot rod"
  meat/plant:       "steak dinner" vs "salad dinner"
  fried/grilled:    "fried chicken" vs "grilled chicken"
  bread/no-bread:   "sandwich" vs "steak"
  street/restaurant: "hot dog stand" vs "restaurant meal"
  processed/fresh:  "sausage" vs "fresh fish"
  fast/slow:        "fast food" vs "gourmet meal"

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_axis_decomp.py
"""
import sys
import torch
import torch.nn.functional as F

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
        W_U = model.lm_head.weight.detach().float().cpu()
    else:
        W_U = model.embed_out.weight.detach().float().cpu()

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=8):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def get_last_hidden(text, L):
        """Get hidden state at last token position at layer L."""
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        return out.hidden_states[L][0, -1, :].float().cpu()

    def get_pos_hidden(text, pos, L):
        """Get hidden state at specific position at layer L."""
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        return out.hidden_states[L][0, pos, :].float().cpu()

    # The target: "The hot dog was" at dog pos (2), at L4 post (= L5 pre)
    # This is where the compound signal lives
    ref_L = _sl(5)[0]  # L5 pre = hidden_states[L4+1] = hidden_states[5]
    # Actually hidden_states[L] is output of layer L-1...
    # hidden_states[5] = output of layer 4 = L4 post = L5 input
    # Let's just use the layer where "fried" appears
    # From sublayer analysis: L4 post has "fried" → hidden_states[5]

    print(f"Reference layer: hidden_states[{ref_L}] (L{ref_L-1} post = L{ref_L} input)")

    # ================================================================
    # 1. Extract semantic axes from food-related contrasts
    # ================================================================
    print(f"\n{'='*100}")
    print(f"1. EXTRACT SEMANTIC AXES from contrastive pairs")
    print(f"   Each pair isolates one food dimension at the last token")
    print(f"{'='*100}")

    # Axis pairs: (label, positive_text, negative_text, description)
    # We use "X was" endings so the last token is the prediction site
    axis_pairs = [
        ("food",
         "The delicious food was",
         "The broken machine was",
         "food vs non-food"),

        ("meat",
         "The grilled steak was",
         "The fresh salad was",
         "meat vs vegetable"),

        ("fried",
         "The deep fried chicken was",
         "The oven roasted chicken was",
         "fried vs not-fried"),

        ("bread",
         "The big sandwich was",
         "The grilled steak was",
         "bread-based vs not"),

        ("street",
         "The street food was",
         "The restaurant meal was",
         "street/casual vs restaurant"),

        ("processed",
         "The processed sausage was",
         "The fresh salmon was",
         "processed vs fresh"),

        ("fast",
         "The fast food was",
         "The gourmet meal was",
         "fast vs gourmet"),

        ("hot_temp",
         "The steaming hot soup was",
         "The ice cold soup was",
         "hot-temperature vs cold"),

        ("savory",
         "The savory dish was",
         "The sweet dessert was",
         "savory vs sweet"),

        ("cheap",
         "The cheap meal was",
         "The expensive dinner was",
         "cheap vs expensive"),

        ("outdoor",
         "The barbecue food was",
         "The kitchen food was",
         "outdoor/grill vs indoor"),

        ("american",
         "The American food was",
         "The Japanese food was",
         "American vs other cuisine"),
    ]

    axes = {}
    for label, pos_text, neg_text, desc in axis_pairs:
        h_pos = get_last_hidden(pos_text, ref_L)
        h_neg = get_last_hidden(neg_text, ref_L)
        direction = h_pos - h_neg
        # Normalize to unit vector
        direction_norm = direction / direction.norm()
        axes[label] = {
            "dir": direction,
            "dir_norm": direction_norm,
            "raw_norm": float(direction.norm()),
        }
        ld = direction @ W_U.T
        print(f"\n  {label:>12} ({desc})")
        print(f"    ||Δh|| = {float(direction.norm()):.1f}")
        print(f"    + pole: [{tk(ld)}]")
        print(f"    - pole: [{tk(-ld)}]")

    # ================================================================
    # 2. Axis independence — are these orthogonal?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"2. AXIS INDEPENDENCE — pairwise cosine")
    print(f"{'='*100}")

    labels = list(axes.keys())
    print(f"  {'':>12}", end="")
    for l in labels:
        print(f" {l[:6]:>7}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:>12}", end="")
        for l2 in labels:
            c = float(F.cosine_similarity(
                axes[l1]["dir"].unsqueeze(0),
                axes[l2]["dir"].unsqueeze(0)))
            print(f" {c:>+7.3f}", end="")
        print()

    # ================================================================
    # 3. Project the hot-dog state onto each axis
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. PROJECT HOT-DOG STATE onto each axis")
    print(f"{'='*100}")

    # Get the hot-dog state at the critical position
    # "The hot dog was" — dog is pos 2, was is pos 3
    # The food signal is at dog pos (pos 2) after L4 MLP
    h_hotdog_dog = get_pos_hidden("The hot dog was", 2, ref_L)
    h_hotdog_was = get_last_hidden("The hot dog was", ref_L)

    # Also get a baseline: "The cold dog was" to compute the
    # contrastive hot-dog signal
    h_colddog_dog = get_pos_hidden("The cold dog was", 2, ref_L)
    h_colddog_was = get_last_hidden("The cold dog was", ref_L)

    delta_dog = h_hotdog_dog - h_colddog_dog
    delta_was = h_hotdog_was - h_colddog_was

    print(f"\n  Full contrastive (hot-cold) at dog pos reads as:")
    ld_full = delta_dog @ W_U.T
    print(f"    [{tk(ld_full)}]")
    print(f"    ||Δh|| = {float(delta_dog.norm()):.1f}")

    print(f"\n  Full contrastive (hot-cold) at was pos reads as:")
    ld_full_was = delta_was @ W_U.T
    print(f"    [{tk(ld_full_was)}]")
    print(f"    ||Δh|| = {float(delta_was.norm()):.1f}")

    # Project onto each axis
    print(f"\n  Projection of hot-cold contrastive onto each axis:")
    print(f"  {'Axis':>12} {'proj_dog':>9} {'proj_was':>9} {'cos_dog':>9} {'cos_was':>9}  axis reads as")

    projections_dog = {}
    projections_was = {}
    for label in labels:
        d = axes[label]["dir_norm"]
        proj_dog = float(torch.dot(delta_dog, d))
        proj_was = float(torch.dot(delta_was, d))
        cos_dog = float(F.cosine_similarity(
            delta_dog.unsqueeze(0), d.unsqueeze(0)))
        cos_was = float(F.cosine_similarity(
            delta_was.unsqueeze(0), d.unsqueeze(0)))
        projections_dog[label] = proj_dog
        projections_was[label] = proj_was
        # What this axis contributes in token space
        axis_contribution = proj_dog * d
        ld = axis_contribution @ W_U.T
        toks = tk(ld, 4)
        print(f"  {label:>12} {proj_dog:>+9.1f} {proj_was:>+9.1f} "
              f"{cos_dog:>+9.3f} {cos_was:>+9.3f}  [{toks}]")

    # ================================================================
    # 4. RECONSTRUCT: sum axis projections → does it match?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. RECONSTRUCTION — sum of axis projections vs full signal")
    print(f"{'='*100}")

    # Reconstruct at dog pos
    reconstructed_dog = torch.zeros_like(delta_dog)
    for label in labels:
        d = axes[label]["dir_norm"]
        proj = projections_dog[label]
        reconstructed_dog += proj * d

    ld_recon = reconstructed_dog @ W_U.T
    residual_dog = delta_dog - reconstructed_dog
    ld_residual = residual_dog @ W_U.T

    cos_recon = float(F.cosine_similarity(
        delta_dog.unsqueeze(0), reconstructed_dog.unsqueeze(0)))
    variance_explained = float(reconstructed_dog.norm()**2 / delta_dog.norm()**2)

    print(f"\n  Dog position:")
    print(f"    Full signal:        [{tk(ld_full)}]")
    print(f"    Reconstructed:      [{tk(ld_recon)}]")
    print(f"    Residual:           [{tk(ld_residual)}]")
    print(f"    cos(full, recon):   {cos_recon:+.4f}")
    print(f"    variance explained: {variance_explained:.1%}")
    print(f"    ||full||={float(delta_dog.norm()):.1f}  "
          f"||recon||={float(reconstructed_dog.norm()):.1f}  "
          f"||residual||={float(residual_dog.norm()):.1f}")

    # Same for was pos
    reconstructed_was = torch.zeros_like(delta_was)
    for label in labels:
        d = axes[label]["dir_norm"]
        proj = projections_was[label]
        reconstructed_was += proj * d

    ld_recon_was = reconstructed_was @ W_U.T
    residual_was = delta_was - reconstructed_was
    ld_residual_was = residual_was @ W_U.T
    cos_recon_was = float(F.cosine_similarity(
        delta_was.unsqueeze(0), reconstructed_was.unsqueeze(0)))
    var_was = float(reconstructed_was.norm()**2 / delta_was.norm()**2)

    print(f"\n  Was position:")
    print(f"    Full signal:        [{tk(ld_full_was)}]")
    print(f"    Reconstructed:      [{tk(ld_recon_was)}]")
    print(f"    Residual:           [{tk(ld_residual_was)}]")
    print(f"    cos(full, recon):   {cos_recon_was:+.4f}")
    print(f"    variance explained: {var_was:.1%}")

    # ================================================================
    # 5. GREEDY AXIS ADDITION — which axes matter most?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"5. GREEDY AXIS ADDITION — add axes one by one, track cos with full signal")
    print(f"{'='*100}")

    # Sort axes by absolute projection magnitude
    sorted_axes = sorted(labels,
                          key=lambda l: abs(projections_dog[l]),
                          reverse=True)

    cumulative = torch.zeros_like(delta_dog)
    print(f"\n  Dog position (adding axes by projection magnitude):")
    print(f"  {'Step':>4} {'Added axis':>12} {'proj':>8} {'cum_cos':>9} {'cum_var%':>9}  cumulative reads as")

    for i, label in enumerate(sorted_axes):
        d = axes[label]["dir_norm"]
        proj = projections_dog[label]
        cumulative = cumulative + proj * d
        c = float(F.cosine_similarity(
            delta_dog.unsqueeze(0), cumulative.unsqueeze(0)))
        v = float(cumulative.norm()**2 / delta_dog.norm()**2)
        ld = cumulative @ W_U.T
        toks = tk(ld, 6)
        print(f"  {i+1:>4} {label:>12} {proj:>+8.1f} {c:>+9.4f} {v:>8.1%}  [{toks}]")

    # ================================================================
    # 6. MULTIPLE REFERENCE POINTS — not just hot-cold
    # ================================================================
    print(f"\n{'='*100}")
    print(f"6. AXIS PROJECTIONS for different hot-dog contrasts")
    print(f"   Same axes, different reference subtraction")
    print(f"{'='*100}")

    other_refs = [
        ("vs cold dog", "The cold dog was", 2),
        ("vs hot cat",  "The hot cat was", 2),
        ("vs hot rod",  "The hot rod was", 2),
        ("vs angry dog", "The angry dog was", 2),
        ("vs bare dog", "The dog was", 1),
    ]

    print(f"\n  {'Reference':>14}", end="")
    for label in labels:
        print(f" {label[:6]:>7}", end="")
    print(f" {'||Δ||':>7}")

    for ref_label, ref_text, ref_pos in other_refs:
        h_ref = get_pos_hidden(ref_text, ref_pos, ref_L)
        delta = h_hotdog_dog - h_ref
        print(f"  {ref_label:>14}", end="")
        for label in labels:
            d = axes[label]["dir_norm"]
            proj = float(torch.dot(delta, d))
            print(f" {proj:>+7.1f}", end="")
        print(f" {float(delta.norm()):>7.1f}")

    # ================================================================
    # 7. RAW LOGIT LENS on each axis direction
    # ================================================================
    print(f"\n{'='*100}")
    print(f"7. WHAT DOES EACH AXIS CONTRIBUTE in token space?")
    print(f"   axis_dir × projection_value → token readout")
    print(f"{'='*100}")

    for label in sorted_axes:
        proj = projections_dog[label]
        d = axes[label]["dir_norm"]
        contribution = proj * d
        ld = contribution @ W_U.T
        print(f"\n  {label:>12} (proj={proj:>+.1f}):")
        print(f"    contributes: [{tk(ld)}]")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
