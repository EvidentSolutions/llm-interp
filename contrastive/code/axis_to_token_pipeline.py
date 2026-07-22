"""
Full pipeline: contrastive axis → token form → noun override.

For each scenario:
1. Multi-contrast at the key position to extract the axis
2. Read through W_U — what token does the axis surface?
3. Causal injection — does the direction change predictions?
4. Token override — does the surfaced token as modifier force the frame?

Scenarios:
  frozen/fresh chicken → temperature-state axis
  deadly/harmless snake → danger axis
  flying/parked car → motion axis
  stolen/returned wallet → legality axis
  ancient/modern building → age axis
  broken/working machine → functionality axis
  pregnant/sleeping cat → biological-state axis
  burning/extinguished fire → combustion axis

Usage: .venv/Scripts/python.exe contrastive/code/axis_to_token_pipeline.py
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
    for p in model.parameters(): p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float()
    else:
        W_U = model.embed_out.weight.detach().float()

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def predict(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV))
        probs = torch.softmax(out.logits[0, -1].float(), -1)
        top = torch.topk(probs, 4)
        return [(tok.decode([int(top.indices[j])]).strip(), round(float(top.values[j]), 3))
                for j in range(4)]

    def get_hidden(text, pos, L):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        h = out.hidden_states[L][0, pos, :].float().cpu()
        del out; torch.cuda.empty_cache()
        return h

    def inject_and_predict(text, direction, scale, pos, layer):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        perturbation = (direction * scale).half().to(DEV)
        def hook_fn(module, args):
            h = args[0].clone()
            h[0, pos, :] += perturbation
            return (h,) + args[1:]
        handle = model.model.layers[layer].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV))
        handle.remove()
        probs = torch.softmax(out.logits[0, -1].float(), -1)
        top = torch.topk(probs, 4)
        return [(tok.decode([int(top.indices[j])]).strip(), round(float(top.values[j]), 3))
                for j in range(4)]

    # Each scenario: (label, target_prompt, contrast_prompts, noun_pos,
    #                  override_template, override_nouns)
    # contrast_prompts: list of prompts to subtract from target
    # The axis = mean(target - each contrast) at noun_pos

    scenarios = [
        {
            "label": "FROZEN (temperature-state)",
            "target": "The frozen chicken was",
            "contrasts": [
                "The fresh chicken was",
                "The warm chicken was",
                "The hot chicken was",
                "The thawed chicken was",
            ],
            "noun_pos": 2,
            "override_template": "The {} {} started to",
            "override_nouns": ["shoe", "rock", "chair", "phone", "brick",
                               "pencil", "hammer", "candle", "clock", "coin",
                               "chicken", "fish", "water", "butter", "soup"],
            "override_target_words": {"melt", "thaw", "defrost", "drip",
                                       "crack", "soften", "warm", "break"},
        },
        {
            "label": "DEADLY (danger)",
            "target": "The deadly spider was",
            "contrasts": [
                "The harmless spider was",
                "The friendly spider was",
                "The common spider was",
                "The small spider was",
            ],
            "noun_pos": 2,
            "override_template": "The {} {} was found near the school. People were",
            "override_nouns": ["shoe", "rock", "chair", "phone", "brick",
                               "pencil", "bottle", "candle", "clock", "coin",
                               "spider", "snake", "mushroom", "plant", "gas"],
            "override_target_words": {"afraid", "scared", "terrified", "warned",
                                       "evacuated", "frightened", "panicked",
                                       "worried", "shocked", "horrified",
                                       "concerned", "alarmed", "urged"},
        },
        {
            "label": "FLYING (motion)",
            "target": "The flying car was",
            "contrasts": [
                "The parked car was",
                "The stationary car was",
                "The broken car was",
                "The old car was",
            ],
            "noun_pos": 2,
            "override_template": "The {} {} crashed into the",
            "override_nouns": ["shoe", "rock", "chair", "phone", "brick",
                               "pencil", "bottle", "helmet", "clock", "coin",
                               "car", "bird", "ball", "plane", "drone"],
            "override_target_words": {"wall", "building", "ground", "window",
                                       "tree", "roof", "car", "fence", "house",
                                       "side", "floor", "door", "road"},
        },
        {
            "label": "STOLEN (legality)",
            "target": "The stolen painting was",
            "contrasts": [
                "The displayed painting was",
                "The famous painting was",
                "The new painting was",
                "The large painting was",
            ],
            "noun_pos": 2,
            "override_template": "He was caught with the {} {} and was",
            "override_nouns": ["shoe", "rock", "chair", "phone", "brick",
                               "pencil", "bottle", "helmet", "clock", "coin",
                               "painting", "diamond", "car", "passport", "weapon"],
            "override_target_words": {"arrested", "charged", "fined", "jailed",
                                       "sentenced", "detained", "prosecuted",
                                       "taken", "brought", "held", "immediately"},
        },
        {
            "label": "ANCIENT (age/time)",
            "target": "The ancient temple was",
            "contrasts": [
                "The modern temple was",
                "The new temple was",
                "The recent temple was",
                "The current temple was",
            ],
            "noun_pos": 2,
            "override_template": "The {} {} was discovered and dated to",
            "override_nouns": ["shoe", "rock", "chair", "phone", "brick",
                               "pencil", "bottle", "helmet", "clock", "coin",
                               "temple", "sword", "bone", "pottery", "scroll"],
            "override_target_words": {"the", "approximately", "around", "about",
                                       "roughly", "be", "over", "more", "nearly"},
        },
        {
            "label": "BURNING (combustion)",
            "target": "The burning building was",
            "contrasts": [
                "The standing building was",
                "The tall building was",
                "The empty building was",
                "The new building was",
            ],
            "noun_pos": 2,
            "override_template": "The {} {} quickly",
            "override_nouns": ["shoe", "rock", "chair", "phone", "brick",
                               "pencil", "bottle", "candle", "carpet", "tire",
                               "building", "house", "forest", "car", "tree"],
            "override_target_words": {"spread", "engulfed", "consumed",
                                       "destroyed", "melted", "ignited",
                                       "caught", "burned", "set", "turned",
                                       "collapsed", "crumbled", "disintegrated"},
        },
    ]

    # Test at multiple layers to find where the axis appears
    test_layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)

    for scenario in scenarios:
        label = scenario["label"]
        target = scenario["target"]
        contrasts = scenario["contrasts"]
        noun_pos = scenario["noun_pos"]

        print(f"\n{'='*100}")
        print(f"  {label}")
        print(f"  Target: \"{target}\"")
        print(f"{'='*100}")

        # === Step 1: Multi-contrast at noun pos ===
        print(f"\n  Step 1: Multi-contrast at noun pos ({noun_pos})")

        dirs_by_layer = {L: [] for L in test_layers}
        for ctext in contrasts:
            for L in test_layers:
                h_t = get_hidden(target, noun_pos, L)
                h_c = get_hidden(ctext, noun_pos, L)
                dirs_by_layer[L].append(h_t - h_c)

        # Find the layer where the axis first becomes token-readable
        print(f"\n  {'L':>4} {'||mean||':>8} {'consistency':>12} W_U reads as")
        best_L = test_layers[0]
        best_cos = 0
        for L in test_layers:
            dirs = dirs_by_layer[L]
            mean_d = torch.stack(dirs).mean(dim=0)
            # Pairwise consistency
            cosines = []
            for i in range(len(dirs)):
                for j in range(i+1, len(dirs)):
                    cosines.append(float(F.cosine_similarity(
                        dirs[i].unsqueeze(0), dirs[j].unsqueeze(0))))
            mean_cos = sum(cosines) / len(cosines) if cosines else 0
            ld = mean_d @ W_U.cpu().T
            print(f"  L{L:>2} {float(mean_d.norm()):>8.1f} {mean_cos:>12.3f}  "
                  f"[{tk(ld, 5)}]")
            if mean_cos > best_cos:
                best_cos = mean_cos
                best_L = L

        # Use the layer with highest consistency
        ref_L = best_L
        mean_dir = torch.stack(dirs_by_layer[ref_L]).mean(dim=0)
        mean_dir_n = mean_dir / mean_dir.norm()
        base_scale = float(mean_dir.norm())

        ld = mean_dir @ W_U.cpu().T
        top_tokens = []
        v, idx = torch.topk(ld.float(), 8)
        for j in range(8):
            top_tokens.append(tok.decode([int(idx[j])]).strip())

        print(f"\n  Best layer: L{ref_L} (consistency={best_cos:.3f})")
        print(f"  Axis reads as: + [{tk(ld, 6)}]")
        print(f"                 - [{tk(-ld, 6)}]")

        # === Step 2: Baseline predictions ===
        print(f"\n  Step 2: Baselines")
        print(f"    {target:>40} → {predict(target)}")
        for ctext in contrasts[:2]:
            print(f"    {ctext:>40} → {predict(ctext)}")

        # === Step 3: Causal injection ===
        print(f"\n  Step 3: Causal injection at L{ref_L} pos {noun_pos}")
        for ctext in contrasts[:2]:
            ids = tok(ctext, add_special_tokens=False)["input_ids"]
            baseline = predict(ctext)
            injected = inject_and_predict(ctext, mean_dir_n.cpu(),
                                           base_scale, noun_pos, ref_L)
            print(f"    {ctext}")
            print(f"      baseline:   {baseline}")
            print(f"      +1× axis:   {injected}")

        # Reverse: subtract from target
        baseline = predict(target)
        subtracted = inject_and_predict(target, mean_dir_n.cpu(),
                                         -base_scale, noun_pos, ref_L)
        print(f"    {target}")
        print(f"      baseline:   {baseline}")
        print(f"      -1× axis:   {subtracted}")

        # === Step 4: Token override test ===
        print(f"\n  Step 4: Token override — do surfaced tokens force the frame?")
        template = scenario["override_template"]
        test_nouns = scenario["override_nouns"]
        target_words = scenario["override_target_words"]

        # Test top-3 tokens from the axis readout as modifiers
        test_modifiers = [t for t in top_tokens[:6]
                          if len(t) > 2 and t.isalpha()][:3]
        # Also add the obvious human-readable token if not in list
        obvious = label.split(" ")[0].lower()
        if obvious not in [t.lower() for t in test_modifiers]:
            test_modifiers.insert(0, obvious)

        for modifier in test_modifiers:
            hits = 0
            for noun in test_nouns:
                text = template.format(modifier, noun)
                preds = predict(text)
                t1 = preds[0][0].lower()
                if t1 in target_words or any(
                        p[0].lower() in target_words for p in preds[:2]):
                    hits += 1
            pct = hits / len(test_nouns) * 100
            print(f"    \"{modifier}\": {hits}/{len(test_nouns)} ({pct:.0f}%)")

        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
