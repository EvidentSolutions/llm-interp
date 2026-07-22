"""
Multi-contrast triangulation at entity positions for two-hop factual recall.

Single-pair readout at the Eiffel Tower position reads gibberish — the
entity representation is superposed with many properties (tall, steel,
landmark, Paris, France...). Multi-contrast against different baselines
should cancel the shared properties and surface what's unique.

Baseline sets:
  1. Other landmarks (Colosseum, Parthenon, Big Ben, Great Wall)
     → cancels "landmark" properties, surfaces France/Paris
  2. Generic tall/steel structures (tallest building, steel bridge, radio tower)
     → cancels physical properties, surfaces proper-name associations

Usage: .venv/Scripts/python.exe contrastive/code/gpt2_multihop_multicontrast.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

DEV = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach().float()


def tk(logits, k=6):
    v, i = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def run(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return out, ids


# ============================================================
# Setup
# ============================================================
FRAME = "The capital of the country where the {} is located is"

TARGET = "Eiffel Tower"
target_prompt = FRAME.format(TARGET)
out_target, ids_target = run(target_prompt)
toks_target = [tok.decode([t]) for t in ids_target]
END_T = len(ids_target) - 1

print(f"Target: {target_prompt}")
print(f"Tokens: {list(enumerate(toks_target))}")

# Find entity token range (where target and baselines differ)
# For "Eiffel Tower": tokens 7-10

# ============================================================
# Baseline set 1: Other famous landmarks
# ============================================================
print(f"\n{'='*120}")
print("BASELINE SET 1: Other landmarks")
print("  What does Eiffel Tower have that other landmarks don't?")
print("  Expect: France/Paris to surface (the location-specific content)")
print("=" * 120)

LANDMARKS = [
    "Colosseum",        # Italy/Rome
    "Parthenon",        # Greece/Athens
    "Taj Mahal",        # India/Delhi
    "Kremlin",          # Russia/Moscow
    "Alhambra",         # Spain/Madrid
]

dh_entity_L = defaultdict(list)  # layer -> list of Δh at entity position
dh_end_L = defaultdict(list)

for baseline in LANDMARKS:
    bp = FRAME.format(baseline)
    out_b, ids_b = run(bp)
    end_b = len(ids_b) - 1
    toks_b = [tok.decode([t]) for t in ids_b]

    # Generation check
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids_b], device=DEV),
            max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = tok.decode(gen[0][len(ids_b):]).strip()[:30]
    print(f"\n  vs {baseline}: predicts '{pred}'")

    # Single-pair readout at END
    for L in [20, 24, 28]:
        h_a = out_target.hidden_states[L][0, END_T, :].float()
        h_b = out_b.hidden_states[L][0, end_b, :].float()
        dh = h_a - h_b
        ld = dh @ W_U.T
        print(f"    END L{L}: [{tk(ld)}]")

    # Collect for averaging — at END
    for L in range(NL + 1):
        h_a = out_target.hidden_states[L][0, END_T, :].float().cpu()
        h_b = out_b.hidden_states[L][0, end_b, :].float().cpu()
        dh_end_L[L].append(h_a - h_b)

    # Collect at entity positions (pos 10 in target = last entity token)
    # Only if same length
    if len(ids_b) == len(ids_target):
        for L in range(NL + 1):
            h_a = out_target.hidden_states[L][0, 10, :].float().cpu()
            h_b = out_b.hidden_states[L][0, 10, :].float().cpu()
            dh_entity_L[L].append(h_a - h_b)

    del out_b
    torch.cuda.empty_cache()

# Averaged readout
print(f"\n  AVERAGED over {len(LANDMARKS)} landmarks:")
print(f"  {'Pos':>5} {'Layer':>5} {'MNorm':>7} {'ANorm':>7} {'Ret':>5}  {'Averaged readout':>55}")

if dh_entity_L:
    n = len(dh_entity_L[0])
    for L in [8, 12, 16, 20, 24, 28]:
        vecs = dh_entity_L[L]
        mn = sum(float(v.norm()) for v in vecs) / len(vecs)
        avg = torch.stack(vecs).mean(dim=0)
        an = float(avg.norm())
        ret = an / mn if mn > 0 else 0
        ld = avg @ W_U.cpu().T
        print(f"  {'ent':>5} L{L:>3} {mn:>7.1f} {an:>7.1f} {ret:>4.0%}  {tk(ld):>55}")

n = len(dh_end_L[0])
for L in [8, 12, 16, 20, 24, 28]:
    vecs = dh_end_L[L]
    mn = sum(float(v.norm()) for v in vecs) / len(vecs)
    avg = torch.stack(vecs).mean(dim=0)
    an = float(avg.norm())
    ret = an / mn if mn > 0 else 0
    ld = avg @ W_U.cpu().T
    print(f"  {'END':>5} L{L:>3} {mn:>7.1f} {an:>7.1f} {ret:>4.0%}  {tk(ld):>55}")


# ============================================================
# Baseline set 2: Generic tall/steel structures (not proper nouns)
# ============================================================
print(f"\n{'='*120}")
print("BASELINE SET 2: Generic structures")
print("  What does Eiffel Tower have that generic structures don't?")
print("  Expect: proper-name associations (Paris, France, Gustave, 1889...)")
print("=" * 120)

GENERIC = [
    "tallest building",
    "steel bridge",
    "radio tower",
    "clock tower",
    "bell tower",
]

dh_gen_end = defaultdict(list)
dh_gen_ent = defaultdict(list)

for baseline in GENERIC:
    bp = FRAME.format(baseline)
    out_b, ids_b = run(bp)
    end_b = len(ids_b) - 1

    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids_b], device=DEV),
            max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = tok.decode(gen[0][len(ids_b):]).strip()[:30]
    print(f"\n  vs {baseline}: predicts '{pred}'")

    for L in [20, 24, 28]:
        h_a = out_target.hidden_states[L][0, END_T, :].float()
        h_b = out_b.hidden_states[L][0, end_b, :].float()
        dh = h_a - h_b
        ld = dh @ W_U.T
        print(f"    END L{L}: [{tk(ld)}]")

    for L in range(NL + 1):
        h_a = out_target.hidden_states[L][0, END_T, :].float().cpu()
        h_b = out_b.hidden_states[L][0, end_b, :].float().cpu()
        dh_gen_end[L].append(h_a - h_b)

    if len(ids_b) == len(ids_target):
        for L in range(NL + 1):
            h_a = out_target.hidden_states[L][0, 10, :].float().cpu()
            h_b = out_b.hidden_states[L][0, 10, :].float().cpu()
            dh_gen_ent[L].append(h_a - h_b)

    del out_b
    torch.cuda.empty_cache()

print(f"\n  AVERAGED over {len(GENERIC)} generic structures:")
print(f"  {'Pos':>5} {'Layer':>5} {'MNorm':>7} {'ANorm':>7} {'Ret':>5}  {'Averaged readout':>55}")

if dh_gen_ent:
    n = len(dh_gen_ent[0])
    for L in [8, 12, 16, 20, 24, 28]:
        vecs = dh_gen_ent[L]
        mn = sum(float(v.norm()) for v in vecs) / len(vecs)
        avg = torch.stack(vecs).mean(dim=0)
        an = float(avg.norm())
        ret = an / mn if mn > 0 else 0
        ld = avg @ W_U.cpu().T
        print(f"  {'ent':>5} L{L:>3} {mn:>7.1f} {an:>7.1f} {ret:>4.0%}  {tk(ld):>55}")

n = len(dh_gen_end[0])
for L in [8, 12, 16, 20, 24, 28]:
    vecs = dh_gen_end[L]
    mn = sum(float(v.norm()) for v in vecs) / len(vecs)
    avg = torch.stack(vecs).mean(dim=0)
    an = float(avg.norm())
    ret = an / mn if mn > 0 else 0
    ld = avg @ W_U.cpu().T
    print(f"  {'END':>5} L{L:>3} {mn:>7.1f} {an:>7.1f} {ret:>4.0%}  {tk(ld):>55}")


# ============================================================
# Baseline set 3: Other French things (cancel France, surface Tower)
# ============================================================
print(f"\n{'='*120}")
print("BASELINE SET 3: Other French landmarks")
print("  What does Eiffel Tower have that other French things don't?")
print("  Expect: France/Paris to CANCEL, tower-specific content to surface")
print("=" * 120)

FRENCH = [
    "Louvre Museum",
    "Arc de Triomphe",
    "Notre Dame Cathedral",
    "Palace of Versailles",
]

dh_fr_end = defaultdict(list)

for baseline in FRENCH:
    bp = FRAME.format(baseline)
    out_b, ids_b = run(bp)
    end_b = len(ids_b) - 1

    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids_b], device=DEV),
            max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = tok.decode(gen[0][len(ids_b):]).strip()[:30]
    print(f"\n  vs {baseline}: predicts '{pred}'")

    for L in [20, 24, 28]:
        h_a = out_target.hidden_states[L][0, END_T, :].float()
        h_b = out_b.hidden_states[L][0, end_b, :].float()
        dh = h_a - h_b
        ld = dh @ W_U.T
        print(f"    END L{L}: [{tk(ld)}]")

    for L in range(NL + 1):
        h_a = out_target.hidden_states[L][0, END_T, :].float().cpu()
        h_b = out_b.hidden_states[L][0, end_b, :].float().cpu()
        dh_fr_end[L].append(h_a - h_b)

    del out_b
    torch.cuda.empty_cache()

print(f"\n  AVERAGED over {len(FRENCH)} French landmarks:")
print(f"  {'Pos':>5} {'Layer':>5} {'MNorm':>7} {'ANorm':>7} {'Ret':>5}  {'Averaged readout':>55}")

n = len(dh_fr_end[0])
for L in [8, 12, 16, 20, 24, 28]:
    vecs = dh_fr_end[L]
    mn = sum(float(v.norm()) for v in vecs) / len(vecs)
    avg = torch.stack(vecs).mean(dim=0)
    an = float(avg.norm())
    ret = an / mn if mn > 0 else 0
    ld = avg @ W_U.cpu().T
    print(f"  {'END':>5} L{L:>3} {mn:>7.1f} {an:>7.1f} {ret:>4.0%}  {tk(ld):>55}")

print(f"\n{'='*120}")
print("DONE")
print("=" * 120)
