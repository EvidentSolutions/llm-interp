"""
Can contrastive projection detect hallucination?

Compare real facts vs plausible-sounding fictions with detailed context.
The hypothesis: when the model hallucinates, the contrastive projection
against a known-factual prompt should show different signatures than
when both prompts are factual.

Usage: .venv/Scripts/python.exe contrastive/code/hallucination_detection.py
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

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach().float()


def predict(text, max_tokens=15):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0][len(ids):]).strip().split("\n")[0][:60]


def get_entropy_and_top(text, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -float((probs * log_probs).sum())
    v, i = torch.topk(probs, k)
    top = [(tok.decode([int(i[j])]).strip(), float(v[j])) for j in range(k)]
    return entropy, top


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def bk(logits, k=5):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def contrastive_trajectory(pa, pb, layers=None, label_a="A", label_b="B"):
    if layers is None:
        layers = [8, 16, 20, 24, 28, 32]
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )

    norms = []
    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        norms.append(norm)
        ld = dh @ W_U.T
        print(f"    L{L:>2} ({norm:.3f}) {label_a}=[{tk(ld)}]  "
              f"{label_b}=[{bk(ld)}]")

    del out_a, out_b
    return norms


# ============================================================
print("=" * 100)
print("1. FICTIONAL PEOPLE — plausible context, model hallucinates")
print("=" * 100)

fictional_people = [
    # (fictional prompt, matched real prompt)
    ("Ludvig von Vogelkirche, born in Prague in 1859, invented the",
     "Nikola Tesla, born in Smiljan in 1856, invented the"),
    ("Heinrich Müller-Brandeis, a chemist working in Vienna in 1903, discovered that",
     "Marie Curie, a chemist working in Paris in 1903, discovered that"),
    ("Professor Aino Järvikoski of the University of Helsinki published her landmark paper on",
     "Professor Albert Einstein of the University of Zurich published his landmark paper on"),
    ("The renowned architect Tomáš Černý designed the",
     "The renowned architect Frank Lloyd Wright designed the"),
]

for pa, pb in fictional_people:
    print(f'\n  Fictional: "{pa}"')
    ans_a = predict(pa)
    e_a, top_a = get_entropy_and_top(pa)
    print(f'    -> "{ans_a}"  H={e_a:.2f}  top1={top_a[0][0]}({top_a[0][1]:.3f})')

    print(f'  Real:      "{pb}"')
    ans_b = predict(pb)
    e_b, top_b = get_entropy_and_top(pb)
    print(f'    -> "{ans_b}"  H={e_b:.2f}  top1={top_b[0][0]}({top_b[0][1]:.3f})')

    print(f"  Contrastive (Fictional - Real):")
    contrastive_trajectory(pa, pb, layers=[16, 24, 28, 32],
                           label_a="Fict", label_b="Real")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("2. FICTIONAL PLACES — plausible geography")
print("=" * 100)

fictional_places = [
    ("The second largest suburb in Tampere Finland, Kaksikanaa, has a population of",
     "The second largest city in Finland, Espoo, has a population of"),
    ("The medieval castle of Dunvraigh, located in the Scottish Highlands, was built in",
     "The medieval castle of Edinburgh, located in Scotland, was built in"),
    ("The river Valkajoki flows through central Finland for",
     "The river Kokemäenjoki flows through central Finland for"),
    ("Mount Silverhorn, the tallest peak in New Zealand's Southern Alps, rises to",
     "Mount Cook, the tallest peak in New Zealand's Southern Alps, rises to"),
]

for pa, pb in fictional_places:
    print(f'\n  Fictional: "{pa}"')
    ans_a = predict(pa)
    e_a, top_a = get_entropy_and_top(pa)
    print(f'    -> "{ans_a}"  H={e_a:.2f}  top1={top_a[0][0]}({top_a[0][1]:.3f})')

    print(f'  Real:      "{pb}"')
    ans_b = predict(pb)
    e_b, top_b = get_entropy_and_top(pb)
    print(f'    -> "{ans_b}"  H={e_b:.2f}  top1={top_b[0][0]}({top_b[0][1]:.3f})')

    print(f"  Contrastive (Fictional - Real):")
    contrastive_trajectory(pa, pb, layers=[16, 24, 28, 32],
                           label_a="Fict", label_b="Real")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("3. SAME FICTIONAL ENTITY — contrast against bare frame")
print("   (Does the model represent 'I don't know this' differently?)")
print("=" * 100)

# Contrast fictional against minimal frame (same structure, no entity)
frame_contrasts = [
    ("Ludvig von Vogelkirche, born in Prague in 1859, invented the",
     "A man, born in a city in 1859, invented the"),
    ("The second largest suburb in Tampere Finland, Kaksikanaa, has a population of",
     "A suburb in Finland has a population of"),
    ("Professor Aino Järvikoski of the University of Helsinki published her landmark paper on",
     "A professor of a university published her landmark paper on"),
]

for pa, pb in frame_contrasts:
    print(f'\n  Fictional:  "{pa}"')
    ans_a = predict(pa)
    e_a, _ = get_entropy_and_top(pa)
    print(f'    -> "{ans_a}"  H={e_a:.2f}')

    print(f'  Bare frame: "{pb}"')
    ans_b = predict(pb)
    e_b, _ = get_entropy_and_top(pb)
    print(f'    -> "{ans_b}"  H={e_b:.2f}')

    print(f"  Contrastive (Fictional - Bare):")
    contrastive_trajectory(pa, pb, layers=[16, 24, 28, 32],
                           label_a="Fict", label_b="Bare")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("4. REAL vs REAL — control (both factual, different answers)")
print("=" * 100)

real_real = [
    ("Nikola Tesla, born in Smiljan in 1856, invented the",
     "Thomas Edison, born in Milan Ohio in 1847, invented the"),
    ("The second largest city in Finland, Espoo, has a population of",
     "The second largest city in Sweden, Gothenburg, has a population of"),
    ("Marie Curie, working in Paris in 1903, discovered that",
     "Ernest Rutherford, working in Manchester in 1911, discovered that"),
]

for pa, pb in real_real:
    print(f'\n  Real A: "{pa}"')
    ans_a = predict(pa)
    e_a, _ = get_entropy_and_top(pa)
    print(f'    -> "{ans_a}"  H={e_a:.2f}')

    print(f'  Real B: "{pb}"')
    ans_b = predict(pb)
    e_b, _ = get_entropy_and_top(pb)
    print(f'    -> "{ans_b}"  H={e_b:.2f}')

    print(f"  Contrastive (A - B):")
    contrastive_trajectory(pa, pb, layers=[16, 24, 28, 32],
                           label_a="A", label_b="B")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("5. ENTROPY COMPARISON — does the model 'know' it doesn't know?")
print("=" * 100)

all_prompts = [
    ("Nikola Tesla, born in Smiljan in 1856, invented the", "real-person"),
    ("Thomas Edison, born in Milan Ohio in 1847, invented the", "real-person"),
    ("Marie Curie, working in Paris in 1903, discovered that", "real-person"),
    ("Albert Einstein, born in Ulm in 1879, developed the", "real-person"),
    ("Ludvig von Vogelkirche, born in Prague in 1859, invented the", "fictional"),
    ("Heinrich Müller-Brandeis, a chemist in Vienna in 1903, discovered that", "fictional"),
    ("Professor Aino Järvikoski of Helsinki published her paper on", "fictional"),
    ("Tomáš Černý, born in Brno in 1872, designed the", "fictional"),
    ("The capital of France is", "well-known-fact"),
    ("The capital of Narnia is", "fictional-place"),
    ("The second largest city in Finland, Espoo, has a population of", "real-place"),
    ("The second largest suburb in Tampere, Kaksikanaa, has a population of", "fictional-place"),
    ("Mount Cook, the tallest peak in New Zealand, rises to", "real-place"),
    ("Mount Silverhorn, the tallest peak in New Zealand, rises to", "fictional-place"),
]

print(f"\n  {'Prompt':<65} {'Type':<15} {'H':>5} {'P(top1)':>7} {'Prediction'}")
for prompt, ptype in all_prompts:
    e, top = get_entropy_and_top(prompt, 3)
    ans = predict(prompt, 8)
    print(f"  {prompt:<65} {ptype:<15} {e:>5.2f} {top[0][1]:>7.3f} {ans[:30]}")

# Compare hidden state norms
print(f"\n  Hidden state norms at last position (L28):")
for prompt, ptype in all_prompts:
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    h28 = out.hidden_states[28][0, -1, :].float()
    norm = float(h28.norm())
    print(f"  {ptype:<15} {norm:>6.1f}  {prompt[:50]}")
    del out

torch.cuda.empty_cache()
print(f"\n{'='*100}")
print("DONE")
