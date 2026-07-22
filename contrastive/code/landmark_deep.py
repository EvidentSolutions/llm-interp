"""
Deep verification of landmark replications.
1. IOI: more cases + Pythia comparison
2. Factual vs imaginary
3. Successor: Saturday discontinuity + circular representation
4. Per-head diffs to identify responsible heads

Usage: .venv/Scripts/python.exe contrastive/code/landmark_deep.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tk(logits, tok_obj, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(
        tok_obj.decode([int(i[j])]).strip()[:12] for j in range(k)
    )


def run_contrastive(model, tok_obj, pa, pb, W_U, NL, layers=None):
    if layers is None:
        layers = [16, 24, 28, 32]
    ids_a = tok_obj(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok_obj(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV),
            output_hidden_states=True,
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV),
            output_hidden_states=True,
        )
    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) [{tk(ld, tok_obj)}]")
    result = (out_a, out_b, ids_a, ids_b)
    return result


def predict(model, tok_obj, text):
    ids = tok_obj(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tok_obj.eos_token_id,
        )
    return tok_obj.decode(gen[0][len(ids) :]).strip()[:20]


# ============================================================
# SECTION 1: IOI — more cases + Pythia
# ============================================================
def run_ioi(model, tok_obj, W_U, NL, model_name):
    print(f"\n  Model: {model_name}")
    ioi_cases = [
        "{}0 and {}1 went to the store. {}0 gave a book to",
        "{}0 and {}1 had lunch together. {}0 passed the salt to",
        "{}0 and {}1 worked on a project. {}0 sent an email to",
    ]
    name_pairs = [
        ("John", "Mary"),
        ("Alice", "Bob"),
        ("Dan", "Eve"),
        ("Tom", "Sarah"),
        ("Mike", "Lisa"),
    ]

    correct = 0
    total = 0
    for template in ioi_cases:
        for n1, n2 in name_pairs:
            pa = template.replace("{}0", n1).replace("{}1", n2)
            pb = template.replace("{}0", n2).replace("{}1", n1)
            ans_a = predict(model, tok_obj, pa)
            ans_b = predict(model, tok_obj, pb)
            ok_a = n2.lower() in ans_a.lower()
            ok_b = n1.lower() in ans_b.lower()
            total += 2
            correct += ok_a + ok_b
            if not ok_a or not ok_b:
                print(f"    FAIL: A=\"{pa[-40:]}\" -> \"{ans_a}\"" +
                      ("" if ok_a else f" (expected {n2})"))
                print(f"          B=\"{pb[-40:]}\" -> \"{ans_b}\"" +
                      ("" if ok_b else f" (expected {n1})"))

    print(f"  IOI accuracy: {correct}/{total} ({100*correct/total:.0f}%)")


print("=" * 100)
print("1. IOI — Extended + Pythia comparison")
print("=" * 100)

# Phi-2
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()

run_ioi(model, tok, W_U, NL, "Phi-2")
del model
torch.cuda.empty_cache()

# Pythia-410M
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-410m-deduped",
    dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

run_ioi(model, tok, model.embed_out.weight.detach(),
        len(model.gpt_neox.layers), "Pythia-410M")
del model
torch.cuda.empty_cache()

# Pythia-1.4B
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-1.4b-deduped",
    dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

run_ioi(model, tok, model.embed_out.weight.detach(),
        len(model.gpt_neox.layers), "Pythia-1.4B")
del model
torch.cuda.empty_cache()

# ============================================================
# SECTION 2: Factual vs Imaginary
# ============================================================
print(f"\n{'='*100}")
print("2. FACTUAL vs IMAGINARY")
print("=" * 100)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()

fact_imag = [
    ("The capital of France is",
     "The capital of Narnia is"),
    ("The Eiffel Tower is located in",
     "The Crystal Palace of Zoria is located in"),
    ("Shakespeare wrote",
     "Zaldor the Wise wrote"),
    ("Einstein developed the theory of",
     "Glorb developed the theory of"),
    ("The capital of Germany is",
     "The capital of Gondor is"),
]

for pa, pb in fact_imag:
    ans_a = predict(model, tok, pa)
    ans_b = predict(model, tok, pb)
    print(f'\n  Factual:   "{pa}" -> "{ans_a}"')
    print(f'  Imaginary: "{pb}" -> "{ans_b}"')
    run_contrastive(model, tok, pa, pb, W_U, NL)

del model
torch.cuda.empty_cache()

# ============================================================
# SECTION 3: Successor — Saturday discontinuity + circular
# ============================================================
print(f"\n{'='*100}")
print("3. SUCCESSOR — Saturday/Sunday + circular representation")
print("=" * 100)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()

# All days
days = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

print("\n  Successor predictions:")
for day in days:
    prompt = f"After {day} comes"
    ans = predict(model, tok, prompt)
    print(f'    "{prompt}" -> "{ans}"')

# Pairwise contrasts and hidden states for circular analysis
print("\n  Hidden states at L28 for all days:")
day_states = {}
for day in days:
    prompt = f"After {day} comes"
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV),
            output_hidden_states=True,
        )
    day_states[day] = (
        out.hidden_states[28][0, -1, :].float().cpu().detach().clone()
    )
    del out

# Pairwise cosine matrix
print("\n  Pairwise cosine (L28):")
print("          ", end="")
for d in days:
    print(f" {d[:3]:>5}", end="")
print()
for d1 in days:
    print(f"  {d1[:3]:>5}:", end="")
    for d2 in days:
        cos = float(
            torch.nn.functional.cosine_similarity(
                day_states[d1].unsqueeze(0),
                day_states[d2].unsqueeze(0),
            )
        )
        print(f" {cos:.3f}", end="")
    print()

# Step vectors: Mon->Tue, Tue->Wed, ... Sat->Sun, Sun->Mon
print("\n  Consecutive step cos (circular?):")
for i in range(len(days)):
    d1 = days[i]
    d2 = days[(i + 1) % len(days)]
    d3 = days[(i + 2) % len(days)]
    step1 = day_states[d2] - day_states[d1]
    step2 = day_states[d3] - day_states[d2]
    cos = float(
        torch.nn.functional.cosine_similarity(
            step1.unsqueeze(0), step2.unsqueeze(0)
        )
    )
    print(f"    {d1[:3]}->{d2[:3]} vs {d2[:3]}->{d3[:3]}: cos={cos:>+.3f}")

# SVD of centered states — dimensionality
mat = torch.stack([day_states[d] for d in days])
mat_c = mat - mat.mean(dim=0)
U, S, V = torch.svd(mat_c)
total_var = S.pow(2).sum()
print("\n  SVD of centered day states:")
for i in range(min(6, len(S))):
    print(f"    S{i+1} = {S[i]:.1f} ({S[i]**2/total_var:.1%})")

# 2D projection — do days form a circle?
V2 = V[:, :2]
proj_2d = mat_c @ V2
print("\n  2D projection (do days form a circle?):")
for i, day in enumerate(days):
    x, y = float(proj_2d[i, 0]), float(proj_2d[i, 1])
    angle = float(
        torch.atan2(torch.tensor(y), torch.tensor(x)) * 180 / 3.14159
    )
    norm = float(proj_2d[i].norm())
    print(f"    {day:>10}: ({x:>+7.1f}, {y:>+7.1f})"
          f"  r={norm:.1f}  angle={angle:>+7.1f}deg")

# Consecutive angular steps
print("\n  Angular steps in 2D:")
for i in range(len(days)):
    a1 = float(torch.atan2(proj_2d[i, 1], proj_2d[i, 0])
               * 180 / 3.14159)
    a2 = float(torch.atan2(proj_2d[(i+1) % len(days), 1],
                            proj_2d[(i+1) % len(days), 0])
               * 180 / 3.14159)
    step = a2 - a1
    if step > 180:
        step -= 360
    if step < -180:
        step += 360
    print(f"    {days[i]:>10} -> {days[(i+1)%len(days)]:<10}:"
          f" {step:>+7.1f}deg")

# Same for months
print("\n  --- MONTHS ---")
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November",
          "December"]

print("  Predictions:")
for month in months:
    prompt = f"After {month} comes"
    ans = predict(model, tok, prompt)
    print(f'    "{prompt}" -> "{ans}"')

month_states = {}
for month in months:
    prompt = f"After {month} comes"
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV),
            output_hidden_states=True,
        )
    month_states[month] = (
        out.hidden_states[28][0, -1, :].float().cpu().detach().clone()
    )
    del out

# SVD and 2D projection for months
mat_m = torch.stack([month_states[m] for m in months])
mat_mc = mat_m - mat_m.mean(dim=0)
U_m, S_m, V_m = torch.svd(mat_mc)
total_var_m = S_m.pow(2).sum()
print("\n  SVD of centered month states:")
for i in range(min(6, len(S_m))):
    print(f"    S{i+1} = {S_m[i]:.1f} ({S_m[i]**2/total_var_m:.1%})")

V2_m = V_m[:, :2]
proj_2d_m = mat_mc @ V2_m
print("\n  2D projection — months:")
for i, month in enumerate(months):
    x, y = float(proj_2d_m[i, 0]), float(proj_2d_m[i, 1])
    angle = float(
        torch.atan2(torch.tensor(y), torch.tensor(x)) * 180 / 3.14159
    )
    norm = float(proj_2d_m[i].norm())
    print(f"    {month:>10}: ({x:>+7.1f}, {y:>+7.1f})"
          f"  r={norm:.1f}  angle={angle:>+7.1f}deg")

# Angular steps for months
print("\n  Angular steps:")
for i in range(len(months)):
    a1 = float(torch.atan2(proj_2d_m[i, 1], proj_2d_m[i, 0])
               * 180 / 3.14159)
    a2 = float(torch.atan2(proj_2d_m[(i+1) % len(months), 1],
                            proj_2d_m[(i+1) % len(months), 0])
               * 180 / 3.14159)
    step = a2 - a1
    if step > 180:
        step -= 360
    if step < -180:
        step += 360
    print(f"    {months[i]:>10} -> {months[(i+1)%len(months)]:<10}:"
          f" {step:>+7.1f}deg")

del model
torch.cuda.empty_cache()

print(f"\n{'='*100}")
print("DONE")
