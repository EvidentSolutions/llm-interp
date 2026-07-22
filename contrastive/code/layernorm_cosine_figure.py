"""
Appendix figure for the LayerNorm-bypass justification (paper Section 2.1).

For each of the six poster cases, sweep every layer and compute the cosine
similarity between two contrastive-projection variants at the final position:
  A) Raw:    (h_c - h_k) @ W_U^T             (what the paper does; bypasses LN_f)
  B) PostLN: (LN_f(h_c) - LN_f(h_k)) @ W_U^T (each state normalized first)

High cosine means the LN_f bypass does not change the token ranking the
projection reads. Saves a vector PDF (one line per case + mean) and prints a
per-layer table.

Usage: .venv/Scripts/python.exe contrastive/code/layernorm_cosine_figure.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
OUT = "bbxnlp_paper/ln_cosine.pdf"

CASES = [
    ("hot dog", "The hot dog was", "The cold dog was"),
    ("IOI", "John and Mary went to the store. John gave a book to",
            "Mary and John went to the store. Mary gave a book to"),
    ("Eiffel", "The Eiffel Tower is located in", "The Colosseum is located in"),
    ("Successor", "After Monday comes", "After Tuesday comes"),
    ("Truth", "Paris is the capital of France. This is",
              "Paris is the capital of Germany. This is"),
    ("Negation", "The dog ran quickly through the park",
                 "The dog did not run quickly through the park"),
]

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach().float()
LN = model.model.final_layernorm


def toks(p):
    return tok(p, add_special_tokens=False)["input_ids"]


def cos_curve(pa, pb):
    with torch.no_grad():
        oa = model(torch.tensor([toks(pa)], device=DEV), output_hidden_states=True)
        ob = model(torch.tensor([toks(pb)], device=DEV), output_hidden_states=True)
    curve = []
    for L in range(NL + 1):
        ha = oa.hidden_states[L][0, -1].float()
        hb = ob.hidden_states[L][0, -1].float()
        lA = (ha - hb) @ W_U.T
        ha_ln = LN(ha.unsqueeze(0).half()).float().squeeze(0)
        hb_ln = LN(hb.unsqueeze(0).half()).float().squeeze(0)
        lB = (ha_ln - hb_ln) @ W_U.T
        curve.append(float(torch.nn.functional.cosine_similarity(
            lA.unsqueeze(0), lB.unsqueeze(0))))
    del oa, ob
    torch.cuda.empty_cache()
    return curve


curves = {}
for name, pa, pb in CASES:
    curves[name] = cos_curve(pa, pb)
    print(f"{name:10s}: L28={curves[name][28]:.4f}  L32={curves[name][32]:.4f}")

# L0 is degenerate (the read-position token is identical across the pair, so
# the differenced state is zero); start the plot at L1.
Ls = list(range(1, NL + 1))
mean = [sum(curves[n][L] for n, _, _ in CASES) / len(CASES) for L in Ls]

print("\n  L   " + "  ".join(f"{n[:7]:>7}" for n, _, _ in CASES) + "    mean")
for i, L in enumerate(Ls):
    row = "  ".join(f"{curves[n][L]:>7.3f}" for n, _, _ in CASES)
    print(f"  {L:>2}  {row}  {mean[i]:>7.3f}")

# ---- figure ----
plt.figure(figsize=(5.2, 3.2))
for name, _, _ in CASES:
    plt.plot(Ls, curves[name][1:], lw=1.0, alpha=0.55, label=name)
plt.plot(Ls, mean, lw=2.4, color="black", label="mean")
plt.axhline(0.98, ls="--", lw=0.8, color="gray")
plt.text(1.5, 0.983, "0.98", color="gray", fontsize=7, va="bottom")
plt.ylim(0.6, 1.01)
plt.xlim(1, NL)
plt.xlabel("Layer")
plt.ylabel(r"$\cos$(raw diff, post-LN diff) at $W_U$")
plt.legend(fontsize=6.5, ncol=2, loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT)
print(f"\nSaved {OUT}")
