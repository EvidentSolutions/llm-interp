"""
Logit lens on individual attention heads — single input, not contrastive.
What does each head's V->O output contain for one prompt?

Usage: .venv/Scripts/python.exe contrastive/code/head_logit_lens.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading microsoft/phi-2...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NH = model.config.num_attention_heads
HD = model.config.hidden_size // NH
W_U = model.lm_head.weight.detach().float().cpu()


def tk(logits, k=8):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def head_logit_lens(text, L=24, label="", heads_to_show=None):
    if heads_to_show is None:
        heads_to_show = [0, 1, 14, 16, 7, 15, 21]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    captured = {}
    dense = model.model.layers[L].self_attn.dense

    def hook_fn(module, inp, out):
        captured[L] = inp[0].detach().float().cpu()

    handle = dense.register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    handle.remove()

    O_w = model.model.layers[L].self_attn.dense.weight.float().cpu()
    dense_in = captured[L][0, -1, :]

    h_full = out.hidden_states[L + 1][0, -1, :].float().cpu()
    h_logits = h_full @ W_U.T

    print(f"  {label}")
    print(f"    Full logit lens (L{L+1}): [{tk(h_logits)}]")

    for h_idx in heads_to_show:
        dh = dense_in[h_idx * HD : (h_idx + 1) * HD]
        O_h = O_w[:, h_idx * HD : (h_idx + 1) * HD]
        contribution = dh @ O_h.T
        logits = contribution @ W_U.T
        n = float(contribution.norm())
        print(f"      H{h_idx:>2} (norm={n:>5.1f}): [{tk(logits)}]")

    del out
    torch.cuda.empty_cache()


print("=" * 100)
print("LOGIT LENS on individual heads at L24 — single input")
print("=" * 100)

# IOI: both inputs separately
prompts = [
    ("John and Mary went to the store. John gave a book to",
     "IOI-A: John...gave to [expect Mary]"),
    ("Mary and John went to the store. Mary gave a book to",
     "IOI-B: Mary...gave to [expect John]"),
    ("John and Tom went to the store. John gave a book to",
     "MM-A: John...gave to [expect Tom]"),
    ("Tom and John went to the store. Tom gave a book to",
     "MM-B: Tom...gave to [expect John]"),
    ("A short man and a tall man went to the store. The short man gave a book to",
     "DESC-A: short...gave to [expect tall]"),
    ("A tall man and a short man went to the store. The tall man gave a book to",
     "DESC-B: tall...gave to [expect short]"),
    ("The capital of France is",
     "Non-IOI: France"),
    ("The hot dog was",
     "Non-IOI: hot dog"),
    ("The cold dog was",
     "Non-IOI: cold dog"),
]

for text, label in prompts:
    print()
    head_logit_lens(text, L=24, label=label)

# Now compare contrastive vs individual logit lens for IOI
print()
print("=" * 100)
print("COMPARISON: logit lens A, logit lens B, contrastive A-B")
print("=" * 100)

pairs = [
    ("John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to",
     "John/Mary"),
    ("John and Tom went to the store. John gave a book to",
     "Tom and John went to the store. Tom gave a book to",
     "John/Tom"),
    ("A short man and a tall man went to the store. The short man gave a book to",
     "A tall man and a short man went to the store. The tall man gave a book to",
     "short/tall"),
]

for pa, pb, label in pairs:
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    captured_a = {}
    captured_b = {}

    L = 24
    dense = model.model.layers[L].self_attn.dense

    def make_hook(store):
        def hook_fn(module, inp, out):
            store[L] = inp[0].detach().float().cpu()
        return hook_fn

    handle = dense.register_forward_hook(make_hook(captured_a))
    with torch.no_grad():
        model(torch.tensor([ids_a], device=DEV))
    handle.remove()

    handle = dense.register_forward_hook(make_hook(captured_b))
    with torch.no_grad():
        model(torch.tensor([ids_b], device=DEV))
    handle.remove()

    O_w = model.model.layers[L].self_attn.dense.weight.float().cpu()

    print(f"\n  --- {label} at L24 ---")
    for h_idx in [1, 14, 16]:
        a_h = captured_a[L][0, -1, h_idx * HD : (h_idx + 1) * HD]
        b_h = captured_b[L][0, -1, h_idx * HD : (h_idx + 1) * HD]
        O_h = O_w[:, h_idx * HD : (h_idx + 1) * HD]

        cont_a = a_h @ O_h.T
        cont_b = b_h @ O_h.T
        cont_diff = (a_h - b_h) @ O_h.T

        log_a = cont_a @ W_U.T
        log_b = cont_b @ W_U.T
        log_d = cont_diff @ W_U.T

        print(f"    H{h_idx:>2}:")
        print(f"      A alone:     [{tk(log_a)}]")
        print(f"      B alone:     [{tk(log_b)}]")
        print(f"      A - B:       [{tk(log_d)}]")

    torch.cuda.empty_cache()

print(f"\n{'='*100}")
print("DONE")
