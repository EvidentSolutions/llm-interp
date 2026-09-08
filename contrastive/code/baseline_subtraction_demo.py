"""
Baseline subtraction (paper Sec 2.5): read a single prompt against the mean
residual of many random text snippets of the same token length.

Produces three logit-lens readings at the final position, per layer:
  raw        : logit lens of the target residual
  average    : logit lens of the MEAN baseline residual (what is subtracted)
  subtracted : logit lens of (target - mean baseline)
Phi-2, fp32. Target = the hot-dog prompt inside the Sec 2.4 preamble.

Usage: .venv/Scripts/python.exe public/contrastive/code/baseline_subtraction_demo.py
"""
import sys, os, json, random
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
TARGET = os.environ.get("TGT","At the ballpark the hot dog was")
N_BASE = 100
LAYERS = [6, 10, 16, 20, 24, 28]
random.seed(0)


import re


def is_prose(s):
    s = s.strip()
    return (len(s) > 40 and s[:1].isupper()
            and sum(c.isalpha() or c in " ,.'-" for c in s) / len(s) > 0.95
            and not any(ch in s for ch in "{}[]<>|=@#\\/*_~`"))


def get_snippets(tok, n, length, prose=True):
    """n text windows of `length` tokens. prose=True -> style-matched English
    prose, taken sentence-initially (the target's position); else raw Pile."""
    texts = []
    pool = None
    src = [("Salesforce/wikitext", "text")] if prose else \
          [("monology/pile-uncopyrighted", "text")]
    for name, field in src:
        try:
            from datasets import load_dataset
            kw = {"name": "wikitext-103-raw-v1"} if "wikitext" in name else {}
            ds = load_dataset(name, split="train", streaming=True, **kw)
            pool = []
            for x in ds:
                t = x.get(field, "")
                for sent in re.split(r"(?<=[.!?])\s+", t):
                    if prose and not is_prose(sent):
                        continue
                    pool.append(sent.strip())
                if len(pool) >= 8000:
                    break
            print(f"  (corpus: {name}, prose={prose}, {len(pool)} sentences)")
            break
        except Exception as e:
            print(f"  ({name} failed: {str(e)[:60]})")
            pool = None
    if not pool:
        print("  (falling back to built-in generic pool)")
        pool = [
            "The weather today is quite pleasant and calm across the region.",
            "She walked to the market to buy some fresh vegetables and bread.",
            "Scientists have long studied the behaviour of distant galaxies.",
            "He opened the door slowly and looked around the empty room.",
            "The committee will meet next week to discuss the annual budget.",
            "Children played in the park while their parents talked nearby.",
            "The new policy is expected to affect thousands of small businesses.",
            "After dinner they sat by the fire and told old family stories.",
            "The river flowed gently past the town toward the distant sea.",
            "Engineers tested the bridge carefully before opening it to traffic.",
        ] * 40
    random.shuffle(pool)
    for t in pool:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= length:
            # sentence-initial window: matches the target's position and style
            texts.append(ids[:length])
        if len(texts) >= n:
            break
    return texts


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.lm_head.weight.detach().float()
    NL = m.config.num_hidden_layers

    tgt_ids = tok(TARGET, add_special_tokens=False)["input_ids"]
    L = len(tgt_ids)
    print(f"Target: {TARGET!r}  ({L} tokens)")

    def final_states(ids):
        with torch.no_grad():
            out = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [out.hidden_states[k][0, -1] for k in range(NL + 1)]

    tgt = final_states(tgt_ids)

    def mean_baseline(prose):
        snippets = get_snippets(tok, N_BASE, L, prose=prose)
        acc = [torch.zeros_like(tgt[0]) for _ in range(NL + 1)]
        for s in snippets:
            fs = final_states(s)
            for k in range(NL + 1):
                acc[k] += fs[k]
        return [a / len(snippets) for a in acc]

    prose_base = mean_baseline(True)
    raw_base = mean_baseline(False)

    def rd(v, k=4):
        i = torch.topk(v @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)

    out = {}
    print(f"\n{'L':>3} | {'raw target':<22} | {'PROSE avg':<20} | {'PROSE-subtracted':<26} | raw-Pile-subtracted")
    for k in LAYERS:
        raw = rd(tgt[k]); pavg = rd(prose_base[k])
        psub = rd(tgt[k] - prose_base[k]); rsub = rd(tgt[k] - raw_base[k])
        print(f"{k:>3} | {raw:<22} | {pavg:<20} | {psub:<26} | {rsub}")
        out[k] = dict(raw=raw, prose_avg=pavg, prose_subtracted=psub,
                      pile_subtracted=rsub)

    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "baseline_subtraction_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
