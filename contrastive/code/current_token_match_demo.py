"""
Current-token matching (paper Sec 2.4), via a 2x2.

Axes: CONTEXT (reading a novel vs watching a lecture) x CURRENT TOKEN (the
synonyms boring vs dull at the final word). We read the SAME context difference
(novel - lecture) four ways and report BOTH poles (+ = novel/reading,
- = lecture/watching) across layers.

Matched current token (both sides end in the same word): the context reads
cleanly on both poles from the early-middle layers. Mismatched current token (a
boring/dull synonym swap): the difference also carries the synonym, whose
suffix morphology (ly/ed/ers/ing) dominates the readout; the context only bleeds
through weakly at the final layers (L30-32), still mixed with fragments. Phi-2,
fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/current_token_match_demo.py
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
P = "I spent the whole afternoon %s and found it extremely %s"
LAYERS = [4, 12, 20, 28, 30, 32]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W = m.lm_head.weight.detach().float()
    NL = m.config.num_hidden_layers

    def hs(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            o = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [o.hidden_states[L][0, -1] for L in range(NL + 1)]

    def pole(v, k=4, neg=False):
        i = torch.topk((-v if neg else v) @ W.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:9] for x in i)

    nb = hs(P % ("reading the novel", "boring"))
    nd = hs(P % ("reading the novel", "dull"))
    lb = hs(P % ("watching the lecture", "boring"))
    ld = hs(P % ("watching the lecture", "dull"))

    pairs = {
        "matched_dull":   (nd, ld),   # both end "dull"
        "matched_boring": (nb, lb),   # both end "boring"
        "mismatch_nb_ld": (nb, ld),   # novel/boring - lecture/dull
        "mismatch_nd_lb": (nd, lb),   # novel/dull - lecture/boring
    }
    out = {}
    for name, (a, b) in pairs.items():
        print(f"\n== {name} ==  (+ novel/reading, - lecture/watching)")
        rows = {}
        for L in LAYERS:
            d = a[L] - b[L]
            rec = dict(norm=round(float(d.norm()), 1),
                       pos=pole(d), neg=pole(d, neg=True))
            rows[L] = rec
            print(f"  L{L:2}: norm={rec['norm']:5}  +[{rec['pos']}]   -[{rec['neg']}]")
        out[name] = rows

    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "current_token_match_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
