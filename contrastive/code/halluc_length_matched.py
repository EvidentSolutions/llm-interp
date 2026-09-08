"""
Length-matched control for the recall-vs-hallucination contrast (paper Sec 4).

The entity pairs in the main text are not token-length matched (e.g., "Nikola
Tesla, ..." is 11 tokens, "Ludvig von Vogelkirche, ..." is 18), so the read token
sits at a different index and rotary position structure need not cancel. This
control rebuilds the contrast with real and fictional prompts of *identical*
token length (same frame, name chosen to match the token count), reads at the
shared final token, and checks whether the finding holds: the real pole reads
factual associations, the fictional pole only name fragments.

For each real anchor we search a list of invented names for one whose full prompt
tokenizes to the same length as the real prompt, then read (real - fictional)
through W_U at the final token. Phi-2, fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/halluc_length_matched.py
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

# (real name, birth year, verb-frame, note on the real fact)
REAL = [
    ("Nikola Tesla", 1856, "invented the", "AC / Tesla coil"),
    ("Marie Curie", 1867, "discovered", "radioactivity / radium"),
    ("Albert Einstein", 1879, "developed the", "relativity"),
    ("Isaac Newton", 1643, "discovered", "gravity / calculus"),
]
# candidate invented names (plausibly non-existent), searched for a length match
FICT = ["Aldous Fenwick", "Elsa Voronova", "Magnus Thorne", "Cornelius Vale",
        "Ingrid Halvorsen", "Bartholomew Crane", "Lucius Ravensworth",
        "Silas Vandermeer", "Rowena Ashcroft", "Emeric Dulac", "Doreen Blackwood",
        "Halvard Kessler", "Petra Almqvist", "Osric Dunmore", "Alba Ferreira",
        "Tobias Winterhalter", "Greta Sundqvist", "Marius Delacroix"]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.lm_head.weight.detach().float()
    NL = m.config.num_hidden_layers

    def ntok(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def final_hidden(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            o = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [o.hidden_states[L][0, -1] for L in range(NL + 1)]

    def rd(v, k=4):
        i = torch.topk(v @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:12] for x in i)

    out = {}
    LAY = [24, 28]
    for name, year, verb, note in REAL:
        real_prompt = f"{name}, born in {year}, {verb}"
        rl = ntok(real_prompt)
        # find a fictional name giving the same total token length
        match = None
        for fn in FICT:
            fp = f"{fn}, born in {year}, {verb}"
            if ntok(fp) == rl:
                match = (fn, fp)
                break
        if match is None:
            print(f"[skip] no length match for {name!r} (len {rl})")
            continue
        fn, fict_prompt = match
        hr = final_hidden(real_prompt)
        hf = final_hidden(fict_prompt)
        print(f"\n=== {name} ({note})  vs  {fn}  |  both {rl} tokens ===")
        rec = {"real": real_prompt, "fictional": fict_prompt, "n_tokens": rl,
               "layers": {}}
        for L in LAY:
            d = hr[L] - hf[L]
            real_pole = rd(d)
            fict_pole = rd(-d)
            print(f"  L{L}: real pole [{real_pole}]   fictional pole [{fict_pole}]")
            rec["layers"][L] = {"real_pole": real_pole, "fictional_pole": fict_pole}
        out[name] = rec

    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "halluc_length_matched.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
