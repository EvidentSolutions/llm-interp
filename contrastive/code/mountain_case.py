"""
The mountain case (paper Sec 4, recall vs hallucination).

Entropy alone cannot separate confident recall from confident hallucination:
the real Mount Cook and the fictional Mount Silverhorn predict a height at
nearly the same entropy. The contrastive projection separates them: the real
mountains' pole reads geographic knowledge, the fictional pole reads only
generic number-range priors.

For each mountain: next-token entropy and the greedy height continuation.
Contrastive poles (real minus Silverhorn, and Silverhorn minus Everest) read
through W_U at the final token. Phi-2, fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/mountain_case.py
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
PROMPTS = {
    "Everest": "Mount Everest, the tallest peak in the Himalayas, rises to",
    "Cook": "Mount Cook, the tallest peak in New Zealand's Southern Alps, rises to",
    "Silverhorn": "Mount Silverhorn, the tallest peak in New Zealand's "
                  "Southern Alps, rises to",
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.lm_head.weight.detach().float()
    NL = m.config.num_hidden_layers

    def analyse(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
            gen = m.generate(torch.tensor([ids], device=DEV), max_new_tokens=6,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        p = torch.softmax(out.logits[0, -1].float(), -1)
        H = -float((p * torch.log(p + 1e-12)).sum())
        greedy = tok.decode(gen[0, len(ids):]).strip()
        hs = [out.hidden_states[L][0, -1] for L in range(NL + 1)]
        return H, greedy, hs

    R = {k: analyse(v) for k, v in PROMPTS.items()}

    def pole(v, k=5, neg=False):
        i = torch.topk((-v if neg else v) @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)

    out = {}
    for k in PROMPTS:
        out[k] = dict(entropy=round(R[k][0], 2), greedy=R[k][1])
    # real poles vs Silverhorn (geographic); Silverhorn pole vs Everest (numbers)
    S = R["Silverhorn"][2]
    out["Everest"]["pole_L24"] = pole(R["Everest"][2][24] - S[24])
    out["Cook"]["pole_L24"] = pole(R["Cook"][2][24] - S[24])
    out["Silverhorn"]["pole_L28"] = pole(S[28] - R["Everest"][2][28])

    print(f"{'mountain':<11} {'H':>5}  greedy continuation")
    for k in PROMPTS:
        print(f"{k:<11} {out[k]['entropy']:>5}  {out[k]['greedy']!r}")
    print("\ncontrastive poles:")
    print(f"  Everest (real, L24):    {out['Everest']['pole_L24']}")
    print(f"  Cook (real, L24):       {out['Cook']['pole_L24']}")
    print(f"  Silverhorn (fict, L28): {out['Silverhorn']['pole_L28']}")

    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "mountain_case.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
