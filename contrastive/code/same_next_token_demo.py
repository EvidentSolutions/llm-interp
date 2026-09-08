"""
Contrast design (method note): to surface mid-to-late-layer computation that a
position STORES for downstream emission (rather than its own next token),
choose contrasts whose read position predicts the SAME next token.

Read at the "France" position of:
  A: "The capital of France"    (France stores capital-of -> Paris)
  B: "The currency of France"   (France stores currency-of -> Euro)
Both predict the same immediate next token ("is"). Because that component is
shared it cancels in the difference, so the logit lens of h_A(France)-h_B(France)
should reveal the stored capital/currency computation, not the shared "is".

Controls:
  - raw single-prompt read at France (should be "is"-dominated for both)
  - a MISMATCHED-next-token pair for comparison:
      A': "The capital of France"   B': "France is a country in"
    where the read positions do NOT predict the same next token.

Phi-2, fp32.
Usage: .venv/Scripts/python.exe public/contrastive/code/same_next_token_demo.py
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
LAYERS = [8, 12, 16, 20, 24, 28, 32]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.lm_head.weight.detach().float()
    NL = m.config.num_hidden_layers

    def states_at(text, anchor="France"):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        toks = [tok.decode([t]) for t in ids]
        pos = max((i for i, t in enumerate(toks) if anchor.lower() in t.lower()),
                  default=len(ids) - 1)
        with torch.no_grad():
            out = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [out.hidden_states[L][0, pos] for L in range(NL + 1)], toks[pos]

    def rd(v, k=6):
        i = torch.topk(v @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)

    A, _ = states_at("The capital of France")
    B, _ = states_at("The currency of France")
    # mismatched-next-token control: France as subject, different continuation
    C, _ = states_at("France is a country in", anchor="France")

    out = {"raw_capital": {}, "raw_currency": {}, "diff_cap_minus_cur": {},
           "diff_cur_minus_cap": {}, "diff_mismatched": {}}
    print("Read at the 'France' position. Matched pair: capital vs currency of France.")
    print(f"\n{'L':>3} | {'raw capital-of':<20} | {'raw currency-of':<20} | "
          f"{'cap - cur':<24} | {'cur - cap':<24} | mismatched ctrl")
    for L in LAYERS:
        rc = rd(A[L]); ru = rd(B[L])
        dcap = rd(A[L] - B[L]); dcur = rd(B[L] - A[L]); dx = rd(A[L] - C[L])
        out["raw_capital"][L] = rc; out["raw_currency"][L] = ru
        out["diff_cap_minus_cur"][L] = dcap; out["diff_cur_minus_cap"][L] = dcur
        out["diff_mismatched"][L] = dx
        print(f"{L:>3} | {rc:<20} | {ru:<20} | {dcap:<24} | {dcur:<24} | {dx}")

    # generality: a second country, same relation contrast
    print("\nGenerality check: capital vs currency of Japan, read at 'Japan'.")
    AJ, _ = states_at("The capital of Japan", anchor="Japan")
    BJ, _ = states_at("The currency of Japan", anchor="Japan")
    out["japan_raw_capital"] = {}
    out["japan_raw_currency"] = {}
    out["japan_cap_minus_cur"] = {}
    out["japan_cur_minus_cap"] = {}
    print(f"{'L':>3} | {'raw cap':<16} | {'raw cur':<16} | {'cap - cur':<26} | cur - cap")
    for L in LAYERS:
        rjc = rd(AJ[L]); rju = rd(BJ[L])
        dcap = rd(AJ[L] - BJ[L]); dcur = rd(BJ[L] - AJ[L])
        out["japan_raw_capital"][L] = rjc
        out["japan_raw_currency"][L] = rju
        out["japan_cap_minus_cur"][L] = dcap
        out["japan_cur_minus_cap"][L] = dcur
        print(f"{L:>3} | {rjc:<16} | {rju:<16} | {dcap:<26} | {dcur}")

    # counterpoint: read the SAME contrast one token later, at "is", where the
    # two prompts now predict DIFFERENT next tokens (Tokyo vs yen). The
    # difference then surfaces those specific answers, not the relation.
    print("\nAnswer-site read: '...Japan is' (next tokens differ: Tokyo vs yen).")
    AI, _ = states_at("The capital of Japan is", anchor="is")
    BI, _ = states_at("The currency of Japan is", anchor="is")
    out["answersite_cap_minus_cur"] = {}
    out["answersite_cur_minus_cap"] = {}
    print(f"{'L':>3} | {'cap - cur (cities)':<34} | cur - cap (currencies)")
    for L in LAYERS:
        dcap = rd(AI[L] - BI[L]); dcur = rd(BI[L] - AI[L])
        out["answersite_cap_minus_cur"][L] = dcap
        out["answersite_cur_minus_cap"][L] = dcur
        print(f"{L:>3} | {dcap:<34} | {dcur}")

    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "same_next_token_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
