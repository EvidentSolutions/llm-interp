"""
Some/all triangulation WITH the denoising envelope (preamble + shared coda).

Prior run (some_all_triangulate_token.py) used BARE prompts read at the token
right after ", so" -- a prediction-divergence position, no register control.
The triangulated readout was register junk (Tool, femin, Cups, subscribe). Per
the denoising-envelope practice, structural/compositional contrasts that read as
junk are often undenoised, not genuinely illegible. Fixes:
  (1) PRE-PROMPT: a shared Pile-natural narrative preamble fixes register and
      lets attention sinks settle before the critical clause.
  (2) SHARED CODA: append an identical, target-neutral coda after the quantifier
      clause and READ AT A CODA-INTERIOR token (identical tokens across branches,
      differing only by the propagated some-vs-all commitment) -- escapes the
      next-token prediction-divergence confound. We KL-check reconvergence.

We print the triangulated readout (full + top20/bot20 token-subspace) per layer
for BARE vs ENVELOPED, so we can see whether the envelope surfaces a legible
scalar-implicature token where the bare read gave junk.

Usage: .venv/Scripts/python.exe contrastive/code/some_all_envelope_token.py
"""
import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = (model.lm_head.weight if hasattr(model, "lm_head")
       else model.embed_out.weight).detach().float().to(DEV)
K = 20

PRE0 = ("In the classroom that afternoon, the teacher sat down at her desk and "
        "looked carefully over the results of the recent test. ")
QUANTS = ["Some", "All", "None", "Most", "Few", "Many"]
STEM = " of the students passed the exam."
# shared, target-neutral coda; read at an INTERIOR token (not the final one).
CODA = " Looking at this outcome, she remarked that the situation was"
READ_BACK = 1   # read this many tokens back from the end (coda-interior)


def build(quant, enveloped):
    s = (PRE0 if enveloped else "") + quant + STEM + CODA
    return s


def run(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    pos = len(ids) - 1 - READ_BACK
    hs = [out.hidden_states[L][0, pos, :].float() for L in range(NL + 1)]
    logit_at_pos = out.logits[0, pos, :].float()
    return hs, logit_at_pos, ids, pos


def toks(v, k=8):
    i = torch.topk(v @ W_U.T, k).indices
    return [tok.decode([int(x)]).strip() for x in i]


def subspace_comp(vec, k=K):
    logits = vec @ W_U.T
    idx = torch.cat([torch.topk(logits, k).indices, torch.topk(-logits, k).indices])
    Q, _ = torch.linalg.qr(W_U[idx].T)
    return Q @ (Q.T @ vec)


def kl(p_logits, q_logits):
    lp = torch.log_softmax(p_logits, -1)
    lq = torch.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum())


for enveloped in (False, True):
    tag = "ENVELOPED (preamble + shared coda, coda-interior read)" if enveloped \
        else "BARE (no preamble, coda-interior read)"
    print("\n" + "#" * 98)
    print(f"#  {tag}")
    print("#" * 98)

    runs = {q: run(build(q, enveloped)) for q in QUANTS}
    tgt = "Some"
    hs_t = runs[tgt][0]
    read_tok = tok.decode([runs[tgt][2][runs[tgt][3]]]).strip()
    print(f"  target='{tgt}'  read token = {read_tok!r}  "
          f"(pos {runs[tgt][3]} of {len(runs[tgt][2])})")

    # KL reconvergence check at the read position: are the branch predictions
    # close there? (small KL => the coda-interior read is denoised, not sitting
    # on a live prediction-divergence).
    print("  KL(some || quant) at read position:")
    for q in QUANTS[1:]:
        print(f"    some||{q:<5}: {kl(runs[tgt][1], runs[q][1]):.3f}")

    baselines = [runs[q][0] for q in QUANTS[1:]]
    print(f"\n  {'L':>3} | {'||dh||/||h||':>11} | full readout // subspace readout")
    for L in range(4, NL + 1, 2):
        dh = hs_t[L] - torch.stack([b[L] for b in baselines]).mean(0)
        rel = float(dh.norm() / hs_t[L].norm())
        sc = subspace_comp(dh)
        print(f"  {L:>3} | {rel:>11.3f} | [{','.join(toks(dh,8))}]  //  "
              f"[{','.join(toks(sc,6))}]")

print("\nDONE")
