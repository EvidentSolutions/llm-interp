"""
Focused reproduction of the paper's S3.5 triangulation-recovery claim
(caught cold: single token-subspace 1% -> multi 77%; some/all -40% -> 101%).

The claim rests on one comparison that must be APPLES-TO-APPLES:
    single-pair token-subspace recovery   vs   triangulated token-subspace recovery
both = project dh onto QR(W_U rows of its own top-K/bottom-K read tokens),
inject only that component, recover the prediction gap.

We sweep EVERY layer, on the exact bare prompts, and also print:
  - full single-dh recovery (unrestricted)   [S3.5 says 69% for caught cold]
  - full tri-dh  recovery (unrestricted)
so we can see whether any "improvement" is triangulation moving causal content
INTO the readable tokens, or merely restricted(single) vs unrestricted(tri).

Usage: .venv/Scripts/python.exe contrastive/code/triangulation_recovery_repro.py
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


def hidden(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return [out.hidden_states[L][0, -1, :].float() for L in range(NL + 1)], \
        out.logits[0, -1].float(), ids


def toks(v, k=6):
    i = torch.topk(v @ W_U.T, k).indices
    return [tok.decode([int(x)]).strip() for x in i]


def subspace_comp(vec, k=K):
    logits = vec @ W_U.T
    idx = torch.cat([torch.topk(logits, k).indices, torch.topk(-logits, k).indices])
    Q, _ = torch.linalg.qr(W_U[idx].T)
    return Q @ (Q.T @ vec)


def inject(base_ids, delta, L, tid, p_b, gap):
    done = [False]

    def hook(m, i, o):
        if done[0]:
            return o
        done[0] = True
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] += delta.half(); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] += delta.half(); return h
    handle = (model.model.layers[L] if L < NL
              else model.model.final_layernorm).register_forward_hook(hook)
    with torch.no_grad():
        out = model(torch.tensor([base_ids], device=DEV))
    handle.remove()
    p = float(torch.softmax(out.logits[0, -1].float(), -1)[tid])
    return (p - p_b) / gap * 100 if gap else 0.0


CASES = [
    dict(name="caught_cold",
         target="She caught a cold",
         baselines=["She caught a fish", "She caught a ball", "She caught a bus",
                    "She caught a thief", "She caught a glimpse"],
         # sense token to recover (may not be argmax of the bare prompt)
         probe=["doctor", "sick", "ill", "flu", "bed", "rest"]),
    dict(name="some_all",
         target="Some of the students passed the exam, so",
         baselines=["All of the students passed the exam, so",
                    "Most of the students passed the exam, so",
                    "Few of the students passed the exam, so",
                    "None of the students passed the exam, so",
                    "Many of the students passed the exam, so"],
         probe=["others", "the", "some", "not"]),
]

for c in CASES:
    hs_t, log_t, _ = hidden(c["target"])
    bases = [hidden(b) for b in c["baselines"]]
    b0_ids = bases[0][2]
    p_t = torch.softmax(log_t, -1)
    p_b0 = torch.softmax(bases[0][1], -1)

    # choose target token two ways: (a) argmax of target, (b) best probe by gap
    argmax_id = int(torch.argmax(p_t))
    probe_ids = []
    for w in c["probe"]:
        wid = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(wid) == 1:
            probe_ids.append(wid[0])
    probe_id = max(probe_ids, key=lambda t: float(p_t[t] - p_b0[t])) if probe_ids else argmax_id

    print("\n" + "=" * 92)
    print(f"CASE {c['name']}   target='{c['target']}'")
    print(f"  argmax next token = '{tok.decode([argmax_id]).strip()}' "
          f"(gap {float(p_t[argmax_id]-p_b0[argmax_id]):+.3f})")
    print(f"  best probe token  = '{tok.decode([probe_id]).strip()}' "
          f"(gap {float(p_t[probe_id]-p_b0[probe_id]):+.3f})")
    print("=" * 92)

    for label, tid in [("argmax", argmax_id), ("probe", probe_id)]:
        p_b = float(p_b0[tid]); gap = float(p_t[tid] - p_b)
        if abs(gap) < 1e-4:
            print(f"\n  [{label} '{tok.decode([tid]).strip()}'] gap~0, skip")
            continue
        print(f"\n  [{label} target '{tok.decode([tid]).strip()}'  gap={gap:+.3f}]")
        print(f"  {'L':>3} {'single_full':>11} {'single_sub':>10} "
              f"{'tri_full':>9} {'tri_sub':>8}   single_read / tri_read")
        for L in range(4, NL + 1, 2):
            dh_s = hs_t[L] - bases[0][0][L]
            dh_t = hs_t[L] - torch.stack([b[0][L] for b in bases]).mean(0)
            sf = inject(b0_ids, dh_s, L, tid, p_b, gap)
            ss = inject(b0_ids, subspace_comp(dh_s), L, tid, p_b, gap)
            tf = inject(b0_ids, dh_t, L, tid, p_b, gap)
            ts = inject(b0_ids, subspace_comp(dh_t), L, tid, p_b, gap)
            sr = ",".join(toks(dh_s, 3))
            tr = ",".join(toks(dh_t, 3))
            flag = "  <<" if (ts - ss > 25 and ts > 40) else ""
            print(f"  {L:>3} {sf:>10.0f}% {ss:>9.0f}% {tf:>8.0f}% {ts:>7.0f}%   "
                  f"[{sr}] / [{tr}]{flag}")

print("\nDONE")
