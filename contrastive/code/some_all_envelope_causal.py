"""
Some/all ENVELOPED causal test: does the readable (token-subspace) component of
the triangulated Delta-h do causal work -- now that the read is denoised?

Companion to some_all_envelope_token.py, which showed the enveloped triangulated
readout is LEGIBLE (remaining/remainder/still/incomplete/uneven/selectively --
the "some but not all -> a remainder remains" scalar signal), unlike the bare
", so" read (Tool/femin/Cups junk). Legibility != causality. Here we inject.

Setup: PRE0 preamble + "{Quant} of the students passed the exam." + shared CODA
ending "...the situation was". The FINAL token ("was") is shared across branches,
so its next-token prediction reflects the propagated some-vs-all commitment (not a
raw lexical divergence). We:
  - dh_tri[L] = h_some[final] - mean_baselines(h[final])   (triangulated)
  - dh_single[L] = h_some[final] - h_all[final]            (single pair)
  - inject full / top20-bot20 token-subspace into the "All" baseline at layer L
  - recover P(target): (p_inj - p_all) / (p_some - p_all) * 100
for a sweep of target tokens (argmax of some + scalar probes) and layers.

If the enveloped token-subspace recovers a real gap (unlike the bare -40%/~0),
the some/all distinction is token-shaped after denoising, not concept-space.

Usage: .venv/Scripts/python.exe contrastive/code/some_all_envelope_causal.py
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
CODA = " Looking at this outcome, she remarked that the situation was"
INJECT_QUANT = "All"  # inject the some-vs-baselines delta into the All prompt

PROBE = ["mixed", "uneven", "concerning", "disappointing", "complicated",
         "unclear", "incomplete", "unsatisfactory", "not", "still", "partly",
         "partially", "somewhat", "positive", "good", "excellent", "great",
         "encouraging", "perfect", "bad", "poor", "better", "improving",
         "worrying", "serious", "difficult", "manageable", "acceptable"]


def run(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    hs = [out.hidden_states[L][0, -1, :].float() for L in range(NL + 1)]
    return hs, out.logits[0, -1, :].float(), ids


def toks(v, k=8):
    i = torch.topk(v @ W_U.T, k).indices
    return [tok.decode([int(x)]).strip() for x in i]


def toks_logits(lg, k=8):
    i = torch.topk(lg, k).indices
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


runs = {q: run(PRE0 + q + STEM + CODA) for q in QUANTS}
hs_some = runs["Some"][0]
inj_ids = runs[INJECT_QUANT][2]
p_some = torch.softmax(runs["Some"][1], -1)
p_inj = torch.softmax(runs[INJECT_QUANT][1], -1)
baselines = [runs[q][0] for q in QUANTS[1:]]

print("\n" + "=" * 96)
print(f"ENVELOPED some/all causal test. inject into {INJECT_QUANT!r} baseline.")
print(f"some next-token top-8: {toks_logits(runs['Some'][1], 8)}")
print(f"{INJECT_QUANT} next-token top-8: {toks_logits(runs[INJECT_QUANT][1], 8)}")
print("=" * 96)

argmax_id = int(torch.argmax(p_some))
cands = [("argmax", argmax_id)]
for w in PROBE:
    wid = tok(" " + w, add_special_tokens=False)["input_ids"]
    if len(wid) == 1:
        cands.append((w, wid[0]))

print(f"\n  {'token':>13} {'gap':>7} | {'bestL':>5} {'triSub%':>8} {'triFull%':>9} "
      f"| {'singleSub%':>10}")
rows = []
for name, tid in cands:
    p_b = float(p_inj[tid]); gap = float(p_some[tid] - p_b)
    if abs(gap) < 2e-3:
        continue
    best = None
    for L in range(4, NL + 1, 2):
        dh_tri = hs_some[L] - torch.stack([b[L] for b in baselines]).mean(0)
        ss = inject(inj_ids, subspace_comp(dh_tri), L, tid, p_b, gap)
        ff = inject(inj_ids, dh_tri, L, tid, p_b, gap)
        if best is None or ss > best[1]:
            best = (L, ss, ff)
    Lb = best[0]
    dh_single = hs_some[Lb] - runs[INJECT_QUANT][0][Lb]
    sp = inject(inj_ids, subspace_comp(dh_single), Lb, tid, p_b, gap)
    rows.append((name, tid, gap, best[0], best[1], best[2], sp))

rows.sort(key=lambda r: -r[4])
for name, tid, gap, L, ss, ff, sp in rows:
    print(f"  {name:>13} {gap:>+7.3f} | {L:>5} {ss:>8.0f} {ff:>9.0f} | {sp:>10.0f}")

if rows:
    name, tid, gap, L, ss, ff, sp = rows[0]
    print("\n" + "=" * 96)
    print(f"BEST READABLE TARGET: {name!r} gap={gap:+.3f} triSub={ss:.0f}% "
          f"triFull={ff:.0f}% singleSub={sp:.0f}% @L{L}")
    dh = hs_some[L] - torch.stack([b[L] for b in baselines]).mean(0)
    print(f"  triangulated readout @L{L}: [{','.join(toks(dh,12))}]")
    print(f"  subspace readout    @L{L}: [{','.join(toks(subspace_comp(dh),12))}]")
    print("  layer profile  triSub% / triFull% / singleSub%:")
    p_b = float(p_inj[tid])
    for Lp in range(4, NL + 1, 2):
        dhp = hs_some[Lp] - torch.stack([b[Lp] for b in baselines]).mean(0)
        s = inject(inj_ids, subspace_comp(dhp), Lp, tid, p_b, gap)
        f = inject(inj_ids, dhp, Lp, tid, p_b, gap)
        dsp = hs_some[Lp] - runs[INJECT_QUANT][0][Lp]
        sg = inject(inj_ids, subspace_comp(dsp), Lp, tid, p_b, gap)
        print(f"    L{Lp:>2}: triSub={s:>+6.0f}%  triFull={f:>+6.0f}%  singleSub={sg:>+6.0f}%")

print("\nDONE")
