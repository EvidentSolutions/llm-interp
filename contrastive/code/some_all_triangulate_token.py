"""
Some/all triangulation: is there a READABLE token whose direction does causal work?

Background: single-pair (some vs all), the top-20/bottom-20 W_U token-subspace
component of dh recovers ~0 of the argmax gap (token_superposition_causal.py gave
-40%). deep_decompose.py's "101%" is the rank-1 SHARED-AXIS recovery, NOT a token-
subspace projection -- apples-to-oranges. Here we ask the honest apples-to-apples
question: after TRIANGULATING against several quantifier baselines, does the
token-subspace (readable) component of the triangulated dh recover any gap -- and
for WHICH target token? The argmax of "...passed the exam, so" is a function word;
the scalar-implicature content ("some but not all") may live on a different token
(not/others/failed/rest). We sweep candidate targets x layers, injecting BOTH the
full triangulated dh and its token-subspace projection, and print the read tokens.

Inject site: the "All..." baseline (strongest contrast). Recovery = (p_inj - p_base)
/ (p_target - p_base) * 100, per candidate target token.

Usage: .venv/Scripts/python.exe contrastive/code/some_all_triangulate_token.py
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

TARGET = "Some of the students passed the exam, so"
BASELINES = [
    "All of the students passed the exam, so",
    "None of the students passed the exam, so",
    "Most of the students passed the exam, so",
    "Few of the students passed the exam, so",
    "Many of the students passed the exam, so",
]
INJECT_BASE = BASELINES[0]  # "All ..." -- strongest contrast

# candidate target tokens: argmax is added dynamically; these probe the
# scalar-implicature continuation "some (but not all) passed, so <...>"
PROBE_WORDS = ["not", "others", "the", "they", "failed", "some", "Some", "many",
               "most", "everyone", "nobody", "while", "it", "there", "rest",
               "he", "she", "we", "you", "those", "half", "only", "still"]


def hidden(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return [out.hidden_states[L][0, -1, :].float() for L in range(NL + 1)], \
        out.logits[0, -1].float(), ids


def toks(v, k=8):
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


hs_t, log_t, _ = hidden(TARGET)
bases = [hidden(b) for b in BASELINES]
inj_ids = bases[0][2]
p_t = torch.softmax(log_t, -1)
p_all = torch.softmax(bases[0][1], -1)  # "All..." baseline distribution

# ---- candidate target tokens: argmax + single-token probes with real gap ----
argmax_id = int(torch.argmax(p_t))
cands = [("argmax", argmax_id)]
for w in PROBE_WORDS:
    wid = tok(" " + w, add_special_tokens=False)["input_ids"]
    if len(wid) == 1:
        cands.append((w, wid[0]))

print("\n" + "=" * 96)
print(f"TARGET: {TARGET!r}")
print(f"INJECT BASE: {INJECT_BASE!r}")
print(f"argmax next tok = {tok.decode([argmax_id]).strip()!r}")
print("=" * 96)

# ---- Show the triangulated readout across layers (what CAN be read) ----
print("\nTRIANGULATED dh = h(some) - mean(all,none,most,few,many); readout per layer:")
print(f"  {'L':>3} | {'||dh||/||h||':>11} | full readout (top-8)   //   subspace readout")
for L in range(4, NL + 1, 2):
    dh = hs_t[L] - torch.stack([b[0][L] for b in bases]).mean(0)
    rel = float(dh.norm() / hs_t[L].norm())
    sc = subspace_comp(dh)
    print(f"  {L:>3} | {rel:>11.3f} | [{','.join(toks(dh,8))}]  //  [{','.join(toks(sc,6))}]")

# ---- For each candidate target token, find the best layer where the
#      SUBSPACE (readable) component recovers the gap ----
print("\n" + "=" * 96)
print("CANDIDATE TARGET TOKENS: gap (some-vs-All) and best-layer recovery")
print("  full = full triangulated dh ; sub = token-subspace(top20/bot20) of it")
print("=" * 96)
print(f"  {'token':>10} {'gap':>7} | {'bestL':>5} {'sub%':>6} {'full%':>6} | "
      f"single-pair sub% (some-all)")
rows = []
for name, tid in cands:
    p_b = float(p_all[tid]); gap = float(p_t[tid] - p_b)
    if abs(gap) < 2e-3:
        continue
    best = None
    for L in range(4, NL + 1, 2):
        dh_tri = hs_t[L] - torch.stack([b[0][L] for b in bases]).mean(0)
        ss = inject(inj_ids, subspace_comp(dh_tri), L, tid, p_b, gap)
        ff = inject(inj_ids, dh_tri, L, tid, p_b, gap)
        if best is None or ss > best[1]:
            best = (L, ss, ff)
    # single-pair (some vs all) token-subspace at the same best layer, for contrast
    Lb = best[0]
    dh_single = hs_t[Lb] - bases[0][0][Lb]
    sp = inject(inj_ids, subspace_comp(dh_single), Lb, tid, p_b, gap)
    rows.append((name, tid, gap, best[0], best[1], best[2], sp))

rows.sort(key=lambda r: -r[4])  # by subspace recovery
for name, tid, gap, L, ss, ff, sp in rows:
    print(f"  {name:>10} {gap:>+7.3f} | {L:>5} {ss:>6.0f} {ff:>6.0f} | {sp:>+6.0f}")

# ---- For the best readable token, dump the readout + a layer profile ----
if rows:
    name, tid, gap, L, ss, ff, sp = rows[0]
    print("\n" + "=" * 96)
    print(f"BEST READABLE TARGET: {name!r} (tok {tok.decode([tid]).strip()!r}) "
          f"gap={gap:+.3f}  sub={ss:.0f}%  full={ff:.0f}%  @L{L}")
    dh = hs_t[L] - torch.stack([b[0][L] for b in bases]).mean(0)
    print(f"  triangulated readout @L{L}: [{','.join(toks(dh,12))}]")
    print(f"  subspace readout    @L{L}: [{','.join(toks(subspace_comp(dh),12))}]")
    print("  layer profile (sub% / full%):")
    p_b = float(p_all[tid])
    for Lp in range(4, NL + 1, 2):
        dhp = hs_t[Lp] - torch.stack([b[0][Lp] for b in bases]).mean(0)
        s = inject(inj_ids, subspace_comp(dhp), Lp, tid, p_b, gap)
        f = inject(inj_ids, dhp, Lp, tid, p_b, gap)
        print(f"    L{Lp:>2}: sub={s:>+6.0f}%  full={f:>+6.0f}%")

print("\nDONE")
