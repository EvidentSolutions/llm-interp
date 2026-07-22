"""
Third verification: do the READABLE tokens carry the causal content?

Verifications 1-2 in the paper show the contrastive DIRECTION is causal
(injection, dose-response) -- but a causal direction is RepE/ActAdd prior art.
The paper's novel claim is about the READOUT: the tokens we read are real
content the model computes with. This script tests exactly that, at the grain
of the readout (tokens), with no commitment to individual neurons.

Core metric -- "causal read-token share":
    project dh onto the QR-orthonormal W_U-row subspace of its own top-N /
    bottom-N read tokens, inject ONLY that token-subspace component, measure
    prediction-gap recovery. share = recovery(read-subspace) / recovery(full dh).
    High share => the causal content lives in the tokens we read (not decoration).

Controls that make it a verification, not a demo (all matched to the SAME
subspace dimension, so it is not a dimension effect):
  - random     : N random W_U rows.
  - shared/freq: top tokens of (h_a + h_b) -- generic-to-both, the cancelled
                 shared component ("properties of any input").
  - mismatched : read tokens from a DIFFERENT contrast (scenario-arbitrary).
  - single vs triangulated dh (compositional cases): does averaging over
    baselines move the causal content INTO the readable tokens?

Per-token burden (directs causality studies):
  - leave-one-out over the top positive read tokens: which specific tokens
    carry the recovery.
  - scenario-specificity: a burden token from case X should NOT be causal in
    case Y.

Usage: .venv/Scripts/python.exe contrastive/code/computationality_verification.py
"""
import sys
import os
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = (model.lm_head.weight if hasattr(model, "lm_head")
       else model.embed_out.weight).detach().float()          # (vocab, d)
W_U_dev = W_U.to(DEV)
VOCAB, D = W_U.shape

K = 20          # tokens per pole for the token subspace (2K dims), matches paper
LOO_K = 8       # top positive tokens to score for per-token burden
N_RAND = 10     # random-null trials
SWEEP = [12, 16, 20, 24, 28]
PREAMBLE = "Consider the following short scene. "


# ---------------------------------------------------------------- helpers
def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


def last_hidden(text):
    """All-layer hidden states at the final position: list length NL+1 of (d,)."""
    ids = ids_of(text)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    hs = [out.hidden_states[L][0, -1, :].float() for L in range(NL + 1)]
    logits = out.logits[0, -1].float()
    return hs, logits, ids


def toks(logits, k=6):
    v, i = torch.topk(logits.float(), k)
    return [tok.decode([int(i[j])]).strip() for j in range(k)]


def project_indices(vec, idx):
    """Component of vec inside span of the W_U rows at token indices idx."""
    rows = W_U_dev[idx]                      # (m, d)
    Q, _ = torch.linalg.qr(rows.T)           # (d, m)
    v = vec.to(DEV)
    return Q @ (Q.T @ v)


def read_indices(vec, k=K):
    """top-k + bottom-k token indices of vec's W_U projection."""
    logits = vec.to(DEV) @ W_U_dev.T
    top = torch.topk(logits, k).indices
    bot = torch.topk(-logits, k).indices
    return torch.cat([top, bot]), top, bot


def inject_recover(base_ids, delta, L, target_id, p_b, gap):
    """Inject delta at layer L, last pos, into base_ids; % recovery of P(target)."""
    injected = [False]

    def hook(module, inp, output):
        if injected[0]:
            return output
        injected[0] = True
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, -1, :] += delta.half().to(DEV)
            return (h,) + output[1:]
        h = output.clone()
        h[0, -1, :] += delta.half().to(DEV)
        return h

    if L < NL:
        handle = model.model.layers[L].register_forward_hook(hook)
    else:
        handle = model.model.final_layernorm.register_forward_hook(hook)
    with torch.no_grad():
        out = model(torch.tensor([base_ids], device=DEV))
    handle.remove()
    p = float(torch.softmax(out.logits[0, -1].float(), -1)[target_id])
    return (p - p_b) / gap * 100.0 if gap != 0 else 0.0


# ---------------------------------------------------------------- cases
def P(s):
    return PREAMBLE + s

CASES = [
    dict(name="caught_cold", kind="compositional",
         target=P("She caught a cold. The next morning she hurried to the"),
         baselines=[P("She caught a fish. The next morning she hurried to the"),
                    P("She caught a ball. The next morning she hurried to the"),
                    P("She caught a bus. The next morning she hurried to the"),
                    P("She caught a thief. The next morning she hurried to the"),
                    P("She caught a glimpse. The next morning she hurried to the")]),
    dict(name="cold_wentto", kind="compositional",   # faithful S3.5 repro frame
         target=P("She caught a cold and went to"),
         baselines=[P("She caught a fish and went to"),
                    P("She caught a ball and went to"),
                    P("She caught a bus and went to"),
                    P("She caught a thief and went to"),
                    P("She caught a glimpse and went to")]),
    dict(name="some_all", kind="compositional",
         target=P("Some of the students passed the exam, so"),
         baselines=[P("All of the students passed the exam, so"),
                    P("Most of the students passed the exam, so"),
                    P("Few of the students passed the exam, so"),
                    P("None of the students passed the exam, so"),
                    P("Many of the students passed the exam, so")]),
    dict(name="hot_dog", kind="compositional",
         target=P("The hot dog was served with"),
         baselines=[P("The cold dog was served with"),
                    P("The angry dog was served with"),
                    P("The old dog was served with"),
                    P("The pet dog was served with"),
                    P("The stray dog was served with")]),
    dict(name="capital_france", kind="entity",
         target=P("The capital of France is"),
         baselines=[P("The capital of Japan is"),
                    P("The capital of Germany is"),
                    P("The capital of Italy is"),
                    P("The capital of Spain is"),
                    P("The capital of Russia is")]),
    dict(name="ioi", kind="entity",
         target=P("John and Mary went to the store. John gave a book to"),
         baselines=[P("Mary and John went to the store. Mary gave a book to")]),
]


def prep(case):
    """Hidden states; target = the model's ACTUAL next-token prediction."""
    hs_t, log_t, _ = last_hidden(case["target"])
    base = []
    for b in case["baselines"]:
        hs_b, log_b, ids_b = last_hidden(b)
        base.append(dict(text=b, hs=hs_b, logits=log_b, ids=ids_b))
    p_t = torch.softmax(log_t, -1)
    p_b0 = torch.softmax(base[0]["logits"], -1)
    # the model's real prediction for the target prompt -- "what it computes"
    target_id = int(torch.argmax(p_t))
    case.update(hs_t=hs_t, log_t=log_t, base=base,
                target_id=target_id,
                target_tok=tok.decode([target_id]).strip(),
                p_t=float(p_t[target_id]), p_b=float(p_b0[target_id]))
    case["gap"] = case["p_t"] - case["p_b"]
    return case


# ---------------------------------------------------------------- run
def dh_single(case, L):
    return case["hs_t"][L] - case["base"][0]["hs"][L]

def dh_tri(case, L):
    return case["hs_t"][L] - torch.stack(
        [b["hs"][L] for b in case["base"]]).mean(0)

def best_layer(case, use_tri=False):
    """Layer with max full-dh recovery -- where the direction is most causal."""
    b0 = case["base"][0]["ids"]
    best, bestrec = SWEEP[0], -1e9
    for L in SWEEP:
        dh = dh_tri(case, L) if use_tri else dh_single(case, L)
        r = inject_recover(b0, dh, L, case["target_id"], case["p_b"], case["gap"])
        if r > bestrec:
            best, bestrec = L, r
    return best, bestrec


results = {}
prepared = [prep(c) for c in CASES]

# generic high-frequency token ranking (neutral context), for the frequency null
_, _gen_logits, _ = last_hidden("The report said that the weather was")
GENERIC_RANK = torch.topk(_gen_logits.to(DEV), 8 * K).indices  # pool of generic toks

print("\n" + "=" * 78)
print("BASELINE SETUP")
print("=" * 78)
for c in prepared:
    print(f"  {c['name']:<15} target='{c['target_tok']}'  "
          f"P(t)={c['p_t']:.3f} P(b)={c['p_b']:.3f} gap={c['gap']:+.3f}  [{c['kind']}]")

for c in prepared:
    name = c["name"]
    b0 = c["base"][0]["ids"]
    tid, p_b, gap = c["target_id"], c["p_b"], c["gap"]
    print("\n" + "=" * 78)
    print(f"CASE: {name}  ({c['kind']})   target token = '{c['target_tok']}'")
    print("=" * 78)

    rec = {}
    L, full_rec = best_layer(c, use_tri=False)
    rec["primary_L"] = L
    print(f"  primary layer L{L} (max single-dh recovery = {full_rec:.0f}%)")

    dh = dh_single(c, L)
    full = inject_recover(b0, dh, L, tid, p_b, gap)

    # own read-token subspace
    own_idx, own_top, _ = read_indices(dh, K)
    own_comp = project_indices(dh, own_idx)
    own = inject_recover(b0, own_comp, L, tid, p_b, gap)

    # frequency subspace: generic high-freq tokens, EXCLUDING this contrast's
    # read tokens and the target token (so it is purely generic content)
    read_set = set(own_idx.tolist()) | {tid}
    freq_idx = torch.tensor([int(t) for t in GENERIC_RANK.tolist()
                             if int(t) not in read_set][:2 * K], device=DEV)
    shared = inject_recover(b0, project_indices(dh, freq_idx), L, tid, p_b, gap)

    # mismatched: read tokens of a different case's single-dh
    other = prepared[(prepared.index(c) + 1) % len(prepared)]
    mm_idx, _, _ = read_indices(dh_single(other, L), K)
    mism = inject_recover(b0, project_indices(dh, mm_idx), L, tid, p_b, gap)

    # random subspace (same dim), averaged
    rand_vals = []
    for _ in range(N_RAND):
        ridx = torch.randint(0, VOCAB, (2 * K,), device=DEV)
        rand_vals.append(inject_recover(b0, project_indices(dh, ridx), L, tid, p_b, gap))
    rnd = sum(rand_vals) / len(rand_vals)

    print(f"  READ tokens (+target pole): {toks(dh.to(DEV) @ W_U_dev.T, 8)}")
    print(f"\n  recovery of P('{c['target_tok']}') at L{L}  [full dh = {full:.0f}%]")
    print(f"    {'read-token subspace':<24} {own:>7.0f}%   share = {own/full*100 if full else 0:>4.0f}% of full")
    print(f"    {'shared/freq subspace':<24} {shared:>7.0f}%")
    print(f"    {'mismatched subspace':<24} {mism:>7.0f}%")
    print(f"    {'random subspace':<24} {rnd:>7.0f}%")
    rec.update(full=full, read=own, shared=shared, mismatched=mism, random=rnd,
               share=own / full * 100 if full else 0)

    # single vs triangulated (compositional only) -- SAME layer L, apples-to-apples
    if c["kind"] == "compositional":
        Lt = L
        dht = dh_tri(c, Lt)
        tri_full = inject_recover(b0, dht, Lt, tid, p_b, gap)
        tri_idx, tri_top, _ = read_indices(dht, K)
        tri_comp = project_indices(dht, tri_idx)
        tri_read = inject_recover(b0, tri_comp, Lt, tid, p_b, gap)
        print(f"\n  TRIANGULATED (L{Lt}, {len(c['base'])} baselines)")
        print(f"    read tokens (+pole): {toks(dht.to(DEV) @ W_U_dev.T, 8)}")
        print(f"    full tri-dh          {tri_full:>7.0f}%")
        print(f"    read-token subspace  {tri_read:>7.0f}%   share = "
              f"{tri_read/tri_full*100 if tri_full else 0:>4.0f}% of full")
        print(f"    => single read-share {own/full*100 if full else 0:.0f}%  "
              f"-> tri read-share {tri_read/tri_full*100 if tri_full else 0:.0f}%")
        rec.update(tri_L=Lt, tri_full=tri_full, tri_read=tri_read,
                   tri_share=tri_read / tri_full * 100 if tri_full else 0)
        loo_src, loo_top = tri_comp, tri_top   # attribute on the clean (tri) read
        loo_full = tri_read
        loo_L = Lt
    else:
        loo_src, loo_top = own_comp, own_top
        loo_full = own
        loo_L = L

    # per-token burden: leave-one-out over top positive read tokens
    print(f"\n  per-token burden (leave-one-out, top-{LOO_K} positive read tokens):")
    burden = []
    top_ids = loo_top[:LOO_K].tolist()
    for j, tkid in enumerate(top_ids):
        keep = [t for t in loo_top[:K].tolist() if t != tkid]
        # rebuild subspace on the SAME source dh but without this token direction
        # (use the loo case's dh: tri for compositional, single for entity)
        src_dh = dh_tri(c, loo_L) if c["kind"] == "compositional" else dh_single(c, loo_L)
        comp_wo = project_indices(src_dh, torch.tensor(keep, device=DEV))
        r_wo = inject_recover(b0, comp_wo, loo_L, tid, p_b, gap)
        drop = loo_full - r_wo
        burden.append((tok.decode([tkid]).strip(), drop))
    for word, drop in sorted(burden, key=lambda x: -x[1]):
        bar = "#" * max(0, int(drop / 3))
        print(f"    {word:<16} {drop:>+6.0f}%  {bar}")
    rec["burden"] = burden
    c["_top_burden_id"] = top_ids[max(range(len(burden)), key=lambda j: burden[j][1])]
    c["_loo_L"] = loo_L

    results[name] = rec
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- specificity
print("\n" + "=" * 78)
print("SCENARIO-SPECIFICITY: burden token of case X, tested in case Y")
print("row = token owner (X), col = case injected into (Y); value = 1-dir recovery %")
print("=" * 78)
names = [c["name"] for c in prepared]
print(f"  {'token/owner':<20}" + "".join(f"{n[:10]:>12}" for n in names))
spec = {}
for cx in prepared:
    xid = cx["_top_burden_id"]
    word = tok.decode([xid]).strip()
    row = []
    for cy in prepared:
        L = cy["_loo_L"]
        dhy = (dh_tri(cy, L) if cy["kind"] == "compositional" else dh_single(cy, L))
        comp = project_indices(dhy, torch.tensor([xid], device=cy["base"][0]["hs"][0].device
                                                 if False else DEV))
        r = inject_recover(cy["base"][0]["ids"], comp, L,
                           cy["target_id"], cy["p_b"], cy["gap"])
        row.append(r)
    spec[cx["name"]] = row
    print(f"  {word+' ('+cx['name'][:6]+')':<20}" + "".join(f"{v:>12.0f}" for v in row))

# ---------------------------------------------------------------- summary + save
print("\n" + "=" * 78)
print("SUMMARY: causal read-token share (own read-subspace / full dh)")
print("=" * 78)
print(f"  {'case':<15}{'kind':<14}{'full%':>7}{'read%':>7}{'share%':>7}"
      f"{'shared%':>8}{'mism%':>7}{'rand%':>7}{'tri-share%':>11}")
for c in prepared:
    r = results[c["name"]]
    ts = f"{r.get('tri_share', float('nan')):.0f}" if "tri_share" in r else "-"
    print(f"  {c['name']:<15}{c['kind']:<14}{r['full']:>7.0f}{r['read']:>7.0f}"
          f"{r['share']:>7.0f}{r['shared']:>8.0f}{r['mismatched']:>7.0f}"
          f"{r['random']:>7.0f}{ts:>11}")

outpath = os.path.join(os.path.dirname(__file__), "..", "data",
                       f"computationality_verification_{MODEL.split('/')[-1]}.json")
with open(outpath, "w", encoding="utf-8") as f:
    json.dump({"model": MODEL, "K": K, "preamble": PREAMBLE,
               "results": results, "specificity": spec,
               "cases": {c["name"]: {"target": c["target"],
                                     "target_tok": c["target_tok"],
                                     "gap": c["gap"]} for c in prepared}},
              f, indent=2)
print(f"\nsaved -> {os.path.abspath(outpath)}")
print("DONE")
