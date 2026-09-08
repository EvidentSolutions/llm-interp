"""
Phase-0 verification of the published triangulation table (paper_v1.tex
tab:triangulation), 2026-07-30.

Provenance (session_report_2026-07-04_some_all_envelope.md §1): the table's two
columns came from two different scripts with two different projections:
  - "Single-pair" = token_superposition_causal.py: project single-pair dh onto
    QR(W_U rows of its own top-20 + bottom-20 tokens), inject at L28, recover
    the argmax gap.
  - "Multi"       = deep_decompose.py: project dh onto the RANK-1 SHARED MEAN
    direction across baselines, inject that (not a token subspace), L28.
The paper's S3.3 narrates both as token-subspace ("constructed the same way,
from the W_U rows of the averaged vector's own top-20 and bottom-20 tokens").
Caught-cold survived the apples-to-apples re-test (07-04); moral and hot-dog
were never re-checked; some/all is known-wrong as printed.

This script computes, for each compositional row, at every even layer:
  single_full  - inject full single-pair dh
  single_sub   - inject token-subspace component of single-pair dh   [column 1]
  rank1_shared - inject (dh . m^) m^, the provenance "Multi" construction
  tri_full     - inject full averaged dh
  tri_sub      - inject token-subspace component of averaged dh      [the claim]
Recovery = (p_inj - p_base)/(p_target - p_base) on the TARGET-PROMPT ARGMAX
token (the published construction), plus a content-probe token as a secondary
read. Injection convention identical to both source scripts: dh from
hidden_states[L], added at model.model.layers[L] output, last position.

Verdict logic per row: the published claim "single X% -> multi Y%" is
apples-to-apples-verified iff single_sub ~ X and tri_sub ~ Y at/near L28.
If instead only rank1_shared reaches Y while tri_sub does not, the published
Multi number rode the shared-axis projection (the 07-04 bug).

Usage: .venv/Scripts/python.exe public/contrastive/code/triangulation_table_verify.py
"""
import sys, os, json
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
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach().float().to(DEV)
K = 20

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "data",
                        "triangulation-table-verify.json")


def hidden(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    hs = [out.hidden_states[L][0, -1, :].float() for L in range(NL + 1)]
    return hs, out.logits[0, -1].float(), ids


def toks(v, k=3):
    i = torch.topk(v @ W_U.T, k).indices
    return [tok.decode([int(x)]).strip() for x in i]


def subspace_comp(vec, k=K):
    logits = vec @ W_U.T
    idx = torch.cat([torch.topk(logits, k).indices,
                     torch.topk(-logits, k).indices])
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
    # Control row: verified apples-to-apples on 07-04; must reproduce ~1 -> ~77.
    # PROVENANCE PROMPTS (deep_decompose CASE 2 / token_superposition
    # caught_cold_fish): the "and went to" frame, NOT the bare prompts the
    # 07-04 repro script used.
    dict(name="caught_cold", published=(1, 77),
         target="She caught a cold and went to",
         baselines=["She caught a fish and went to",
                    "She caught a ball and went to",
                    "She caught a bus and went to",
                    "She caught a thief and went to",
                    "She caught a glimpse and went to"],
         probe=["doctor", "sick", "ill", "flu", "bed", "rest"]),
    # Published 4 -> 75. Provenance: deep_decompose CASE 4 (4 baselines,
    # inject into "paid").
    dict(name="theft_moral", published=(4, 75),
         target="He slipped a bottle under his coat and walked out without paying. He",
         baselines=[
             "He picked up a bottle, went to the register and paid. He",
             "He slipped a bottle under his coat then returned it to the shelf. He",
             "He slipped a bottle under his coat, forgetting he already paid. He",
             "He browsed the shelves, picked up a bottle, and put it back. He"],
         probe=["evade", "caught", "arrested", "guilty", "police", "hoped"]),
    # Published 11 -> 49. Provenance: deep_decompose CASE 1 (8 baselines,
    # inject into "cold_dog").
    dict(name="hotdog_prov8", published=(11, 49),
         target="The hot dog was",
         baselines=["The cold dog was", "The hot cat was", "The old dog was",
                    "The angry dog was", "The hot rod was",
                    "The hot chocolate was", "The dog was", "A hot dog was"],
         probe=["cooked", "delicious", "tasty", "grilled", "crispy"]),
    # Same row, clean frame-matched baseline set (the paper's S3.2 eatability
    # family) - reported alongside, not the provenance.
    dict(name="hotdog_clean5", published=(11, 49),
         target="The hot dog was",
         baselines=["The cold dog was", "The angry dog was", "The old dog was",
                    "The pet dog was", "The stray dog was"],
         probe=["cooked", "delicious", "tasty", "grilled", "crispy"]),
    # Known-wrong as printed (-40 -> 101, mixed projections). Documented here
    # bare; the honest number per 07-04 needs the denoising envelope.
    dict(name="some_all", published=(-40, 101),
         target="Some of the students passed the exam, so",
         baselines=["All of the students passed the exam, so",
                    "Most of the students passed the exam, so",
                    "Few of the students passed the exam, so",
                    "None of the students passed the exam, so",
                    "Many of the students passed the exam, so"],
         probe=["others", "some", "not", "the"]),
]

results = {}
for c in CASES:
    hs_t, log_t, _ = hidden(c["target"])
    bases = [hidden(b) for b in c["baselines"]]
    b0_ids = bases[0][2]
    p_t = torch.softmax(log_t, -1)
    p_b0 = torch.softmax(bases[0][1], -1)

    argmax_id = int(torch.argmax(p_t))
    probe_ids = [tok(" " + w, add_special_tokens=False)["input_ids"]
                 for w in c["probe"]]
    probe_ids = [w[0] for w in probe_ids if len(w) == 1]
    # repair 2026-07-30: require a real gap (>=0.01) so the recovery ratio's
    # denominator is meaningful; the first run picked 'flu' at gap +0.0004
    # and produced junk ratios.
    probe_ids = [t for t in probe_ids if abs(float(p_t[t] - p_b0[t])) >= 0.01]
    probe_id = (max(probe_ids, key=lambda t: abs(float(p_t[t] - p_b0[t])))
                if probe_ids else argmax_id)

    print("\n" + "=" * 100)
    print(f"CASE {c['name']}   published single->multi: "
          f"{c['published'][0]}% -> {c['published'][1]}%")
    print(f"  target='{c['target']}'")
    print(f"  inject-into baseline = '{c['baselines'][0]}'  "
          f"({len(c['baselines'])} baselines)")
    print(f"  argmax = '{tok.decode([argmax_id]).strip()}' "
          f"(gap {float(p_t[argmax_id] - p_b0[argmax_id]):+.3f})   "
          f"probe = '{tok.decode([probe_id]).strip()}' "
          f"(gap {float(p_t[probe_id] - p_b0[probe_id]):+.3f})")
    print("=" * 100)

    case_res = {"target": c["target"], "baselines": c["baselines"],
                "published": c["published"], "layers": {}}
    for label, tid in [("argmax", argmax_id), ("probe", probe_id)]:
        p_b = float(p_b0[tid]); gap = float(p_t[tid] - p_b)
        if abs(gap) < 1e-4:
            print(f"\n  [{label}] gap~0, skip")
            continue
        print(f"\n  [{label} '{tok.decode([tid]).strip()}'  gap={gap:+.3f}]")
        print(f"  {'L':>3} {'sgl_full':>8} {'sgl_sub':>8} {'rank1':>8} "
              f"{'tri_full':>8} {'tri_sub':>8}   tri_sub reads")
        for L in range(4, NL + 1, 2):
            dh_s = hs_t[L] - bases[0][0][L]
            dh_t = hs_t[L] - torch.stack([b[0][L] for b in bases]).mean(0)
            # provenance "Multi": rank-1 projection of dh_s on the normalized
            # mean over per-baseline deltas (deep_decompose Step 3)
            all_dh = torch.stack([hs_t[L] - b[0][L] for b in bases])
            m_hat = all_dh.mean(0) / all_dh.mean(0).norm()
            rank1 = (dh_s @ m_hat) * m_hat
            row = dict(
                sgl_full=inject(b0_ids, dh_s, L, tid, p_b, gap),
                sgl_sub=inject(b0_ids, subspace_comp(dh_s), L, tid, p_b, gap),
                rank1=inject(b0_ids, rank1, L, tid, p_b, gap),
                tri_full=inject(b0_ids, dh_t, L, tid, p_b, gap),
                tri_sub=inject(b0_ids, subspace_comp(dh_t), L, tid, p_b, gap),
            )
            reads = ",".join(toks(dh_t, 3))
            print(f"  {L:>3} {row['sgl_full']:>7.0f}% {row['sgl_sub']:>7.0f}% "
                  f"{row['rank1']:>7.0f}% {row['tri_full']:>7.0f}% "
                  f"{row['tri_sub']:>7.0f}%   [{reads}]")
            case_res["layers"].setdefault(str(L), {})[label] = {
                **{k: round(v, 1) for k, v in row.items()},
                "tri_reads": toks(dh_t, 5)}
        results[c["name"]] = case_res

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print(f"\nSaved {OUT_JSON}\nDONE")
