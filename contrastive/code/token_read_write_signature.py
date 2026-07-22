"""
Computational vs output-manifesting tokens: a structural (weights-only) signature.

Hypothesis (Olli): tokens lie on a scale. "Computational" tokens (negation 'n't,
function words) are READ by many downstream detectors, sustained into late layers
-- they participate in computation. "Output-manifesting" tokens (entities, content
nouns) are mostly WRITTEN late and then unembedded, not read afterward.

Measured per token t with unit W_U row u_t (residual aligns with W_U in depth):
    READ(t,L)  = || fc1_L @ u_t ||        detector response of layer L to u_t
    WRITE(t,L) = || fc2_L^T @ u_t ||       ability of layer L to write along u_t
Reported RELATIVE to random unit directions (enrichment), so we are not just
rediscovering norm/frequency. No forward passes, no scenario, no intervention.

Usage: .venv/Scripts/python.exe contrastive/code/token_read_write_signature.py
"""
import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach().to(DEV)                       # (V, d)
FC1 = [model.model.layers[L].mlp.fc1.weight.detach().to(DEV) for L in range(NL)]  # (dm,d)
FC2 = [model.model.layers[L].mlp.fc2.weight.detach().to(DEV) for L in range(NL)]  # (d,dm)

BINS = [("early", range(0, 11)), ("mid", range(11, 22)), ("late", range(22, NL))]


def read_write(u):
    """u: unit (d,). Returns per-layer READ and WRITE norms."""
    rd = torch.tensor([torch.linalg.vector_norm(FC1[L] @ u) for L in range(NL)])
    wr = torch.tensor([torch.linalg.vector_norm(FC2[L].T @ u) for L in range(NL)])
    return rd, wr


# random-direction baseline (per layer), averaged
torch.manual_seed(0)
d = W_U.shape[1]
RB_read = torch.zeros(NL); RB_write = torch.zeros(NL)
NR = 40
for _ in range(NR):
    u = torch.randn(d, device=DEV); u = u / u.norm()
    rd, wr = read_write(u)
    RB_read += rd; RB_write += wr
RB_read /= NR; RB_write /= NR


def single_token(word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


GROUPS = {
    "negation":   [" not", " no", " never", " n't", "n't", " none", " nor"],
    "function":   [" the", " of", " and", " is", " to", " a", " that", " but",
                   " because", " if", " than", " which"],
    "content":    [" Paris", " France", " Tesla", " mustard", " doctor", " dog",
                   " Rome", " Einstein", " banana", " table", " river", " coffee"],
}


def summarize(word):
    tid = single_token(word)
    if tid is None:
        return None
    u = W_U[tid].clone(); u = u / u.norm()
    rd, wr = read_write(u)
    # enrichment over random baseline, per bin
    out = {"tok": word.strip(), "id": tid}
    for name, rng in BINS:
        idx = list(rng)
        out[f"R_{name}"] = float((rd[idx] / RB_read[idx]).mean())
        out[f"W_{name}"] = float((wr[idx] / RB_write[idx]).mean())
    out["RW_late"] = out["R_late"] / max(out["W_late"], 1e-6)
    out["R_late_over_early"] = out["R_late"] / max(out["R_early"], 1e-6)
    return out


print(f"\nRandom-direction baseline READ norm  (early/mid/late): "
      f"{RB_read[:11].mean():.1f} / {RB_read[11:22].mean():.1f} / {RB_read[22:].mean():.1f}")
print(f"Random-direction baseline WRITE norm (early/mid/late): "
      f"{RB_write[:11].mean():.1f} / {RB_write[11:22].mean():.1f} / {RB_write[22:].mean():.1f}")

hdr = (f"\n{'token':>12} | {'R_early':>7} {'R_mid':>6} {'R_late':>6} | "
       f"{'W_early':>7} {'W_mid':>6} {'W_late':>6} | {'R_late/W_late':>13} {'R_late/early':>12}")
group_means = {}
for gname, words in GROUPS.items():
    print("\n" + "=" * 96)
    print(f"GROUP: {gname}")
    print("=" * 96)
    print(hdr)
    rows = []
    for w in words:
        s = summarize(w)
        if s is None:
            print(f"{w.strip():>12} |  (multi-token, skipped)")
            continue
        rows.append(s)
        print(f"{s['tok']:>12} | {s['R_early']:>7.2f} {s['R_mid']:>6.2f} {s['R_late']:>6.2f} | "
              f"{s['W_early']:>7.2f} {s['W_mid']:>6.2f} {s['W_late']:>6.2f} | "
              f"{s['RW_late']:>13.2f} {s['R_late_over_early']:>12.2f}")
    if rows:
        import statistics as st
        group_means[gname] = {k: st.mean(r[k] for r in rows)
                              for k in ["R_early", "R_mid", "R_late",
                                        "W_early", "W_mid", "W_late",
                                        "RW_late", "R_late_over_early"]}

print("\n" + "=" * 96)
print("GROUP MEANS (enrichment over random direction; >1 = more than a random dir)")
print("=" * 96)
print(hdr)
for g, m in group_means.items():
    print(f"{g:>12} | {m['R_early']:>7.2f} {m['R_mid']:>6.2f} {m['R_late']:>6.2f} | "
          f"{m['W_early']:>7.2f} {m['W_mid']:>6.2f} {m['W_late']:>6.2f} | "
          f"{m['RW_late']:>13.2f} {m['R_late_over_early']:>12.2f}")
print("\nDONE")
