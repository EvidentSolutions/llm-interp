"""
Are contrastive readouts just f(prompts, W_U), or do they surface tokens the
model's OTHER weights are built to READ (computation nailed to weights)?

readout = topk((h_c - h_k) @ W_U.T)  is by construction a function of the two
prompts and W_U. The only way it can carry information about COMPUTATION is if the
tokens it surfaces have directions that downstream weight circuits (attention QK,
MLP fc1) READ -- a pure-weight property, independent of the prompts.

Per token t (u_t = unit W_U row) we score, from WEIGHTS ONLY:
    KEY(t)  = mean_late-heads ||W_K^h u_t|| / random-baseline   (attn readability)
    FC1(t)  = mean_late      ||fc1_L  u_t|| / random-baseline   (MLP detector read)
A high score = the model is built to READ this token's direction downstream (it is
a "computational" token); a score ~1 = read no more than a random direction (it is
output-manifesting, seen only by the unembedding).

For each contrast we compare the weight-read score of:
    (R) readout tokens      = top-k of (h_c - h_k) @ W_U.T
    (P) raw-prompt tokens   = top-k of  h_c        @ W_U.T   (what c alone predicts)
    (X) random tokens
If R > P > ~X, the SUBTRACTION preferentially surfaces weight-read (computational)
content -- the readout is nailed to weights, not just W_U geometry. We expect this
for STRUCTURAL contrasts (negation, quantifier, register) and NOT for FACTUAL ones
(their readout is an output token, consumed only by the unembedding).

Usage: .venv/Scripts/python.exe contrastive/code/readout_weight_nailed.py
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
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
NH = model.config.num_attention_heads
HD = model.config.hidden_size // NH
d = model.config.hidden_size
W_U = model.lm_head.weight.detach().to(DEV)
LATE = list(range(22, NL))
WK = [model.model.layers[L].self_attn.k_proj.weight.detach().to(DEV) for L in LATE]
FC1 = [model.model.layers[L].mlp.fc1.weight.detach().to(DEV) for L in LATE]

# random baselines (per late layer) for KEY (per head) and FC1
torch.manual_seed(0)
NR = 40
RB_key = torch.zeros(len(LATE), NH, device=DEV)
RB_fc1 = torch.zeros(len(LATE), device=DEV)
for _ in range(NR):
    u = torch.randn(d, device=DEV); u = u / u.norm()
    for li in range(len(LATE)):
        RB_key[li] += (WK[li] @ u).view(NH, HD).norm(dim=1)
        RB_fc1[li] += torch.linalg.vector_norm(FC1[li] @ u)
RB_key /= NR; RB_fc1 /= NR


def weight_read_vec(u):
    """KEY and FC1 late-enrichment for an arbitrary direction u (pure weights).
    Measures how much downstream (late) attention/MLP READ the direction u."""
    u = u / u.norm()
    key = torch.stack([(WK[li] @ u).view(NH, HD).norm(dim=1) for li in range(len(LATE))])
    fc1 = torch.stack([torch.linalg.vector_norm(FC1[li] @ u) for li in range(len(LATE))])
    return float((key / RB_key).mean()), float((fc1 / RB_fc1).mean())


def weight_read(tid):
    """KEY and FC1 late-enrichment for one token id (pure weights)."""
    return weight_read_vec(W_U[tid])


def score_set(ids):
    ks, fs = zip(*[weight_read(t) for t in ids])
    return sum(ks) / len(ks), sum(fs) / len(fs)


def hidden_last(text, L):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return out.hidden_states[L][0, -1, :].float()


def topk_tokens(vec, k=8):
    return torch.topk(vec @ W_U.T, k).indices.tolist()


CONTRASTS = [
    ("STRUCT negation", "She has finished the work", "She has not finished the work"),
    ("STRUCT quantifier", "Some of the students passed the exam",
     "All of the students passed the exam"),
    ("STRUCT tense", "He walks to the store every day",
     "He walked to the store every day"),
    ("STRUCT modal", "The bridge will collapse", "The bridge might collapse"),
    ("FACT capital", "The capital of France is", "The capital of Germany is"),
    ("FACT color", "The banana is", "The sky is"),
    ("FACT entity", "The theory of relativity was developed by",
     "The theory of evolution was developed by"),
]

def hidden_all(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return [out.hidden_states[Lh][0, -1, :].float() for Lh in range(NL + 1)]


def realword_frac(ids):
    n = 0
    for t in ids:
        s = tok.decode([t]).strip()
        if len(s) >= 2 and s.isalpha():
            n += 1
    return n / len(ids)


# anchors: what a purely-computational vs purely-output token scores
anchor_fn = score_set([tok(" " + w, add_special_tokens=False)["input_ids"][0]
                       for w in ["the", "and", "of", "to", "that", "is"]])
anchor_ct = score_set([tok(" " + w, add_special_tokens=False)["input_ids"][0]
                       for w in ["Paris", "France", "doctor", "banana", "table", "river"]])
print(f"\nWeight-read enrichment (late {LATE[0]}..{NL-1}) over random direction.")
print(f"ANCHOR function-words: KEY={anchor_fn[0]:.2f} FC1={anchor_fn[1]:.2f}   "
      f"ANCHOR content-words: KEY={anchor_ct[0]:.2f} FC1={anchor_ct[1]:.2f}")
print("(a readout 'nailed to computation' should score near the function anchor;")
print(" a readout that is just an output token should score near the content anchor)\n")
print(f"{'contrast':>18} {'L*':>3} {'realwd':>6} | {'KEY':>5} {'FC1':>5} | readout tokens")
print("-" * 100)

print("\n### TOKEN-LEVEL: weight-read of the top-k readout TOKENS (per contrast, at")
print("    the layer whose readout is most coherent). This throws away direction.\n")
import statistics as st
agg = {"STRUCT": [], "FACT": []}
for name, c, k in CONTRASTS:
    Hc = hidden_all(c); Hk = hidden_all(k)
    best = None
    for Lh in range(8, NL + 1):
        rout = topk_tokens(Hc[Lh] - Hk[Lh])
        rf = realword_frac(rout)
        if best is None or rf >= best[0]:
            best = (rf, Lh, rout)
    rf, Lstar, rout = best
    Rk, Rf = score_set(rout)
    words = ",".join(tok.decode([t]).strip() for t in rout[:6])
    kind = name.split()[0]
    agg[kind].append((Rk, Rf))
    print(f"{name:>18} {Lstar:>3} {rf:>6.2f} | KEY={Rk:>5.2f} FC1={Rf:>5.2f} | [{words}]")
print("-" * 100)
for kind in ["STRUCT", "FACT"]:
    ks = [x[0] for x in agg[kind]]
    print(f"MEAN {kind:>6}: KEY={st.mean(ks):.2f}")

# ---- DIRECTION-LEVEL: is the contrast DIRECTION itself read downstream, and is
#      that read SPECIFIC to the matched contrast (vs a mismatched difference)? --
print("\n### DIRECTION-LEVEL at L=20. wr(dh)=how much downstream QK/MLP read the")
print("    direction. match = h_c - h_k (paired); MIS = h_c - h_k' from an")
print("    UNRELATED contrast (same c, wrong baseline); full = wr(h_c); rand=1.00.")
print("    If match ~ MIS, the read is generic to 'any difference', NOT the")
print("    specific contrast. If match < MIS, subtraction desuperposes (cancels")
print("    the shared read-heavy background, leaving the clean distinction).\n")
L = 20
H = [(hidden_all(c), hidden_all(k)) for _, c, k in CONTRASTS]
print(f"{'contrast':>18} | {'match':>6} {'MIS':>6} {'full hc':>8}")
dir_agg = {"STRUCT": [], "FACT": []}
for i, (name, c, k) in enumerate(CONTRASTS):
    Hc, Hk = H[i]
    j = (i + 3) % len(CONTRASTS)          # an unrelated contrast's baseline
    dh_m = Hc[L] - Hk[L]
    dh_x = Hc[L] - H[j][1][L]
    wm, _ = weight_read_vec(dh_m)
    wx, _ = weight_read_vec(dh_x)
    wf, _ = weight_read_vec(Hc[L])
    dir_agg[name.split()[0]].append((wm, wx))
    print(f"{name:>18} | {wm:>6.2f} {wx:>6.2f} {wf:>8.2f}")
print("-" * 46)
for kind in ["STRUCT", "FACT"]:
    rows = dir_agg[kind]
    print(f"MEAN {kind:>6}: match={st.mean(r[0] for r in rows):.2f}  "
          f"MIS={st.mean(r[1] for r in rows):.2f}")
print("\nDONE")
