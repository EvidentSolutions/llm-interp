"""
Attention-read signature of a token (weights-only, no forward pass).

Fork-1 of the computational-vs-output-manifesting question. The MLP read/write
probe (token_read_write_signature.py) was NULL: READ==WRITE for all groups.
Hypothesis (Olli): the distinction lives in the ATTENTION channel, because "the
things we are looking for are just the things that move between positions."

For a token t we take its late-residual signature u_t = unit W_U row (a token
placed in the stream looks like its W_U row before unembedding). We ask, per head
h, three weights-only questions:
    KEY(t,h)   = || (W_K^h) u_t ||     how attendable a position holding u_t is
    QUERY(t,h) = || (W_Q^h) u_t ||     how much u_t drives a query
    MOVE(t,h)  = || W_O[:,h] (W_V^h) u_t ||   how much u_t is BROADCAST (OV move)
Rotary preserves per-head norm, so we ignore it here.

MOVE is the operative one: a "computational"/register token gets moved between
positions (read by attention, broadcast onward); an "output-manifesting" token is
written late and unembedded, not moved.

Reported as enrichment over random unit directions, per layer-bin, PLUS a
SELECTIVITY count: how many individual heads read u_t sharply (enrichment>thr).
Aggregate norm hid the effect last time; sharp-head counts should not.

Groups include REGISTER tokens (format/genre/style markers -- "global variable"
like), tested against function, negation, and content.

Usage: .venv/Scripts/python.exe contrastive/code/token_attention_read_signature.py
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
W_U = model.lm_head.weight.detach().to(DEV)                       # (V, d)

WQ = [model.model.layers[L].self_attn.q_proj.weight.detach().to(DEV) for L in range(NL)]
WK = [model.model.layers[L].self_attn.k_proj.weight.detach().to(DEV) for L in range(NL)]
WV = [model.model.layers[L].self_attn.v_proj.weight.detach().to(DEV) for L in range(NL)]
WO = [model.model.layers[L].self_attn.dense.weight.detach().to(DEV) for L in range(NL)]

BINS = [("early", range(0, 11)), ("mid", range(11, 22)), ("late", range(22, NL))]
SEL_THR = 3.0   # a head "sharply reads" u_t if MOVE enrichment > this


def per_head_norms(mat, u):
    """mat: (NH*HD, d) proj. Returns (NL?) no -- per head norm of (mat @ u)."""
    v = mat @ u                      # (NH*HD,)
    return v.view(NH, HD).norm(dim=1)  # (NH,)


def move_per_head(L, u):
    """OV broadcast per head: ||W_O[:, h_slice] @ (W_V^h u)||."""
    v = (WV[L] @ u).view(NH, HD)          # (NH, HD)
    out = torch.empty(NH, device=DEV)
    for h in range(NH):
        sl = WO[L][:, h * HD:(h + 1) * HD]   # (d, HD)
        out[h] = (sl @ v[h]).norm()
    return out                              # (NH,)


def signature(u):
    """Per-(layer,head) KEY, QUERY, MOVE norms. Returns 3 tensors (NL, NH)."""
    key = torch.empty(NL, NH, device=DEV)
    qry = torch.empty(NL, NH, device=DEV)
    mov = torch.empty(NL, NH, device=DEV)
    for L in range(NL):
        key[L] = per_head_norms(WK[L], u)
        qry[L] = per_head_norms(WQ[L], u)
        mov[L] = move_per_head(L, u)
    return key, qry, mov


# random-direction baseline (per layer,head), averaged over NR dirs
torch.manual_seed(0)
NR = 40
RB_key = torch.zeros(NL, NH, device=DEV)
RB_qry = torch.zeros(NL, NH, device=DEV)
RB_mov = torch.zeros(NL, NH, device=DEV)
print(f"Building random baseline over {NR} directions...")
for _ in range(NR):
    u = torch.randn(d, device=DEV); u = u / u.norm()
    k, q, m = signature(u)
    RB_key += k; RB_qry += q; RB_mov += m
RB_key /= NR; RB_qry /= NR; RB_mov /= NR


def single_token(word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


GROUPS = {
    "register": [" Chapter", " Abstract", " Dear", " def", " import", " Note",
                 " Section", " http", " www", " Figure", " Table", " Fig"],
    "punct":    [".", ",", ":", ";", "?", "!", ")", "(", '"', "'", "-", "\n"],
    "negation": [" not", " no", " never", " none", " nor", " cannot"],
    "function": [" the", " of", " and", " is", " to", " a", " that", " but",
                 " because", " if", " than", " which"],
    "content":  [" Paris", " France", " Tesla", " mustard", " doctor", " dog",
                 " Rome", " Einstein", " banana", " table", " river", " coffee"],
}


def summarize(word):
    tid = single_token(word)
    if tid is None:
        return None
    u = W_U[tid].clone(); u = u / u.norm()
    key, qry, mov = signature(u)
    e_key = key / RB_key; e_qry = qry / RB_qry; e_mov = mov / RB_mov
    out = {"tok": repr(word.strip()) if word.strip() else repr(word), "id": tid}
    for name, rng in BINS:
        idx = list(rng)
        out[f"MOVE_{name}"] = float(e_mov[idx].mean())
        out[f"KEY_{name}"] = float(e_key[idx].mean())
    # selectivity: how many (layer,head) sharply move u_t, in late layers
    late = list(BINS[2][1])
    out["n_sharp_late"] = int((e_mov[late] > SEL_THR).sum())
    out["max_move"] = float(e_mov.max())
    return out


hdr = (f"\n{'token':>10} | {'MOVE_e':>6} {'MOVE_m':>6} {'MOVE_l':>6} | "
       f"{'KEY_e':>6} {'KEY_m':>6} {'KEY_l':>6} | {'sharp_l':>7} {'maxMOVE':>7}")


def row(s):
    return (f"{s['tok']:>10} | {s['MOVE_early']:>6.2f} {s['MOVE_mid']:>6.2f} "
            f"{s['MOVE_late']:>6.2f} | {s['KEY_early']:>6.2f} {s['KEY_mid']:>6.2f} "
            f"{s['KEY_late']:>6.2f} | {s['n_sharp_late']:>7d} {s['max_move']:>7.2f}")


print(f"\nSELECTIVITY threshold = MOVE enrichment > {SEL_THR} (late layers "
      f"{list(BINS[2][1])[0]}..{NL-1}, {len(list(BINS[2][1]))*NH} head slots)")

group_means = {}
import statistics as st
for gname, words in GROUPS.items():
    print("\n" + "=" * 78)
    print(f"GROUP: {gname}")
    print("=" * 78)
    print(hdr)
    rows = []
    for w in words:
        s = summarize(w)
        if s is None:
            print(f"{repr(w.strip()):>10} |  (multi-token, skipped)")
            continue
        rows.append(s)
        print(row(s))
    if rows:
        keys = ["MOVE_early", "MOVE_mid", "MOVE_late", "KEY_early", "KEY_mid",
                "KEY_late", "n_sharp_late", "max_move"]
        group_means[gname] = {k: st.mean(r[k] for r in rows) for k in keys}

print("\n" + "=" * 78)
print("GROUP MEANS (enrichment over random direction)")
print("=" * 78)
print(hdr)
for g, m in group_means.items():
    print(f"{g:>10} | {m['MOVE_early']:>6.2f} {m['MOVE_mid']:>6.2f} "
          f"{m['MOVE_late']:>6.2f} | {m['KEY_early']:>6.2f} {m['KEY_mid']:>6.2f} "
          f"{m['KEY_late']:>6.2f} | {m['n_sharp_late']:>7.1f} {m['max_move']:>7.2f}")
print("\nDONE")
