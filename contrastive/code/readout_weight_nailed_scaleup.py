"""
Scaled-up + cross-architecture test of: is a contrastive readout DIRECTION read
by the model's downstream weights (computation nailed to weights), or is it just
f(prompts, W_U) (an output shadow only the unembedding sees)?

Per contrast we take dh = h_c - h_k at the last position, extracted at a mid
layer, and ask how much LATE-layer attention keys (W_K) and MLP detectors (fc1)
READ that direction -- a pure-weight quantity, normalized so a random direction
scores 1.0. The decisive control is a MISMATCH null: h_c minus h_k' from an
UNRELATED contrast. If matched ~ mismatched, the read is generic to "any
difference"; if matched > mismatched, the specific contrast is consumed.

MODE=word     (default): ~15 STRUCT vs ~15 FACT one-line contrasts (use Phi-2).
MODE=scenario         : the 4 Pythia-410M trajectory scenarios incl. plagiarism.

Usage:
  MODEL=microsoft/phi-2 MODE=word .venv/Scripts/python.exe contrastive/code/readout_weight_nailed_scaleup.py
  MODEL=EleutherAI/pythia-410m-deduped MODE=scenario .venv/Scripts/python.exe contrastive/code/readout_weight_nailed_scaleup.py
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
MODE = os.environ.get("MODE", "word")
print(f"Loading {MODEL} (MODE={MODE})...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

# ---- architecture adapters -------------------------------------------------
if hasattr(model, "gpt_neox"):                       # Pythia / GPT-NeoX
    ARCH = "neox"
    LAYERS = model.gpt_neox.layers
    W_U = model.embed_out.weight.detach().to(DEV)
    NH = model.config.num_attention_heads
    d = model.config.hidden_size
    HS = d // NH

    def get_WK(L):
        w = LAYERS[L].attention.query_key_value.weight.detach().to(DEV)  # (3d,d)
        w = w.view(NH, 3 * HS, d)
        return w[:, HS:2 * HS, :].reshape(NH * HS, d)                    # (d,d)

    def get_WFC1(L):
        return LAYERS[L].mlp.dense_h_to_4h.weight.detach().to(DEV)
else:                                                # Phi-2 style
    ARCH = "phi"
    LAYERS = model.model.layers
    W_U = model.lm_head.weight.detach().to(DEV)
    d = model.config.hidden_size

    def get_WK(L):
        return LAYERS[L].self_attn.k_proj.weight.detach().to(DEV)

    def get_WFC1(L):
        return LAYERS[L].mlp.fc1.weight.detach().to(DEV)

NL = len(LAYERS)
LATE = list(range(int(NL * 0.70), NL))
WK = {L: get_WK(L) for L in LATE}
FC1 = {L: get_WFC1(L) for L in LATE}
print(f"arch={ARCH} NL={NL} d={d} late-layers={LATE[0]}..{NL-1}")

# random baselines (full norm, per late layer) over NR dirs
torch.manual_seed(0)
NR = 40
RB_key = {L: 0.0 for L in LATE}
RB_fc1 = {L: 0.0 for L in LATE}
for _ in range(NR):
    u = torch.randn(d, device=DEV); u = u / u.norm()
    for L in LATE:
        RB_key[L] += float(torch.linalg.vector_norm(WK[L] @ u))
        RB_fc1[L] += float(torch.linalg.vector_norm(FC1[L] @ u))
for L in LATE:
    RB_key[L] /= NR; RB_fc1[L] /= NR


def weight_read_vec(u):
    u = u / u.norm()
    k = sum(float(torch.linalg.vector_norm(WK[L] @ u)) / RB_key[L] for L in LATE) / len(LATE)
    f = sum(float(torch.linalg.vector_norm(FC1[L] @ u)) / RB_fc1[L] for L in LATE) / len(LATE)
    return k, f


def hidden_at(text, L):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return out.hidden_states[L][0, -1, :].float()


def topk_tokens(vec, k=6):
    idx = torch.topk(vec @ W_U.T, k).indices.tolist()
    return ",".join(tok.decode([t]).strip() for t in idx)


WORD = [
    ("STRUCT", "She has finished the work", "She has not finished the work"),
    ("STRUCT", "Some of the students passed the exam", "All of the students passed the exam"),
    ("STRUCT", "He walks to the store every day", "He walked to the store every day"),
    ("STRUCT", "The bridge will collapse", "The bridge might collapse"),
    ("STRUCT", "The results were conclusive", "The results were not conclusive"),
    ("STRUCT", "Few people attended the meeting", "Many people attended the meeting"),
    ("STRUCT", "We need more time to finish", "We need less time to finish"),
    ("STRUCT", "If it rains we will cancel", "When it rains we will cancel"),
    ("STRUCT", "You must submit the form", "You may submit the form"),
    ("STRUCT", "The claim is true", "The claim is not true"),
    ("STRUCT", "They are building a house", "They were building a house"),
    ("STRUCT", "All of the answers were right", "None of the answers were right"),
    ("STRUCT", "The machine can run without power", "The machine cannot run without power"),
    ("STRUCT", "She bought the car yesterday", "She bought a car yesterday"),
    ("STRUCT", "The child played in the yard", "The children played in the yard"),
    ("FACT", "The capital of France is", "The capital of Germany is"),
    ("FACT", "The banana is", "The sky is"),
    ("FACT", "The theory of relativity was developed by", "The theory of evolution was developed by"),
    ("FACT", "The capital of Italy is", "The capital of Japan is"),
    ("FACT", "Romeo and Juliet was written by", "War and Peace was written by"),
    ("FACT", "The chemical symbol for gold is", "The chemical symbol for iron is"),
    ("FACT", "The largest planet in the solar system is", "The smallest planet in the solar system is"),
    ("FACT", "The currency of Japan is the", "The currency of India is the"),
    ("FACT", "In Brazil the official language is", "In Egypt the official language is"),
    ("FACT", "The capital of Spain is", "The capital of Russia is"),
    ("FACT", "The telephone was invented by", "The light bulb was invented by"),
    ("FACT", "A dog says", "A cat says"),
    ("FACT", "Egypt is located on the continent of", "Brazil is located on the continent of"),
    ("FACT", "The longest river in the world is the", "The longest river in Europe is the"),
    ("FACT", "The tallest mountain in the world is", "The tallest mountain in Africa is"),
]

SCEN = [
    ("plagiarism",
     "The student submitted a paper containing entire paragraphs copied from published articles without attribution. The plagiarism software flagged a 78 percent match. The university board voted to have the student",
     "The student submitted an original paper with proper citations throughout. The plagiarism software flagged a 2 percent match. The university board voted to have the student"),
    ("veterinary",
     "The small dog had ingested a large quantity of dark chocolate several hours earlier. By the time the owner arrived at the clinic, the animal was experiencing violent seizures and its kidneys were failing. After examining the dog, the veterinarian told the owner that the animal needed to be",
     "The large dog had eaten a small piece of milk chocolate a few minutes ago. The owner brought it in as a precaution and the animal was alert and playful. After examining the dog, the veterinarian told the owner that the animal needed to be"),
    ("software",
     "The application crashed intermittently under high load. Memory profiling revealed a steady increase in heap usage with no corresponding deallocation. After 72 hours of continuous operation, the server",
     "The application ran smoothly under high load. Memory profiling revealed stable heap usage with proper garbage collection. After 72 hours of continuous operation, the server"),
    ("cybersecurity",
     "The server had not been patched in over a year and was running a known vulnerable version of Apache. Logs showed repeated connection attempts from foreign IP addresses followed by large outbound data transfers. The company announced it had been",
     "The server was fully patched and running the latest version of Apache with all security updates. Logs showed normal traffic patterns with no anomalies. The company announced it had been"),
]

CONTRASTS = WORD if MODE == "word" else SCEN

if MODE == "scenario":
    # sweep the extraction layer across the trajectory; the plagiarism
    # computation lives mid-stack (punishment vocab), not at one early layer.
    def hid_all(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [out.hidden_states[L][0, -1, :].float() for L in range(NL + 1)]
    HC = [hid_all(c) for _, c, k in CONTRASTS]
    HK = [hid_all(k) for _, c, k in CONTRASTS]
    sweep = list(range(6, NL, 2))
    print(f"Extraction-layer sweep; match/MIS weight-read (downstream {LATE[0]}..{NL-1}).")
    print("match>MIS at a layer = the contrast direction is specifically read "
          "there.\n")
    hdr = "layer".rjust(6) + "".join(f"{n[:11]:>13}" for n, _, _ in CONTRASTS)
    print(hdr); print("-" * len(hdr))
    for L in sweep:
        cells = []
        for i, (name, c, k) in enumerate(CONTRASTS):
            j = (i + 2) % len(CONTRASTS)
            wm = weight_read_vec(HC[i][L] - HK[i][L])[0]
            wx = weight_read_vec(HC[i][L] - HK[j][L])[0]
            cells.append(f"{wm:>5.2f}/{wx:<5.2f}")
        print(f"{L:>6}" + "".join(f"{c:>13}" for c in cells))
    print("\n(numbers are match/MIS; look for plagiarism match pulling above MIS")
    print(" at a mid layer where cybersecurity does not)")
    # also show the readout at the layer of peak match for plagiarism
    print("\nReadouts along the plagiarism trajectory (c-k):")
    for L in range(8, NL, 2):
        print(f"  L{L:>2}: [{topk_tokens(HC[0][L] - HK[0][L], 6)}]")
    print("\nDONE")
    sys.exit(0)

EXTRACT = int(NL * 0.55)
print(f"extraction layer L={EXTRACT}; mismatch offset rotates baselines.\n")

# precompute hidden states
Hc = [hidden_at(c, EXTRACT) for _, c, k in CONTRASTS]
Hk = [hidden_at(k, EXTRACT) for _, c, k in CONTRASTS]

import statistics as st
rows_by_kind = {}
print(f"{'contrast':>14} | {'match':>6} {'MIS':>6} {'full':>6} | readout (c-k)")
print("-" * 90)
for i, (name, c, k) in enumerate(CONTRASTS):
    j = (i + max(3, len(CONTRASTS) // 2)) % len(CONTRASTS)
    dh_m = Hc[i] - Hk[i]
    dh_x = Hc[i] - Hk[j]
    wm = weight_read_vec(dh_m)[0]
    wx = weight_read_vec(dh_x)[0]
    wf = weight_read_vec(Hc[i])[0]
    rows_by_kind.setdefault(name.split()[0] if MODE == "word" else "SCEN", []).append((wm, wx))
    tag = name if MODE == "scenario" else name
    print(f"{tag:>14} | {wm:>6.2f} {wx:>6.2f} {wf:>6.2f} | [{topk_tokens(dh_m)}]")

print("-" * 90)
if MODE == "word":
    for kind in ["STRUCT", "FACT"]:
        rows = rows_by_kind[kind]
        m = st.mean(r[0] for r in rows); x = st.mean(r[1] for r in rows)
        nwin = sum(1 for r in rows if r[0] > r[1])
        print(f"MEAN {kind:>6} (n={len(rows)}): match={m:.2f}  MIS={x:.2f}  "
              f"gap={m-x:+.2f}  match>MIS in {nwin}/{len(rows)}")
else:
    rows = rows_by_kind["SCEN"]
    m = st.mean(r[0] for r in rows); x = st.mean(r[1] for r in rows)
    print(f"MEAN SCEN (n={len(rows)}): match={m:.2f}  MIS={x:.2f}  gap={m-x:+.2f}")
    print("(per-scenario match-MIS gap shown above is the computational signal;")
    print(" plagiarism/veterinary expected high, cybersecurity expected low)")
print("\nDONE")
