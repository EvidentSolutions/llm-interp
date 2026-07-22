"""
Truth-direction triviality diagnostic (audit for paper Section 3.2).

Question: is the "truth direction" injection a trivial late-layer
output-override (writing the " true"/" false" token onto the logits) or a
manipulation of a genuine internal representation?

Three controls added to the truth_direction_scaleup.py setup:
  1. W_U cosine: cos(truth_dir, WU[" true"] - WU[" false"]). If high, the
     direction largely IS the output-token difference axis.
  2. Layer sweep: build the L-specific truth direction at several layers and
     inject there. If it only flips late (L20+) and does nothing mid, that is
     the output-override signature. If it builds from mid-layers, that argues
     for a manipulable representation.
  3. Restricted-subspace test (as triangulation Section 3.5): inject only the
     top-20/bot-20 W_U-token-subspace component of the truth direction. Report
     whether that component carries the whole flip, and whether the subspace is
     dominated by the " true"/" false" tokens themselves.

Usage: .venv/Scripts/python.exe contrastive/code/truth_direction_diagnostic.py
"""
import sys, os, itertools
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
LAYERS_SWEEP = [4, 8, 12, 16, 20, 24, 28]
SCALE = 2.0

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL)
layers = model.model.layers
N_LAYERS = len(layers)

# unembed matrix + final layernorm
W_U = model.lm_head.weight.data.float()          # [vocab, d]
ln_f = model.model.final_layernorm

TRUE = tok(" true", add_special_tokens=False)["input_ids"][0]
FALSE = tok(" false", add_special_tokens=False)["input_ids"][0]

PAIRS = [
    ("Dogs are mammals", "Dogs are reptiles"),
    ("The Sun is a star", "The Sun is a planet"),
    ("Ice is cold", "Ice is hot"),
    ("Fish live in water", "Fish live in fire"),
    ("Gold is a metal", "Gold is a gas"),
    ("Birds have feathers", "Birds have scales"),
    ("The Earth is round", "The Earth is flat"),
    ("Lions are animals", "Lions are plants"),
    ("Snow is white", "Snow is black"),
    ("Fire is hot", "Fire is cold"),
    ("Humans breathe air", "Humans breathe water"),
    ("Iron is a metal", "Iron is a liquid"),
    ("Cats are animals", "Cats are vegetables"),
    ("The ocean is salty", "The ocean is sweet"),
    ("Bees make honey", "Bees make milk"),
]


def toks(p):
    return tok(p, add_special_tokens=False)["input_ids"]


def last_vec(statement, L):
    p = statement + ". This statement is"
    with torch.no_grad():
        o = model(torch.tensor([toks(p)], device=DEV), output_hidden_states=True)
    return o.hidden_states[L][0, -1].float()


def predicts_true(statement, d, sign, L):
    p = statement + ". This statement is"

    def hook(m, i, o):
        oo = o[0].clone()
        oo[:, -1, :] = oo[:, -1, :] + sign * d.to(oo.dtype)
        return (oo,) + o[1:]
    h = layers[L - 1].register_forward_hook(hook)
    with torch.no_grad():
        lg = model(torch.tensor([toks(p)], device=DEV)).logits[0, -1].float()
    h.remove()
    return lg[TRUE].item() > lg[FALSE].item()


def dirs_at(L):
    return [last_vec(t, L) - last_vec(f, L) for t, f in PAIRS]


def loo(dirs, i, scale=SCALE):
    return torch.stack([dirs[j] for j in range(len(dirs)) if j != i]).mean(0) * scale


def flip_rates(dirs, L, project=None):
    f2t = t2f = 0
    for i, (t, f) in enumerate(PAIRS):
        d = loo(dirs, i)
        if project is not None:
            d = project(d)
        if predicts_true(f, d, +1, L):
            f2t += 1
        if not predicts_true(t, d, -1, L):
            t2f += 1
    return f2t, t2f


def project_token_subspace(d, k=20):
    """Project d onto span of the top-k and bottom-k W_U rows by W_U@d,
    QR-orthogonalised. Returns the component of d in that subspace."""
    scores = W_U @ d                      # [vocab]
    top = torch.topk(scores, k).indices
    bot = torch.topk(-scores, k).indices
    idx = torch.cat([top, bot])
    B = W_U[idx].T                        # [d, 2k]
    Q, _ = torch.linalg.qr(B)             # [d, 2k] orthonormal
    return Q @ (Q.T @ d)


def main():
    print(f"Model: {MODEL}  ({N_LAYERS} layers)  N={len(PAIRS)} pairs  scale {SCALE}x")
    print(f"TRUE id {TRUE} ({tok.decode([TRUE])!r})  FALSE id {FALSE} ({tok.decode([FALSE])!r})")

    # ---- Control 1: W_U cosine ----------------------------------------
    wu_diff = (W_U[TRUE] - W_U[FALSE])
    print("\n[1] W_U cosine of the truth direction with WU[' true']-WU[' false']:")
    for L in LAYERS_SWEEP:
        dirs = dirs_at(L)
        mean_dir = torch.stack(dirs).mean(0)
        c = torch.nn.functional.cosine_similarity(mean_dir, wu_diff, 0).item()
        # also within-set consistency at this layer
        coss = [torch.nn.functional.cosine_similarity(dirs[i], dirs[j], 0).item()
                for i, j in itertools.combinations(range(len(dirs)), 2)]
        print(f"   L{L:2d}: cos(dir, WU_true-WU_false) = {c:+.3f}   "
              f"mean pairwise cos among dirs = {sum(coss)/len(coss):+.2f}")

    # ---- Control 2: layer sweep (full direction) ----------------------
    print("\n[2] Layer sweep - inject L-specific LOO direction at that layer:")
    dirs_by_L = {}
    for L in LAYERS_SWEEP:
        dirs = dirs_at(L)
        dirs_by_L[L] = dirs
        f2t, t2f = flip_rates(dirs, L)
        print(f"   L{L:2d}: false->true {f2t:2d}/{len(PAIRS)}, "
              f"true->false {t2f:2d}/{len(PAIRS)}")

    # ---- Control 3: restricted subspace at each layer -----------------
    print("\n[3] Restricted top-20/bot-20 W_U-subspace component only:")
    for L in LAYERS_SWEEP:
        dirs = dirs_by_L[L]
        # fraction of the mean direction's norm captured by the subspace
        mean_dir = torch.stack(dirs).mean(0)
        proj = project_token_subspace(mean_dir)
        frac = (proj.norm() / mean_dir.norm()).item()
        f2t, t2f = flip_rates(dirs, L, project=project_token_subspace)
        # is the subspace dominated by true/false tokens?
        scores = W_U @ mean_dir
        top = torch.topk(scores, 20).indices.tolist()
        bot = torch.topk(-scores, 20).indices.tolist()
        tf_in_top = TRUE in top or FALSE in top
        tf_in_bot = TRUE in bot or FALSE in bot
        top_toks = [tok.decode([t]) for t in top[:8]]
        bot_toks = [tok.decode([t]) for t in bot[:8]]
        print(f"   L{L:2d}: subspace captures {frac*100:4.1f}% of |dir|  "
              f"flips f->t {f2t:2d}/{len(PAIRS)} t->f {t2f:2d}/{len(PAIRS)}  "
              f"[true/false in top:{tf_in_top} bot:{tf_in_bot}]")
        print(f"        top+: {top_toks}")
        print(f"        top-: {bot_toks}")


if __name__ == "__main__":
    main()
