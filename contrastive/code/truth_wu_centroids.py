"""
Are ' true'/' false' the W_U centroids of the truth-valence groups?

The truth direction's late-layer readable component decodes to a positive
valence group ("observable, universal, measurable...") and a negative group
("incorrect, wrong, misguided...") that are NOT the ' true'/' false' output
tokens. This asks whether ' true'/' false' nonetheless sit at the centres of
those groups in W_U row space, i.e. whether the valence constellation is just
a cloud around the two output tokens.

For the late-layer (L24/L28) mean truth direction:
  - take top-20 (positive) and bot-20 (negative) W_U-scored tokens,
  - centroid of each group's W_U rows,
  - cosine of WU[' true'] to (+centroid, -centroid) and WU[' false'] likewise,
  - rank of ' true'/' false' among all vocab by cosine-to-centroid,
  - and how the group centroids relate to the WU true-false axis itself.

Usage: .venv/Scripts/python.exe contrastive/code/truth_wu_centroids.py
"""
import sys, os
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
LAYERS = [20, 24, 28]

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL)
layers = model.model.layers
W_U = model.lm_head.weight.data.float()          # [vocab, d]
Wn = W_U / W_U.norm(dim=1, keepdim=True)          # unit rows

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


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, 0).item()


def rank_by_cos(vec):
    """rank of ' true'/' false' among all vocab by cosine to vec (0 = closest)."""
    sims = Wn @ (vec / vec.norm())
    order = torch.argsort(sims, descending=True)
    pos = {t.item(): i for i, t in enumerate(order)}
    return pos[TRUE], pos[FALSE], sims[TRUE].item(), sims[FALSE].item()


def main():
    print(f"Model: {MODEL}   TRUE id {TRUE}  FALSE id {FALSE}")
    wu_true, wu_false = W_U[TRUE], W_U[FALSE]
    axis = wu_true - wu_false
    print(f"cos(WU_true, WU_false) = {cos(wu_true, wu_false):+.3f}   "
          f"|WU_true|={wu_true.norm():.2f} |WU_false|={wu_false.norm():.2f}\n")

    for L in LAYERS:
        dirs = [last_vec(t, L) - last_vec(f, L) for t, f in PAIRS]
        mean_dir = torch.stack(dirs).mean(0)
        scores = W_U @ mean_dir
        top = torch.topk(scores, 20).indices
        bot = torch.topk(-scores, 20).indices
        pos_c = W_U[top].mean(0)     # positive-valence group centroid
        neg_c = W_U[bot].mean(0)     # negative-valence group centroid

        print(f"=== L{L} ===")
        print(f"  top+ : {[tok.decode([t]) for t in top[:10].tolist()]}")
        print(f"  top- : {[tok.decode([t]) for t in bot[:10].tolist()]}")

        # Is ' true' the centroid of the positive group / ' false' of negative?
        print(f"  cos(WU_true , +centroid) = {cos(wu_true, pos_c):+.3f}   "
              f"cos(WU_true , -centroid) = {cos(wu_true, neg_c):+.3f}")
        print(f"  cos(WU_false, +centroid) = {cos(wu_false, pos_c):+.3f}   "
              f"cos(WU_false, -centroid) = {cos(wu_false, neg_c):+.3f}")

        # rank of true/false relative to each centroid
        tr_p, fa_p, str_p, sfa_p = rank_by_cos(pos_c)
        tr_n, fa_n, str_n, sfa_n = rank_by_cos(neg_c)
        print(f"  +centroid: ' true' rank {tr_p} (cos {str_p:+.3f}), "
              f"' false' rank {fa_p} (cos {sfa_p:+.3f})")
        print(f"  -centroid: ' true' rank {tr_n} (cos {str_n:+.3f}), "
              f"' false' rank {fa_n} (cos {sfa_n:+.3f})")

        # within-group tightness + how the centroid-axis aligns with true-false axis
        cax = pos_c - neg_c
        print(f"  cos(+cent - -cent, WU_true - WU_false) = {cos(cax, axis):+.3f}")
        # mean cos of group members to their own centroid (tightness)
        tp = torch.stack([cos(W_U[t], pos_c) for t in top]).mean() if False else \
             sum(cos(W_U[t], pos_c) for t in top) / len(top)
        tn = sum(cos(W_U[t], neg_c) for t in bot) / len(bot)
        print(f"  within-group tightness: +grp {tp:+.3f}, -grp {tn:+.3f}\n")


if __name__ == "__main__":
    main()
