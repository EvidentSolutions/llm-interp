"""
Dense per-layer injection-recovery sweep for the four validation cases,
plotted as a line over ALL layers (paper Fig, replacing the sparse table
that a reviewer flagged as possibly cherry-picked through omission).

Recovery(L) = (P_inj - P_control) / (P_target - P_control) * 100, where
P_target is the target token's probability under the context prompt,
P_control under the control prompt, and P_inj after injecting
dh = h_context[L] - h_control[L] into the control's residual stream at the
last position of layer L. Identical construction to null_and_causal.py; here
every layer is swept and drawn.

Phi-2, fp32. Writes recovery_by_layer.{pdf,json} into ../docs and ../data.
Usage: .venv/Scripts/python.exe public/contrastive/code/recovery_sweep_plot.py
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

# (label, context prompt, control prompt, target token) -- exactly the four
# published cases from null_and_causal.py / tab:recovery.
CASES = [
    ("hot dog → delicious", "The hot dog was", "The cold dog was", "delicious"),
    ("IOI → Mary",
     "John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to", "Mary"),
    ("Eiffel → Paris", "The Eiffel Tower is located in",
     "The Colosseum is located in", "Paris"),
    ("Successor → Tuesday", "After Monday comes", "After Tuesday comes",
     "Tuesday"),
]
# Okabe-Ito colorblind-safe categorical palette, fixed order.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


def target_id(tok, w):
    for cand in (" " + w, w):
        t = tok(cand, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            return t[0]
    return tok(" " + w, add_special_tokens=False)["input_ids"][-1]


def main():
    print(f"Loading {MODEL} (fp32)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    NL = model.config.num_hidden_layers

    def recover(ctx, ctrl, tgt):
        ida = tok(ctx, add_special_tokens=False)["input_ids"]
        idb = tok(ctrl, add_special_tokens=False)["input_ids"]
        tid = target_id(tok, tgt)
        with torch.no_grad():
            oa = model(torch.tensor([ida], device=DEV), output_hidden_states=True)
            ob = model(torch.tensor([idb], device=DEV), output_hidden_states=True)
        p_a = float(torch.softmax(oa.logits[0, -1].float(), -1)[tid])
        p_b = float(torch.softmax(ob.logits[0, -1].float(), -1)[tid])
        gap = p_a - p_b
        recs = []
        for L in range(NL + 1):
            dh = (oa.hidden_states[L][0, -1, :] - ob.hidden_states[L][0, -1, :]).detach()
            done = [False]

            def hook(m, i, o):
                if done[0]:
                    return o
                done[0] = True
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] += dh; return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] += dh; return h
            mod = (model.model.layers[L] if L < NL
                   else model.model.final_layernorm)
            handle = mod.register_forward_hook(hook)
            with torch.no_grad():
                oi = model(torch.tensor([idb], device=DEV))
            handle.remove()
            p_i = float(torch.softmax(oi.logits[0, -1].float(), -1)[tid])
            recs.append(100 * (p_i - p_b) / gap if gap != 0 else 0.0)
        return recs, p_a, p_b

    data = {}
    for label, ctx, ctrl, tgt in CASES:
        recs, p_a, p_b = recover(ctx, ctrl, tgt)
        data[label] = recs
        anchors = {L: round(recs[L], 0) for L in (12, 20, 24, 28) if L <= NL}
        print(f"{label}: P_ctx={p_a:.3f} P_ctrl={p_b:.3f}  anchors {anchors}")

    docs = os.path.join(os.path.dirname(__file__), "..", "docs")
    with open(os.path.join(os.path.dirname(__file__), "..", "data",
              "recovery_by_layer.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = list(range(NL + 1))
    for (label, *_), c in zip(CASES, COLORS):
        ys = data[label]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=3.2, label=label)
    ax.axhline(100, color="#888888", lw=1, ls="--", zorder=0)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("Prediction-gap recovery (%)")
    ax.set_xlim(0, NL + 3)
    ax.grid(True, axis="y", color="#eeeeee", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(docs, "recovery_by_layer.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
