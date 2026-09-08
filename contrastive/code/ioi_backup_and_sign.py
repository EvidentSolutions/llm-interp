"""
IOI sharpening (paper Sec 5.1): self-repair/backup heads + name-mover sign.

The published section claims the flagged L24 heads (H14, H1, H16) are
name-movers (H1 a sink-head artifact), and that single-head ablation is
near-zero because the decision "rides the cumulative residual stream." Two
post-publication results sharpen this:

  (A) Self-repair / Hydra effect (McGrath 2023; Wang 2022 backup name-movers;
      Rushing & Nanda 2024): ablating a primary name-mover activates dormant
      BACKUP name-movers that take over, which is *why* single-head ablation
      is small -- a measurable compensation, not diffuse distribution.
  (B) Copy-suppression / negative name-movers (McDougall 2023): heads that
      attend to the IO token and *suppress* it. The paper's filter (high
      contrastive norm + attends-to-IO) cannot tell these from name-movers,
      because both attend to IO. The distinguishing test is the SIGN of the
      head's direct write to the IO-vs-S logit.

We measure per-head Direct Logit Attribution (DLA) at END:
    DLA(L,h) = [ oproj_input[END, head_slice] @ W_o[:, head_slice] ] . (W_U[IO] - W_U[S])
i.e. the head's own additive contribution to the (IO - S) logit difference.
  SIGN CHECK: DLA > 0 => name-mover (boosts IO); DLA < 0 => copy-suppression.
  BACKUP TEST: ablate the top positive-DLA heads (zero their END slice), then
    recompute every head's DLA; heads whose DLA rises are backups (self-repair).

Phi-2, fp32. Usage:
  .venv/Scripts/python.exe public/contrastive/code/ioi_backup_and_sign.py
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

CASES = [
    ("When John and Mary went to the store, John gave a drink to", "Mary", "John"),
    ("When Alice and Bob went to the park, Alice gave a gift to", "Bob", "Alice"),
    ("When Dan and Eve went to the office, Dan gave a letter to", "Eve", "Dan"),
    ("When Sarah and Tom went to the beach, Sarah gave a towel to", "Tom", "Sarah"),
]
L_FLAG = 24
FLAGGED = [14, 1, 16]   # the paper's flagged heads at L24


def main():
    print(f"Loading {MODEL} (fp32)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager").to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    NL = model.config.num_hidden_layers
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
    W_U = model.lm_head.weight.detach().float()

    def sid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    def dla_all(ids, dir_vec, ablate=None):
        """Per-(L,h) DLA at END onto dir_vec. ablate: set of (L,h) to zero at END."""
        ablate = ablate or set()
        inbuf = {}
        handles = []
        for L in range(NL):
            op = layers[L].self_attn.dense
            handles.append(op.register_forward_pre_hook(
                (lambda L: lambda m, x: _cap_and_ablate(L, x, inbuf, ablate, HD))(L)))
        with torch.no_grad():
            model(torch.tensor([ids], device=DEV))
        for h in handles:
            h.remove()
        dvec = dir_vec.detach().float().cpu()
        dla = torch.zeros(NL, NH)
        for L in range(NL):
            O_w = layers[L].self_attn.dense.weight.detach().float().cpu()  # [hid, NH*HD]
            end = inbuf[L]  # [NH*HD] at END, cpu
            for h in range(NH):
                sl = slice(h * HD, (h + 1) * HD)
                contrib = end[sl] @ O_w[:, sl].T
                dla[L, h] = float(contrib @ dvec)
        return dla

    # ---- baseline DLA averaged over cases ----
    dla_sum = torch.zeros(NL, NH)
    per_case_flagged = []
    for prompt, io, s in CASES:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        d = (W_U[sid(io)] - W_U[sid(s)])
        dla = dla_all(ids, d)
        dla_sum += dla
        per_case_flagged.append([round(float(dla[L_FLAG, h]), 2) for h in FLAGGED])
    dla_avg = dla_sum / len(CASES)

    print("\n=== SIGN CHECK: per-head DLA at END onto (IO - S) logit direction ===")
    print("  (DLA>0 = name-mover / boosts IO;  DLA<0 = copy-suppression / suppresses IO)")
    flat = [(float(dla_avg[L, h]), L, h) for L in range(NL) for h in range(NH)]
    flat.sort(key=lambda r: -r[0])
    print("  top name-movers (highest +DLA):")
    for v, L, h in flat[:8]:
        print(f"    L{L}.H{h:<3} DLA={v:+.2f}")
    print("  top copy-suppressors (most -DLA):")
    for v, L, h in flat[-8:][::-1]:
        print(f"    L{L}.H{h:<3} DLA={v:+.2f}")

    print(f"\n  the paper's flagged L{L_FLAG} heads:")
    for i, h in enumerate(FLAGGED):
        per = [c[i] for c in per_case_flagged]
        verdict = ("name-mover" if dla_avg[L_FLAG, h] > 0.3 else
                   "copy-suppression" if dla_avg[L_FLAG, h] < -0.3 else
                   "~neutral (sink/other)")
        print(f"    L{L_FLAG}.H{h:<3} DLA={float(dla_avg[L_FLAG, h]):+.2f} "
              f"per-case {per}  -> {verdict}")

    # ---- BACKUP TEST: ablate top-k primary name-movers, recompute DLA ----
    K = 3
    primaries = [(L, h) for _, L, h in flat[:K]]
    print(f"\n=== BACKUP / SELF-REPAIR: ablate top-{K} name-movers "
          f"{[f'L{L}.H{h}' for L, h in primaries]} at END, recompute DLA ===")
    rises = torch.zeros(NL, NH)
    base_sum = torch.zeros(NL, NH)
    abl_sum = torch.zeros(NL, NH)
    prim_dla_base = 0.0
    prim_dla_abl = 0.0
    for prompt, io, s in CASES:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        d = (W_U[sid(io)] - W_U[sid(s)])
        base = dla_all(ids, d)
        abl = dla_all(ids, d, ablate=set(primaries))
        base_sum += base
        abl_sum += abl
        prim_dla_base += sum(float(base[L, h]) for L, h in primaries)
        prim_dla_abl += sum(float(abl[L, h]) for L, h in primaries)
    base_avg = base_sum / len(CASES)
    abl_avg = abl_sum / len(CASES)
    # only count non-primary heads; a rise = compensation
    rises = abl_avg - base_avg
    for L, h in primaries:
        rises[L, h] = -999  # exclude primaries themselves
    rl = [(float(rises[L, h]), float(base_avg[L, h]), float(abl_avg[L, h]), L, h)
          for L in range(NL) for h in range(NH) if rises[L, h] > -900]
    rl.sort(key=lambda r: -r[0])
    total_backup_rise = sum(max(0.0, r[0]) for r in rl)
    print(f"  primaries' own summed DLA: {prim_dla_base/len(CASES):+.2f} (base) "
          f"-> {prim_dla_abl/len(CASES):+.2f} (ablated, ~0 expected)")
    print(f"  heads whose name-mover DLA RISES most under ablation (backups):")
    print(f"    {'head':<9} {'base':>6} {'ablated':>8} {'rise':>6}")
    for rise, b, a, L, h in rl[:8]:
        tag = " [backup]" if rise > 0.1 else ""
        print(f"    L{L}.H{h:<3} {b:>+6.2f} {a:>+8.2f} {rise:>+6.2f}{tag}")
    print(f"  total positive DLA recovered by backups: {total_backup_rise:+.2f} "
          f"(vs primaries' lost {prim_dla_base/len(CASES) - prim_dla_abl/len(CASES):+.2f})")

    out = {
        "flagged_dla": {f"L{L_FLAG}.H{h}": round(float(dla_avg[L_FLAG, h]), 3)
                        for h in FLAGGED},
        "top_name_movers": [{"L": L, "H": h, "dla": round(v, 3)}
                            for v, L, h in flat[:8]],
        "top_suppressors": [{"L": L, "H": h, "dla": round(v, 3)}
                            for v, L, h in flat[-8:][::-1]],
        "primaries_ablated": [f"L{L}.H{h}" for L, h in primaries],
        "backups": [{"L": L, "H": h, "base": round(b, 3), "ablated": round(a, 3),
                     "rise": round(rise, 3)} for rise, b, a, L, h in rl[:8]],
        "primary_dla_base": round(prim_dla_base / len(CASES), 3),
        "primary_dla_ablated": round(prim_dla_abl / len(CASES), 3),
        "total_backup_rise": round(total_backup_rise, 3),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "ioi_backup_and_sign.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


def _cap_and_ablate(L, x, inbuf, ablate, HD):
    t = x[0].clone()
    for (aL, ah) in ablate:
        if aL == L:
            t[0, -1, ah * HD:(ah + 1) * HD] = 0.0
    inbuf[L] = t[0, -1].detach().float().cpu()
    return (t,) + tuple(x[1:])


if __name__ == "__main__":
    main()
