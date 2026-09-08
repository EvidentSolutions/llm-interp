"""
Can the paper's OWN tools (preamble + multi-contrast triangulation) get a
clean noun-position read of the food compound in Pythia/Qwen, where a single
pair fails? (Mitigation attempt for the cross-model section.)

At the noun ("dog") position, read the food-compound direction:
  single-pair : h(hot dog) - h(cold dog)
  triangulated: mean over X in {cold,angry,old,pet,stray} of h(hot dog)-h(X dog)
Both inside the shared preamble. Report top-k tokens + the best food rank
per layer, for Phi-2 (sanity), Pythia-1.4b, Qwen2.5-1.5b. fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/compound_noun_triangulate.py
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRE = os.environ.get("PRE", "Everyone agreed that ")
MODELS = ["microsoft/phi-2", "EleutherAI/pythia-1.4b", "Qwen/Qwen2.5-1.5B"]
FOOD = ["fried", "cooked", "delicious", "tasty", "crispy", "grilled", "edible",
        "flavor", "food", "sausage", "meat", "eaten", "juicy", "savory",
        "roasted", "seasoned", "burger", "snack", "bun", "ketchup"]
ADJ = ["cold", "angry", "old", "pet", "stray"]   # triangulation baselines


def get_layers_unembed(m):
    if hasattr(m, "gpt_neox"):
        return m.gpt_neox.layers, m.embed_out.weight
    return m.model.layers, m.lm_head.weight


def food_variants(tok):
    ids = set()
    for w in FOOD:
        for cand in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            t = tok(cand, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
    return list(ids)


def run(MODEL):
    print(f"\n{'#'*90}\n# {MODEL}\n{'#'*90}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    layers, W_U = get_layers_unembed(m)
    W_U = W_U.detach().float()
    NL = m.config.num_hidden_layers
    food_ids = set(food_variants(tok))

    def hidden_at_dog(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        toks = [tok.decode([t]) for t in ids]
        # find "dog" position
        dpos = next((i for i, t in enumerate(toks) if "dog" in t.lower()), len(ids) - 2)
        with torch.no_grad():
            out = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [out.hidden_states[L][0, dpos] for L in range(NL + 1)]

    h_hot = hidden_at_dog(PRE + "the hot dog was")
    h_base = {a: hidden_at_dog(PRE + f"the {a} dog was") for a in ADJ}

    def reads(vec, k=6):
        i = torch.topk(vec @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)

    def food_rank(vec):
        order = torch.argsort(vec @ W_U.T, descending=True).tolist()[:300]
        rr = [order.index(t) for t in food_ids if t in order]
        return min(rr) if rr else None

    print(f"  noun-position read, single-pair (hot-cold) vs triangulated (mean over {ADJ})")
    print(f"  {'L':>3} | {'sp rank':>7} {'single-pair reads':<40} | {'tri rank':>7} triangulated reads")
    out = {"single": {}, "tri": {}}
    for L in range(NL + 1):
        sp = h_hot[L] - h_base["cold"][L]
        tri = torch.stack([h_hot[L] - h_base[a][L] for a in ADJ]).mean(0)
        rs, rt = food_rank(sp), food_rank(tri)
        out["single"][L] = rs
        out["tri"][L] = rt
        mark = " <=" if (rt is not None and rt < 10) else ""
        print(f"  {L:>3} | {str(rs):>7} [{reads(sp):<38}] | {str(rt):>7} [{reads(tri)}]{mark}")
    best_sp = min([v for v in out["single"].values() if v is not None], default=None)
    best_tri = min([v for v in out["tri"].values() if v is not None], default=None)
    L_sp = next((L for L in range(NL+1) if out["single"][L] is not None and out["single"][L] < 10), None)
    L_tri = next((L for L in range(NL+1) if out["tri"][L] is not None and out["tri"][L] < 10), None)
    print(f"  -> best food rank: single-pair {best_sp} (top-10 first at L{L_sp}); "
          f"triangulated {best_tri} (top-10 first at L{L_tri})")
    del m
    torch.cuda.empty_cache()
    return dict(best_single=best_sp, best_tri=best_tri, L_single=L_sp, L_tri=L_tri)


def main():
    res = {}
    for M in MODELS:
        try:
            res[M] = run(M)
        except Exception as e:
            import traceback; traceback.print_exc()
            res[M] = {"error": str(e)}
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "compound_noun_triangulate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nSUMMARY:")
    for M, r in res.items():
        print(f"  {M}: {r}")


if __name__ == "__main__":
    main()
