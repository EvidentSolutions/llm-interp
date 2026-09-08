"""
Cross-model replication of the compound-noun two-hop circuit (paper Sec 4.1).

The Phi-2 finding: for "The hot dog was" vs "The cold dog was", the MLP at
the NOUN position recognizes the food compound at some layer L_mlp (food
vocabulary first appears in the MLP output, not attention), and one or more
attention heads then ROUTE that signal from the noun to the "was" prediction
position at a later layer L_route. A two-hop MLP->attention chain.

This script re-runs that decomposition on an arbitrary model and reports,
architecture-agnostically:
  1. Per-position contrastive trace at the noun and "was" positions across layers.
  2. Sub-layer attn-vs-MLP contrastive norm at the noun position -> which
     block writes the compound, and at which layer (L_mlp).
  3. Whether food vocabulary appears in the MLP output but not the attention
     output at L_mlp (the recognition signature).
  4. Routing: per-head decomposition at "was" -> which head carries food
     content forward, and its attention weight onto the noun (L_route).

Residual algebra used (valid for BOTH parallel residual (Phi-2, Pythia-NeoX)
and sequential residual (Qwen2)):
    hidden_states[L+1] = hidden_states[L] + attn_out[L] + mlp_out[L]
so with attn_out captured at the o-projection output,
    mlp_out[L] = hidden_states[L+1] - hidden_states[L] - attn_out[L].

Models (set via MODEL env, or loops over the default battery):
  microsoft/phi-2            (PhiForCausalLM,     MHA, GELU fc1/fc2, parallel)
  EleutherAI/pythia-1.4b     (GPTNeoXForCausalLM, MHA, GELU h_to_4h, parallel)
  Qwen/Qwen2.5-1.5B          (Qwen2ForCausalLM,   GQA, SwiGLU,       sequential)

Usage:
  .venv/Scripts/python.exe public/contrastive/code/compound_noun_multimodel.py
  MODEL=EleutherAI/pythia-1.4b .venv/Scripts/python.exe .../compound_noun_multimodel.py
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FOOD = ["fried", "cooked", "delicious", "tasty", "crispy", "grilled",
        "edible", "flavor", "food", "sausage", "meat", "eaten"]
DEFAULT_BATTERY = ["microsoft/phi-2", "EleutherAI/pythia-1.4b",
                   "Qwen/Qwen2.5-1.5B"]


def get_arch(model):
    """Return (layers_list, attn_attr, oproj_attr, unembed_weight) for the model."""
    m = model
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        layers = m.model.layers                       # Phi-2, Qwen2
    elif hasattr(m, "gpt_neox"):
        layers = m.gpt_neox.layers                    # Pythia
    else:
        raise RuntimeError("unknown layer container")
    attn0 = layers[0]
    attn_attr = "self_attn" if hasattr(attn0, "self_attn") else "attention"
    a = getattr(attn0, attn_attr)
    oproj_attr = "o_proj" if hasattr(a, "o_proj") else "dense"
    if hasattr(m, "lm_head"):
        W_U = m.lm_head.weight.detach().float()
    else:
        W_U = m.embed_out.weight.detach().float()
    return layers, attn_attr, oproj_attr, W_U


def main():
    models = [os.environ["MODEL"]] if "MODEL" in os.environ else DEFAULT_BATTERY
    all_results = {}
    for MODEL in models:
        try:
            all_results[MODEL] = run_model(MODEL)
        except Exception as e:
            print(f"\n!!! {MODEL} FAILED: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()
    out = os.path.join(os.path.dirname(__file__), "..", "data",
                       "compound_noun_multimodel.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {out}")


def run_model(MODEL):
    print(f"\n{'#'*100}\n# {MODEL}\n{'#'*100}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp32: these models are small (<=2.7B) and fp32 avoids the late-layer
    # massive-activation overflow that produces NaN logit projections.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager").to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers, ATTN, OPROJ, W_U = get_arch(model)
    W_Uc = W_U.cpu()
    NL = model.config.num_hidden_layers
    NH = model.config.num_attention_heads
    HID = model.config.hidden_size
    HD = HID // NH

    def toks(p):
        return tok(p, add_special_tokens=False)["input_ids"]

    def sid(w):
        for cand in (" " + w, w):
            t = tok(cand, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                return t[0]
        return None

    def tk(vec, k=6):
        i = torch.topk(vec.float(), k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)

    # Food direction: space-prefixed anchors (prediction-position convention).
    food_ids = [x for x in (sid(w) for w in FOOD) if x is not None]
    fd = W_Uc[food_ids].mean(0)
    fd = fd / fd.norm()

    # Detection set for food_rank: include non-space and capitalized single-token
    # variants, since the noun-internal representation may surface a different
    # token id than the space-prefixed prediction-position one.
    def all_variants(w):
        out = []
        for cand in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            t = tok(cand, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                out.append(t[0])
        return out
    food_det = set()
    for w in FOOD:
        food_det.update(all_variants(w))
    food_det = list(food_det)

    # PRE: shared preamble for read stability (paper Sec 2.4). Bare reads at
    # unstable positions decode to junk; a shared preamble that cancels under
    # subtraction moves the read into a stable region. Default on for a fair
    # cross-model test; set PRE="" to reproduce the bare-prompt behavior.
    PRE = os.environ.get("PRE", "Everyone agreed that ")
    hp, cp = PRE + "the hot dog was", PRE + "the cold dog was"
    hp_ids, cp_ids = toks(hp), toks(cp)
    # dynamic positions (tokenizers differ on leading space / BOS)
    try:
        dog_pos = hp_ids.index(sid("dog"))
    except ValueError:
        dog_pos = 2
    was_pos = len(hp_ids) - 1
    print(f"arch: layers={NL} heads={NH} head_dim={HD} attn={ATTN}.{OPROJ}")
    print(f"tokens hot: {[tok.decode([t]) for t in hp_ids]}  dog@{dog_pos} was@{was_pos}")
    print(f"food anchors used: {len(food_ids)}")

    def sublayer_states(ids):
        """Return per-layer dict with pre/attn/mlp/post residual vectors."""
        attn_outs = {}
        handles = []
        for L in range(NL):
            oproj = getattr(getattr(layers[L], ATTN), OPROJ)
            handles.append(oproj.register_forward_hook(
                (lambda L: lambda m, i, o: attn_outs.__setitem__(
                    L, (o[0] if isinstance(o, tuple) else o).detach().float()))(L)))
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
        for h in handles:
            h.remove()
        res = {}
        for L in range(NL):
            pre = out.hidden_states[L][0].float()
            post = out.hidden_states[L + 1][0].float()
            attn = attn_outs[L][0].float()
            mlp = post - pre - attn
            res[L] = dict(pre=pre.cpu(), attn=attn.cpu(),
                          mlp=mlp.cpu(), post=post.cpu())
        return res

    ts = sublayer_states(hp_ids)
    cs = sublayer_states(cp_ids)
    result = dict(NL=NL, NH=NH, dog_pos=dog_pos, was_pos=was_pos)

    def food_rank(vec):
        """Best (smallest) rank of any food anchor in this projection, or None."""
        order = torch.argsort(vec.float(), descending=True)
        pos = {int(t): i for i, t in enumerate(order.tolist()[:200])}
        rr = [pos[t] for t in food_det if t in pos]
        return min(rr) if rr else None

    # ---- 1: cumulative-residual trace at the NOUN position ----
    # (the paper's "post-MLP: fried appears" read is the cumulative residual,
    #  not the isolated MLP write vector, which is off-manifold.)
    print(f"\n=== cumulative residual trace at noun position ===")
    print(f"  {'L':>3} {'d_post':>7} {'foodrank':>8}  reads")
    noun_post = {}
    for L in range(NL):
        dh = ts[L]["post"][dog_pos] - cs[L]["post"][dog_pos]
        fr = food_rank(dh @ W_Uc.T)
        noun_post[L] = dict(norm=round(float(dh.norm()), 2), food_rank=fr)
        if L < min(NL, 14):
            print(f"  L{L:>2} {float(dh.norm()):>7.1f} {str(fr):>8}  "
                  f"[{tk(dh @ W_Uc.T, 5)}]")
    # L_recog = first layer where a food anchor enters the noun's top-10 residual
    L_recog = next((L for L in range(NL)
                    if noun_post[L]["food_rank"] is not None
                    and noun_post[L]["food_rank"] < 10), None)
    result["noun_post"] = noun_post
    result["L_recog"] = L_recog

    # ---- 2: sub-layer attn-vs-MLP norm at the noun position ----
    print(f"\n=== sub-layer attn-vs-MLP write at noun position ===")
    print(f"  {'L':>3} {'d_attn':>7} {'d_mlp':>7} {'mlp%':>5}")
    sub = {}
    for L in range(NL):
        d_attn = ts[L]["attn"][dog_pos] - cs[L]["attn"][dog_pos]
        d_mlp = ts[L]["mlp"][dog_pos] - cs[L]["mlp"][dog_pos]
        na, nm = float(d_attn.norm()), float(d_mlp.norm())
        mlp_pct = 100 * nm / (na + nm + 1e-9)
        sub[L] = dict(d_attn=round(na, 2), d_mlp=round(nm, 2),
                      mlp_pct=round(mlp_pct, 1))
        if L < min(NL, 14):
            print(f"  L{L:>2} {na:>7.1f} {nm:>7.1f} {mlp_pct:>4.0f}%")
    result["noun_sublayer"] = sub
    # is the recognition write MLP-dominated? (avg mlp% over L_recog-1..L_recog)
    if L_recog is not None:
        win = [sub[L]["mlp_pct"] for L in range(max(0, L_recog - 1), L_recog + 1)]
        result["recog_mlp_pct"] = round(sum(win) / len(win), 1)
    print(f"\n  -> food enters noun residual at L_recog = {L_recog}"
          + (f"; that write is {result['recog_mlp_pct']:.0f}% MLP"
             if L_recog is not None else ""))

    # ---- 3: cumulative-residual trace at 'was' (prediction site) ----
    print(f"\n=== contrastive trace at 'was' (prediction site) ===")
    print(f"  {'L':>3} {'d_post':>7} {'foodrank':>8}  reads")
    was_trace = {}
    for L in range(NL):
        dh = ts[L]["post"][was_pos] - cs[L]["post"][was_pos]
        fr = food_rank(dh @ W_Uc.T)
        was_trace[L] = dict(norm=round(float(dh.norm()), 2), food_rank=fr)
        if L % 2 == 0 or L == NL - 1:
            print(f"  L{L:>2} {float(dh.norm()):>7.1f} {str(fr):>8}  "
                  f"[{tk(dh @ W_Uc.T, 5)}]")
    L_was = next((L for L in range(NL)
                  if was_trace[L]["food_rank"] is not None
                  and was_trace[L]["food_rank"] < 10), None)
    result["was_trace"] = was_trace
    result["L_was"] = L_was
    print(f"\n  -> food reaches 'was' at L_was = {L_was}  (noun L_recog={L_recog})")

    # ---- 4: routing heads at 'was' ----
    print(f"\n=== routing heads at 'was' (per-head food write + attn->noun) ===")
    # capture o_proj INPUT (concatenated head outputs) at 'was', plus attentions
    inbuf = {}
    handles = []
    for L in range(NL):
        oproj = getattr(getattr(layers[L], ATTN), OPROJ)
        handles.append(oproj.register_forward_pre_hook(
            (lambda L: lambda m, x: inbuf.__setitem__(
                L, x[0][0].detach().float().cpu()))(L)))
    with torch.no_grad():
        outH = model(torch.tensor([hp_ids], device=DEV), output_attentions=True)
    for h in handles:
        h.remove()
    inbuf_c = {}
    handles = []
    for L in range(NL):
        oproj = getattr(getattr(layers[L], ATTN), OPROJ)
        handles.append(oproj.register_forward_pre_hook(
            (lambda L: lambda m, x: inbuf_c.__setitem__(
                L, x[0][0].detach().float().cpu()))(L)))
    with torch.no_grad():
        _ = model(torch.tensor([cp_ids], device=DEV))
    for h in handles:
        h.remove()

    rows = []
    for L in range(NL):
        oproj = getattr(getattr(layers[L], ATTN), OPROJ)
        O_w = oproj.weight.detach().float().cpu()   # [hid, NH*HD]
        if inbuf[L].shape[-1] != NH * HD:
            continue  # unexpected o_proj input width; skip head split
        for h in range(NH):
            sl = slice(h * HD, (h + 1) * HD)
            dhead = (inbuf[L][was_pos, sl] - inbuf_c[L][was_pos, sl]) @ O_w[:, sl].T
            a2n = float(outH.attentions[L][0, h, was_pos, dog_pos])
            rows.append((float(dhead @ fd), L, h, a2n, dhead))
    rows.sort(key=lambda r: -r[0])
    print(f"  global top food-writers to 'was':")
    print(f"  {'head':<9} {'food':>6} {'attn->noun':>10}  reads")
    route = []
    for fs, L, h, a2n, dhead in rows[:8]:
        print(f"  L{L}.H{h:<3} {fs:>+6.2f} {a2n:>10.2f}  [{tk(dhead @ W_Uc.T, 5)}]")
        route.append(dict(L=L, H=h, food=round(fs, 2), attn_to_noun=round(a2n, 2)))
    result["routing_top"] = route

    # the EARLY routing hop: heads in the window (L_recog, L_was] that both
    # write food forward and attend was->noun -- the paper's two-hop mechanism.
    lo = (L_recog + 1) if L_recog is not None else 1
    hi = (L_was + 1) if L_was is not None else min(NL, lo + 6)
    early = [(fs, L, h, a2n, dh) for (fs, L, h, a2n, dh) in rows
             if lo <= L <= hi and a2n >= 0.20 and fs > 0]
    early.sort(key=lambda r: (r[1], -r[0]))   # earliest layer first, then food
    print(f"\n  early routing hop, window L{lo}-L{hi} (attn->noun>=0.20):")
    ehop = []
    for fs, L, h, a2n, dhead in early[:5]:
        print(f"  L{L}.H{h:<3} {fs:>+6.2f} {a2n:>10.2f}  [{tk(dhead @ W_Uc.T, 5)}]")
        ehop.append(dict(L=L, H=h, food=round(fs, 2), attn_to_noun=round(a2n, 2)))
    result["routing_early"] = ehop
    result["L_route"] = ehop[0]["L"] if ehop else (route[0]["L"] if route else None)
    if ehop:
        print(f"\n  -> two-hop: MLP recognizes @ L_recog={L_recog}, "
              f"routing head L{ehop[0]['L']}.H{ehop[0]['H']} "
              f"(attn->noun {ehop[0]['attn_to_noun']:.2f}) carries food to 'was' "
              f"(arrives L_was={L_was})")
    else:
        print(f"\n  -> no early routing head in window; food may arrive at 'was' "
              f"by a distributed/later path (L_recog={L_recog}, L_was={L_was})")

    del model
    torch.cuda.empty_cache()
    return result


def food_frac(logits, food_ids, k=10):
    """True if any food anchor is in the top-k of this projection."""
    top = set(int(x) for x in torch.topk(logits.float(), k).indices)
    return bool(top & set(food_ids))


if __name__ == "__main__":
    main()
