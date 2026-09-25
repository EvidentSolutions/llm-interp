"""
Pass-1 scan for the selection-free neuron census.

Design: superposition/docs/plan_neuron_census.md (sections 2-3).

Samples 50 random MLP neurons per layer at 8 depth-stratified layers
(no selection on coherence/mc/contrast), scans ~2000 Pile documents,
and records for each neuron:
  - static properties: mc_WU(fc1), mc_WE(fc1), fc1 tip coherence,
    mc_WU(fc2), with gaussian null calibration
  - a true top-K heap of firing events (activation, doc_id, position)
  - top-tail activation values (for the 0.05% / 0.5% firing-threshold
    quantiles)
  - a strided reservoir of activations for distribution stats

Pre-activations a_n = w_n . x + b_n are computed from the INPUT of
mlp.fc1 (LN already applied by the model), captured via forward
pre-hook. No fc1 outputs are stored.

Outputs (superposition/data/):
  census-docs-<model>.json         the exact documents scanned
  census-pass1-<model>[-random].json   per-neuron results for pass 2
  census-pass1-ckpt-<model>[-random].pt   resumable checkpoint

Null #4 (random-init model control, Heap et al. 2025):
  RANDOM_INIT=1 .venv/Scripts/python.exe superposition/code/census_pass1_scan.py

Usage: .venv/Scripts/python.exe superposition/code/census_pass1_scan.py
"""
import sys, os, json, heapq, math, time
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
RANDOM_INIT = os.environ.get("RANDOM_INIT", "0") == "1"

LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
N_PER_LAYER = 50
SEED = 0

N_DOCS = int(os.environ.get("N_DOCS", 2000))
SKIP = 1000          # skip first N stream docs for diversity
MAX_TOKENS = 256
MIN_TOKENS = 32

EVENT_K = 400        # top firing events kept per neuron (with doc/pos)
VAL_K = 6000         # top activation values kept per neuron (quantiles)
RES_STRIDE = 20      # every RES_STRIDE-th doc sampled fully for stats
CKPT_EVERY = 200

TOPK_READ = 30       # tokens defining the read family / tip coherence
N_NULL = 1000        # gaussian directions for mc null calibration

DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data"))
TAG = MODEL.split("/")[-1] + ("-random" if RANDOM_INIT else "")
if N_DOCS != 2000:
    TAG += f"-d{N_DOCS}"       # keep smoke tests out of the real files
DOCS_PATH = os.path.join(
    DATA_DIR,
    f"census-docs-{MODEL.split('/')[-1]}"
    + (f"-d{N_DOCS}" if N_DOCS != 2000 else "") + ".json")
OUT_PATH = os.path.join(DATA_DIR, f"census-pass1-{TAG}.json")
CKPT_PATH = os.path.join(DATA_DIR, f"census-pass1-ckpt-{TAG}.pt")


# ----------------------------------------------------------------- docs

def collect_docs(tokenizer):
    """Collect the document set once, save to JSON, reuse across variants
    (trained and random-init runs must scan identical text)."""
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} cached documents from {DOCS_PATH}")
        return docs
    print("Streaming Pile documents...")
    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    docs = []
    for i, ex in enumerate(ds):
        if i < SKIP:
            continue
        if len(docs) >= N_DOCS:
            break
        text = ex.get("text", "")
        if len(text) < 100:
            continue
        text = text[:2000]
        n_tok = len(tokenizer(text, add_special_tokens=False,
                              max_length=MAX_TOKENS,
                              truncation=True)["input_ids"])
        if n_tok < MIN_TOKENS:
            continue
        docs.append(text)
        if len(docs) % 500 == 0:
            print(f"  {len(docs)}/{N_DOCS}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"Saved {len(docs)} documents to {DOCS_PATH}")
    return docs


# ------------------------------------------------------------ sampling

def sample_neurons(model):
    """Uniform random neurons per layer, fixed seed. NO selection."""
    n_mlp = model.model.layers[LAYERS[0]].mlp.fc1.weight.shape[0]
    rng = np.random.RandomState(SEED)
    sel = {L: np.sort(rng.choice(n_mlp, N_PER_LAYER, replace=False))
           for L in LAYERS}
    print("Sampled neurons (seed %d, %d per layer, n_mlp=%d):" %
          (SEED, N_PER_LAYER, n_mlp))
    for L in LAYERS:
        print(f"  L{L}: {sel[L][:8].tolist()} ...")
    return sel


# ------------------------------------------------------- static props

def static_properties(model, tok, sel):
    """mc_WU/mc_WE of fc1 reads, tip coherence, mc_WU of fc2 writes,
    plus gaussian-null calibration. Recorded for stratification only."""
    W_U = model.lm_head.weight.detach().float().to(DEV)
    W_E = model.model.embed_tokens.weight.detach().float().to(DEV)
    W_Un = W_U / W_U.norm(dim=1, keepdim=True)
    W_En = W_E / W_E.norm(dim=1, keepdim=True)
    d_model = W_U.shape[1]

    # gaussian null for max-cos
    g = torch.randn(N_NULL, d_model, device=DEV)
    g = g / g.norm(dim=1, keepdim=True)
    null_wu = (W_Un @ g.T).abs().max(dim=0).values
    null_we = (W_En @ g.T).abs().max(dim=0).values
    null = {
        "mc_wu_null_mean": null_wu.mean().item(),
        "mc_wu_null_p95": null_wu.quantile(0.95).item(),
        "mc_we_null_mean": null_we.mean().item(),
        "mc_we_null_p95": null_we.quantile(0.95).item(),
    }
    print(f"Null mc: WU p95={null['mc_wu_null_p95']:.4f} "
          f"WE p95={null['mc_we_null_p95']:.4f}")

    props = {}
    for L in LAYERS:
        fc1 = model.model.layers[L].mlp.fc1.weight.detach().float().to(DEV)
        fc2 = model.model.layers[L].mlp.fc2.weight.detach().float().to(DEV)
        for n in sel[L]:
            n = int(n)
            r = fc1[n] / fc1[n].norm()
            w = fc2[:, n] / fc2[:, n].norm()
            cos_wu = W_Un @ r
            cos_we = W_En @ r
            mc_wu = cos_wu.abs().max().item()
            mc_we = cos_we.abs().max().item()
            top = torch.topk(cos_wu.abs(), TOPK_READ).indices
            # tip coherence: mean pairwise cos among top-30 read tokens
            V = W_Un[top]
            G = V @ V.T
            coh = ((G.sum() - G.diag().sum())
                   / (TOPK_READ * (TOPK_READ - 1))).item()
            read_tokens = [tok.decode([t]).strip()[:16] for t in top[:10]]
            read_signs = ["+" if cos_wu[t] > 0 else "-" for t in top[:10]]
            mc_wu_fc2 = (W_Un @ w).abs().max().item()
            props[f"L{L}_n{n}"] = {
                "layer": L, "neuron": n,
                "mc_wu_fc1": round(mc_wu, 4),
                "mc_we_fc1": round(mc_we, 4),
                "tip_coherence": round(coh, 4),
                "mc_wu_fc2": round(mc_wu_fc2, 4),
                "read_family_top10": [s + t for s, t in
                                      zip(read_signs, read_tokens)],
                "read_family_ids": top.tolist(),
            }
        del fc1, fc2
    return props, null


# ------------------------------------------------------------ scanning

def main():
    t0 = time.time()
    print(f"Device: {DEV}  Model: {MODEL}  random_init={RANDOM_INIT}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    if RANDOM_INIT:
        cfg = AutoConfig.from_pretrained(MODEL)
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_config(cfg)
        model = model.to(torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    docs = collect_docs(tok)
    sel = sample_neurons(model)
    props, null = static_properties(model, tok, sel)

    # per-layer selected read weights for fast pre-activation computation
    W_sel, b_sel = {}, {}
    for L in LAYERS:
        fc1 = model.model.layers[L].mlp.fc1
        idx = torch.tensor(sel[L], device=DEV)
        W_sel[L] = fc1.weight.detach()[idx].clone()          # (50, d) fp16
        b_sel[L] = (fc1.bias.detach()[idx].clone().float()
                    if fc1.bias is not None
                    else torch.zeros(len(idx), device=DEV))

    n_neurons = len(LAYERS) * N_PER_LAYER
    keys = [f"L{L}_n{int(n)}" for L in LAYERS for n in sel[L]]

    # state (resumable)
    start_doc = 0
    event_heaps = [[] for _ in range(n_neurons)]   # min-heaps of (act, doc, pos)
    val_bufs = [[] for _ in range(n_neurons)]      # top-tail activation values
    val_cuts = np.full(n_neurons, -np.inf)
    res_bufs = [[] for _ in range(n_neurons)]      # strided reservoir (np.f16)
    n_positions = 0

    if os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, weights_only=False)
        if ck["keys"] == keys and ck["n_docs_total"] == len(docs):
            start_doc = ck["next_doc"]
            event_heaps = ck["event_heaps"]
            val_bufs = ck["val_bufs"]
            val_cuts = ck["val_cuts"]
            res_bufs = ck["res_bufs"]
            n_positions = ck["n_positions"]
            print(f"Resumed from checkpoint at doc {start_doc}")
        else:
            print("Checkpoint incompatible with current config — ignoring.")

    # hooks: compute (seq, 50) pre-activations from the fc1 INPUT
    acts = {}

    def mk_hook(L):
        def hook(mod, args):
            x = args[0][0]                                   # (seq, d) fp16
            a = (x @ W_sel[L].T).float() + b_sel[L]          # (seq, 50)
            acts[L] = a
        return hook

    handles = [model.model.layers[L].mlp.fc1.register_forward_pre_hook(
        mk_hook(L)) for L in LAYERS]

    def save_ckpt(next_doc):
        torch.save({
            "keys": keys, "n_docs_total": len(docs), "next_doc": next_doc,
            "event_heaps": event_heaps, "val_bufs": val_bufs,
            "val_cuts": val_cuts, "res_bufs": res_bufs,
            "n_positions": n_positions,
        }, CKPT_PATH)
        print(f"  [ckpt] doc {next_doc}, {time.time()-t0:.0f}s elapsed")

    print(f"Scanning docs {start_doc}..{len(docs)-1}")
    for doc_idx in range(start_doc, len(docs)):
        ids = tok(docs[doc_idx], add_special_tokens=False,
                  max_length=MAX_TOKENS, truncation=True)["input_ids"]
        with torch.no_grad():
            model(torch.tensor([ids], device=DEV))
        n_positions += len(ids)

        for li, L in enumerate(LAYERS):
            A = acts[L].cpu().numpy()                        # (seq, 50)
            for j in range(N_PER_LAYER):
                g = li * N_PER_LAYER + j
                col = A[:, j]

                # --- top-K event heap (true heap, not first-seen)
                h = event_heaps[g]
                thr = h[0][0] if len(h) >= EVENT_K else -np.inf
                for pos in np.nonzero(col > thr)[0]:
                    v = float(col[pos])
                    if len(h) < EVENT_K:
                        heapq.heappush(h, (v, doc_idx, int(pos)))
                    elif v > h[0][0]:
                        heapq.heapreplace(h, (v, doc_idx, int(pos)))

                # --- top-tail values for quantile thresholds
                vb = val_bufs[g]
                big = col[col > val_cuts[g]]
                if big.size:
                    vb.extend(big.tolist())
                    if len(vb) > 3 * VAL_K:
                        vb.sort(reverse=True)
                        del vb[VAL_K:]
                        val_cuts[g] = vb[-1]

                # --- strided reservoir for distribution stats
                if doc_idx % RES_STRIDE == 0:
                    res_bufs[g].append(col.astype(np.float16))

        if (doc_idx + 1) % 50 == 0:
            print(f"  doc {doc_idx+1}/{len(docs)}  "
                  f"({n_positions} positions)")
        if (doc_idx + 1) % CKPT_EVERY == 0:
            save_ckpt(doc_idx + 1)

    for hd in handles:
        hd.remove()
    save_ckpt(len(docs))

    # ------------------------------------------------------- results
    print("\nAssembling results...")
    results = {
        "model": MODEL, "random_init": RANDOM_INIT, "seed": SEED,
        "layers": LAYERS, "n_per_layer": N_PER_LAYER,
        "n_docs": len(docs), "n_positions": n_positions,
        "max_tokens": MAX_TOKENS, "event_k": EVENT_K,
        "docs_file": os.path.basename(DOCS_PATH),
        "null_calibration": null,
        "neurons": {},
    }

    k0005 = max(1, int(round(0.0005 * n_positions)))   # top 0.05%
    k0050 = max(1, int(round(0.0050 * n_positions)))   # top 0.5%

    for g, key in enumerate(keys):
        events = sorted(event_heaps[g], reverse=True)
        vals = sorted(val_bufs[g], reverse=True)
        res = (np.concatenate(res_bufs[g]).astype(np.float32)
               if res_bufs[g] else np.zeros(1, np.float32))

        thr_0005 = vals[k0005 - 1] if len(vals) >= k0005 else vals[-1]
        thr_0050 = vals[k0050 - 1] if len(vals) >= k0050 else vals[-1]

        # decoded contexts for the top 40 events (eyeballing / report)
        top_ctx = []
        for v, d, p in events[:40]:
            ids = tok(docs[d], add_special_tokens=False,
                      max_length=MAX_TOKENS, truncation=True)["input_ids"]
            lo, hi = max(0, p - 10), min(len(ids), p + 4)
            ctx = tok.decode(ids[lo:hi]).replace("\n", " ").strip()[:100]
            trig = tok.decode([ids[p]])[:16]
            top_ctx.append([round(v, 3), d, p, trig, ctx])

        results["neurons"][key] = {
            **props[key],
            "act_mean": round(float(res.mean()), 4),
            "act_std": round(float(res.std()), 4),
            "act_median": round(float(np.median(res)), 4),
            "frac_pos": round(float((res > 0).mean()), 4),
            "thr_top0.05pct": round(float(thr_0005), 4),
            "thr_top0.5pct": round(float(thr_0050), 4),
            "n_events_kept": len(events),
            "events": [[round(v, 4), d, p] for v, d, p in events],
            "top40_contexts": top_ctx,
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Saved {OUT_PATH}")

    # ------------------------------------------------- console summary
    print("\n=== Per-layer static summary ===")
    for L in LAYERS:
        ks = [k for k in keys if results["neurons"][k]["layer"] == L]
        mc = np.array([results["neurons"][k]["mc_wu_fc1"] for k in ks])
        coh = np.array([results["neurons"][k]["tip_coherence"] for k in ks])
        above = (mc > null["mc_wu_null_p95"]).mean()
        print(f"  L{L:>2}: mc_wu_fc1 med={np.median(mc):.3f} "
              f"frac>null_p95={above:.2f}  coh med={np.median(coh):.3f}")

    print("\n=== Example neurons (2 per layer, by mc_wu_fc1) ===")
    for L in LAYERS:
        ks = sorted([k for k in keys if results["neurons"][k]["layer"] == L],
                    key=lambda k: -results["neurons"][k]["mc_wu_fc1"])
        for k in [ks[0], ks[len(ks) // 2]]:
            r = results["neurons"][k]
            fam = " ".join(r["read_family_top10"][:5])
            print(f"  {k} mc={r['mc_wu_fc1']:.3f} reads: {fam}")
            for v, d, p, trig, ctx in r["top40_contexts"][:3]:
                print(f"      {v:>+8.2f} [{trig:>12}] {ctx}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
