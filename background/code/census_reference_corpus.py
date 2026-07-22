"""Corpus-invariance of the carried reference b_L (background paper, PLAN
item 1 follow-up; Olli 2026-07-21).

The paper's split-half stability of b_L (cos 0.94-0.99) is WITHIN one
naturalistic-prose corpus (census-docs-phi-2.json) -- it shows sampling
stability, not distributional invariance. b_L has never been measured on a
drastically different register. This script measures it on CODE (this repo's
own .py files) vs the prose corpus and asks whether the carried reference is
the same constant.

b_L(L) = mean over tokens of the post-LN input to the gate linear at layer L.
m_hat(L) = unit vector of the mean decoder-layer residual input (the Sec 2
"nearly a single direction" object). Both are measured per corpus.

Headline metrics, per model x layer:
  - cos(b_L^prose, b_L^code) and cos(m_hat^prose, m_hat^code): is the
    reference the SAME direction across register? Twin floor: the same
    cosines on a from_config random-init twin (b_L there is tiny/noise, so
    its cross-corpus cosine is the chance floor).
  - within-corpus resting decomposition (resting_med, ref-dominant where a
    bias exists, rho(resting, duty)) on EACH corpus: does the mechanism hold
    on code too?
  - cross-transfer: rho(resting^prose, duty^code) and rho(resting^code,
    duty^prose): does the reference derived on one register rank-predict the
    gate duty on the other? (transfer = the operating point is register-
    invariant; failure = register-conditioned.)

Interpretation the paper cares about:
  - cos(b_L) HIGH across register -> the reference is model-intrinsic, a
    maintained internal constant, not a running statistic of the input; the
    gate operating points are register-invariant. Strengthens Sec 5.
  - cos(b_L) LOW / resting fails to transfer -> b_L is register-conditioned;
    connects to the atomicity/bias-bank thread (E54) and WS-G register
    channel -- split-half over prose halves would hide exactly this.

Built-in archs only (the migration-scale replication set; no Phi-2 / no
trust_remote_code). Laptop-friendly.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_corpus.py
       SMOKE=1 -> first model, few docs.
"""
import sys
import os
import gc
import glob
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

MODELS = [
    ("gpt2-medium", True),
    ("EleutherAI/pythia-1.4b-deduped", False),
    ("Qwen/Qwen2.5-1.5B", True),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", False),
]
if SMOKE:
    MODELS = MODELS[:1]

N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16
CODE_CHUNK = 1800   # chars per code doc, ~matched to the prose docs
DEPTH_FRACS = [0.20, 0.35, 0.50, 0.65]

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROSE = os.path.join(DATA, "census-docs-phi-2.json")
CODE_DUMP = os.path.join(DATA, "census-docs-code-repo.json")
OUT = os.path.join(DATA, "census-reference-corpus.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def build_code_corpus(n_docs):
    """Deterministic and DIVERSE: one chunk from each of n_docs distinct
    sorted .py files under the repo (skip the docstring header by taking the
    chunk starting mid-file). Dumped for reproducibility."""
    files = sorted(glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True))
    files = [f for f in files if os.path.basename(f)
             != "census_reference_corpus.py"]
    docs = []
    for f in files:
        try:
            src = open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        src = src.strip()
        if len(src) < 600:            # need real code past the header
            continue
        start = min(len(src) // 4, 800)   # skip the leading docstring
        chunk = src[start:start + CODE_CHUNK]
        if len(chunk) >= 400:
            docs.append(chunk)
        if len(docs) >= n_docs:
            break
    return docs


def get_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("unknown architecture")


def get_gate(layer):
    mlp = layer.mlp
    for name in ("c_fc", "dense_h_to_4h", "gate_proj", "fc1"):
        if hasattr(mlp, name):
            return getattr(mlp, name), name
    raise RuntimeError("unknown MLP")


def gate_W_b(gate_mod, name):
    W = gate_mod.weight.detach().float().cpu()
    if name == "c_fc":
        W = W.t().contiguous()
    b = gate_mod.bias
    b = b.detach().float().cpu() if b is not None else None
    return W, b


@torch.no_grad()
def collect(m, sel_layers, docs_text, tok):
    """One forward pass over a corpus -> per-layer b_L (mean gate input),
    m_hat_raw (mean residual input), duty (per-unit frac pre-act>0)."""
    layers = get_layers(m)
    d = m.config.hidden_size
    ln_in, gate_out, resid_in = {}, {}, {}
    handles = []
    for L in sel_layers:
        gmod, _ = get_gate(layers[L])

        def mk_in(L=L):
            def f(mod, inp):
                ln_in[L] = inp[0][0, SKIP:].detach().float()
            return f

        def mk_out(L=L):
            def f(mod, inp, out):
                gate_out[L] = out[0, SKIP:].detach().float()
            return f

        def mk_res(L=L):
            def f(mod, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                resid_in[L] = hs[0, SKIP:].detach().float()
            return f
        handles.append(gmod.register_forward_pre_hook(mk_in()))
        handles.append(gmod.register_forward_hook(mk_out()))
        handles.append(layers[L].register_forward_pre_hook(
            mk_res(), with_kwargs=True))

    s_ln = {L: torch.zeros(d, device=DEV) for L in sel_layers}
    s_res = {L: torch.zeros(d, device=DEV) for L in sel_layers}
    duty_num = {L: None for L in sel_layers}
    npos = 0
    for txt in docs_text:
        ids = tok(txt, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] < SKIP + 48:
            continue
        m(input_ids=ids)
        T = ln_in[sel_layers[0]].shape[0]
        npos += T
        for L in sel_layers:
            s_ln[L] += ln_in[L].sum(0)
            s_res[L] += resid_in[L].sum(0)
            pc = (gate_out[L] > 0).float().sum(0)
            duty_num[L] = pc if duty_num[L] is None else duty_num[L] + pc
    for h in handles:
        h.remove()
    b_ln = {L: (s_ln[L] / npos).cpu() for L in sel_layers}
    m_raw = {L: (s_res[L] / npos).cpu() for L in sel_layers}
    duty = {L: (duty_num[L] / npos).cpu().numpy() for L in sel_layers}
    return b_ln, m_raw, duty, npos


@torch.no_grad()
def run_model(hf_id, which, sel_layers, corpora, tok):
    cfg = AutoConfig.from_pretrained(hf_id)
    if which == "twin":
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, low_cpu_mem_usage=True)
    m = m.to(DEV).eval()
    layers = get_layers(m)

    coll = {}
    for cname, docs in corpora.items():
        coll[cname] = collect(m, sel_layers, docs, tok)

    out = {"per_layer": {}}
    cnames = list(corpora.keys())
    for L in sel_layers:
        gmod, gname = get_gate(layers[L])
        W, b = gate_W_b(gmod, gname)
        row = {"gate": gname, "has_bias": b is not None}
        rest = {}
        duty = {}
        for cname in cnames:
            b_ln, m_raw, dty, npos = coll[cname]
            ref = W @ b_ln[L]
            bnc = b.cpu() if b is not None else torch.zeros(ref.shape[0])
            resting = ref + bnc
            rest[cname] = resting.numpy()
            duty[cname] = dty[L]
            row[f"resting_med_{cname}"] = round(float(resting.median()), 4)
            row[f"duty_med_{cname}"] = round(float(np.median(dty[L])), 4)
            row[f"rho_{cname}"] = round(spearman(resting.numpy(), dty[L]), 3)
            if b is not None:
                row[f"ref_dom_{cname}"] = round(float(
                    (ref.abs() > bnc.abs()).float().mean()), 4)
        # cross-corpus direction cosines (prose vs code assumed 2 corpora)
        a, c = cnames[0], cnames[1]
        ba, bc = coll[a][0][L], coll[c][0][L]
        ma, mc = coll[a][1][L], coll[c][1][L]
        row["cos_bL"] = round(float(F.cosine_similarity(
            ba, bc, dim=0)), 4)
        row["cos_mhat"] = round(float(F.cosine_similarity(
            ma, mc, dim=0)), 4)
        row["bL_norm_ratio"] = round(float(
            bc.norm() / ba.norm().clamp(min=1e-9)), 3)
        # cross-transfer of the resting order
        row[f"rho_rest{a}_duty{c}"] = round(
            spearman(rest[a], duty[c]), 3)
        row[f"rho_rest{c}_duty{a}"] = round(
            spearman(rest[c], duty[a]), 3)
        out["per_layer"][str(L)] = row

    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}")
    prose = json.load(open(PROSE, encoding="utf-8"))[:N_DOCS]
    code = build_code_corpus(N_DOCS)
    json.dump(code, open(CODE_DUMP, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"prose docs={len(prose)}  code docs={len(code)} "
          f"(dumped {os.path.basename(CODE_DUMP)})")
    corpora = {"prose": prose, "code": code}

    rec = {"config": {"smoke": SMOKE, "n_docs": N_DOCS, "skip": SKIP,
                      "code_chunk": CODE_CHUNK, "depth_fracs": DEPTH_FRACS},
           "models": {}}
    for hf_id, tied in MODELS:
        print(f"\n==== {hf_id} (tied={tied}) ====")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            cfg = AutoConfig.from_pretrained(hf_id)
            n_layers = cfg.num_hidden_layers
            sel = sorted(set(min(max(int(n_layers * f), 1), n_layers - 1)
                             for f in DEPTH_FRACS))
            torch.manual_seed(SEED)
            twin = run_model(hf_id, "twin", sel, corpora, tok)
            trained = run_model(hf_id, "trained", sel, corpora, tok)
            rec["models"][hf_id] = {"tied": tied, "n_layers": n_layers,
                                    "sel_layers": sel,
                                    "trained": trained, "twin": twin}
            for L in sel:
                e = trained["per_layer"][str(L)]
                tw = twin["per_layer"][str(L)]
                print(f"  L{L} [{e['gate']}]: cos(b_L) prose~code="
                      f"{e['cos_bL']} (twin {tw['cos_bL']})  "
                      f"cos(m_hat)={e['cos_mhat']}  |b_L|code/prose="
                      f"{e['bL_norm_ratio']}")
                print(f"      rho(rest,duty): prose={e['rho_prose']} "
                      f"code={e['rho_code']} | transfer "
                      f"rest_prose->duty_code={e['rho_restprose_dutycode']} "
                      f"rest_code->duty_prose={e['rho_restcode_dutyprose']}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rec["models"][hf_id] = {"error": repr(ex)}
            print(f"  !! {hf_id} failed: {ex!r}")
            gc.collect()
            torch.cuda.empty_cache()

    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
