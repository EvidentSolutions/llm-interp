"""Read-weight factorization: does a gate's operating point separate from
its content selectivity? (background_paper/PLAN.md sec 8, pre-registered
2026-07-22).

Decompose each fc1 read weight  w_n = c_n * bhat_L + w_n^perp,
c_n = w_n . bhat_L. Ask whether the OPERATING POINT (duty/sparsity) is
carried by the scalar parallel coupling c_n while CONTENT SELECTIVITY
lives in w_n^perp.

Leg 0 (construction validity): reproduce rho(resting, duty) at
L6/10/14/22 = 0.943/0.959/0.960/0.954 +-0.02 and frac(|w.b_L|>|b_n|)>0.999.
Leg 1 (operating-point norm fraction): f_n = |c_n|/||w_n|| vs twin floor.
Leg 2 (double dissociation): rho(c,duty) vs rho(resting,duty) vs
    rho(resting/s,duty); content share ||w^perp||/||w||.
Leg 3 (coupling-content link by layer): corr(c_n, fw_score_n), fw_score
    = mean over function-word UNEMBEDDING rows of cos(e_t, w_n^perp).
    Predict INDEPENDENT mid-stack (L6/10/14), LINKED near readout (L22/26).

Floor: from_config random twin. Usage:
  .venv/Scripts/python.exe superposition/code/census_readweight_factorization.py
  SMOKE=1 fast pass.
"""
import sys
import os
import gc
import json
import time
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

LAYERS = [6, 10, 14, 22, 26]
N_DOCS = 6 if SMOKE else 20
MAXLEN = 256
SKIP = 16
# construction-validity targets (from census_reference_mechanism.py, committed)
TARGET_RHO = {6: 0.943, 10: 0.959, 14: 0.960, 22: 0.954}

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-readweight-factorization.json")

# function words (leading-space forms + bare punctuation); single-token kept
FW_WORDS = [" the", " and", " a", " to", " of", " in", " is", " that",
            " it", " for", " as", " with", " on", " '", "'s", ",", ".",
            " ,", " ."]


@torch.no_grad()
def load_model(which):
    if which == "random":
        m = AutoModelForCausalLM.from_pretrained(
            TWIN, dtype=torch.float16, low_cpu_mem_usage=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    return m.to(DEV).eval()


class Cap:
    """Capture fc1 input (ln) and fc1 output (z = preactivation), pos>=SKIP."""

    def __init__(self, model, layers):
        self.model, self.layers = model, layers
        self.ln, self.z = {}, {}
        self.h = []

    def __enter__(self):
        for L in self.layers:
            fc1 = self.model.model.layers[L].mlp.fc1

            def mk_ln(L=L):
                def f(mod, inp):
                    self.ln[L] = inp[0][0, SKIP:].detach().float()
                return f

            def mk_z(L=L):
                def f(mod, inp, out):
                    self.z[L] = out[0, SKIP:].detach().float()
                return f
            self.h.append(fc1.register_forward_pre_hook(mk_ln()))
            self.h.append(fc1.register_forward_hook(mk_z()))
        return self

    def __exit__(self, *a):
        for hh in self.h:
            hh.remove()


def fw_token_rows(m, tok):
    """Unit-normalized unembedding rows for single-token function words."""
    ids = []
    seen = set()
    for w in FW_WORDS:
        enc = tok(w, add_special_tokens=False)["input_ids"]
        if len(enc) == 1 and enc[0] not in seen:
            ids.append(enc[0])
            seen.add(enc[0])
    W_U = m.lm_head.weight.detach().float().cpu()   # (vocab, hidden)
    rows = W_U[ids]                                  # (nfw, hidden)
    rows = rows / rows.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return rows, ids


@torch.no_grad()
def run_model(which, docs_ids, tok):
    t0 = time.time()
    print(f"--- {which} ---")
    m = load_model(which)
    d = m.config.hidden_size
    fw_rows, fw_ids = fw_token_rows(m, tok)

    # ---- single pass: accumulate ln mean + z stats per gate ----
    s_ln = {L: torch.zeros(d, device=DEV) for L in LAYERS}
    zsum, zsq, poscount = {}, {}, {}
    dff = {}
    npos = 0
    for ids in docs_ids:
        with Cap(m, LAYERS) as c:
            m(input_ids=ids)
        npos += c.z[LAYERS[0]].shape[0]
        for L in LAYERS:
            s_ln[L] += c.ln[L].sum(0)
            z = c.z[L]
            if L not in zsum:
                dff[L] = z.shape[1]
                zsum[L] = torch.zeros(z.shape[1], device=DEV)
                zsq[L] = torch.zeros(z.shape[1], device=DEV)
                poscount[L] = torch.zeros(z.shape[1], device=DEV)
            zsum[L] += z.sum(0)
            zsq[L] += (z * z).sum(0)
            poscount[L] += (z > 0).float().sum(0)

    rec = {"layers": {}, "config_dff": {}}
    for L in LAYERS:
        b_ln = (s_ln[L] / npos).cpu()                       # (d,)
        bhat = b_ln / b_ln.norm().clamp(min=1e-9)
        W1 = m.model.layers[L].mlp.fc1.weight.detach().float().cpu()  # (dff,d)
        bn = m.model.layers[L].mlp.fc1.bias.detach().float().cpu()    # (dff,)

        wnorm = W1.norm(dim=1)                              # (dff,)
        c_n = W1 @ bhat                                     # (dff,) = coupling
        w_perp = W1 - c_n.unsqueeze(1) * bhat.unsqueeze(0)  # (dff, d)
        wperp_norm = w_perp.norm(dim=1)
        f_n = c_n.abs() / wnorm.clamp(min=1e-9)             # operating-pt tax
        content_share = wperp_norm / wnorm.clamp(min=1e-9)

        ref = W1 @ b_ln                                     # (dff,) = w.b_L
        resting = ref + bn
        duty = (poscount[L] / npos).cpu()
        z_mean = (zsum[L] / npos).cpu()
        z_var = (zsq[L] / npos).cpu() - z_mean ** 2
        s_n = z_var.clamp(min=1e-12).sqrt()

        # construction check: z_mean == resting by identity
        rest_recon = float((z_mean - resting).abs().max())

        cn_np = c_n.numpy()
        rest_np = resting.numpy()
        duty_np = duty.numpy()
        rests_np = (resting / s_n).numpy()

        # Leg 3: coupling-content link
        wperp_hat = w_perp / wperp_norm.unsqueeze(1).clamp(min=1e-9)
        fw_score = (fw_rows @ wperp_hat.t()).mean(0).numpy()   # (dff,)

        rec["config_dff"][str(L)] = int(dff[L])
        rec["layers"][str(L)] = {
            # Leg 0
            "frac_ref_dominant": round(float(
                (ref.abs() > bn.abs()).float().mean()), 5),
            "spearman_resting_duty": round(spearman(rest_np, duty_np), 3),
            "rest_recon_maxabs": round(rest_recon, 5),
            # Leg 1
            "f_n_median": round(float(f_n.median()), 5),
            "f_n_mean": round(float(f_n.mean()), 5),
            "f_n_p90": round(float(np.percentile(f_n.numpy(), 90)), 5),
            # Leg 2
            "spearman_coupling_duty": round(spearman(cn_np, duty_np), 3),
            "spearman_restingoverS_duty": round(
                spearman(rests_np, duty_np), 3),
            "content_share_median": round(float(content_share.median()), 4),
            "content_share_min": round(float(content_share.min()), 4),
            "corr_contentshare_coupling": round(
                pearson(content_share.numpy(), np.abs(cn_np)), 3),
            # Leg 3
            "pearson_coupling_fwscore": round(pearson(cn_np, fw_score), 3),
            "spearman_coupling_fwscore": round(spearman(cn_np, fw_score), 3),
            "fw_score_median": round(float(np.median(fw_score)), 4),
        }
        del W1, w_perp, wperp_hat
    rec["n_fw_tokens"] = len(fw_ids)
    rec["npos"] = int(npos)

    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {which} done ({time.time()-t0:.0f}s)")
    return rec


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  docs={N_DOCS}  layers={LAYERS}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    docs_ids = []
    for t in docs:
        ids = tok(t, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] >= SKIP + 48:
            docs_ids.append(ids)
    print(f"{len(docs_ids)} docs usable")
    rec = {"config": {"smoke": SMOKE, "n_docs": len(docs_ids), "seed": SEED,
                      "layers": LAYERS}}
    rec["trained"] = run_model("trained", docs_ids, tok)
    rec["twin"] = run_model("random", docs_ids, tok)
    json.dump(rec, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")

    # ---- pre-registered verdict readout ----
    tr, tw = rec["trained"]["layers"], rec["twin"]["layers"]
    print("\n== Leg 0 construction validity (rho resting-duty vs target; "
          "frac_ref_dom; z-recon):")
    ok0 = True
    for L in LAYERS:
        a = tr[str(L)]
        tgt = TARGET_RHO.get(L)
        mark = ""
        if tgt is not None:
            dev = abs(a["spearman_resting_duty"] - tgt)
            mark = "OK" if (dev <= 0.02 and a["frac_ref_dominant"] > 0.999) \
                else "FAIL"
            ok0 = ok0 and mark == "OK"
        print(f"  L{L}: rho={a['spearman_resting_duty']} (tgt {tgt}) "
              f"fracRefDom={a['frac_ref_dominant']} "
              f"recon={a['rest_recon_maxabs']} {mark}")
    print(f"  Leg0 gate: {'PASS' if ok0 else 'FAIL'}")

    print("\n== Leg 1 operating-point norm fraction f_n (trained vs twin):")
    for L in LAYERS:
        a, b = tr[str(L)], tw[str(L)]
        ratio = a["f_n_median"] / max(b["f_n_median"], 1e-9)
        print(f"  L{L}: median={a['f_n_median']} (twin {b['f_n_median']}, "
              f"{ratio:.2f}x)  mean={a['f_n_mean']} p90={a['f_n_p90']}")

    print("\n== Leg 2 double dissociation "
          "(rho: coupling / resting / resting-over-s; content share):")
    for L in LAYERS:
        a = tr[str(L)]
        print(f"  L{L}: {a['spearman_coupling_duty']} / "
              f"{a['spearman_resting_duty']} / "
              f"{a['spearman_restingoverS_duty']}   "
              f"content_share med={a['content_share_median']} "
              f"min={a['content_share_min']} "
              f"corr(share,|c|)={a['corr_contentshare_coupling']}")

    print("\n== Leg 3 coupling-content link (corr c_n vs fw_score):")
    for L in LAYERS:
        a = tr[str(L)]
        print(f"  L{L}: pearson={a['pearson_coupling_fwscore']} "
              f"spearman={a['spearman_coupling_fwscore']} "
              f"fw_score_med={a['fw_score_median']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
