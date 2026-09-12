"""The machinery transition below step 512 (plan_mid_stack_empirical_basis
.md 3ai, pre-registered 2026-07-17).

3w found induction + gate sparsification + the inhibitory reference
forming "jointly" at 512->1000; the lit sweep (Xu 2026) bounded that to
grid resolution. Pythia's log2 checkpoints below 512 were never measured.
Per checkpoint {1,2,4,8,16,32,64} (+ {0,128,256,512} as gates): induction
score, duty cycle, light inhibitory drive share vs b_L, split-half const
share, readout PR rank, mean W_U-row relgain, mc_wu(fc2) median.

Branches: JOINT-AT-FINER-GRAIN (all flat through 64 except the known
const early-mover) vs STAGGERED (components move at different sub-512
steps -- the Xu-style separation, in-house).

Gates: steps 0/128/256/512 reproduce 3w/3u within doc noise; induction
at floor at step 0.

Usage: .venv/Scripts/python.exe superposition/code/census_formation_prelude.py
       SMOKE=1 (steps 0 and 64 only, 4 docs).
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "EleutherAI/pythia-410m-deduped"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

STEPS = [0, 64] if SMOKE else [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BAND = [5, 8, 11]
N_DOCS = 4 if SMOKE else 16
MAXLEN = 256
SKIP = 16
N_INDSEQ = 3 if SMOKE else 6
N_SHARE = 100 if SMOKE else 200
N_ROWGAIN = 300 if SMOKE else 1000

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-formation-prelude.json")


@torch.no_grad()
def weights_lite(m, rng):
    """Readout PR rank, mean W_U-row relgain, mc_fc2 med per band layer."""
    d = m.config.hidden_size
    WU = m.embed_out.weight.detach().float()
    WUn = (WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
           ).to(DEV, torch.float16)
    gamma_f = m.gpt_neox.final_layer_norm.weight.detach().float().to(DEV)
    A = (WU.to(DEV) * gamma_f[None, :]).float()
    frob2 = float((A * A).sum() - (A.sum(dim=1) ** 2).sum() / d)
    g_rand = float(np.sqrt(frob2 / d))
    C = torch.eye(d, device=DEV) - torch.full((d, d), 1.0 / d, device=DEV)
    MtM = C @ (A.T @ A) @ C
    ev = torch.linalg.eigvalsh(MtM).clamp(min=0)
    pr_rank = float((ev.sum() ** 2) / (ev ** 2).sum())
    del MtM, ev, C

    def gain(v):
        vc = v - v.mean()
        return float((A @ vc).norm() / v.norm().clamp(min=1e-9))
    row_ids = rng.choice(WU.shape[0], N_ROWGAIN, replace=False)
    rg = []
    for t in row_ids:
        u = WU[t].to(DEV)
        u = u / u.norm().clamp(min=1e-9)
        rg.append(gain(u) / g_rand)
    out = {"readout_pr_rank": round(pr_rank, 1),
           "mean_row_relgain": round(float(np.mean(rg)), 4),
           "mc_fc2_med": {}}
    for L in BAND:
        f2 = m.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.detach()
        f2u = (f2.float() / f2.float().norm(dim=0, keepdim=True
                                            ).clamp(min=1e-9)
               ).to(DEV, torch.float16)
        mc2 = torch.empty(f2.shape[1])
        for s in range(0, f2.shape[1], 2048):
            mc2[s:s + 2048] = (WUn @ f2u[:, s:s + 2048]
                               ).abs().max(dim=0).values.float().cpu()
        out["mc_fc2_med"][str(L)] = round(float(mc2.median()), 4)
        del f2u
        torch.cuda.empty_cache()
    del A, WUn
    return out


@torch.no_grad()
def induction_score(m, vocab, rng):
    """Prefix-matching score on repeated random sequences (3w statistic)."""
    K = 48
    scores_mean, scores_max = [], []
    for _ in range(N_INDSEQ):
        seq = rng.randint(100, vocab - 100, K).tolist()
        ids = torch.as_tensor([seq + seq], device=DEV)
        out = m(input_ids=ids, output_attentions=True)
        per_head = []
        for att in out.attentions:
            a = att[0].float()
            idx_q = torch.arange(K + 1, 2 * K, device=DEV)
            idx_k = idx_q - K + 1
            per_head.append(a[:, idx_q, idx_k].mean(dim=1))
        ph = torch.cat(per_head)
        scores_mean.append(float(ph.mean()))
        scores_max.append(float(ph.max()))
    return round(float(np.mean(scores_mean)), 4), \
        round(float(np.mean(scores_max)), 4)


@torch.no_grad()
def forwards_lite(m, docs_ids, rng):
    """Duty, split-half const share, light b-share, per band layer."""
    caps = {}
    handles = []

    def mk_ln(L):
        def f(mod, inp):
            caps[("ln", L)] = inp[0][0, SKIP:].detach().float()
        return f

    def mk_act(L):
        def f(mod, inp):
            caps[("act", L)] = inp[0][0, SKIP:].detach().float()
        return f
    for L in BAND:
        lyr = m.gpt_neox.layers[L]
        handles.append(lyr.mlp.dense_h_to_4h.register_forward_pre_hook(
            mk_ln(L)))
        handles.append(lyr.mlp.dense_4h_to_h.register_forward_pre_hook(
            mk_act(L)))

    duty = {L: None for L in BAND}
    bsum = {L: None for L in BAND}
    SsumA = {L: None for L in BAND}
    SsumB = {L: None for L in BAND}
    S2sum = {L: 0.0 for L in BAND}
    share_x = {L: [] for L in BAND}
    npos = cA = cB = 0
    for di, ids in enumerate(docs_ids):
        m(input_ids=ids)
        T = caps[("act", BAND[0])].shape[0]
        npos += T
        for L in BAND:
            a, x = caps[("act", L)], caps[("ln", L)]
            pc = (a > 0).float().sum(0)
            duty[L] = pc if duty[L] is None else duty[L] + pc
            bsum[L] = x.sum(0) if bsum[L] is None else bsum[L] + x.sum(0)
            if di < 6:
                share_x[L].append((a.cpu(), x.cpu()))
            f2 = m.gpt_neox.layers[L].mlp.dense_4h_to_h.weight \
                .detach().float()
            S_full = a.to(DEV) @ f2.T
            S2sum[L] += float((S_full ** 2).sum())
            if di % 2 == 0:
                SsumA[L] = S_full.sum(0) if SsumA[L] is None \
                    else SsumA[L] + S_full.sum(0)
            else:
                SsumB[L] = S_full.sum(0) if SsumB[L] is None \
                    else SsumB[L] + S_full.sum(0)
        if di % 2 == 0:
            cA += T
        else:
            cB += T
    for h in handles:
        h.remove()

    out = {}
    for L in BAND:
        rec = {"duty_med": round(float((duty[L] / npos).median()), 4)}
        muA = SsumA[L] / max(cA, 1)
        muB = (SsumB[L] / max(cB, 1)) if SsumB[L] is not None else muA * 0
        rec["const_share"] = round(float(
            (muA @ muB) / (S2sum[L] / npos + 1e-9)), 4)
        b = bsum[L] / npos
        bh = (b / b.norm().clamp(min=1e-9)).cpu()
        f1 = m.gpt_neox.layers[L].mlp.dense_h_to_4h.weight.detach().float()
        take = rng.choice(f1.shape[0], N_SHARE, replace=False)
        A = torch.cat([t[0] for t in share_x[L]], 0)
        X = torch.cat([t[1] for t in share_x[L]], 0)
        shares = []
        for n in take:
            acts_n = A[:, n]
            top = torch.topk(acts_n, min(32, acts_n.shape[0])).indices
            w = f1[n].cpu()
            xs = X[top]
            drive = xs @ w
            sb = (float(w @ bh)) * (xs @ bh)
            ok = drive.abs() > 1e-3
            if ok.sum() > 0:
                shares.append(float((sb[ok] / drive[ok]).median()))
        rec["drive_share_b_med"] = round(float(np.median(shares)), 4) \
            if shares else None
        out[str(L)] = rec
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  steps={STEPS}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    docs_ids = []
    for t in docs:
        ids = tok(t, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] >= SKIP + 48:
            docs_ids.append(ids)
    print(f"{len(docs_ids)} docs usable")

    rec = {"config": {"smoke": SMOKE, "steps": STEPS, "band": BAND,
                      "n_docs": len(docs_ids), "seed": SEED}}
    if os.path.exists(OUT):
        try:
            old = json.load(open(OUT, encoding="utf-8"))
            if old.get("config", {}).get("smoke") == SMOKE:
                rec = old
                print(f"resuming: have {list(rec.get('steps', {}).keys())}")
        except Exception:
            pass
    rec.setdefault("steps", {})

    for step in STEPS:
        key = str(step)
        if key in rec["steps"]:
            print(f"step{step}: cached, skip")
            continue
        tS = time.time()
        print(f"loading step{step} ...")
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=f"step{step}", dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager").to(DEV).eval()
        rng = np.random.RandomState(SEED + step % 99991)
        wt = weights_lite(m, rng)
        im, ix = induction_score(m, m.config.vocab_size,
                                 np.random.RandomState(SEED + 7))
        ft = forwards_lite(m, docs_ids, np.random.RandomState(SEED + 11))
        rec["steps"][key] = {"weights": wt, "induction_mean": im,
                             "induction_maxhead": ix, "forwards": ft}
        json.dump(rec, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        f8 = ft.get("8", {})
        print(f"step{step} ({time.time()-tS:.0f}s): ind={ix} "
              f"duty={f8.get('duty_med')} "
              f"share_b={f8.get('drive_share_b_med')} "
              f"const={f8.get('const_share')} "
              f"prR={wt['readout_pr_rank']} "
              f"rowrg={wt['mean_row_relgain']} "
              f"mc2_L11={wt['mc_fc2_med'].get('11')}")
        del m
        gc.collect()
        torch.cuda.empty_cache()

    # ordering summary
    print("\nstep  ind_max  duty_L8  share_b_L8  const_L8  prR  rowrg")
    for s in sorted((int(k) for k in rec["steps"]), key=int):
        r = rec["steps"][str(s)]
        f8 = r["forwards"].get("8", {})
        print(f"{s:>6} {r['induction_maxhead']:>7} "
              f"{f8.get('duty_med'):>8} {f8.get('drive_share_b_med'):>10} "
              f"{f8.get('const_share'):>9} "
              f"{r['weights']['readout_pr_rank']:>6} "
              f"{r['weights']['mean_row_relgain']:>6}")
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
