"""Checkpoint trajectory: token signaling first, the dark channel on top?
(plan_mid_stack_empirical_basis.md 3u, Olli's developmental hypothesis,
pre-registered 2026-07-16.)

Pythia-410M-deduped public checkpoints as the sandbox-replacement: per
checkpoint measure (weights tier) mc_wu of writes/reads, quadrant fractions
vs step-0 floors, relgain per quadrant + mean W_U-row relgain (the 3s
standing tool), readout effective rank; (forwards tier, 16 docs) duty
cycle, quadrant energy shares of the MLP update, const share, and the
formation of the empirical read basis (small centered D dictionary) vs the
W_U read basis.

GPTNeoX specifics: parallel residual -- the MLP reads
post_attention_layernorm(x); lm head = embed_out (untied); fc1/fc2 =
mlp.dense_h_to_4h / dense_4h_to_h.

Gates: reproduce the 07-05 trajectory's step-1000 write bloom (GATE 1);
step143000 must show the Phi-2-shaped endstate (GATE 2). step0 = the run's
own random init = the floor for everything.

Resumable: skips checkpoints already present in the output JSON.

Usage: .venv/Scripts/python.exe superposition/code/census_checkpoint_trajectory.py
       SMOKE=1 fast pass (3 checkpoints, fewer docs/samples).
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

STEPS = [0, 1000, 143000] if SMOKE else [0, 128, 1000, 2000, 4000, 8000,
                                         16000, 64000, 143000]
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23]
N_DOCS = 6 if SMOKE else 16
MAXLEN = 256
SKIP = 16
N_RELGAIN = 60 if SMOKE else 150       # sampled columns/quadrant
N_ROWGAIN = 500 if SMOKE else 1000     # sampled W_U rows
N_FC1 = 100 if SMOKE else 200          # sampled fc1 rows/layer (read bases)
N_DTOK = 300                           # empirical dictionary tokens

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-checkpoint-trajectory.json")


@torch.no_grad()
def load_ckpt(step):
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=f"step{step}", dtype=torch.float16,
        low_cpu_mem_usage=True)
    return m.to(DEV).eval()


@torch.no_grad()
def weights_tier(m, rng, floors=None):
    """mc_wu fc1/fc2, conc16, quadrants, relgain, readout rank."""
    d = m.config.hidden_size
    WU = m.embed_out.weight.detach().float()
    WUn = (WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
           ).to(DEV, torch.float16)
    gamma_f = m.gpt_neox.final_layer_norm.weight.detach().float().to(DEV)
    A = (WU.to(DEV) * gamma_f[None, :]).to(torch.float16)

    def gain(v):
        vc = (v - v.mean()).half()
        return float((A @ vc).float().norm() / v.norm().clamp(min=1e-9))
    Af = A.float()
    frob2 = float((Af * Af).sum() - (Af.sum(dim=1) ** 2).sum() / d)
    g_rand = float(np.sqrt(frob2 / d))
    # readout effective rank: PR of eig(M^T M)
    C = torch.eye(d, device=DEV) - torch.full((d, d), 1.0 / d, device=DEV)
    MtM = C @ (Af.T @ Af) @ C
    ev = torch.linalg.eigvalsh(MtM).clamp(min=0)
    pr_rank = float((ev.sum() ** 2) / (ev ** 2).sum())
    del Af, MtM, ev, C
    # mean W_U row relgain
    row_ids = rng.choice(WU.shape[0], N_ROWGAIN, replace=False)
    rg = []
    for t in row_ids:
        u = WU[t].to(DEV)
        u = u / u.norm().clamp(min=1e-9)
        rg.append(gain(u) / g_rand)
    out = {"g_rand": round(g_rand, 4),
           "mean_row_relgain": round(float(np.mean(rg)), 4),
           "readout_pr_rank": round(pr_rank, 1), "layers": {}}
    new_floors = {}
    for L in LAYERS:
        lyr = m.gpt_neox.layers[L]
        f2 = lyr.mlp.dense_4h_to_h.weight.detach()          # (d, dff)
        f2u = (f2.float() / f2.float().norm(dim=0, keepdim=True
                                            ).clamp(min=1e-9)
               ).to(DEV, torch.float16)
        mc2 = torch.empty(f2.shape[1])
        for s in range(0, f2.shape[1], 2048):
            mc2[s:s + 2048] = (WUn @ f2u[:, s:s + 2048]
                               ).abs().max(dim=0).values.float().cpu()
        f1 = lyr.mlp.dense_h_to_4h.weight.detach()          # (dff, d)
        f1u = (f1.float() / f1.float().norm(dim=1, keepdim=True
                                            ).clamp(min=1e-9)
               ).to(DEV, torch.float16)
        mc1 = (f1u @ WUn.T.half()).abs().max(dim=1).values.float().cpu() \
            if False else (WUn @ f1u.T).abs().max(dim=0).values.float().cpu()
        # conc16 over downstream gamma-folded fc1
        top16 = None
        for Lp in [x for x in LAYERS if x > L]:
            lp = m.gpt_neox.layers[Lp]
            g = lp.post_attention_layernorm.weight.detach().float()
            r = lp.mlp.dense_h_to_4h.weight.detach().float() * g[None, :]
            r = (r / r.norm(dim=1, keepdim=True).clamp(min=1e-9)
                 ).to(DEV, torch.float16)
            cs = (r @ f2u).abs()
            cand = torch.topk(cs, 16, dim=0).values
            top16 = cand if top16 is None else torch.topk(
                torch.cat([top16, cand], dim=0), 16, dim=0).values
            del r, cs, cand
        cc = top16.mean(dim=0).float().cpu() if top16 is not None else \
            torch.zeros(f2.shape[1])
        new_floors[L] = (float(np.percentile(mc2.numpy(), 95)),
                         float(np.percentile(cc.numpy(), 95)))
        rec = {"mc_fc2_med": round(float(mc2.median()), 4),
               "mc_fc1_med": round(float(mc1.median()), 4)}
        if floors is not None and L < LAYERS[-1]:
            fl = floors[L]
            wr, rc_ = mc2 > fl[0], cc > fl[1]
            q = {"jumble_conc": torch.nonzero((~wr) & rc_).squeeze(1),
                 "readable_conc": torch.nonzero(wr & rc_).squeeze(1)}
            rec["frac_jc"] = round(float(len(q["jumble_conc"]))
                                   / f2.shape[1], 4)
            rec["frac_rc"] = round(float(len(q["readable_conc"]))
                                   / f2.shape[1], 4)
            rec["quads"] = {g_: q[g_].numpy().tolist() for g_ in q}
            for g_ in ("jumble_conc", "readable_conc"):
                idx = q[g_]
                if len(idx) == 0:
                    rec[f"relgain_{g_}"] = None
                    continue
                take = idx[torch.as_tensor(rng.choice(
                    len(idx), min(N_RELGAIN, len(idx)), replace=False))]
                gs = [gain(f2u[:, j].float()) / g_rand
                      for j in take.tolist()]
                rec[f"relgain_{g_}"] = round(float(np.median(gs)), 4)
        out["layers"][str(L)] = rec
        del f2u, f1u, top16
        torch.cuda.empty_cache()
    del A, WUn
    return out, new_floors


@torch.no_grad()
def forwards_tier(m, docs_ids, quads, WU_cpu):
    """Duty cycle, energy shares, const share, read-basis formation."""
    d = m.config.hidden_size
    caps = {}
    handles = []

    def mk_fc1in(L):
        def f(mod, inp):
            caps[("ln", L)] = inp[0][0, SKIP:].detach().float()
        return f

    def mk_acts(L):
        def f(mod, inp):
            caps[("act", L)] = inp[0][0, SKIP:].detach().float()
        return f
    for L in LAYERS:
        lyr = m.gpt_neox.layers[L]
        handles.append(lyr.mlp.dense_h_to_4h.register_forward_pre_hook(
            mk_fc1in(L)))
        handles.append(lyr.mlp.dense_4h_to_h.register_forward_pre_hook(
            mk_acts(L)))

    # frequent tokens over the doc set (for D)
    cnt = {}
    for ids in docs_ids:
        for t in ids[0, SKIP:].tolist():
            cnt[t] = cnt.get(t, 0) + 1
    dtoks = [t for t, c in sorted(cnt.items(), key=lambda x: -x[1])
             if c >= 5][:N_DTOK]
    tok2i = {t: i for i, t in enumerate(dtoks)}

    duty = {L: None for L in LAYERS}
    npos = 0
    Dsum = {L: torch.zeros(len(dtoks), d, device=DEV) for L in LAYERS}
    Dcnt = torch.zeros(len(dtoks), device=DEV)
    esum = {L: {"jc": 0.0, "rc": 0.0, "full": 0.0} for L in LAYERS}
    SsumA = {L: torch.zeros(d, device=DEV) for L in LAYERS}
    SsumB = {L: torch.zeros(d, device=DEV) for L in LAYERS}
    S2sum = {L: 0.0 for L in LAYERS}
    cA = cB = 0
    for di, ids in enumerate(docs_ids):
        m(input_ids=ids)
        T = caps[("act", LAYERS[0])].shape[0]
        npos += T
        toks = ids[0, SKIP:].tolist()
        sel_rows, sel_pos = [], []
        for p, t in enumerate(toks):
            if t in tok2i:
                sel_rows.append(tok2i[t])
                sel_pos.append(p)
        for L in LAYERS:
            a = caps[("act", L)]
            pc = (a > 0).float().sum(0)
            duty[L] = pc if duty[L] is None else duty[L] + pc
            if sel_pos:
                Dsum[L].index_add_(0, torch.as_tensor(sel_rows, device=DEV),
                                   caps[("ln", L)][sel_pos].to(DEV))
            f2 = m.gpt_neox.layers[L].mlp.dense_4h_to_h.weight \
                .detach().float()
            S_full = a.to(DEV) @ f2.T                       # (T, d)
            esum[L]["full"] += float((S_full ** 2).sum())
            S2sum[L] += float((S_full ** 2).sum())
            if di % 2 == 0:
                SsumA[L] += S_full.sum(0)
            else:
                SsumB[L] += S_full.sum(0)
            if L in quads:
                for gk, g_ in (("jc", "jumble_conc"),
                               ("rc", "readable_conc")):
                    idx = quads[L][g_]
                    if len(idx) == 0:
                        continue
                    it = torch.as_tensor(idx, device=DEV)
                    S_g = a.to(DEV)[:, it] @ f2[:, it].T
                    esum[L][gk] += float((S_g ** 2).sum())
        if sel_pos:
            Dcnt.index_add_(0, torch.as_tensor(sel_rows, device=DEV),
                            torch.ones(len(sel_rows), device=DEV))
        if di % 2 == 0:
            cA += T
        else:
            cB += T
    for h in handles:
        h.remove()

    rng = np.random.RandomState(SEED + 5)
    out = {}
    for L in LAYERS:
        rec = {"duty_med": round(float((duty[L] / npos).median()), 4)}
        if esum[L]["full"] > 0:
            rec["eshare_jc"] = round(esum[L]["jc"] / esum[L]["full"], 4)
            rec["eshare_rc"] = round(esum[L]["rc"] / esum[L]["full"], 4)
        muA, muB = SsumA[L] / max(cA, 1), SsumB[L] / max(cB, 1)
        rec["const_share"] = round(float(
            (muA @ muB) / (S2sum[L] / npos + 1e-9)), 4)
        # read-basis formation: sampled fc1 rows vs centered D and W_U
        valid = Dcnt >= 5
        Dm = Dsum[L][valid] / Dcnt[valid].unsqueeze(1)
        Dc = Dm - Dm.mean(0, keepdim=True)
        Dc = Dc / Dc.norm(dim=1, keepdim=True).clamp(min=1e-9)
        f1 = m.gpt_neox.layers[L].mlp.dense_h_to_4h.weight.detach().float()
        take = rng.choice(f1.shape[0], N_FC1, replace=False)
        R = f1[take].to(DEV)
        R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-9)
        cs_d = (R @ Dc.T).abs()
        top8_d = torch.topk(cs_d, min(8, cs_d.shape[1]), dim=1
                            ).values.mean(1)
        WUn = WU_cpu / WU_cpu.norm(dim=1, keepdim=True).clamp(min=1e-9)
        cs_w = (WUn.to(DEV, torch.float16) @ R.T.half()).abs()
        top8_w = torch.topk(cs_w.float(), 8, dim=0).values.mean(0)
        rec["read_emp_top8"] = round(float(top8_d.median()), 4)
        rec["read_wu_top8"] = round(float(top8_w.median()), 4)
        rec["n_dtok_valid"] = int(valid.sum())
        out[str(L)] = rec
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  steps={STEPS}  docs={N_DOCS}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    docs_ids = []
    for t in docs:
        ids = tok(t, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] >= SKIP + 48:
            docs_ids.append(ids)
    print(f"{len(docs_ids)} docs usable")

    rec = {"config": {"smoke": SMOKE, "steps": STEPS, "layers": LAYERS,
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

    floors = rec.get("_floors")
    if floors is not None:
        floors = {int(k): tuple(v) for k, v in floors.items()}
    for step in STEPS:
        key = str(step)
        if key in rec["steps"] and (step == 0 or floors is not None):
            print(f"step{step}: cached, skip")
            continue
        tS = time.time()
        print(f"loading step{step} ...")
        m = load_ckpt(step)
        rng = np.random.RandomState(SEED + step % 99991)
        wt, new_floors = weights_tier(
            m, rng, floors=floors if step > 0 else None)
        if step == 0:
            floors = new_floors
            rec["_floors"] = {str(k): list(v) for k, v in floors.items()}
        quads = {}
        if step > 0:
            for L in LAYERS[:-1]:
                lr = wt["layers"][str(L)]
                if "quads" in lr:
                    quads[L] = lr.pop("quads")
        WU_cpu = m.embed_out.weight.detach().float()
        ft = forwards_tier(m, docs_ids, quads, WU_cpu)
        rec["steps"][key] = {"weights": wt, "forwards": ft}
        json.dump(rec, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        lmid = str(LAYERS[3])
        w = wt["layers"][lmid]
        f = ft[lmid]
        print(f"step{step} ({time.time()-tS:.0f}s): "
              f"mc2={w['mc_fc2_med']} mc1={w['mc_fc1_med']} "
              f"jc%={w.get('frac_jc')} rg_jc={w.get('relgain_jumble_conc')} "
              f"rowrg={wt['mean_row_relgain']} prR={wt['readout_pr_rank']} "
              f"| duty={f['duty_med']} ejc={f.get('eshare_jc')} "
              f"emp8={f['read_emp_top8']} wu8={f['read_wu_top8']}")
        del m, WU_cpu
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
