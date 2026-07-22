"""Formation-window microscopy (plan_mid_stack_empirical_basis.md 3w,
pre-registered 2026-07-16).

Fine checkpoint grid over the carpet's formation window (Pythia-410M-deduped
steps 0..16000). Four legs:
  1 fate-mapping at L8 -- per-neuron quadrant state + relgain per checkpoint,
    transition matrices + fc2 rotation (conversion vs recruitment);
  2 induction co-timing -- prefix-matching score per checkpoint;
  3 reference co-formation -- cos(S_g, b) sign split + light inhibitory
    drive share per checkpoint;
  4 causal load during formation -- V2-lite: zero jumble_conc at L8 vs
    size-matched random, final-position KL.

Gates: cached checkpoints reproduce 3u (jc%, duty, ejc at L8); induction
score ~0 at steps 0/128. Resumable per checkpoint (rotations need the
previous checkpoint in the same process run; resume loses one transition,
noted).

Usage: .venv/Scripts/python.exe superposition/code/census_formation_microscopy.py
       SMOKE=1 fast pass (cached checkpoints only).
"""
import sys
import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "EleutherAI/pythia-410m-deduped"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")

STEPS = [0, 1000, 4000] if SMOKE else [0, 128, 256, 512, 1000, 2000, 3000,
                                       4000, 6000, 8000, 16000]
GRID = [2, 5, 8, 11, 14, 17, 20, 23]     # downstream grid for conc16
BAND = [5, 8, 11]                        # carpet band, measured
L_FATE = 8
N_DOCS = 6 if SMOKE else 16
MAXLEN = 256
SKIP = 16
N_SHARE = 100 if SMOKE else 200         # sampled neurons for drive share
N_INDSEQ = 4 if SMOKE else 8            # repeated random sequences

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-formation-microscopy.json")

STATES = ["readable_conc", "jumble_conc", "readable_diff", "jumble_diff"]


@torch.no_grad()
def load_ckpt(step):
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=f"step{step}", dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager")
    return m.to(DEV).eval()


@torch.no_grad()
def layer_metrics(m, WUn, L):
    """mc_wu_fc2 + conc16 for layer L (3u machinery)."""
    f2 = m.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.detach()
    f2u = (f2.float() / f2.float().norm(dim=0, keepdim=True).clamp(min=1e-9)
           ).to(DEV, torch.float16)
    mc = torch.empty(f2.shape[1])
    for s in range(0, f2.shape[1], 2048):
        mc[s:s + 2048] = (WUn @ f2u[:, s:s + 2048]
                          ).abs().max(dim=0).values.float().cpu()
    top16 = None
    for Lp in [x for x in GRID if x > L]:
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
    cc = top16.mean(dim=0).float().cpu()
    del top16
    torch.cuda.empty_cache()
    return mc, cc, f2u


@torch.no_grad()
def relgain_all(m, f2u):
    """Per-column relgain through the effective unembedding, batched."""
    d = m.config.hidden_size
    WU = m.embed_out.weight.detach().float()
    gamma = m.gpt_neox.final_layer_norm.weight.detach().float().to(DEV)
    A = (WU.to(DEV) * gamma[None, :]).to(torch.float16)
    Af = A.float()
    frob2 = float((Af * Af).sum() - (Af.sum(dim=1) ** 2).sum() / d)
    g_rand = float(np.sqrt(frob2 / d))
    del Af
    cm = f2u.float().mean(0)
    out = torch.empty(f2u.shape[1])
    for s in range(0, f2u.shape[1], 1024):
        blk = (f2u[:, s:s + 1024].float() - cm[s:s + 1024][None, :]).half()
        G = (A @ blk).float()
        out[s:s + 1024] = (G.norm(dim=0)
                           / f2u[:, s:s + 1024].float().norm(dim=0
                                                             ).clamp(min=1e-9)
                           ).cpu() / g_rand
        del G, blk
    del A
    torch.cuda.empty_cache()
    return out, g_rand


@torch.no_grad()
def induction_score(m, vocab, rng):
    """Prefix-matching score on repeated random sequences."""
    K = 48
    scores_mean, scores_max = [], []
    for _ in range(N_INDSEQ):
        seq = rng.randint(100, vocab - 100, K).tolist()
        ids = torch.as_tensor([seq + seq], device=DEV)
        out = m(input_ids=ids, output_attentions=True)
        per_head = []
        for att in out.attentions:                      # (1,H,T,T)
            a = att[0].float()
            idx_q = torch.arange(K + 1, 2 * K, device=DEV)
            idx_k = idx_q - K + 1
            per_head.append(a[:, idx_q, idx_k].mean(dim=1))   # (H,)
        ph = torch.cat(per_head)
        scores_mean.append(float(ph.mean()))
        scores_max.append(float(ph.max()))
    return round(float(np.mean(scores_mean)), 4), \
        round(float(np.mean(scores_max)), 4)


class ZeroCols:
    """Zero a set of activation channels entering fc2 at layer L."""

    def __init__(self, m, L, idx):
        self.m, self.L = m, L
        self.idx = torch.as_tensor(idx, device=DEV)
        self.h = None

    def __enter__(self):
        def f(mod, inp):
            x = inp[0].clone()
            x[:, :, self.idx] = 0
            return (x,)
        self.h = self.m.gpt_neox.layers[self.L].mlp.dense_4h_to_h \
            .register_forward_pre_hook(f)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def forwards_tier(m, docs_ids, quads_band, jc8, rand8):
    """Duty, energy shares, b_L, cos(S,b), drive share, V2-lite at L8."""
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
    npos = 0
    esum = {L: {"jc": 0.0, "rc": 0.0, "full": 0.0} for L in BAND}
    Ssum = {L: {"jc": None, "rc": None} for L in BAND}
    kl_j, kl_r = [], []
    share_x = {L: [] for L in BAND}     # (acts, ln) snapshots for share leg
    for di, ids in enumerate(docs_ids):
        out = m(input_ids=ids)
        clean_last = F.log_softmax(out.logits[0, -1].float(), dim=-1)
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
            esum[L]["full"] += float((S_full ** 2).sum())
            for gk in ("jc", "rc"):
                idx = quads_band[L][gk]
                if len(idx) == 0:
                    continue
                it = torch.as_tensor(idx, device=DEV)
                S_g = a.to(DEV)[:, it] @ f2[:, it].T
                esum[L][gk] += float((S_g ** 2).sum())
                Ssum[L][gk] = S_g.sum(0) if Ssum[L][gk] is None \
                    else Ssum[L][gk] + S_g.sum(0)
        # Leg 4: V2-lite at L8
        with ZeroCols(m, L_FATE, jc8):
            oj = m(input_ids=ids)
        with ZeroCols(m, L_FATE, rand8):
            orr = m(input_ids=ids)
        for o, acc in ((oj, kl_j), (orr, kl_r)):
            lg = F.log_softmax(o.logits[0, -1].float(), dim=-1)
            acc.append(float(F.kl_div(lg, clean_last, log_target=True,
                                      reduction="sum")))
    for h in handles:
        h.remove()

    rng = np.random.RandomState(SEED + 11)
    out = {"kl_jumble": round(float(np.median(kl_j)), 5),
           "kl_rand": round(float(np.median(kl_r)), 5)}
    for L in BAND:
        rec = {"duty_med": round(float((duty[L] / npos).median()), 4)}
        if esum[L]["full"] > 0:
            rec["eshare_jc"] = round(esum[L]["jc"] / esum[L]["full"], 4)
            rec["eshare_rc"] = round(esum[L]["rc"] / esum[L]["full"], 4)
        b = bsum[L] / npos
        bh = b / b.norm().clamp(min=1e-9)
        for gk in ("jc", "rc"):
            S = Ssum[L][gk]
            rec[f"cos_S{gk}_b"] = None if S is None else round(float(
                F.cosine_similarity(S / npos, b, dim=0)), 4)
        # light inhibitory drive share over sampled neurons
        f1 = m.gpt_neox.layers[L].mlp.dense_h_to_4h.weight.detach().float()
        take = rng.choice(f1.shape[0], N_SHARE, replace=False)
        A = torch.cat([t[0] for t in share_x[L]], 0)      # (P, dff) cpu
        X = torch.cat([t[1] for t in share_x[L]], 0)      # (P, d) cpu
        shares = []
        bh_c = bh.cpu()
        for n in take:
            acts_n = A[:, n]
            top = torch.topk(acts_n, min(32, acts_n.shape[0])).indices
            w = f1[n].cpu()
            xs = X[top]
            drive = xs @ w
            sb = (float(w @ bh_c)) * (xs @ bh_c)
            ok = drive.abs() > 1e-3
            if ok.sum() > 0:
                shares.append(float((sb[ok] / drive[ok]).median()))
        rec["drive_share_b_med"] = round(float(np.median(shares)), 4) \
            if shares else None
        out[f"L{L}"] = rec
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
                      "l_fate": L_FATE, "n_docs": len(docs_ids),
                      "seed": SEED}}
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

    prev_f2u = None
    prev_states = None
    prev_step = None
    rng0 = np.random.RandomState(SEED)
    for step in STEPS:
        key = str(step)
        if key in rec["steps"] and (step == 0 or floors is not None):
            print(f"step{step}: cached, skip (transitions from it lost "
                  f"unless recomputed)")
            prev_f2u, prev_states, prev_step = None, None, None
            continue
        tS = time.time()
        print(f"loading step{step} ...")
        m = load_ckpt(step)
        WU = m.embed_out.weight.detach().float()
        WUn = (WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9)
               ).to(DEV, torch.float16)
        srec = {"weights": {}, "fate": {}}
        # floors from step 0
        if step == 0:
            floors = {}
            for L in BAND:
                mc, cc, _ = layer_metrics(m, WUn, L)
                floors[L] = (float(np.percentile(mc.numpy(), 95)),
                             float(np.percentile(cc.numpy(), 95)))
            rec["_floors"] = {str(k): list(v) for k, v in floors.items()}
        # band metrics + quadrants
        quads_band = {}
        for L in BAND:
            mc, cc, f2u = layer_metrics(m, WUn, L)
            fl = floors[L]
            wr, rc_ = mc > fl[0], cc > fl[1]
            quads_band[L] = {
                "jc": torch.nonzero((~wr) & rc_).squeeze(1).tolist(),
                "rc": torch.nonzero(wr & rc_).squeeze(1).tolist()}
            srec["weights"][f"L{L}"] = {
                "mc_med": round(float(mc.median()), 4),
                "frac_jc": round(len(quads_band[L]["jc"]) / len(mc), 4),
                "frac_rc": round(len(quads_band[L]["rc"]) / len(mc), 4)}
            if L == L_FATE:
                states = (wr.int() * 1 + rc_.int() * 2).numpy()
                # 0=jd, 1=rd, 2=jc, 3=rc  -> remap to STATES order
                remap = {3: 0, 2: 1, 1: 2, 0: 3}
                states = np.vectorize(remap.get)(states)
                rg, g_rand = relgain_all(m, f2u)
                srec["fate"] = {
                    "states": states.tolist(),
                    "state_counts": {STATES[i]: int((states == i).sum())
                                     for i in range(4)},
                    "relgain_med_by_state": {
                        STATES[i]: round(float(np.median(
                            rg.numpy()[states == i])), 4)
                        for i in range(4) if (states == i).sum() > 0},
                    "g_rand": round(g_rand, 3)}
                if prev_states is not None:
                    tm = np.zeros((4, 4), dtype=int)
                    for a, b in zip(prev_states, states):
                        tm[a, b] += 1
                    rot = (prev_f2u.float() * f2u.float()).sum(0).cpu()
                    srec["fate"]["transition_from_prev"] = {
                        "prev_step": prev_step, "matrix": tm.tolist(),
                        "rot_med_stay": round(float(np.median(
                            rot.numpy()[prev_states == states])), 4),
                        "rot_med_change": round(float(np.median(
                            rot.numpy()[prev_states != states])), 4)
                        if (prev_states != states).sum() > 0 else None}
                prev_f2u = f2u.clone()
                prev_states = states
                prev_step = step
        # induction
        im, ix = induction_score(m, m.config.vocab_size,
                                 np.random.RandomState(SEED + 3))
        srec["induction_mean"] = im
        srec["induction_maxhead"] = ix
        # forwards + V2-lite
        jc8 = quads_band[L_FATE]["jc"]
        n_j = max(len(jc8), 1)
        all_idx = np.arange(4096)
        rand8 = rng0.choice(all_idx, min(n_j, 4096), replace=False).tolist()
        srec["forwards"] = forwards_tier(m, docs_ids, quads_band,
                                         jc8 if jc8 else [0], rand8)
        rec["steps"][key] = srec
        json.dump(rec, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        f8 = srec["forwards"]["L8"]
        print(f"step{step} ({time.time()-tS:.0f}s): "
              f"jc%={srec['weights']['L8']['frac_jc']} "
              f"ind={im}/{ix} duty={f8['duty_med']} "
              f"ejc={f8.get('eshare_jc')} cosSjc_b={f8.get('cos_Sjc_b')} "
              f"share_b={f8.get('drive_share_b_med')} "
              f"KLj={srec['forwards']['kl_jumble']} "
              f"KLr={srec['forwards']['kl_rand']}")
        del m, WU, WUn
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
