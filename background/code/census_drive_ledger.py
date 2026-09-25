"""Drive-ledger decomposition: computed token content at firing events
(plan_drive_ledger.md).

The domain-decoding postmortem: "firing drive is token-shaped" is the
UNTRAINED DEFAULT (random-init 58% decodable vs trained 17%) because an
untrained residual is nothing but current-token identity. This instrument
decomposes the drive w.x at each firing event into an ORDERED orthogonal
ledger and isolates the component that is zero by construction in an
untrained network:

  1 baseline   per-corpus mean residual b_L (event doc classified code/prose/
               pooled with the stage-1 classifier)
  2 present    centered pooled-dict rows of tokens within +-W of the firing
               position (W in {0,2,8}; W=8 primary)
  3 computed   top-30 centered pooled-dict rows selected from the RESIDUE of
               x after blocks 1-2 (selection never touches w); tokens present
               in >20% of events excluded
  4 unexplained

Nulls: (a) corpus-pool random-token families for block 3 (4 draws, W=8);
(b) the random-init model end to end (its dicts, baselines, events, fc1).
Controls: SELF vs long-range CONJUNCTION neurons from c2-read-ablation.json.
Byproduct: per-layer computational-vocabulary census from pooled residues.

Usage: .venv/Scripts/python.exe superposition/code/census_drive_ledger.py
       SMOKE=1 MAX_PER_LAYER=N for a fast pass.
"""
import sys
import os
import json
import time
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import census_c2_domain_dict as dd

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
MAX_PER_LAYER = int(os.environ.get("MAX_PER_LAYER", "0"))

LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
N_EVENTS = 40
MAX_TOKENS = 256
MIN_EVENTS = 5
MIN_CNT = 15
TOPK_COMP = 30          # computed-family size
WINDOWS = [0, 2, 8]
W_PRIMARY = 8
N_NULL_FAM = 4          # corpus-pool random-token draws (W=8 only)
PRESENT_EXCL = 0.20     # exclude from computed if present in >20% of events
N_BASE_DOCS = 200 if SMOKE else 4000

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
BIGDOCS = os.path.join(DATA_DIR, "pile-big-80000.json")
DOCS = os.path.join(DATA_DIR, "census-docs-phi-2.json")
RECLASS = os.path.join(DATA_DIR, "census-pass2-reclass.json")
ABLATION = os.path.join(DATA_DIR, "c2-read-ablation.json")
BASE_OUT = os.path.join(DATA_DIR, "drive-ledger-baselines.pt")
OUT = os.path.join(DATA_DIR, "census-drive-ledger.json")

CFG = {
    "trained": {"pass1": "census-pass1-phi-2.json",
                "dicts": "c2-domain-dicts.pt"},
    "random": {"pass1": "census-pass1-phi-2-random.json",
               "dicts": "c2-domain-dicts-random.pt"},
}


def load_model(tag):
    if tag == "random":
        cfg = AutoConfig.from_pretrained(MODEL)
        torch.manual_seed(SEED)      # EXACT pass-1 replication
        m = AutoModelForCausalLM.from_config(cfg).to(torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True)
    m = m.to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def build_baselines(model, tok):
    """Per-corpus mean of input_ln_L(resid): code / prose / pooled."""
    rng = np.random.default_rng(SEED)
    docs = json.load(open(BIGDOCS, encoding="utf-8"))
    if SMOKE:
        docs = docs[:4000]
    entries = dd.select_docs(docs, rng)[:N_BASE_DOCS]
    d = model.config.hidden_size
    sums = {c: {L: torch.zeros(d, dtype=torch.float64, device=DEV)
                for L in LAYERS} for c in ("code", "prose")}
    cnts = {c: 0 for c in ("code", "prose")}
    lns = {L: model.model.layers[L].input_layernorm for L in LAYERS}
    for i, dom, sp in entries:
        ids = tok(docs[i], truncation=True, max_length=MAX_TOKENS)["input_ids"]
        if len(ids) < 8:
            continue
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
        for L in LAYERS:
            sums[dom][L] += lns[L](out.hidden_states[L])[0].double().sum(0)
        cnts[dom] += len(ids)
        del out
    b = {}
    for c in ("code", "prose"):
        b[c] = {L: (sums[c][L] / max(1, cnts[c])).float().cpu() for L in LAYERS}
    b["pooled"] = {L: ((sums["code"][L] + sums["prose"][L]) /
                       max(1, cnts["code"] + cnts["prose"])).float().cpu()
                   for L in LAYERS}
    print(f"  baselines from {cnts} positions")
    return b


def doc_class(text):
    s = dd.code_score(text)
    if s > dd.CODE_THR:
        return "code"
    if s < dd.PROSE_THR:
        return "prose"
    return "pooled"


def ledger_shares(x, w, b_vec, Dpres, Dcomp):
    """Ordered orthogonal ledger. Returns (share_base, share_pres, share_comp).
    x, w: (d,). b_vec: (d,). Dpres: (p,d) may be empty. Dcomp: (k,d)."""
    den = float(x @ w)
    if abs(den) < 1e-6:
        return None
    M = torch.cat([b_vec.unsqueeze(0), Dpres], 0) if Dpres.shape[0] \
        else b_vec.unsqueeze(0)
    Q12, _ = torch.linalg.qr(M.T)                 # (d, 1+p)
    cx, cw = Q12.T @ x, Q12.T @ w
    contrib = cx * cw
    s_base = float(contrib[0]) / den
    s_pres = float(contrib[1:].sum()) / den
    Dc_o = Dcomp - (Dcomp @ Q12) @ Q12.T          # orthogonalize block 3
    Q3, _ = torch.linalg.qr(Dc_o.T)
    s_comp = float(((Q3.T @ x) * (Q3.T @ w)).sum()) / den
    return s_base, s_pres, s_comp


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))
    doc_cls = [doc_class(d) for d in docs]
    reclass = json.load(open(RECLASS, encoding="utf-8"))
    abl = json.load(open(ABLATION, encoding="utf-8"))

    # ablation controls: SELF vs long-range CONJUNCTION (nec offset < -W_PRIMARY)
    abl_class, abl_longrange = {}, set()
    for grp in ("trained_C2", "trained_legible"):
        for k, e in abl[grp].items():
            abl_class[k] = e["class"]
            offs = []
            for c in e["contexts"]:
                nt = c["nec_tokens"]
                if isinstance(nt, str):
                    nt = json.loads(nt)
                offs.extend(o[0] for o in nt)
            if e["class"] == "CONJUNCTION" and any(o < -W_PRIMARY for o in offs):
                abl_longrange.add(k)
    print(f"ablation controls: {len(abl_class)} labeled, "
          f"{len(abl_longrange)} long-range CONJ")

    baselines = torch.load(BASE_OUT, weights_only=False) \
        if os.path.exists(BASE_OUT) else {}
    all_out = {"config": {"windows": WINDOWS, "w_primary": W_PRIMARY,
                          "topk_comp": TOPK_COMP, "n_null_fam": N_NULL_FAM,
                          "present_excl": PRESENT_EXCL, "smoke": SMOKE},
               "models": {}}

    for tag in ("trained", "random"):
        print(f"\n================ {tag} model ================")
        rng = np.random.RandomState(SEED)
        model = load_model(tag)

        if tag not in baselines:
            print("building baselines...")
            baselines[tag] = build_baselines(model, tok)
            torch.save(baselines, BASE_OUT)
        b_all = {c: {L: baselines[tag][c][L].to(DEV) for L in LAYERS}
                 for c in ("code", "prose", "pooled")}

        emp = torch.load(os.path.join(DATA_DIR, CFG[tag]["dicts"]),
                         weights_only=False)
        keep, D_all, C = emp["keep"], emp["D"], emp["cnts"]
        cnt = C["code"] + C["prose"]
        vr = torch.nonzero(cnt >= MIN_CNT).squeeze(1)
        vt = [keep[int(r)] for r in vr]
        cnt_v = cnt[vr].double().numpy()
        p_sample = cnt_v / cnt_v.sum()          # corpus-pool token frequencies
        Dcen = {}
        for L in LAYERS:
            Dv = D_all["pooled"][L][vr].float().to(DEV)
            Dcen[L] = Dv - Dv.mean(0, keepdim=True)
        # token id -> valid row lut
        lut = torch.full((max(vt) + 1,), -1, dtype=torch.long)
        for i, t in enumerate(vt):
            lut[t] = i

        p1 = json.load(open(os.path.join(DATA_DIR, CFG[tag]["pass1"]),
                            encoding="utf-8"))
        neurons = p1["neurons"]
        types = reclass["trained" if tag == "trained" else "random"]
        keys = [k for k in neurons if k in types]
        if MAX_PER_LAYER:
            kept, per = [], {}
            for k in keys:
                L = neurons[k]["layer"]
                if per.get(L, 0) < MAX_PER_LAYER:
                    kept.append(k)
                    per[L] = per.get(L, 0) + 1
            keys = kept
        print(f"{len(keys)} neurons")

        w_dir = {}
        for L in LAYERS:
            fc1 = model.model.layers[L].mlp.fc1.weight.detach().float().to(DEV)
            for k in keys:
                if neurons[k]["layer"] == L:
                    w_dir[k] = fc1[neurons[k]["neuron"]].clone()
            del fc1

        # ---------------- event capture ----------------
        doc_needs = {}
        for k in keys:
            for v, d, pos in neurons[k]["events"][:N_EVENTS]:
                doc_needs.setdefault(d, []).append((k, neurons[k]["layer"], pos))
        need = sorted(doc_needs)
        print(f"need {len(need)} docs")
        ev = {k: [] for k in keys}      # (x cpu, window_valid_rows, doc_class)
        acts, tokids = {}, {}

        def mk_hook(L):
            def hook(mod, args):
                acts[L] = args[0][0].detach().float()
            return hook
        handles = [model.model.layers[L].mlp.fc1.register_forward_pre_hook(
            mk_hook(L)) for L in LAYERS]
        for i, d in enumerate(need):
            ids = tok(docs[d], add_special_tokens=False,
                      max_length=MAX_TOKENS, truncation=True)["input_ids"]
            with torch.no_grad():
                model(torch.tensor([ids], device=DEV))
            for k, L, pos in doc_needs[d]:
                if pos >= acts[L].shape[0]:
                    continue
                lo, hi = max(0, pos - max(WINDOWS)), min(len(ids), pos + max(WINDOWS) + 1)
                wtoks = {}          # offset -> valid row (or skip)
                for j in range(lo, hi):
                    t = ids[j]
                    r = int(lut[t]) if t < lut.numel() else -1
                    if r >= 0:
                        wtoks.setdefault(j - pos, r)
                ev[k].append((acts[L][pos].cpu(), wtoks, doc_cls[d]))
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(need)} ({time.time()-t0:.0f}s)")
        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()

        # ---------------- per-neuron ledger ----------------
        results = {}
        vocab_acc = {L: torch.zeros(len(vt), dtype=torch.float64, device=DEV)
                     for L in LAYERS}
        vocab_n = {L: 0 for L in LAYERS}
        for ki, k in enumerate(keys):
            if len(ev[k]) < MIN_EVENTS:
                continue
            L = neurons[k]["layer"]
            w = w_dir[k]
            X = torch.stack([e[0] for e in ev[k]]).to(DEV)
            nE = X.shape[0]

            # present rows per event per window; presence counts for exclusion
            pres_rows = {W: [] for W in WINDOWS}
            present_frac = np.zeros(len(vt))
            for x_, wtoks, cls_ in ev[k]:
                seen = set(wtoks.values())
                for r in seen:
                    present_frac[r] += 1.0 / nE
                for W in WINDOWS:
                    rows = sorted({r for o, r in wtoks.items() if abs(o) <= W})
                    pres_rows[W].append(torch.tensor(rows, dtype=torch.long))
            b_ev = [b_all[e[2]][L] for e in ev[k]]

            # residues under primary window -> computed-family selection
            score = torch.zeros(len(vt), device=DEV)
            score_even = torch.zeros(len(vt), device=DEV)   # split-half sel
            resid_list = []
            for ei in range(nE):
                rows = pres_rows[W_PRIMARY][ei]
                M = torch.cat([b_ev[ei].unsqueeze(0), Dcen[L][rows.to(DEV)]], 0)
                Q, _ = torch.linalg.qr(M.T)
                r_ = X[ei] - Q @ (Q.T @ X[ei])
                r_ = r_ / r_.norm().clamp(min=1e-9)
                resid_list.append(r_)
                sc = (Dcen[L] @ r_).abs()
                score += sc
                if ei % 2 == 0:
                    score_even += sc
            score /= nE
            excl = torch.tensor(present_frac > PRESENT_EXCL, device=DEV)
            score[excl] = -1.0
            score_even[excl] = -1.0
            comp_rows = torch.topk(score, TOPK_COMP).indices
            comp_rows_even = torch.topk(score_even, TOPK_COMP).indices

            # vocab census accumulation (pooled residue energy)
            for r_ in resid_list:
                vocab_acc[L] += (Dcen[L] @ r_).double() ** 2
            vocab_n[L] += nE

            rec = {"layer": L, "neuron": neurons[k]["neuron"],
                   "type": types[k], "n_events": nE,
                   "abl_class": abl_class.get(k, "-"),
                   "longrange": k in abl_longrange}

            # ledger per window
            for W in WINDOWS:
                sh = []
                for ei in range(nE):
                    rows = pres_rows[W][ei].to(DEV)
                    s = ledger_shares(X[ei], w, b_ev[ei],
                                      Dcen[L][rows], Dcen[L][comp_rows])
                    if s:
                        sh.append(s)
                if not sh:
                    continue
                sh = np.array(sh)
                rec[f"W{W}"] = {"base": round(float(np.median(sh[:, 0])), 4),
                                "pres": round(float(np.median(sh[:, 1])), 4),
                                "comp": round(float(np.median(sh[:, 2])), 4)}

            # split-half: family from even events, share on odd events only
            sh_odd = []
            for ei in range(1, nE, 2):
                rows = pres_rows[W_PRIMARY][ei].to(DEV)
                s = ledger_shares(X[ei], w, b_ev[ei],
                                  Dcen[L][rows], Dcen[L][comp_rows_even])
                if s:
                    sh_odd.append(s[2])
            if sh_odd:
                rec["comp_split"] = round(float(np.median(sh_odd)), 4)

            # corpus-pool random-token null (primary window)
            nulls = []
            for _ in range(N_NULL_FAM):
                nr = torch.tensor(rng.choice(len(vt), TOPK_COMP,
                                             replace=False, p=p_sample),
                                  device=DEV)
                sh = []
                for ei in range(nE):
                    rows = pres_rows[W_PRIMARY][ei].to(DEV)
                    s = ledger_shares(X[ei], w, b_ev[ei],
                                      Dcen[L][rows], Dcen[L][nr])
                    if s:
                        sh.append(s[2])
                if sh:
                    nulls.append(float(np.median(sh)))
            if not nulls:
                nulls = [0.0]
            rec["comp_null"] = round(float(np.mean(nulls)), 4)
            rec["comp_toks"] = [vt[int(j)] for j in comp_rows[:10]]
            results[k] = rec
            if (ki + 1) % 100 == 0:
                print(f"  neuron {ki+1}/{len(keys)} ({time.time()-t0:.0f}s)")

        # ---------------- summaries ----------------
        print(f"\nscored {len(results)} ({time.time()-t0:.0f}s)")

        def summ(ks, label):
            if not ks:
                return None
            g = lambda f: np.array([results[k][f] for k in ks])
            out = {"n": len(ks)}
            line = f"--- {tag} {label} (n={len(ks)}) "
            for W in WINDOWS:
                ks_w = [k for k in ks if f"W{W}" in results[k]]
                for c in ("base", "pres", "comp"):
                    v = np.median([results[k][f"W{W}"][c] for k in ks_w])
                    out[f"W{W}_{c}"] = float(v)
                line += (f"| W{W} b/p/c="
                         f"{out[f'W{W}_base']:+.2f}/"
                         f"{out[f'W{W}_pres']:+.2f}/"
                         f"{out[f'W{W}_comp']:+.2f} ")
            out["comp_null"] = float(np.median(g("comp_null")))
            cs = [results[k]["comp_split"] for k in ks
                  if "comp_split" in results[k]]
            if cs:
                out["comp_split"] = float(np.median(cs))
                line += f"| split={out['comp_split']:+.2f} "
            exc = np.array([results[k][f"W{W_PRIMARY}"]["comp"] -
                            results[k]["comp_null"] for k in ks
                            if f"W{W_PRIMARY}" in results[k]])
            out["comp_minus_null_med"] = float(np.median(exc))
            line += (f"| null={out['comp_null']:+.2f} "
                     f"comp-null={out['comp_minus_null_med']:+.3f}")
            print(line)
            return out

        S = {"ALL": summ(list(results), "ALL")}
        for t in ("A", "A_prime", "C1", "C2"):
            S[t] = summ([k for k in results if results[k]["type"] == t], t)
        S["ctrl_SELF"] = summ(
            [k for k in results if results[k]["abl_class"] == "SELF"],
            "ctrl SELF")
        S["ctrl_longrange"] = summ(
            [k for k in results if results[k]["longrange"]],
            "ctrl long-range CONJ")

        # per-layer computed share (for cross-model p95 calibration)
        per_layer = {}
        for L in LAYERS:
            v = [results[k][f"W{W_PRIMARY}"]["comp"] for k in results
                 if results[k]["layer"] == L and f"W{W_PRIMARY}" in results[k]]
            if v:
                per_layer[str(L)] = {"n": len(v), "med": float(np.median(v)),
                                     "p95": float(np.percentile(v, 95))}
        print("per-layer comp med: " + " ".join(
            f"L{L}:{per_layer[str(L)]['med']:+.2f}" for L in LAYERS
            if str(L) in per_layer))

        # vocab census: top tokens per layer
        vocab = {}
        for L in LAYERS:
            if vocab_n[L] == 0:
                continue
            e = (vocab_acc[L] / vocab_n[L]).float()
            topv, topi = torch.topk(e, 20)
            vocab[str(L)] = [[tok.decode([vt[int(i)]]).strip()[:16],
                              round(float(v), 5)]
                             for v, i in zip(topv, topi)]
        print("vocab census L14 top-10:",
              " ".join(t for t, _ in vocab.get("14", [])[:10]))

        all_out["models"][tag] = {"summary": S, "per_layer": per_layer,
                                  "vocab_census": vocab, "neurons": results}
        del Dcen, w_dir, ev
        torch.cuda.empty_cache()

    # ---------------- cross-model calibration ----------------
    tr = all_out["models"]["trained"]
    rd = all_out["models"]["random"]
    print("\n=========== CALIBRATION: trained vs random-init ===========")
    calib = {}
    for t in ("A", "A_prime", "C1", "C2"):
        ks = [k for k, r in tr["neurons"].items() if r["type"] == t
              and f"W{W_PRIMARY}" in r]
        above = [k for k in ks
                 if str(tr["neurons"][k]["layer"]) in rd["per_layer"]
                 and tr["neurons"][k][f"W{W_PRIMARY}"]["comp"] >
                 rd["per_layer"][str(tr["neurons"][k]["layer"])]["p95"]]
        calib[t] = {"n": len(ks), "above_rand_p95": len(above)}
        print(f"  trained {t:8s} n={len(ks):3d} "
              f"comp>rand_p95: {len(above)} ({100*len(above)/max(1,len(ks)):.0f}%)")
    all_out["calibration"] = calib

    json.dump(all_out, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT}  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
