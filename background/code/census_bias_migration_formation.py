"""Bias migration as a TIME-COURSE, not an endpoint (background paper, PLAN
follow-up; Olli 2026-07-21 -- 'did we study how the bias terms develop during
training?').

Sec 5's migration claim -- the gate resting inhibition w.b_L + b_n is carried
by the reference coupling w.b_L, not the explicit gate bias b_n -- is measured
only at the two ENDPOINTS: the from_config random-init twin (b_n = 0 at init,
w.b_L ~= 0) vs the final trained model (w.b_L dominant). The word 'migration'
is thus an inference from endpoints, never an observed trajectory of the two
static terms. The Formation section (Sec 6) tracks the reference DRIVE share
(how much b contributes to activation) across Pythia checkpoints, but NOT this
partition of the two resting-inhibition parameters.

This script closes that gap. Pythia-410M-deduped has bias-carrying GELU gates
(dense_h_to_4h has a real b_n) and public log-spaced checkpoints, so the
partition |w.b_L| vs |b_n| is meaningful and can be tracked step by step.
Per checkpoint x mid-stack layer, the STATIC decomposition of the scale script
(reused verbatim): ref = w.b_L (b_L = corpus-mean post-LN gate input from one
forward pass), bias = b_n (weight), resting = ref + bias, plus duty and
rho(resting, duty).

The question is the SHAPE of the two curves:
  - CLEAN MIGRATION: b_n stays pinned near its small init while |w.b_L| grows
    monotonically -> ref-dominant fraction climbs 0 -> ~1, the resting
    operating point is built entirely in the coupling.
  - OVERTAKE: b_n first moves (early), then |w.b_L| grows past it -> migration
    is a hand-off, not a pure coupling build. Reported either way.
step0 = the run's own random init = the twin floor for free (resting ~0,
duty ~0.5).

Static decomposition + one forward pass per checkpoint. Resumable (skips
checkpoints already in the output JSON). Pythia only (the one bias-carrying
family with public checkpoints).

Usage: .venv/Scripts/python.exe superposition/code/census_bias_migration_formation.py
       SMOKE=1 -> 3 checkpoints, few docs.
Downloads one Pythia-410M checkpoint (~800MB) per step not already cached.
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

# log-spaced checkpoints spanning init -> formation window (512->1000) -> end
STEPS = [0, 512, 143000] if SMOKE else [
    0, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 43000, 143000]

N_DOCS = 6 if SMOKE else 30
MAXLEN = 256
SKIP = 16
DEPTH_FRACS = [0.20, 0.35, 0.50, 0.65]   # mid-stack, matches scale/decomp

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-bias-migration-formation.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def get_gate(layer):
    """Pythia GELU gate: mlp.dense_h_to_4h (has bias)."""
    return layer.mlp.dense_h_to_4h, "dense_h_to_4h"


@torch.no_grad()
def run_one(m, docs_text, tok, sel_layers):
    layers = m.gpt_neox.layers
    d = m.config.hidden_size

    ln_in, gate_out = {}, {}
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
        handles.append(gmod.register_forward_pre_hook(mk_in()))
        handles.append(gmod.register_forward_hook(mk_out()))

    s_ln = {L: torch.zeros(d, device=DEV) for L in sel_layers}
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
            pc = (gate_out[L] > 0).float().sum(0)
            duty_num[L] = pc if duty_num[L] is None else duty_num[L] + pc
    for h in handles:
        h.remove()
    b_ln = {L: s_ln[L] / npos for L in sel_layers}

    per_layer = {}
    for L in sel_layers:
        gmod, gname = get_gate(layers[L])
        W = gmod.weight.detach().float()
        b = gmod.bias.detach().float()
        ref = (W @ b_ln[L]).cpu()
        bnc = b.cpu()
        resting = ref + bnc
        duty = (duty_num[L] / npos).cpu().numpy()
        per_layer[str(L)] = {
            "gate": gname,
            "ref_med": round(float(ref.median()), 4),          # signed w.b_L
            "bias_med": round(float(bnc.median()), 4),         # signed b_n
            "resting_med": round(float(resting.median()), 4),
            "abs_ref_med": round(float(ref.abs().median()), 4),
            "abs_bias_med": round(float(bnc.abs().median()), 4),
            "frac_resting_neg": round(float((resting < 0).float().mean()), 4),
            "frac_ref_dominant": round(float(
                (ref.abs() > bnc.abs()).float().mean()), 4),
            "ref_over_bias": round(float(ref.abs().median()
                                         / bnc.abs().median().clamp(min=1e-9)),
                                   2),
            "spearman_resting_duty": round(spearman(resting.numpy(), duty), 3),
            "duty_med": round(float(np.median(duty)), 4),
        }
    return per_layer


@torch.no_grad()
def main():
    t0 = time.time()
    print(f"Device: {DEV}  SMOKE={SMOKE}  steps={STEPS}")
    docs_text = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    tok = AutoTokenizer.from_pretrained(MODEL)
    n_layers = 24                       # pythia-410m, fixed across checkpoints
    sel = sorted(set(min(max(int(n_layers * f), 1), n_layers - 1)
                     for f in DEPTH_FRACS))

    rec = {"config": {"smoke": SMOKE, "model": MODEL, "steps": STEPS,
                      "n_docs": len(docs_text), "skip": SKIP,
                      "depth_fracs": DEPTH_FRACS, "sel_layers": sel,
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
    print(f"layers {sel}")

    for step in STEPS:
        key = str(step)
        if key in rec["steps"]:
            print(f"step{step}: cached, skip")
            continue
        tS = time.time()
        print(f"loading step{step} ...")
        try:
            # float32: deep-layer trained gate inputs reach massive-activation
            # magnitudes (sun2024/gallego2025) that overflow fp16 -> nan b_L.
            m = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=f"step{step}", dtype=torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager").to(DEV).eval()
            torch.manual_seed(SEED)
            pl = run_one(m, docs_text, tok, sel)
            rec["steps"][key] = pl
            json.dump(rec, open(OUT, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"step{step} ({time.time()-tS:.0f}s):")
            for L in sel:
                e = pl[str(L)]
                print(f"  L{L}: |ref|={e['abs_ref_med']} "
                      f"|bias|={e['abs_bias_med']} "
                      f"ref/bias={e['ref_over_bias']} "
                      f"ref_dom={e['frac_ref_dominant']} "
                      f"rest={e['resting_med']} "
                      f"rho={e['spearman_resting_duty']} "
                      f"duty={e['duty_med']}")
            del m
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"  !! step{step} failed: {ex!r}")
            gc.collect()
            torch.cuda.empty_cache()

    # trajectory summary: one mid-stack layer (0.5 depth), curves over steps
    Lmid = sel[2]
    print(f"\n==== trajectory at L{Lmid} (0.5 depth) ====")
    print(f"  {'step':>7} {'|ref|':>7} {'|bias|':>7} {'r/b':>6} "
          f"{'refdom':>7} {'rest':>8} {'rho':>6} {'duty':>6}")
    steps_present = sorted((int(k) for k in rec["steps"]),
                           key=lambda s: s)
    for s in steps_present:
        e = rec["steps"][str(s)].get(str(Lmid))
        if e is None:
            continue
        print(f"  {s:>7} {e['abs_ref_med']:>7} {e['abs_bias_med']:>7} "
              f"{e['ref_over_bias']:>6} {e['frac_ref_dominant']:>7} "
              f"{e['resting_med']:>8} {e['spearman_resting_duty']:>6} "
              f"{e['duty_med']:>6}")

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
