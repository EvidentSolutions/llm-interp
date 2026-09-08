import sys, os
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained("microsoft/phi-2")
m=AutoModelForCausalLM.from_pretrained("microsoft/phi-2",dtype=torch.float32,low_cpu_mem_usage=True).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
W=m.lm_head.weight.detach().float(); NL=m.config.num_hidden_layers
def dh(a,b):
    ia=tok(a,add_special_tokens=False)["input_ids"]; ib=tok(b,add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        oa=m(torch.tensor([ia],device=DEV),output_hidden_states=True); ob=m(torch.tensor([ib],device=DEV),output_hidden_states=True)
    return [oa.hidden_states[L][0,-1]-ob.hidden_states[L][0,-1] for L in range(NL+1)]
def rd(v,k=8):
    i=torch.topk(v@W.T,k).indices; return ", ".join(tok.decode([int(x)]).strip()[:9] for x in i)
matched=dh("The hot dog was","The cold dog was")
mismatch=dh("The hot dog was","the cold dog was")
cap=dh("The cold dog was","the cold dog was")
for L in [16,20,24,28]:
    mv=matched[L]/matched[L].norm(); qv=mismatch[L]/mismatch[L].norm()
    print(f"L{L}: cos(matched,mismatched)={float(mv@qv):+.3f}  ||matched||={float(matched[L].norm()):.1f} ||cap-only||={float(cap[L].norm()):.1f}")
    print(f"   matched   : {rd(matched[L])}")
    print(f"   mismatched: {rd(mismatch[L])}")
    print(f"   cap-only  : {rd(cap[L])}")
