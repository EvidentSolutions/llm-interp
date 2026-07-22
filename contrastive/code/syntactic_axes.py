"""
Syntactic and semantic axis discovery via contrastive directions.

1. Extract number axis from multiple pairs, test transfer
2. Extract tense axis, gender axis
3. Test composition (are axes independent?)
4. Test cross-context transfer (does the axis work in new frames?)

Usage: .venv/Scripts/python.exe contrastive/code/syntactic_axes.py
"""
import sys
import torch
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")


def get_last_hidden(model, tok, text, layer):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    return out.hidden_states[layer][0, -1, :].float()


def topk(logits, tok, k=6):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    W_U = model.lm_head.weight.detach()
    L = 24  # analysis layer

    # ================================================================
    # 1. NUMBER AXIS — extract, test transfer, test composition
    # ================================================================
    print(f"\n{'='*100}")
    print(f"1. NUMBER AXIS (layer {L})")
    print(f"{'='*100}")

    # Training pairs: extract number direction
    number_train = [
        ("The dog was", "The dogs were"),
        ("The cat was", "The cats were"),
        ("The house was", "The houses were"),
        ("The car was", "The cars were"),
    ]

    # Held-out pairs: test transfer
    number_test = [
        ("The bird was", "The birds were"),
        ("The tree was", "The trees were"),
        ("The book was", "The books were"),
        ("The phone was", "The phones were"),
        ("The boy was", "The boys were"),
        ("The girl was", "The girls were"),
        # Different frame
        ("A dog ran", "The dogs ran"),
        ("The big dog sat", "The big dogs sat"),
        ("My cat slept", "My cats slept"),
        ("One horse galloped", "Two horses galloped"),
    ]

    # Extract number directions from training pairs
    train_dirs = []
    for sing, plur in number_train:
        h_s = get_last_hidden(model, tok, sing, L)
        h_p = get_last_hidden(model, tok, plur, L)
        d = h_s - h_p
        d = d / d.norm()
        train_dirs.append(d)
        print(f"  {sing:>25} vs {plur:<25} W_U: [{topk((d @ W_U.float().T), tok)}]")

    # Mean number direction
    number_dir = torch.stack(train_dirs).mean(dim=0)
    number_dir = number_dir / number_dir.norm()
    print(f"\n  Mean number direction through W_U:")
    print(f"    sing pole: [{topk((number_dir @ W_U.float().T), tok)}]")
    nd_neg = -number_dir
    print(f"    plur pole: [{topk((nd_neg @ W_U.float().T), tok)}]")

    # Pairwise cosines of training directions
    print(f"\n  Training direction pairwise cos:")
    for i, (s1, _) in enumerate(number_train):
        for j, (s2, _) in enumerate(number_train):
            if j > i:
                cos = float(torch.nn.functional.cosine_similarity(
                    train_dirs[i].unsqueeze(0), train_dirs[j].unsqueeze(0)))
                print(f"    {s1.split()[1]}-{s2.split()[1]}: {cos:.3f}")

    # Test transfer: project held-out pairs onto the mean number direction
    print(f"\n  Transfer test (projection of held-out Δh onto mean number dir):")
    print(f"  Positive = singular, negative = plural")
    for sing, plur in number_test:
        h_s = get_last_hidden(model, tok, sing, L)
        h_p = get_last_hidden(model, tok, plur, L)
        d = h_s - h_p
        proj = float(d @ number_dir)
        cos = float(torch.nn.functional.cosine_similarity(
            d.unsqueeze(0), number_dir.unsqueeze(0)))
        print(f"    {sing:>30} vs {plur:<30} proj={proj:>+7.1f} cos={cos:>+.3f}")

    # ================================================================
    # 2. TENSE AXIS
    # ================================================================
    print(f"\n{'='*100}")
    print(f"2. TENSE AXIS (layer {L})")
    print(f"{'='*100}")

    tense_train = [
        ("The dog runs", "The dog ran"),
        ("The cat sleeps", "The cat slept"),
        ("The man walks", "The man walked"),
        ("The bird sings", "The bird sang"),
    ]

    tense_test = [
        ("The girl dances", "The girl danced"),
        ("The boy jumps", "The boy jumped"),
        ("The car stops", "The car stopped"),
        ("She writes", "She wrote"),
        ("He eats", "He ate"),
        ("The horse runs", "The horse ran"),
    ]

    tense_dirs = []
    for pres, past in tense_train:
        h_pres = get_last_hidden(model, tok, pres, L)
        h_past = get_last_hidden(model, tok, past, L)
        d = h_pres - h_past
        d = d / d.norm()
        tense_dirs.append(d)
        print(f"  {pres:>25} vs {past:<25} W_U: [{topk((d @ W_U.float().T), tok)}]")

    tense_dir = torch.stack(tense_dirs).mean(dim=0)
    tense_dir = tense_dir / tense_dir.norm()
    print(f"\n  Mean tense direction through W_U:")
    print(f"    present pole: [{topk((tense_dir @ W_U.float().T), tok)}]")
    td_neg = -tense_dir
    print(f"    past pole:    [{topk((td_neg @ W_U.float().T), tok)}]")

    print(f"\n  Training direction pairwise cos:")
    for i in range(len(tense_train)):
        for j in range(i + 1, len(tense_train)):
            cos = float(torch.nn.functional.cosine_similarity(
                tense_dirs[i].unsqueeze(0), tense_dirs[j].unsqueeze(0)))
            n1 = tense_train[i][0].split()[-1]
            n2 = tense_train[j][0].split()[-1]
            print(f"    {n1}-{n2}: {cos:.3f}")

    print(f"\n  Transfer test:")
    for pres, past in tense_test:
        h_pres = get_last_hidden(model, tok, pres, L)
        h_past = get_last_hidden(model, tok, past, L)
        d = h_pres - h_past
        proj = float(d @ tense_dir)
        cos = float(torch.nn.functional.cosine_similarity(
            d.unsqueeze(0), tense_dir.unsqueeze(0)))
        print(f"    {pres:>30} vs {past:<30} proj={proj:>+7.1f} cos={cos:>+.3f}")

    # ================================================================
    # 3. GENDER AXIS
    # ================================================================
    print(f"\n{'='*100}")
    print(f"3. GENDER AXIS (layer {L})")
    print(f"{'='*100}")

    gender_train = [
        ("The boy was", "The girl was"),
        ("The man was", "The woman was"),
        ("He was", "She was"),
        ("The king was", "The queen was"),
    ]

    gender_test = [
        ("The father was", "The mother was"),
        ("The brother was", "The sister was"),
        ("The husband was", "The wife was"),
        ("His dog was", "Her dog was"),
        ("The actor was", "The actress was"),
        ("The prince was", "The princess was"),
    ]

    gender_dirs = []
    for male, female in gender_train:
        h_m = get_last_hidden(model, tok, male, L)
        h_f = get_last_hidden(model, tok, female, L)
        d = h_m - h_f
        d = d / d.norm()
        gender_dirs.append(d)
        print(f"  {male:>25} vs {female:<25} W_U: [{topk((d @ W_U.float().T), tok)}]")

    gender_dir = torch.stack(gender_dirs).mean(dim=0)
    gender_dir = gender_dir / gender_dir.norm()
    print(f"\n  Mean gender direction through W_U:")
    print(f"    male pole:   [{topk((gender_dir @ W_U.float().T), tok)}]")
    gd_neg = -gender_dir
    print(f"    female pole: [{topk((gd_neg @ W_U.float().T), tok)}]")

    print(f"\n  Training direction pairwise cos:")
    for i in range(len(gender_train)):
        for j in range(i + 1, len(gender_train)):
            cos = float(torch.nn.functional.cosine_similarity(
                gender_dirs[i].unsqueeze(0), gender_dirs[j].unsqueeze(0)))
            n1 = gender_train[i][0].split()[-1]
            n2 = gender_train[j][0].split()[-1]
            print(f"    {n1}-{n2}: {cos:.3f}")

    print(f"\n  Transfer test:")
    for male, female in gender_test:
        h_m = get_last_hidden(model, tok, male, L)
        h_f = get_last_hidden(model, tok, female, L)
        d = h_m - h_f
        proj = float(d @ gender_dir)
        cos = float(torch.nn.functional.cosine_similarity(
            d.unsqueeze(0), gender_dir.unsqueeze(0)))
        print(f"    {male:>30} vs {female:<30} proj={proj:>+7.1f} cos={cos:>+.3f}")

    # ================================================================
    # 4. AXIS INDEPENDENCE — are number, tense, gender orthogonal?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"4. AXIS INDEPENDENCE")
    print(f"{'='*100}")

    cos_nt = float(torch.nn.functional.cosine_similarity(
        number_dir.unsqueeze(0), tense_dir.unsqueeze(0)))
    cos_ng = float(torch.nn.functional.cosine_similarity(
        number_dir.unsqueeze(0), gender_dir.unsqueeze(0)))
    cos_tg = float(torch.nn.functional.cosine_similarity(
        tense_dir.unsqueeze(0), gender_dir.unsqueeze(0)))

    print(f"  number-tense:  cos = {cos_nt:>+.4f}")
    print(f"  number-gender: cos = {cos_ng:>+.4f}")
    print(f"  tense-gender:  cos = {cos_tg:>+.4f}")
    print(f"  (0 = orthogonal, ±1 = aligned)")

    # ================================================================
    # 5. SEMANTIC AXES — animacy, size, valence
    # ================================================================
    print(f"\n{'='*100}")
    print(f"5. SEMANTIC AXES (layer {L})")
    print(f"{'='*100}")

    # Animacy
    animacy_pairs = [
        ("The dog was", "The rock was"),
        ("The cat was", "The table was"),
        ("The bird was", "The chair was"),
        ("The man was", "The building was"),
    ]

    animacy_dirs = []
    print("\n  ANIMACY:")
    for anim, inanim in animacy_pairs:
        h_a = get_last_hidden(model, tok, anim, L)
        h_i = get_last_hidden(model, tok, inanim, L)
        d = h_a - h_i
        d = d / d.norm()
        animacy_dirs.append(d)
        print(f"  {anim:>25} vs {inanim:<25} W_U: [{topk((d @ W_U.float().T), tok)}]")

    animacy_dir = torch.stack(animacy_dirs).mean(dim=0)
    animacy_dir = animacy_dir / animacy_dir.norm()

    print(f"  Pairwise cos:")
    for i in range(len(animacy_dirs)):
        for j in range(i + 1, len(animacy_dirs)):
            cos = float(torch.nn.functional.cosine_similarity(
                animacy_dirs[i].unsqueeze(0), animacy_dirs[j].unsqueeze(0)))
            print(f"    {animacy_pairs[i][0].split()[1]}-{animacy_pairs[j][0].split()[1]}: {cos:.3f}")

    # Valence (positive vs negative)
    valence_pairs = [
        ("The man was happy", "The man was sad"),
        ("The day was good", "The day was bad"),
        ("The news was great", "The news was terrible"),
        ("The food was delicious", "The food was disgusting"),
    ]

    valence_dirs = []
    print("\n  VALENCE:")
    for pos, neg in valence_pairs:
        h_p = get_last_hidden(model, tok, pos, L)
        h_n = get_last_hidden(model, tok, neg, L)
        d = h_p - h_n
        d = d / d.norm()
        valence_dirs.append(d)
        print(f"  {pos:>35} vs {neg:<35} W_U: [{topk((d @ W_U.float().T), tok)}]")

    valence_dir = torch.stack(valence_dirs).mean(dim=0)
    valence_dir = valence_dir / valence_dir.norm()

    print(f"  Pairwise cos:")
    for i in range(len(valence_dirs)):
        for j in range(i + 1, len(valence_dirs)):
            cos = float(torch.nn.functional.cosine_similarity(
                valence_dirs[i].unsqueeze(0), valence_dirs[j].unsqueeze(0)))
            print(f"    {i}-{j}: {cos:.3f}")

    # ================================================================
    # 6. ALL AXES PAIRWISE — full independence matrix
    # ================================================================
    print(f"\n{'='*100}")
    print(f"6. FULL INDEPENDENCE MATRIX")
    print(f"{'='*100}")

    axes = {
        "number": number_dir,
        "tense": tense_dir,
        "gender": gender_dir,
        "animacy": animacy_dir,
        "valence": valence_dir,
    }

    print(f"  {'':>10}", end="")
    for name in axes:
        print(f" {name:>10}", end="")
    print()

    for n1, d1 in axes.items():
        print(f"  {n1:>10}", end="")
        for n2, d2 in axes.items():
            cos = float(torch.nn.functional.cosine_similarity(
                d1.unsqueeze(0), d2.unsqueeze(0)))
            print(f" {cos:>+10.3f}", end="")
        print()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
