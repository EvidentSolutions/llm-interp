"""Shared training corpus loading for the MinOut sweeps.

The same corpus file lives in different places on the two machines used for
these runs (local workstation vs RunPod), so the path is resolved by search
rather than hardcoded. Every arm in a sweep must see the identical corpus and
identical shuffle seed, otherwise the comparison is confounded by data order.
"""
import os
import json
import random
import torch

REPO = os.path.dirname(os.path.abspath(__file__))

# Search order: minout/data, RunPod layout, sibling superposition project.
CORPUS_CANDIDATES = [
    os.path.join(REPO, "..", "data", "pile-big-80000.json"),
    os.path.join(REPO, "..", "..", "superposition", "data",
                 "pile-big-80000.json"),
    "/workspace/ffn_sweep/data/pile-big-80000.json",
]


def resolve_corpus(name="pile-big-80000.json"):
    """Return the first existing path for the named corpus file."""
    cands = [os.path.join(os.path.dirname(c), name)
             for c in CORPUS_CANDIDATES]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"corpus {name!r} not found. Looked in:\n  " +
        "\n  ".join(os.path.abspath(c) for c in cands))


class LocalPileDataset(torch.utils.data.Dataset):
    """Pre-tokenized fixed-length chunks from a JSON list of documents.

    Documents are shuffled with `seed` before tokenization, then packed
    end-to-end and cut into seq_len+1 chunks (input and shifted label share
    the tensor). Chunk boundaries therefore depend on the seed, so all arms
    in a comparison must use the same one.
    """

    def __init__(self, tokenizer, seq_len, seed, path=None):
        path = path or resolve_corpus()
        with open(path, encoding="utf-8") as f:
            docs = json.load(f)
        rng = random.Random(seed)
        rng.shuffle(docs)
        self.chunks = []
        buffer = []
        for doc in docs:
            ids = tokenizer(doc, add_special_tokens=False).input_ids
            buffer.extend(ids)
            while len(buffer) >= seq_len + 1:
                self.chunks.append(
                    torch.tensor(buffer[:seq_len + 1], dtype=torch.long))
                buffer = buffer[seq_len:]
        self.n_tokens = len(self.chunks) * seq_len
        print(f"  Corpus: {len(self.chunks)} chunks "
              f"({self.n_tokens/1e6:.1f}M tokens) from {len(docs)} docs "
              f"[{os.path.basename(path)}]", flush=True)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def epochs_at(steps, batch_size, seq_len, n_tokens):
    """How many times the corpus is revisited after `steps` optimizer steps."""
    return steps * batch_size * seq_len / max(n_tokens, 1)


class BinPileDataset(torch.utils.data.Dataset):
    """Pre-tokenized flat uint16 corpus (built by build_corpus_1b.py), memmapped.

    Same contract as LocalPileDataset -- fixed seq_len+1 rows, deterministic
    chunk order under `seed` -- but tokenization happened once at build time,
    so run startup is O(1) and RAM is the memmap page cache, not the corpus.
    Every arm of a sweep must use the same (path, seed): the shuffle is over
    chunk indices, so identical seeds give identical data order.
    """

    def __init__(self, path, seq_len=512, seed=1234):
        import numpy as np
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        n = (len(self.tokens) - 1) // seq_len
        rng = random.Random(seed)
        self.order = list(range(n))
        rng.shuffle(self.order)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        import numpy as np
        s = self.order[i] * self.seq_len
        row = np.asarray(self.tokens[s:s + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(row)
