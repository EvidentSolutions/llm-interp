# A Visibility Threshold for Top-k Logit-Lens Readouts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21461944.svg)](https://doi.org/10.5281/zenodo.21461944)

Code, data, and the paper for a closed-form account of when a planted
signal is visible in the top-*k* of a logit-lens readout.

**Paper:** [`paper_visibility.pdf`](paper_visibility.pdf) &nbsp;·&nbsp;
**Preprint DOI:** [10.5281/zenodo.21461944](https://doi.org/10.5281/zenodo.21461944) &nbsp;·&nbsp;
**Author:** Olli Tuomi, Evident Solutions Oy
([ORCID 0009-0006-2042-1576](https://orcid.org/0009-0006-2042-1576))

## Overview

Projecting a hidden state through the unembedding *W<sub>U</sub>* and reading the
top-*k* tokens only surfaces a direction if its logit clears the bath of
competing tokens. The paper derives the crossing point in closed form, tests it
with planted-signal sweeps across seven model families, checks the isotropy
assumption on real hidden states, and validates it against a battery of eight
minimal contrast pairs.

## Repository structure

| Path | Contents |
|------|----------|
| `paper_visibility.pdf` | Compiled paper |
| `paper_visibility.tex`, `paper_visibility.bib` | LaTeX source and bibliography |
| `code/` | Experiment scripts |
| `data/` | JSON result artifacts (full per-point numbers) |

Scripts and the artifacts they write (all run with fixed seed 0):

| Script (`code/`) | Artifact (`data/`) | Role |
|---|---|---|
| `census_visibility_threshold.py` | `census-visibility-threshold.json` | Phi-2 curves, triangulation, real anchor |
| `census_visibility_threshold_scale.py` | `census-visibility-threshold-scale.json` | Five-model sweep, gates, regression |
| `census_visibility_threshold_scale_posthoc.py` | `census-visibility-threshold-scale-posthoc.json` | Kurtosis diagnostic |
| `census_visibility_anchor_scale.py` | `census-visibility-anchor-scale.json` | Real anchor per scale |
| `census_visibility_revision.py` | `census-visibility-revision.json` | Extra model families, real-vector isotropy check, eight-pair battery |

## Reproducing

### Build the paper

```bash
pdflatex paper_visibility
bibtex   paper_visibility
pdflatex paper_visibility
pdflatex paper_visibility
```

### Run the experiments

The scripts load public checkpoints (EleutherAI Pythia deduped suite,
`microsoft/phi-2`, `gpt2-medium`, `bigscience/bloom-560m`,
`Qwen/Qwen2.5-0.5B` and `-1.5B`, `TinyLlama/TinyLlama_v1.1`) via
`AutoModelForCausalLM` and write their JSON artifact to `data/`, e.g.:

```bash
python code/census_visibility_threshold.py
```

## Citation

```bibtex
@misc{tuomi2026visibility,
  author    = {Tuomi, Olli},
  title     = {A Visibility Threshold for Top-k Logit-Lens Readouts},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21461944},
  url       = {https://doi.org/10.5281/zenodo.21461944}
}
```

## License

Code is released under the [MIT License](../LICENSE). The paper text and figures
are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
