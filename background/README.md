# Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21498411.svg)](https://doi.org/10.5281/zenodo.21498411)

Code, data, and the paper for the finding that a transformer's corpus-mean
residual direction is not a nuisance to subtract but a functional reference: the
constant the MLP gate population couples to when it sets its operating point.

**Paper:** [`paper_background.pdf`](paper_background.pdf) &nbsp;·&nbsp;
**Preprint DOI:** [10.5281/zenodo.21498411](https://doi.org/10.5281/zenodo.21498411) &nbsp;·&nbsp;
**Author:** Olli Tuomi, Evident Solutions Oy
([ORCID 0009-0006-2042-1576](https://orcid.org/0009-0006-2042-1576))

## Overview

An exact decomposition of resting MLP-gate pre-activations in Phi-2 shows that
the resting inhibition of >99.9% of gates is carried by the coupling
*w·b<sub>L</sub>* to the carried mean direction *b<sub>L</sub>* — at 50–60× the
explicit bias parameter — and that this resting term rank-predicts each neuron's
firing rate at ρ ≈ 0.95. Removing the stream's projection on the reference
multiplies above-threshold firing ~9× dose-monotonically (a random-initialised
twin is flat), and replacement tests show the effect is carried by the
*direction*, not its magnitude or token content. The decomposition replicates on
four further families spanning both gate types (GELU with an explicit gate bias,
bias-free SwiGLU), and across an eight-model scan is present in every GELU and
SiLU family — absent only in OPT, where an opposing LayerNorm bias cancels the
carried reference.

## Repository structure

| Path | Contents |
|------|----------|
| `paper_background.pdf` | Compiled paper |
| `paper_background.tex`, `paper_background.bib` | LaTeX source and bibliography |
| `code/` | Experiment scripts |
| `data/` | JSON result artifacts (full per-point numbers) |

Scripts and the artifacts they write (public checkpoints, fixed seeds):

| Script (`code/`) | Artifact (`data/`) | Paper location |
|---|---|---|
| `census_background_reference.py` | `census-background-reference.json` | Object / function reference |
| `census_background_identity.py` | `census-background-identity.json` | Identity, co-removal |
| `census_reference_mechanism.py` | `census-reference-mechanism.json` | Bias, maintenance, the knob |
| `census_checkpoint_trajectory.py` | `census-checkpoint-trajectory.json` | Formation |
| `census_formation_microscopy.py` | `census-formation-microscopy.json` | Formation |
| `census_reference_geometry.py` | `census-reference-geometry.json` | Geometric comparisons |
| `census_crossmodel_causal.py` | `census-crossmodel-causal.json` | Cross-model dose-response, entropy V |
| `census_bias_migration_scale.py` | `census-bias-migration-scale.json` | Cross-model table |
| `census_bias_migration_formation.py` | `census-bias-migration-formation.json` | Coupling time-course |
| `census_bias_migration_nonlinearity.py` | `census-bias-migration-nonlinearity.json` | Eight-model scan, OPT boundary |
| `census_reference_norm_decomp.py` | `census-reference-norm-decomp.json` | Residual-vs-LN-bias decomposition |
| `census_suppressor_reference.py` | `census-suppressor-reference.json` | Read-side alignment |
| `census_reference_corpus.py` | `census-reference-corpus.json` | Register adaptivity |
| `census_readweight_factorization.py` | `census-readweight-factorization.json` | Per-gate anatomy |

Shared inputs also in `data/`: `census-docs-phi-2.json` (naturalistic-prose
corpus), `census-docs-code-repo.json` (code corpus, register comparison), and
`c2-inhibitory-reader.json` (suppressor token directions).

## Reproducing

### Build the paper

```bash
pdflatex paper_background
bibtex   paper_background
pdflatex paper_background
pdflatex paper_background
```

### Run the experiments

The scripts load public checkpoints (primary model `microsoft/phi-2`; the
cross-model scans add GPT-2, Pythia, Qwen2.5, TinyLlama, OPT and other families)
via `AutoModelForCausalLM` and write their JSON artifact to `data/`, e.g.:

```bash
python code/census_background_reference.py
```

Note: `census_suppressor_reference.py` additionally reads a large precomputed
token-statistics cache (`c2-empirical-dict.pt`, ~180 MB) that is not bundled
here; its reported numbers are already in `data/census-suppressor-reference.json`.

## Citation

```bibtex
@misc{tuomi2026background,
  author    = {Tuomi, Olli},
  title     = {Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21498411},
  url       = {https://doi.org/10.5281/zenodo.21498411}
}
```

## License

Code is released under the [MIT License](../LICENSE). The paper text and figures
are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
