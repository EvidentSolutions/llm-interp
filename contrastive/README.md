# Contrastive Projection: Reading Transformer Internals Through Desuperposition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20843137.svg)](https://doi.org/10.5281/zenodo.20843137)

Code, data, and the paper for **Contrastive Projection**, a training-free method
for reading what separates two transformer inputs in token space.

**Paper:** [`docs/paper_v1.pdf`](docs/paper_v1.pdf) &nbsp;·&nbsp;
**Preprint DOI:** [10.5281/zenodo.20843137](https://doi.org/10.5281/zenodo.20843137) &nbsp;·&nbsp;
**Author:** Olli Tuomi, Evident Solutions Oy
([ORCID 0009-0006-2042-1576](https://orcid.org/0009-0006-2042-1576))

## Overview

Subtracting the hidden states of two matched inputs at each layer and projecting
the difference through the unembedding matrix *W<sub>U</sub>* produces a
layer-by-layer readout of what separates them in token space. The subtraction
*desuperposes* the residual stream: it cancels the shared content and isolates the
axis of variation — features invisible to the logit lens on either input alone.

The method is validated three ways (causal injection recovering the prediction
gap, dose-response directional specificity, and MLP-neuron gating) and applied to
Phi-2 (2.7B) to trace compound-noun recognition, the IOI and factual-recall
circuits, recall-vs-hallucination, scalar implicature, and metaphor processing
across 15 semantic axes and four models.

## Repository structure

| Path | Contents |
|------|----------|
| `docs/paper_v1.pdf` | Compiled paper |
| `docs/paper_v1.tex`, `docs/paper_v1.bib` | LaTeX source and bibliography |
| `docs/paper_v1.md` | Markdown version of the paper |
| `code/` | Experiment scripts (≈56 Python files + `run_all.sh`) |

## Reproducing

### Build the paper

```bash
cd docs
pdflatex paper_v1
bibtex   paper_v1
pdflatex paper_v1
pdflatex paper_v1
```

(or `latexmk -pdf paper_v1`). Requires the `mathpazo` and `bera` font packages.

### Run the experiments

The scripts load a Hugging Face model via `AutoModelForCausalLM` and register
forward hooks; the primary model is `microsoft/phi-2`. To run the full suite:

```bash
MODEL=microsoft/phi-2 bash code/run_all.sh
```

Individual experiments can be run directly, e.g. `python code/ioi_path_trace.py`.

## Citation

```bibtex
@misc{tuomi2026contrastive,
  author    = {Tuomi, Olli},
  title     = {Contrastive Projection: Reading Transformer Internals Through Desuperposition},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20843137},
  url       = {https://doi.org/10.5281/zenodo.20843137}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper text and figures are
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
