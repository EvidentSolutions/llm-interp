# The Carried Reference Tunes a Switch–Multiplier Continuum in Gated (GLU) MLPs

*A companion note to [Transformer MLP Gate Thresholds Are Couplings to a Carried
Reference Direction](../background/).*

Code, data, and the paper for the finding that in a gated (GLU) transformer MLP,
a gate unit's coupling to the carried reference direction *b<sub>L</sub>* (from
the companion paper) is its coordinate on a continuous switch–multiplier
spectrum, and that trained SwiGLU spreads units across the whole spectrum,
shifting toward the multiplier end with depth.

**Paper:** [`background_glu.pdf`](background_glu.pdf) &nbsp;·&nbsp;
**DOI:** [10.5281/zenodo.22813298](https://doi.org/10.5281/zenodo.22813298) &nbsp;·&nbsp;
**arXiv:** submitted, under review (arXiv ID pending) &nbsp;·&nbsp;
**Author:** Olli Tuomi, Evident Solutions Oy
([ORCID 0009-0006-2042-1576](https://orcid.org/0009-0006-2042-1576))

## Overview

Expanding the SiLU nonlinearity about its knee gives a gate unit two regimes
selected by its resting operating point, the coupling of its gate row to
*b<sub>L</sub>*: a deep-negative coupling rests the gate closed and it fires
only above threshold (a thresholded *switch*); a coupling near zero rests the
gate at the knee, where the unit computes, to leading order, a bilinear product
of its two branch reads (a *multiplier*). Trained SwiGLU populates the whole
spectrum, and the mix shifts toward the multiplier end with depth. Supporting
this: the reference coupling localises to the gate branch (across four SwiGLU
models the gate-projection rows align with *b<sub>L</sub>* at 4–34× the
value-projection rows) and sits in the reference's high-variance subspace; a
bit-identical-init training ladder shows the elementwise nonlinearity is what
creates the switch end (adding a gate relocates *b<sub>L</sub>*, swapping the
elementwise function does not), and among four feed-forward variants only the
gated one spans both ends of the spectrum with independent key and value reads;
freeing the reference from the value branch returns it a content dimension. The
reading extends to the experts of a mixture-of-experts, where the reference
becomes expert-conditional. The account is geometric and makes no loss/quality
claim.

## Repository structure

| Path | Contents |
|------|----------|
| `background_glu.pdf` | Compiled paper |
| `background_glu.tex`, `background_glu.bib` | LaTeX source and bibliography |
| `code/` | Experiment scripts |
| `data/` | JSON result artifacts (full per-point numbers) |

Scripts and the artifacts they write (public checkpoints, fixed seeds):

| Script (`code/`) | Artifact (`data/`) | Paper location |
|---|---|---|
| `census_gated_gate_structure.py` | `census-gated-gate-structure.json` | Table 2, gate/value read geometry |
| `census_gated_massive_control.py` | `census-gated-massive-control.json` | Massive-coordinate control (§ The coupling localises to the gate branch) |
| `census_gated_branch_xmodel.py` | `census-gated-branch-xmodel.json` | Table 3, cross-model replication |
| `census_gated_uncoupled_smollm2.py` | `census-gated-uncoupled-smollm2.json` | Table 3, the SmolLM2 soft-spot population shift |
| `census_bL_shape_xarch.py` | `census-bL-shape-xarch.json` | Cross-architecture *b<sub>L</sub>*-shape measurement |
| `census_gated_coupling_dist.py` | `census-gated-coupling-dist.json` | Table 1, switch/multiplier split |
| `census_gated_uncoupled_units.py` | `census-gated-uncoupled-units.json` | Table 1, content-gate characterisation |
| `census_gate_bilinear_fidelity.py` | `census-gate-bilinear-fidelity.json` | Behavioural bilinear-fidelity check |
| `census_gated_value_content.py` | `census-gated-value-content.json` | Content-test control (both branches carry content) |
| `ladder_gate_reference.py` | `ladder-gate-reference.json` | Table 4, the relocation ladder |
| `census_ladder_gate_spectrum.py` | `ladder-gate-spectrum.json` | Table 5, cross-arm spectrum span |
| `ladder_content_capacity.py` | `ladder-content-capacity.json` | Table 6, content-capacity reframe |
| `measure_seed1_relocation.py` | `seed1-relocation.json` | Seed replicate (n=2) |
| `census_moe_expert_reference.py` | `census-moe-expert-reference.json` | MoE pilot (OLMoE-1B-7B) |
| `census_moe_shared_vs_routed.py` | `census-moe-shared-vs-routed.json` | MoE shared-vs-routed contrast (Qwen1.5-MoE) |
| `ffn_variants.py` | — | Feed-forward variants used by the training ladder (incl. `DoubleGatedMLP`) |
| `run_glu_matched.py` | — | Trains the bit-identical-init four-arm ladder |
| `ladder_probe.py`, `corpus.py` | — | Shared infrastructure (snapshot loading, evaluation, the Pile dataset reader) used by the ladder scripts above |

Shared input also in `data/`: `pile-big-80000.json`, a 200-document prefix of
the project's Pile sample. Every script above reads only its first 16–24
documents, in original order, so this prefix reproduces the published numbers
exactly at a fraction of the size of the full 80,000-document file.

## Reproducing

### Build the paper

```bash
pdflatex background_glu
bibtex   background_glu
pdflatex background_glu
pdflatex background_glu
```

### Run the experiments

Most scripts load public checkpoints (Qwen2.5-1.5B/3B, TinyLlama-1.1B,
SmolLM2-1.7B, OLMoE-1B-7B, Qwen1.5-MoE-A2.7B) via `AutoModelForCausalLM` and
`data/pile-big-80000.json`, writing their JSON artifact to `data/`, e.g.:

```bash
python code/census_gated_gate_structure.py
```

The training-ladder scripts (`ladder_gate_reference.py`,
`ladder_content_capacity.py`, `census_ladder_gate_spectrum.py`,
`measure_seed1_relocation.py`) depend on the bit-identical-initialisation
four-arm ladder trained with `run_glu_matched.py` and its multi-gigabyte
checkpoint snapshots, which are not bundled here. Their reported numbers are in
`data/ladder-gate-reference.json`, `data/ladder-content-capacity.json`,
`data/ladder-gate-spectrum.json`, and `data/seed1-relocation.json`;
`run_glu_matched.py`, `ffn_variants.py`, `ladder_probe.py`, and `corpus.py` are
included so the training and evaluation setup is inspectable even without the
checkpoints.

## Citation

Archived on Zenodo; also on arXiv (under review, arXiv ID pending — add the
`eprint` field once assigned).

```bibtex
@misc{tuomi2026backgroundglu,
  author    = {Tuomi, Olli},
  title     = {The Carried Reference Tunes a Switch--Multiplier Continuum in
               Gated (GLU) MLPs},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22813298},
  url       = {https://doi.org/10.5281/zenodo.22813298}
}
```

## License

Code is released under the [MIT License](../LICENSE). The paper text and figures
are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
