# Errata: *The Carried Reference Tunes a Switch–Multiplier Continuum in Gated (GLU) MLPs*

Applies to **Version 1** (September 2026),
DOI [10.5281/zenodo.22813298](https://doi.org/10.5281/zenodo.22813298).

These corrections come from a check of every reported number against the JSON
artifacts in `data/`, carried out on 2026-09-24 while preparing the combined
paper that supersedes this note. None of them changes a conclusion. Several
correct a number, and a few scope a figure to the model and layers it comes
from. The source file for the pre-normalisation footnote (`bl_and_noise.py`,
`bl-noise.json`) was not in the v1 release and has now been added.

## Corrections to stated results

| Where | Published | Corrected | Source (`data/`) |
|---|---|---|---|
| §2 | Gate coupling "scales with the row norm (Spearman 0.55–0.66)" | The paragraph describes Qwen2.5-3B, where the value is **0.57–0.62 at L18 and L27 and 0.15 at L9**. The published range pools Qwen2.5-1.5B and TinyLlama and leaves out Qwen2.5-3B L9 | `census-gated-coupling-dist.json`, `per_layer.{L}.gate.corr_absc_norm` |
| §2 | Knee units: "median R² 0.55–0.83 and best-fit slope ≈ 0.4" | For Qwen2.5-3B: **R² 0.55–0.82, slope ≈ 0.4 at L18 and L27 and 0.21 at L9**. The 0.83 comes from Qwen2.5-1.5B | `census-gate-bilinear-fidelity.json`, `r2_uncoupled_med`, `slope_uncoupled_med` |
| §2, Table 1 prose and caption | Switch units have \|mean\|/sd "well above one"; multiplier units "fire about a third of the time (duty ≈ 1/3)" and are input-driven ("that ratio near zero") | The prose overstated its own table: switch units **3.2 at L9, 1.1–1.3 deeper**; multiplier units **duty 0.19 at L9, about 1/3 deeper**, with the ratio **0.8 at L9 and 0.4 deeper** | `census-gated-uncoupled-units.json`, `constancy_med`, `duty_med` |
| §4 | SwiGLU gate-over-value ratio "rising to about 13 mid-stack" | **Peaks near 13 in the early-to-middle layers** (13.3 at layer 2 of 12; 11.9, 10.1 and 6.6 in the next three; median over layers 4.9) | `ladder-gate-reference.json`, `150000.arms.swiglu.gate_over_val_bL` |
| §3 | With the top 64 variance coordinates removed, "the ratio falls to about 1.3" | **About 1.3 at L18 and L27; 2.0 at L9** | `census-gated-massive-control.json`, `results.{L}.rest_top64.ratio` |
| §7 Two seeds | Second seed at step 12000, "about 80% of the effect" | This figure cannot be derived from the shipped data. By step 12000 the first seed had shown **92%** of its final swiglu–bilinear drop (0.895 → 0.224, final 0.168) | `ladder-gate-reference.json` |
| §5 | "the value branch's effective rank is about 1.5% of d above the GELU read matrix's" | **1.7%** (effective rank 0.8367 against 0.82) | `ladder-content-capacity.json` |

## Minor numerical and wording corrections

| Where | Published | Corrected | Source (`data/`) |
|---|---|---|---|
| §2 | "The operating point g₀, equivalently a unit's cosine to b_L" | g₀ is the unit's cosine to b_L **up to the row norm**, not equivalent to it | — |
| §3, Table 2 | Value-row high-variance energy "0.015" | **0.015–0.017** (L27: 0.0168) | `census-gated-gate-structure.json` |
| §3, Table 3 | Qwen2.5-3B gate/value ratios 33.0 / 8.1 / 7.1 (vs Table 2's 34.4 / 8.4 / 7.1) | Not an error: Table 3 uses a separate calibration sample. The v1 text did not say so | `census-gated-branch-xmodel.json` vs `census-gated-gate-structure.json` |
| §4 Double-gated arm | Each branch carries b_L "at about half the single-gate strength" | "About half" holds for the per-row coupling (ratio 0.56). On shared/iso the ratio is 0.39, and on the top-PC cosine it is 0.78 | `ladder-content-capacity.json`, step 20000 |
| §5 | b_L carries read energy "about 17–20×" the isotropic share | **16–20×** (squared-ReLU: 16.4) | `ladder-content-capacity.json`, `shared_over_iso` |
| §5 | In the value branch b_L is the 78th principal component, "at the isotropic floor" | **Near** the isotropic floor, at **1.7×** it (Table 6 itself shows 1.7) | `ladder-content-capacity.json` |
| §7 Both branches carry content | "value-over-gate top-1 flip 0.84–1.18 and ΔNLL 0.5–1.8×" | Correct; the ΔNLL ratio is a ratio of **mean** ΔNLL (0.51–1.78) | `census-gated-value-content.json`, `value_over_gate_dnll` |
| §4 | Bit-identical initialisation ("maximum absolute weight difference 0") | Correct, and now verified: at step 0 every matched weight (gate/value vs the bilinear branches, GELU vs squared-ReLU read, down-projections, and all 100 shared parameters) differs by exactly 0. The step-0 checkpoints are not bundled | local checkpoints `*_step0.pt` |
