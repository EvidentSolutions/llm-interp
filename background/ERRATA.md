# Errata: *Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction*

Applies to **Version 2** (September 2026),
DOI [10.5281/zenodo.22723588](https://doi.org/10.5281/zenodo.22723588).

These corrections come from a check of every reported number against the JSON
artifacts in `data/`, carried out on 2026-09-24 while preparing the combined
paper that supersedes this one. Several of the scripts behind the paper's numbers
were not included in the v2 release. They have now been added to `code/` and
`data/` (see the README), so each correction below can be checked against a
shipped file. A second check, on 2026-09-25, compared the paper's statements
about cited work against the full text of each cited paper; two citations were
wrong (see "Citation corrections" at the end).

None of the corrections changes a conclusion of the paper. Several change a
stated number, and a few narrow the scope of a claim. Section names follow the
v2 paper.

## Corrections to stated results

| Where | Published | Corrected | Source (`data/`) |
|---|---|---|---|
| Abstract; §3 Result | Removal releases firing at "30–40×" the norm-matched control | **23–59×** (37× at L6, 59× at L10, 23× at L14). The published range appears to have been read off Table 1's rounded control values | `census-background-reference.json`, `trained.manip.L{6,10,14}_a0.0` vs `_rand0`, `dfrac05` |
| Abstract; §5 | Reference coupling at "50–60×" the explicit bias parameter | **48–56×** (ratio of median magnitudes at the tabulated layers; the definition used for the cross-model 15–18× and 38–43×) | `census-reference-mechanism.json`, `trained.legA.{L}.abs_ref_med / abs_bias_med` |
| §5 | ">99.9% of gates at every measured layer" | At every **tabulated** layer (L6–L22). The shallowest layer measured, L2, is at 98.5% | same file, `legA.2.frac_ref_dominant` |
| §5 | Twin: "w·b_L ≈ 0 (median magnitude ≤ 0.02) … the informative twin statistic is the absent coupling" | The twin's coupling has **no shared sign** rather than no magnitude: median ≤ 0.01 but typical magnitude 0.46–0.53, with half the gates on each side. The informative statistic is that the twin population has no shared resting inhibition | same file, `twin.legA.{L}.ref_med`, `abs_ref_med`, `frac_resting_neg` |
| §4 Table 2 | KL column: 0.521 / 0.963 / 0.502 / 0.402 (text: shuffled KL "0.96") | The KL column came from a different run than the shipped artifact. Shipped values: **0.516 / 0.950 / 0.484 / 0.395**, shuffled 0.95. The Δfrac and cosine columns and the 31× / 130× ratios are unchanged | `census-background-identity.json`, `trained.variants.*.kl` |
| §6 Co-removal | Releases firing at every depth, "at L18 more than at the shallowest site" | Releases firing at every depth, **most at the shallowest site** (+0.100 at L6, +0.075 at L18) | same file, `variants.corem_ge6.dfrac05` |
| §3 Cross-model | Weakening the reference raises entropy "with the minimum at the natural magnitude" | Holds for **Phi-2 and Pythia**. In Qwen2.5-1.5B the amplification side is noisy and dips below it (the script's own verdict: PARTIAL/NOISY, `min_at_1` false) | `census-crossmodel-causal.json`, `models.*.v_decision` |
| §5 Cross-model | Resting term rank-predicts duty at "ρ ≥ 0.96 across mid-stack layers" | **ρ ≥ 0.91** (GPT-2-medium's shallowest sampled layer is 0.913; Table 4 itself shows 0.91–0.97) | `census-bias-migration-scale.json` |
| §5 Boundary case | Eight-model scan: "ref-dominant 99.5–100% where a bias exists, ρ ≥ 0.89 throughout" | **99.4–100%**, and **ρ ≥ 0.91 except at Qwen2.5-1.5B's shallowest layer (0.62)**, which the preceding paragraph already names | `census-bias-migration-nonlinearity.json` |
| §7 Development | Pythia coupling "peaks near 0.85–0.95 around step 4000" | Reaches 0.84–0.95 **by** step 4000 and **peaks at step 8000** (16000 at the shallowest layer) | `census-bias-migration-formation.json`, `abs_ref_med` |
| §1 | "a share of about −0.40 of trained drive at every layer measured" | **−0.40 pooled** over the eight layers measured; per layer **−0.31 to −0.55**. Twin: −0.11 to +0.01 per layer | `census-drive-ledger.json`, `models.trained.summary.ALL.W0_base` and per-neuron `W0.base` |
| §2 | Reference cosines across inputs, "where the twin has one direction for all of them" | The twin keeps one direction for prose, code and random tokens (≥ 0.99) but only partly for a repeated token (0.51–0.52) | `census-reference-position-shape.json`, `random.{L}.identity.prose.*` |
| §3 Input-side demonstration | Resting term's "rank correlation to measured mean pre-activation falling from 0.98 to 0.16" at L24 | **0.95 → 0.16** at L24 (0.98 is the L8/L16 prose value), and the statistic is a Pearson correlation, not a rank correlation | `census-reference-text-statistics.json`, `leg1.L24.{prose,repeat}.corr_rest_meanz` |
| §3 Matched-energy scale | Enforcing "an exactly zero empirical mean on the injection (residual ≤ 0.01)" | The residual is ≤ 0.01 in three of the four settings and **0.06** in the fourth (ε = 0.2). The along-b̂ arms balance sign counts, not magnitudes | `census-noise-zero-mean.json`, rows `along_bL_bal` |

## Minor numerical and wording corrections

| Where | Published | Corrected | Source (`data/`) |
|---|---|---|---|
| §1 | "median duty cycle is 0.076 against 0.48" | Median over **400 sampled gates** (full-population per-layer medians are 0.008–0.094) | `census-pass1-phi-2.json`, `census-pass1-phi-2-random.json` |
| §2 | Resting term within "a band of about 0.7 nats" | Pre-activations are not in nats; **a band of about 0.7** | — |
| §2 What holds the fraction | Gain control removes "85% / 78% / 66%" | **85% / 77% / 66%** (L16: 0.775) | `census-reference-variance-sources.json`, `L16.ln_gain_removes` |
| §2 Rotating band | "adjacent layers sit at ≈ 0.97" | Layers **four apart** sit at ≈ 0.97; adjacent layers at 0.99 | `census-reference-statistic-nulls.json`, `trained.leg_b.cos` |
| §2 Content | "about 1.7× the isotropic floor" | **About 1.5–1.7×** | `census-background-reference.json`, `anatomy.{L}.b_mc` (trained vs twin) |
| §2 Sources | Mean writes at "cosines +0.2 to +0.7 … norms ~1–3 against update RMS 15–38" | Cosines **+0.10 to +0.77** and norms **~1–4 through L27**; the last four layers write larger (norms 7–15) and less consistently aligned. The update-RMS figure is not recorded in any artifact and is withdrawn | `census-background-reference.json`, `trained.components` |
| §3 Table 1 caption | Twin "\|Δfrac\| ≤ 0.002" | **≤ 0.003** | `census-background-reference.json`, twin `manip` |
| §3 Output damage | Twin raw mean "8–11× larger in norm (168 vs. 15–21)" | **8–10×** (121–208 vs. 15–21) | same file, twin `m_raw_norm` |
| §3 Matched-energy scale | "~1000× more than removing a random direction" | **670–4200×** across layers (about 1150× at L10) | `census-noise-floor.json` |
| §5 | Bias alone orders duty at "ρ = 0.81" | **0.81–0.83** | `census-reference-duty-controls.json` |
| §5 Parameter share | OPT "−17 to −8%" | **−16 to −8%**, opt-350m only (the opt-1.3b run failed) | `census-reference-bias-share-xmodel.json` |
| §6 Attention feed | Twin channels "0.93 vs. 0.87" | **0.96 vs. 0.86** | `census-reference-mechanism.json`, `twin.legC.10.attn` |
| §5 Table 3 caption | Bootstrap intervals for ρ | Percentile intervals; resampling biases ρ slightly low, so some point estimates sit just above their interval (e.g. L14 0.960 vs [0.953, 0.959]) | `census-reference-mechanism.json` |
| Related work | Twin reference vs sink "cos = 0.98" | **0.99** (0.985–0.987) | `census-reference-geometry.json` |
| Related work | Sink overlap "0.08 → 0.33 in the principal-subspace metric" | This trend is in the principal subspace of the **raw-residual** covariance, not the layer-normalised one quoted just before it (−0.41 to −0.14) | `census-reference-geometry-mahalanobis.json`, `sigma_res.topr.r512` |
| Limitations | Raw-residual metric "returns the same value for all three proxies to three decimals" | **Within 0.004** | same file, `sigma_res.mcs` |
| Limitations | Whitening variant "ranking cos(b_L, sink) above cos(b_L, m_L)" at "9,360 positions" | **8,868 positions**. The inversion occurs only at the weakest regularisation (λ = 0.01, at L6 and L18); at the other two the positive control stays on top at 0.70–0.80 | same file, `trained.n_far_positions`, `sigma_ln.whit` |

## Citation corrections

| Where | Published | Corrected |
|---|---|---|
| §7 Development | "…by the time the induction transition — itself a multi-phase event (Minegishi et al., 2025; Xu, 2026) — completes." | Neither source supports "multi-phase" for the induction transition. Minegishi et al. study a synthetic in-context meta-learning task and contrast its multi-phase circuit emergence with "the single-phases change in induction heads", the opposite of what the sentence attributes to them. Xu (2026) shows that capability-circuit formation and attention-sink formation are separate transitions, and that induction-circuit membership turns over during training, but does not split the induction transition itself into phases. The aside and both citations should be removed: "…by the time the induction transition completes." |
| §7 Development | "…training's first act on the MLP write, which destroys the init constant (constant share 0.40 → 0.15 by step 128) (Gallego-Feliciano et al., 2025)." | The measurement is this paper's own (`census-formation-prelude.json`). Gallego-Feliciano et al. do not report it: they measure massive activations in the residual-stream hidden state only, explicitly excluding intermediate MLP states. The citation does not support the sentence and should be removed. What they do show about early training in the same model family (massive activations are absent at initialisation and, in shallow and deep layers, rise to an early peak) may be cited as a comparison. |
