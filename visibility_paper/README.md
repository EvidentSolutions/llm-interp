# A Visibility Threshold for Top-k Logit-Lens Readouts

Draft paper. Build: `pdflatex paper_visibility && bibtex
paper_visibility && pdflatex paper_visibility && pdflatex
paper_visibility`.

Source experiments (all in `superposition/`, seed 0; each script
writes its JSON artifact to `superposition/data/`):

| section | script | artifact |
|---|---|---|
| §3 Phi-2 curves, triangulation | `census_visibility_threshold.py` | `census-visibility-threshold.json` |
| §4 five-model sweep + gates | `census_visibility_threshold_scale.py` | `census-visibility-threshold-scale.json` |
| §4 kurtosis diagnostic | `census_visibility_threshold_scale_posthoc.py` | `census-visibility-threshold-scale-posthoc.json` |
| §5 real anchor per scale | `census_visibility_anchor_scale.py` | `census-visibility-anchor-scale.json` |

Lab-notebook provenance: `superposition/docs/plan_mid_stack_empirical_basis.md`
§3af–§3ah; session reports 2026-07-16 and 2026-07-17.

## Revision legs (pre-registered 2026-07-20, from external review)

Script: `census_visibility_revision.py` →
`census-visibility-revision.json`. Three legs, run before submission:

- **Leg M — more models.** Extend the sweep with gpt2-medium (tied,
  d=1024, V≈50k), bloom-560m (tied, d=1024, V≈251k),
  Qwen2.5-0.5B/1.5B (tied, d=896/1536, V≈152k), TinyLlama-1.1B
  (d=2048, V=32k). Each runs the identical planted-signal sweep +
  bath-ceiling gate. Branches: passers join the regression (n grows
  from 3); failers extend the anisotropy bound (reported, excluded,
  tied/untied noted). BONUS: V now spans 32k–251k, so the predicted
  ceiling sqrt(2 ln 2V) varies 4.66–5.13σ — the V-term of the closed
  form gets its first test on passers (branch: CEILING-TRACKS-V vs
  CEILING-FLAT).
- **Leg E — assumption (i) on real vectors.** For the three isotropic
  models at the anchor layers: take real hidden states (and the real
  contrast δ), remove the top-15 W_U-row span, and measure the
  remainder's bath statistics through row-normalised W_U (excess
  kurtosis + max-σ vs prediction). Branches: ISO-OK (kurtosis ≈ 0,
  ceiling ≈ prediction → assumption (i) empirically supported for
  real reads) / HEAVY-TAILED (report as a measured caveat: the
  formula's bar is optimistic on raw states by the measured factor;
  note if the contrast δ is cleaner than raw states, as common-mode
  cancellation predicts).
- **Leg A — anchor battery.** 8 minimal contrast pairs (the original
  hot/cold dog + 7 new: soup temperature, food/tool, animal/vehicle,
  negation, king/queen, summer/winter, coffee/water) on all five
  original models at the pre-registered layers. Statistic: per-model
  median f_real, median margin, n signal-present (same 3× null rule).
  Branches: BATTERY-CONFIRMS (isotropic-model median margins within
  2× of the hot/cold anchor's) / HETEROGENEOUS (spread > 2× — report
  the distribution, soften §5's "roughly constant") /
  ANCHOR-ATYPICAL (hot/cold is an outlier vs the battery median).

Paper edits queued with the legs: n>3 regression + V-term sentence
(Leg M), assumption-(i) paragraph (Leg E), battery table replacing the
single-anchor table (Leg A), an order-statistic-constant intuition
paragraph (arithmetic only: rank-15 relaxation ln(2V/k)/ln(2V) ≈ 0.77
× measured-ceiling correction (4.46/4.80)² ≈ 0.86 → ≈ 0.66 predicted
vs 0.56 measured; residual attributed to signal-side fluctuation at
the 50% crossing — presented as intuition, not derivation), the
"Use of AI Assistants" section (per bbxnlp), and two audit fixes
(the 8.34σ-vs-8.60σ re-draw parenthetical; the Pythia-1b "vs."
soften).
