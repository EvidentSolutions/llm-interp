# Contrastive Projection — changes in version 3

This changelog is kept separate from the paper body. Use it for the version notes
/ arXiv comment field, not inside the PDF.

## Version 3

Version 3 reframes the method as a reading tool and stops arguing for claims
beyond it. The difference vector's causal status is inherited from RepE rather
than re-established per contrast; the analysis of whether the *surfaced tokens*
carry the causal content, and the associated triangulation-recovery tables, is
removed, since it neither held cleanly nor was needed. Multi-contrast
triangulation is now presented as a reading technique. Causal grounding is shown
once, on the compound-noun circuit. The IOI, factual-recall, successor,
parametric-recall, dose-response, and truth-direction demonstrations are cut; the
first two relied on circuit-specific vocabulary that this paper does not otherwise
need.

Added: a cross-architecture replication of the compound-noun circuit (Phi-2,
Pythia-1.4B, Qwen2.5-1.5B), a cross-seed control showing that the token-face a
computation wears is network-specific while the distinction it encodes is
preserved, and a section on what the readout reads.
