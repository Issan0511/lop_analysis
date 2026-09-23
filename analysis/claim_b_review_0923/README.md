# Claim B verification and outline alternatives (2026-09-23)

Requested by Issa: verify the proposed central thesis, obtain an independent Fable review through Claude Code, and propose several narrative structures.

This is a research synthesis with deterministic algebra/implementation checks; no new benchmark training runs.

- `python3 -m analysis.claim_b_review_0923.verify_math`: invokes the actual KKT1 implementation in float64; verifies scale covariance, derivative invariance, failure under clipping/stale EMA, the zero-derivative point and unit-slope tail, and a constructive SGD counterexample to an unrestricted inevitability claim.
- `results/claim_b_review_0923/verification.md`: Japanese verification and source links, mirrored to Vault.
- `results/claim_b_review_0923/outlines.md` / `.json`: four optional six-section outlines; old board IDs validated against `compressed_blocks.json`.
- `source_manifest.json`: exact Vault note hashes and repository base.
- `fable_prompt.md`: read-only reviewer instructions. Invocation explicitly uses `--model fable`; actual model is recorded from the response metadata. Original CLI response/logs are archived with hashes in `backup_manifest.json`.

No existing manuscript, PDF, 38-card board data, user memo, or browser board state is replaced. Outlines are proposals, not author decisions. Figure references retain current corrected-manuscript numbering pending adoption.
