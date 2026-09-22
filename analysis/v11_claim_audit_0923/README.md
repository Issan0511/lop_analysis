# V11 claim audit and structure board (2026-09-23)

This is an authoring aid for the current Japanese manuscript. It does not write to Obsidian or change scientific claims automatically. The audit is based on the immutable manuscript snapshot in `results/v11_claim_audit_0923/`, Vault source notes, and targeted checks of existing specifications, outputs, and implementations. No new training or literature review was run.

## Outputs

- `output/structure/V11_structure_board_0923.html`: self-contained offline board. Open in a browser. Move any of the 38 sentence cards directly to a numbered position or use up/down, edit the proposed summaries, choose main text / appendix / park, inspect the original paragraphs, evidence links, and audit findings.
- `results/v11_claim_audit_0923/audit.md`: detailed report mirrored to Vault `可塑性喪失/論文作成/V11主張と証拠の監査_0923.md`.
- `results/v11_claim_audit_0923/compressed_blocks.json`: sentence summaries and line ranges in the source manuscript.
- `results/v11_claim_audit_0923/verification.json`: source hashes and link verification.
- `results/v11_claim_audit_0923/board_data.json`: embedded board data.

The browser saves decisions locally. Export JSON to retain an editable backup and Markdown to pass an outline back to the assistant or put it in the Vault. A different browser or origin has separate storage. No shared database, external API, or automatic synchronization is used. Obsidian links require Obsidian and the `obsidian-research` Vault on the user's machine.

## Rebuild

From the repository root:

```bash
python3 analysis/v11_claim_audit_0923/assemble_audit.py
python3 analysis/v11_claim_audit_0923/build.py
```

The build resolves Vault note names at `/home/issan/Projects/obsidian-research`. It reads the frozen manuscript snapshot, not the latest edited manuscript. The source manuscript and previous PDF were intentionally left unchanged; the audit lists corrections to apply when the author settles the outline.

The initial board order follows the manuscript and is not a newly approved outline. Pending B2/B8 and decisions 9a/9b remain pending. Existing decisions 10b and 11 are not reopened automatically.

## Verification

`node analysis/v11_claim_audit_0923/check_state.cjs` checks the actual state validator and Markdown serializer, including nine malformed imports that must not overwrite edits. Browser QA and its limits are recorded in `results/v11_claim_audit_0923/ui_verification.json`. Number selection and arrows are the verified reorder controls.
