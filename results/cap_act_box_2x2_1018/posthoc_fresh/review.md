# Follow-up implementation review

Reviewed locally by the implementing root agent before new fresh-cap outcomes. This is a self-review, not an independent review.

- Original spec/source/result hashes remain unchanged. The follow-up lives in a separate analysis directory so the original source-closure glob is unchanged. Completed original manifests were verified, and reused compact outputs were compared to immutable commit 14ad56af.
- The explicit fresh engine copies the original step, task-training loop and CUDA capture/rewind block exactly (AST equality). Only selecting one late task, advancing prefix batch RNG without training, initializing fixed cap radii, and fresh-specific metadata differ. CPU replay against the unchanged host reproduces both activations' final fresh accuracy and batch hashes.
- Each fresh run starts with P0, zero moments/counter and initial activation state. The graph warm-up is rewound. Radii are active after every actual update; the initial forward has no projection. Only W1/W2 are projected; CPU checks preserve biases, output weights and moments and verify the exact shrink rule.
- Original task-1 radii match between ref/cap for both activations. Production requires identical reconstructed batch/class/label/input/initial hashes and exact old task-29 fresh accuracy at original CSV precision. Live R10 graph/eager checks and projection-bound checks gate production.
- Five task observations are averaged within each seed. Inference uses 10 paired seeds and six simultaneous contrasts. The history interaction includes optimizer and parameter history and does not uniquely identify a mechanism. Non-significance does not prove equivalent effects.
- Production and CNN serialize through the existing shared lock. Images remain on CUDA between fresh conditions. Raw outputs are preserved and copied with SHA-256 verification; existing arms are not overwritten.
