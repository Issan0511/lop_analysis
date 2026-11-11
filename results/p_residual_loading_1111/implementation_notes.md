# Implementation and preregistered checks

Spec committed as359d0453 before trained-cell smoke. MLP ELU seed0 task-end1 and ViT SL seed10252010 task-end100 each completed the prescribed50update smoke:12 logical curves,10 runs,2 shared per model. Synthetic factorial/body-invariance/S-P-ZC identity passed. Smoke observations are excluded from main summaries. No effect-dependent design change.

Read-only review confirmed canonical MLP single-P and Pall RNGs, factorial/bodyAdam retention, same-batch delayed suffix, and exact S/P-ZC reuse. Before main, resume validation was strengthened to require source archive hashes as well as script/spec/helper hashes. This does not alter training. S-history reconstruction checks saved weights/labels; Pall checks all nine saved measurement fields at each reconstructed task, not unavailable weights.

T90 numerical detail: canonical MLP eval stores a float32 fraction; exactly1080/1200 correct is represented as float32(.9)=.8999999761581421. Its threshold is the same float32(.9), preserving the mathematical90% decision. ViT float64 fraction uses .9. No tolerance-based expansion of the threshold. Recorded sustained T90 is only sustained over observed points to the horizon; endpoint-only support is reported.

Main evaluation has no early stopping. The source-set compatibility check was added after smoke, so smoke and main script hashes intentionally differ by this guard. Both source versions can be recovered from the saved smoke provenance and the copied initial /tmp script only if needed; main code is the committed version.
