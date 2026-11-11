# Implementation and preregistered checks

Spec committed as359d0453 before trained-cell smoke. MLP ELU seed0 task-end1 and ViT SL seed10252010 task-end100 each completed the prescribed50update smoke:12 logical curves,10 runs,2 shared per model. Synthetic factorial/body-invariance/S-P-ZC identity passed. Smoke observations are excluded from main summaries. No effect-dependent design change.

Read-only review confirmed canonical MLP single-P and Pall RNGs, factorial/bodyAdam retention, same-batch delayed suffix, and exact S/P-ZC reuse. Before main, resume validation was strengthened to require source archive hashes as well as script/spec/helper hashes. This does not alter training. S-history reconstruction checks saved weights/labels; Pall checks all nine saved measurement fields at each reconstructed task, not unavailable weights.

T90 numerical detail: canonical MLP eval stores a float32 fraction; exactly1080/1200 correct is represented as float32(.9)=.8999999761581421. Its threshold is the same float32(.9), preserving the mathematical90% decision. ViT float64 fraction uses .9. No tolerance-based expansion of the threshold. Recorded sustained T90 is only sustained over observed points to the horizon; endpoint-only support is reported.

Main evaluation has no early stopping. The source-set compatibility check was added after smoke, so smoke and main script hashes intentionally differ by this guard. The exact smoke source is preserved at smoke_mlp/source_run.py with its hash checked against all smoke JSON; main code is the committed version.

Independent synthetic continuation test also restored the25-update fork and reran the same suffix to50: curve, final parameters and Adam hashes matched exactly. No real-cell training was used for that supplementary test.

Independent summary review verified nonconstant paired bootstrap effects with identical source-wise resampling, guarded ratio=1 for identical source effects, rejection of unpaired S/P seeds, separate first/sustained censoring, elapsed offsets, and endpoint-only support. A synthetic820-logical/690-direct full design passed; missing rows, changed body Adam, broken shared curves, eager ViT engine, and mismatched source hashes were rejected. Synthetic primary outcomes produced the registered supported/rejected/unresolved/not-evaluated classifications. This review did not inspect main results. Reproduction: python -m analysis.p_residual_loading_1111.verify_summary.

## Final-data audit and interpretation review

All820 logical curves /690 executed trajectories /130 registered shared identities passed the full validator. All41 jobs exited0; no seed was added, removed, or rerun on effect direction.80 source archive hashes matched.190 unique archived full MLP curves (250 logical comparisons) matched exactly;300 S-history task endpoints matched weights/labels;290 Pall replay tasks matched all nine archived record fields. The760-row MLP interim numerical analysis and final registered metrics agree. COMPLETE.json records counts and resources:48min GPU worker elapsed,0.94CPU core-hour proxy, within both caps.

Independent final interpretation review retained the registered unresolved verdict. ViT Ptau0 KC−ZC has positive U differences in only2/5seeds; first-T90 difference+30 has CI[−90,155], while sustained-T90 difference−30 has CI[−140,105]. Neither replaces the primary U contrast. The positive conditional KK−ZK result is reported alongside the cleared-head-Adam primary contrast, not substituted for it. The P/S ratio supports weakening of this specific intervention effect, not a causal fraction of total delay. All5 ViT tau25 parameter contrasts are negative; later zeroing also destroys newly learned readout structure, so this is not proof that an initial cost vanished or that the body was permanently damaged.

Plots were visually inspected. Main training files remained fixed at0c874fe9 throughout execution; later changes concern analysis, reports and figures only. Raw, smoke source and logs are archived with a full SHA256 manifest.
