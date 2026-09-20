# S4 implementation and execution record

Issa adopted the registered design and predictions before implementation (`5705359`). The statistical definitions, 12 arms, seeds 0–9, t1/t10 branches and 400 epochs remain those in `specs/spec_resp_cifar_ee_0920.md`.

The common runtime reproduces the original host's float32 CUDA operations and Adam updates. Frozen branch outputs are recomputed using the current minibatch shape. `AnchorAdd` preserves the frozen value's signed zero when the difference is zero and passes the difference's derivative unchanged; ELU's derivative is still computed by native autograd.

Run admission tests with the shared environment's Python and `analysis/resp_cifar_ee_0920/checks.py`. Every attempt is retained. The collector requires all named checks and mutation controls and stores their source hashes. The full host comparison uses disjoint seeds 100–109, R=10 and 2×400 epochs. Actual main-prefix equality is an additional runtime gate.

Commit and push verified code, then execute `analysis/resp_cifar_ee_0920/launch.sh`. The runner checks source hashes, a clean implementation, input identity and the GPU lock before proceeding. To pause, create `results/resp_cifar_ee_0920/STOP`; resume the same command after removing it. Complete state is saved at epoch boundaries. No changing seeds or settings on resume.

After every arm is complete, run `analysis/resp_cifar_ee_0920/report.py --src results/resp_cifar_ee_0920`. Intermediate outcomes are not used to change the design. All checks are by the implementing agent; there is no independent audit.
