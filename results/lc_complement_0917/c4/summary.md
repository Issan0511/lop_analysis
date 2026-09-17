# C4: recovery time is bounded below, not above

Cauchy–Schwarz gives |Δθ| ≤ η (1−β1)/sqrt((1−β2)(1−β1²/β2)). For (.9,.999) the factor is 7.270292, not 3.16 for arbitrary gradient histories. With fixed ||μ||₁=104 and η=.001, depth 16 requires at least 21 updates. This necessary condition gives no finite upper bound on recovery time. Zero gradients, unfavorable directions, or persistent transport can prevent recovery. For layer 2 with moving inputs, Δz̄=Δw·μ+w·Δμ+Δw·Δμ+Δb; the bound above controls only Δw·μ+Δb.
