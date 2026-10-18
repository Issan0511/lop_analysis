# Width and position in the original cap experiment

Post hoc descriptive readout requested by Issa. Width s is the population standard deviation of preactivation over the task images, calculated separately per unit. These are means over units and seeds; r is mean(m_i/s_i), not mean(m_i)/mean(s_i).

| Box | Activation | Arm | Task | Layer | Width s | Position m/s | Mean m | Dead fraction |
|---|---|---|---:|---:|---:|---:|---:|---:|
| F51 | ELU | ref | 1 | 1 | 5.744 | 0.054 | 0.418 | 0.000 |
| F51 | ELU | ref | 1 | 2 | 3.197 | 0.099 | 0.569 | 0.000 |
| F51 | ELU | ref | 3 | 1 | 6.718 | -0.011 | 0.175 | 0.000 |
| F51 | ELU | ref | 3 | 2 | 3.694 | -0.128 | -0.151 | 0.000 |
| F51 | ELU | ref | 11 | 1 | 9.061 | -0.068 | -0.209 | 0.000 |
| F51 | ELU | ref | 11 | 2 | 4.746 | -0.390 | -1.324 | 0.000 |
| F51 | ELU | ref | 21 | 1 | 10.444 | -0.044 | -0.235 | 0.000 |
| F51 | ELU | ref | 21 | 2 | 5.447 | -0.444 | -1.854 | 0.000 |
| F51 | ELU | ref | 29 | 1 | 10.251 | -0.062 | -0.345 | 0.000 |
| F51 | ELU | ref | 29 | 2 | 5.458 | -0.429 | -1.784 | 0.000 |
| F51 | ELU | cap12 | 1 | 1 | 5.744 | 0.054 | 0.418 | 0.000 |
| F51 | ELU | cap12 | 1 | 2 | 3.197 | 0.099 | 0.569 | 0.000 |
| F51 | ELU | cap12 | 3 | 1 | 5.784 | -0.008 | 0.146 | 0.000 |
| F51 | ELU | cap12 | 3 | 2 | 2.989 | -0.066 | 0.063 | 0.000 |
| F51 | ELU | cap12 | 11 | 1 | 6.749 | -0.048 | -0.027 | 0.000 |
| F51 | ELU | cap12 | 11 | 2 | 3.359 | -0.337 | -0.751 | 0.000 |
| F51 | ELU | cap12 | 21 | 1 | 7.001 | -0.027 | 0.005 | 0.000 |
| F51 | ELU | cap12 | 21 | 2 | 3.466 | -0.332 | -0.731 | 0.000 |
| F51 | ELU | cap12 | 29 | 1 | 6.783 | -0.026 | 0.057 | 0.000 |
| F51 | ELU | cap12 | 29 | 2 | 3.328 | -0.362 | -0.749 | 0.000 |
| F51 | ReLU | ref | 1 | 1 | 6.349 | 0.118 | 0.668 | 0.000 |
| F51 | ReLU | ref | 1 | 2 | 3.075 | 0.514 | 2.084 | 0.005 |
| F51 | ReLU | ref | 3 | 1 | 5.557 | -0.043 | 0.058 | 0.000 |
| F51 | ReLU | ref | 3 | 2 | 2.772 | 0.323 | 1.295 | 0.002 |
| F51 | ReLU | ref | 11 | 1 | 8.507 | -0.065 | 0.014 | 0.000 |
| F51 | ReLU | ref | 11 | 2 | 4.391 | -0.888 | -2.943 | 0.040 |
| F51 | ReLU | ref | 21 | 1 | 9.714 | -0.057 | 0.027 | 0.000 |
| F51 | ReLU | ref | 21 | 2 | 5.739 | -1.258 | -6.618 | 0.297 |
| F51 | ReLU | ref | 29 | 1 | 9.518 | -0.061 | 0.033 | 0.000 |
| F51 | ReLU | ref | 29 | 2 | 5.828 | -1.308 | -7.102 | 0.382 |
| RC | ELU | ref | 1 | 1 | 16.976 | -0.123 | -2.231 | 0.000 |
| RC | ELU | ref | 1 | 2 | 12.910 | -0.489 | -6.868 | 0.000 |
| RC | ELU | ref | 10 | 1 | 195.410 | -0.092 | -10.611 | 0.000 |
| RC | ELU | ref | 10 | 2 | 407.381 | -1.940 | -782.157 | 0.121 |
| RC | ELU | ref | 30 | 1 | 226.526 | -0.086 | -10.648 | 0.000 |
| RC | ELU | ref | 30 | 2 | 470.937 | -1.960 | -914.048 | 0.323 |
| RC | ELU | ref | 50 | 1 | 239.042 | -0.083 | -10.227 | 0.000 |
| RC | ELU | ref | 50 | 2 | 501.446 | -1.961 | -973.631 | 0.379 |

F51 ELU width does grow (roughly 1.8x/1.7x from hard task 1 to 29), but much less than RC ELU (14x/39x from task 1 to 50), and is nearly flat from F51 task 21 to 29. F51 ELU also stays much shallower in relative position. This is consistent with the absence of large loss in F51 ELU, but does not identify width alone as its cause. RC and F51 differ in task distribution, learning rate, update budget and endpoint. The evaluated image classes also change across F51 tasks.

F51 ReLU has comparable late width to F51 ELU, yet has a more negative second-layer relative position and substantial death. Width alone therefore does not describe the whole contrast.
