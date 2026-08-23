# determinism check

- result: **PASS**
- method: same NPZ inputsから、セル割当とRNGをリセットして解析全体を2回構築
- outer replay: 同一commit・同一NPZでCLIを独立に再実行し、下記7 CSVのSHA-256が全て再一致
- bootstrap: B=10000, seed=20260831

| CSV | SHA-256 |
|---|---|
| exposures.csv | `2edc9aa82185843d8fd7f9663380b60590cd75027b27601f19546b39ef7b126b` |
| primary_cells.csv | `70757c62625aa5ab3174e48115826deec553e9db00a539eacd3832a7cccf74fd` |
| primary_rates.csv | `356671d3122c2d1885e912ff65ab3f8233ba706387aa6b73723456b26765695a` |
| verdict.csv | `588f216ba72ad8126317ce4f696678d8391b0db2132cb444b07ca59f60cafea1` |
| secondary_results.csv | `dd67a8eee750752198ceb98f3e66c298635841e418bc2ae2e7692e5cf21369ad` |
| repeat_exposure.csv | `950a4c834f417e3e10ce85923df2b1bd857872b3e815cd5ca81403b8fd838632` |
| sanity.csv | `d09598369d700004b725df2fac5d81bdae67af8790fde0efc43eef63bd27bccb` |
