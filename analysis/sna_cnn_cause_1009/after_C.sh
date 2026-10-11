#!/bin/bash
# When bundle C has finished: resume bundle A (paused with SIGSTOP to let C run alone) and start
# bundle D (spec §9.3) beside it.
cd "$(dirname "$0")/../.." || exit 1
R=results/sna_cnn_cause_1009
until [ -f "$R/C/provenance.json" ]; do sleep 30; done
kill -CONT 1556836
analysis/sna_cnn_cause_1009/launch.sh D "S:a3-c0.6-c3-c3,S:a3-c0.6-c0.6-c0.6+fcamp0.7" 10-19 30
