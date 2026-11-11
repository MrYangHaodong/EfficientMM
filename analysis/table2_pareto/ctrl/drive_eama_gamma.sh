#!/bin/bash
# drive_eama_gamma.sh: SeMA + EAMA at Table 2's lambda with SeMARC-style gamma costs (ctrl/eama_gamma.py), one process per
# dataset on GPUs 2/3. IEMOCAP fold 0 was run first, by hand, as the test (FOLDS=0 ctrl/envrun.sh iemocap 2 ctrl/eama_gamma.py).
T=/files1/haodong/test/new/table2_pareto; cd $T || exit 1; mkdir -p ctrl/logs
run() { FOLDS=$3 ctrl/envrun.sh $1 $2 ctrl/eama_gamma.py > ctrl/logs/eama_gamma_$1.log 2>&1; echo "DONE $1 rc=$? $(date '+%m-%d %H:%M')"; }
run iemocap 2 1,2 &
run cmi 3 "" &
run czu_mhad 2 "" &
run mmfi 3 "" &
run eav 2 "" &
run utd_mhad 3 "" &
wait
echo "ALL_EAMA_GAMMA_DONE $(date '+%m-%d %H:%M')"
