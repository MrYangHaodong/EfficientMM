#!/bin/bash
# chain_czu2.sh: start the CZU-MHAD queue (two workers per GPU) only after all four CMI workers have finished; a third
# process beside two CMI jobs ran out of GPU memory on 2026-11-09 11:23.
T=/files1/haodong/test/new/table2_pareto; cd $T
until [ "$(cat logs/w_heavy_*.out 2>/dev/null | grep -c WORKER_DONE)" -ge 4 ]; do sleep 60; done
for g in 2 3; do
  ./worker.sh logs/q_czu.txt $g 4000 >> logs/w_czu_$g.a.out 2>&1 &
  sleep 30
  ./worker.sh logs/q_czu.txt $g 4000 >> logs/w_czu_$g.b.out 2>&1 &
  sleep 30
done
wait
echo "CZU_DONE $(date '+%m-%d %H:%M')"
