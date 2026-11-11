#!/bin/bash
# chain_czu3.sh: start two CZU-MHAD workers on a GPU as soon as that GPU's two CMI workers have finished (GPU 2 first).
T=/files1/haodong/test/new/table2_pareto; cd $T
for g in 2 3; do
  until [ "$(cat logs/w_heavy_$g.*.out 2>/dev/null | grep -c WORKER_DONE)" -ge 2 ]; do sleep 30; done
  ./worker.sh logs/q_czu.txt $g 4000 >> logs/w_czu_$g.a.out 2>&1 &
  sleep 30
  ./worker.sh logs/q_czu.txt $g 4000 >> logs/w_czu_$g.b.out 2>&1 &
done
wait
echo "CZU_DONE $(date '+%m-%d %H:%M')"
