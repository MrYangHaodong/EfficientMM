#!/bin/bash
# run_sweep.sh: 2 heavy workers (CMI, CZU-MHAD) + 1 light worker (MM-Fi, UTD-MHAD, EAV) on each of GPUs 2 and 3.
T=/files1/haodong/test/new/table2_pareto; cd $T
for g in 2 3; do
  ./worker.sh logs/q_heavy.txt $g 10000 >> logs/w_heavy_$g.a.out 2>&1 &
  sleep 20
  ./worker.sh logs/q_heavy.txt $g 10000 >> logs/w_heavy_$g.b.out 2>&1 &
  sleep 20
  ./worker.sh logs/q_light.txt $g 3000 >> logs/w_light_$g.out 2>&1 &
done
wait
echo "SWEEP_DONE $(date '+%m-%d %H:%M')"
