#!/bin/bash
# chain_czu.sh: CZU-MHAD queue workers. One per GPU now (small jobs fit beside the CMI workers); two more once all four
# CMI workers have finished.
T=/files1/haodong/test/new/table2_pareto; cd $T
./worker.sh logs/q_czu.txt 3 2500 >> logs/w_czu_3.a.out 2>&1 &
sleep 30
./worker.sh logs/q_czu.txt 2 2500 >> logs/w_czu_2.a.out 2>&1 &
until [ "$(cat logs/w_heavy_*.out 2>/dev/null | grep -c WORKER_DONE)" -ge 4 ]; do sleep 60; done
./worker.sh logs/q_czu.txt 2 2500 >> logs/w_czu_2.b.out 2>&1 &
sleep 30
./worker.sh logs/q_czu.txt 3 2500 >> logs/w_czu_3.b.out 2>&1 &
wait
echo "CZU_DONE $(date '+%m-%d %H:%M')"
