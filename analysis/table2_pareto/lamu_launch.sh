#!/bin/bash
# lamu_launch.sh: SeMARC at uniform cost (gamma = 0) on extra lambda values that fill the gaps of the existing grid
# (.05, .1, .2, .5, 1, 2), for the lambda curve of fig/pareto_lambda_controllers_mean (lam_curves.py).
# Same sweep.py / run_one.sh / worker.sh as the gamma x lambda sweep, GPUs 2 and 3 only. Outputs res/lamu_l<lam>_<ds>_g0.csv,
# outside the sweep_* glob of plot_pareto.sweep_configs(), so the earlier figures and tables do not change.
# Heavy queue: CMI (~8.4 GB, ~220 s per fit) and IEMOCAP; small queue: CZU-MHAD (~2 GB, ~70 s), EAV, MM-Fi, UTD-MHAD.
# Lambda order = priority: the 424-550 ms gap first (.7, .8, .6, .9), then 1.5, .4, 1.25, .3.
T=/files1/haodong/test/new/table2_pareto; cd $T || exit 1
LAMS="0.7 0.8 0.6 0.9 1.5 0.4 1.25 0.3"
: > logs/q_lamu_heavy.txt; : > logs/q_lamu_small.txt
for l in $LAMS; do
  for ds in cmi iemocap; do echo "$ds 0 $l lamu_l$l" >> logs/q_lamu_heavy.txt; done
  for ds in czu_mhad eav mmfi utd_mhad; do echo "$ds 0 $l lamu_l$l" >> logs/q_lamu_small.txt; done
done
./worker.sh logs/q_lamu_heavy.txt 3 10000 >> logs/w_lamu_heavy_3.a.out 2>&1 &
sleep 45
./worker.sh logs/q_lamu_heavy.txt 2 10000 >> logs/w_lamu_heavy_2.a.out 2>&1 &
sleep 45
./worker.sh logs/q_lamu_small.txt 3 3000 >> logs/w_lamu_small_3.out 2>&1 &
sleep 20
./worker.sh logs/q_lamu_small.txt 2 3000 >> logs/w_lamu_small_2.out 2>&1 &
sleep 90
./worker.sh logs/q_lamu_heavy.txt 3 12000 >> logs/w_lamu_heavy_3.b.out 2>&1 &
wait
echo "LAMU_DONE $(date '+%m-%d %H:%M')"
