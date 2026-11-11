#!/bin/bash
# drive_jafa.sh: Joint-AFA r_cost sweep. Per dataset: raw previews on GPU 2 (one dataset at a time, after waiting for
# 8 GB free), then CPU trainings of that dataset's folds in the background (2 processes per fold, 2 threads each).
C=/files1/haodong/test/new/table2_pareto/ctrl; cd $C || exit 1; mkdir -p logs
RA=0,0.02,0.05,0.1,0.2,0.5; RB=0.01,0.03,0.075,0.15,0.3
for ds in cmi eav utd_mhad iemocap czu_mhad mmfi; do
  F=0,1,2; [ $ds = utd_mhad ] && F=3,4,5
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)" -lt 8000 ]; do sleep 30; done
  MODE=previews FOLDS=$F ./envrun.sh $ds 2 jafa_sweep.py > logs/previews_$ds.log 2>&1
  echo "PREVIEWS $ds rc=$? $(date '+%m-%d %H:%M')"
  for f in ${F//,/ }; do
    [ -f previews/${ds}_f$f.npz ] || { echo "MISSING previews $ds f$f"; continue; }
    for g in A B; do
      R=$RA; [ $g = B ] && R=$RB
      (MODE=train DEV=cpu THREADS=2 FOLDS=$f RCOSTS=$R ./envrun.sh $ds cpu jafa_sweep.py > logs/train_${ds}_f${f}_$g.log 2>&1
       echo "TRAINED $ds f$f $g rc=$? $(date '+%m-%d %H:%M')") &
    done
  done
done
wait
echo "ALL_JAFA_DONE $(date '+%m-%d %H:%M')"
