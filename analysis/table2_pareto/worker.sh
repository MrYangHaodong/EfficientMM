#!/bin/bash
# worker.sh <queue> <gpu> <min free MiB>: pops "ds gamma lams tag [try]" lines and runs run_one.sh on GPU 2 or 3.
# Waits for <min free MiB> on its GPU before each job; a job that died of CUDA OOM goes back to the queue (max 3 tries).
# Nets already fitted by a killed job are reloaded, not refitted (train_or_load, force_retrain=False).
Q=$1; GPU=$2; NEED=$3; T=/files1/haodong/test/new/table2_pareto; cd $T
case $GPU in 2|3) ;; *) echo "GPU must be 2 or 3"; exit 2;; esac
while true; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)" -lt "$NEED" ]; do sleep 30; done
  line=$(flock $Q.lock bash -c "head -n1 $Q; sed -i 1d $Q")
  [ -z "$line" ] && break
  set -- $line
  try=${5:-0}
  out=$(./run_one.sh $1 $2 $3 $GPU $4)
  echo "$out gpu$GPU try=$try"
  if ! echo "$out" | grep -q "rc=0"; then
    if grep -qi "out of memory" "logs/$4_$1_g$2.log" && [ "$try" -lt 3 ]; then
      sleep 45
      flock $Q.lock bash -c "echo '$1 $2 $3 $4 $((try + 1))' >> $Q"
      echo "REQUEUED $1 g$2 $3 (OOM)"
    else
      echo "FAILED $1 g$2 $3"
    fi
  fi
done
echo "WORKER_DONE $Q gpu$GPU $(date '+%m-%d %H:%M')"
