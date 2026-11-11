#!/bin/bash
# run_jafa.sh <ds> <gpu> <folds e.g. 0,1,2> <rcosts e.g. 0,0.01> <tag>: one jafa_sweep.py process on GPU 2 or 3 only.
# Per-dataset environment = the deployed backbone and prefix cache (same as ../run_one.sh); idle extraction table.
DS=$1; GPU=$2; FOLDS=$3; RC=$4; TAG=$5
case $GPU in 2|3) ;; *) echo "GPU must be 2 or 3"; exit 2;; esac
C=/files1/haodong/test/new/table2_pareto/ctrl; PY=/home/egg8711/miniconda3/envs/maestro/bin/python
D=/files1/haodong/Efficient-Multimodal-Bottleneck/runs/dropout_robustness_seqA_dropaug; EA=/files1/haodong/test/new/energy_adaptive/checkpoints
PK=$DS; FX=""
case $DS in
  iemocap)  BB=$D/2026-07-19_iemocap_pv3_branch_16_seqAbranch; SUB=botneck_16;;
  cmi)      BB=$D/2026-07-20_cmi_pv3_branch_16_seqAbranch; SUB=botneck_16;;
  czu_mhad) BB=$D/2026-07-19_czu_mhad_pv3_branch_16_seqAbranch; SUB=botneck_16;;
  eav)      BB=$D/2026-08-01_eav_pv3_branch_16_parity_seqAbranch; SUB=botneck_16_parity;;
  mmfi)     BB=$EA/seqA_mmfi_rebalanced/2026-07-24_mmfi_pv3_branch_16_seqAbranch; SUB=botneck_16_mmfi_rebal; PK=mmfi@rebal;;
  utd_mhad) BB=$EA/seqA_utd_mhad_rebal422; SUB=botneck_16_utd_mhad_rebal422; PK=utd_mhad@rebal422; FX="RLB_FOLDS=3,4,5";;
  *) echo "bad ds $DS"; exit 2;;
esac
mkdir -p $C/logs
env PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  RLB_RUNSET=t2p RLB_CACHE_SUBDIR=$SUB RLB_CKPT_OVERRIDE=$BB RLB_PROV_KEY=$PK $FX \
  RLB_COST_FILE=/files1/haodong/test/new/energy_adaptive/results_per_modality_extract_cuda_idle.json \
  DS=$DS FOLDS=$FOLDS RCOSTS=$RC \
  $PY $C/jafa_sweep.py > $C/logs/${TAG}_${DS}_f${FOLDS//,/}.log 2>&1
rc=$?
echo "DONE $TAG $DS f$FOLDS rc=$rc $(date '+%m-%d %H:%M')"
