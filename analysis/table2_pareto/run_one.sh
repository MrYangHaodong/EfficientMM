#!/bin/bash
# run_one.sh <ds> <gamma> <lams> <gpu> <tag>: one sweep.py process (one dataset, one gamma) on one GPU (2 or 3 only).
# Per-dataset environment = the deployed one (rebuttal_abl/post_unit.sh, deployed_provenance.json), run set t2p.
DS=$1; G=$2; LAMS=$3; GPU=$4; TAG=$5
case $GPU in 2|3) ;; *) echo "GPU must be 2 or 3"; exit 2;; esac
T=/files1/haodong/test/new/table2_pareto; PY=/home/egg8711/miniconda3/envs/maestro/bin/python
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
if [ "$G" = 0 ]; then CE="RLB_COST_MODE=uniform"
else CE="RLB_COST_MODE=normpow RLB_ABS_GAMMA=$G RLB_COST_FILE=/files1/haodong/test/new/energy_adaptive/results_per_modality_extract_cuda_idle.json"; fi
env PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  RLB_RUNSET=t2p RLB_CACHE_SUBDIR=$SUB RLB_CKPT_OVERRIDE=$BB RLB_PROV_KEY=$PK $FX $CE \
  DS=$DS PROV_KEY=$PK LAMS=$LAMS OUT=$T/res/${TAG}_${DS}_g${G}.csv SN_DIR=$T/res/sn/$TAG \
  $PY $T/sweep.py > $T/logs/${TAG}_${DS}_g${G}.log 2>&1
rc=$?
echo "DONE $TAG $DS g$G rc=$rc $(date '+%m-%d %H:%M')"
