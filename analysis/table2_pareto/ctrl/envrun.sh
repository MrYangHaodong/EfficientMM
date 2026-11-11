#!/bin/bash
# envrun.sh <ds> <gpu: 2 | 3 | cpu> <script.py> [args...]: run a python script in the deployed per-dataset environment
# (deployed backbone, prefix cache, provenance key, folds; idle extraction table), as ../run_one.sh does for sweep.py.
# GPUs 2 and 3 only; 'cpu' hides every GPU. Extra environment variables pass through (e.g. FOLDS, RCOSTS, THREADS).
DS=$1; DEV=$2; shift 2
case $DEV in 2|3) VIS=$DEV;; cpu) VIS="";; *) echo "device must be 2, 3 or cpu"; exit 2;; esac
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
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
TH=${THREADS:-4}
exec env PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$VIS OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH \
  RLB_RUNSET=t2p RLB_CACHE_SUBDIR=$SUB RLB_CKPT_OVERRIDE=$BB RLB_PROV_KEY=$PK $FX \
  RLB_COST_FILE=/files1/haodong/test/new/energy_adaptive/results_per_modality_extract_cuda_idle.json \
  DS=$DS $PY "$@"
