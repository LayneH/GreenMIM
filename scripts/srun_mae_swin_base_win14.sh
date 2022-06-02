#!/usr/bin/env bash

set -x

METHOD=green_mim_swin_base_patch4_win14_dec5121b1
DSET='in1k'
DATA_PATH="/mnt/cache/share/images/"
LR=1.5e-4
WD=0.05
BS=256
EP=800
MASK_RATIO=0.75
WARM_EP=40
PARTITION=$1
NTASKS=$2
PY_ARGS=${@:3}
JOB_NAME=${JOB_NAME:-"green_mim"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PYTHON=${PYTHON:-"python"}
CKPT_DIR=./ckpts/${METHOD}/${DSET}/ep${EP}_${WARM_EP}_lr${LR}_bs${BS}x${NTASKS}_wd${WD}_m${MASK_RATIO}_${JOB_NAME}

mkdir -p ./ckpts
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

GLOG_vmodule=MemcachedClient=-1 \
srun -p ${PARTITION} \
    --mpi=pmi2 --gres=gpu:${GPUS_PER_NODE} \
    -n${NTASKS} --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --job-name=${JOB_NAME} --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${PYTHON} -u main_pretrain.py \
        --output_dir ${CKPT_DIR} \
        --batch_size ${BS} \
        --model ${METHOD} \
        --norm_pix_loss \
        --mask_ratio ${MASK_RATIO} \
        --epochs ${EP} \
        --warmup_epochs ${WARM_EP} \
        --blr ${LR} \
        --weight_decay ${WD} \
        --data_path ${DATA_PATH} \
        --save_freq 20 \
        ${PY_ARGS} \
        2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log
