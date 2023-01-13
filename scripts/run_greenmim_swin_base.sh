#!/usr/bin/env bash

set -x

METHOD=green_mim_swin_base_patch4_dec512b1
DSET='in1k'
DATA_PATH="/srv/datasets/pytorch/ImageNet"
LR=1.5e-4
WD=0.05
BS=586
EP=200
MASK_RATIO=0.75
WARM_EP=40
PY_ARGS=${@:1}
PORT=${PORT:-5389}
NPROC=${NPROC:-8}
PYTHON=${PYTHON:-"python"}
CKPT_DIR=./ckpts/${METHOD}/${DSET}/ep${EP}_${WARM_EP}_lr${LR}_bs${BS}x${NPROC}_wd${WD}_m${MASK_RATIO}_${JOB_NAME}

mkdir -p ./ckpts
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" \
${PYTHON} -u -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NPROC} \
    main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --model ${METHOD} \
    --norm_pix_loss \
    --mask_ratio ${MASK_RATIO} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --data_path ${DATA_PATH} \
    --save_freq 25 \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log
