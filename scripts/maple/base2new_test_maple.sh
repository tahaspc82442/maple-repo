#!/bin/bash

# cd ../..

# custom config
DATA=/raid/biplab/taha
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx # vit_b16_c2_ep5_batch4_2ctx_cross_datasets vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=3
SUB=new

# COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
# MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}  # changed dataset to PatternNet
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED} \
      --model-dir output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
      --load-epoch ${LOADEP} \
      --eval-only \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${SUB}
else
    echo "Evaluating model"
    echo "Running the first phase job and saving the output to ${DIR}"

    python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED} \
      --model-dir output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
      --load-epoch ${LOADEP} \
      --eval-only \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
