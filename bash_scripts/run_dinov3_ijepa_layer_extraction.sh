#!/bin/bash

set -e

export MY_ENV="${MY_ENV:-tiziano_mac_mini}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIEXEC="${MPIEXEC:-mpiexec}"
N_PROCESSES="${N_PROCESSES:-3}"
SCRIPT="python_scripts/scripts/run_hf_vit_layer_extraction.py"

for MODEL_NAME in dino_v3_l ijepa_vith14_1k; do
    "${PYTHON_BIN}" "${SCRIPT}" \
        --model_name "${MODEL_NAME}" \
        --prepare_only

    "${MPIEXEC}" -np "${N_PROCESSES}" "${PYTHON_BIN}" "${SCRIPT}" \
        --model_name "${MODEL_NAME}" \
        --folder_name talia_20each_tizi \
        --img_size 224 \
        --batch_size 8 \
        --pooling mean
done

# EOF
