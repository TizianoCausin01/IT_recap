#!/bin/bash

set -e

export MY_ENV="${MY_ENV:-tiziano_mac_mini}"
PROJECT_ROOT="/Users/tizianocausin/Desktop/IT_recap"
DOCKER_CONFIG_DIR="${DOCKER_CONFIG:-/private/tmp/it-recap-docker-config}"

cd "${PROJECT_ROOT}"

bash bash_scripts/download_convrnn_checkpoint.sh rgc_intermediate

env DOCKER_CONFIG="${DOCKER_CONFIG_DIR}" docker run --rm --platform linux/amd64 \
    -v /Users/tizianocausin/Desktop/IT_recap:/work \
    -v /Users/tizianocausin/livingstone_lab_local:/Users/tizianocausin/livingstone_lab_local:ro \
    -v /Users/tizianocausin/IT_recap_local:/Users/tizianocausin/IT_recap_local \
    -w /work \
    it-recap-convrnn-tf1 \
    python python_scripts/scripts/extract_convrnn_features.py \
        --model_name rgc_intermediate \
        --layers conv9,conv10 \
        --folder_name talia_20each_tizi \
        --pooling mean \
        --batch_size 16 \
        --image_pres neural

/Users/tizianocausin/Desktop/metrics_II/.venv/bin/python \
    python_scripts/scripts/run_static_dRSA_convrnn_it.py \
        --target_model_name rgc_intermediate \
        --target_layer conv10 \
        --signal_RDM_metric correlation \
        --model_RDM_metric correlation \
        --RSA_metric spearman

# EOF
