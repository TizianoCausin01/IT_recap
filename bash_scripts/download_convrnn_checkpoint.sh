#!/bin/bash

set -e

MODEL_NAME="${1:-rgc_intermediate}"
BASE_URL="https://convrnnckpts.s3-us-west-1.amazonaws.com"
CKPT_DIR="third_party/convrnns/ckpts/${MODEL_NAME}"

mkdir -p "${CKPT_DIR}"
curl -fLo "${CKPT_DIR}/model.ckpt.data-00000-of-00001" "${BASE_URL}/${MODEL_NAME}/model.ckpt.data-00000-of-00001"
curl -fLo "${CKPT_DIR}/model.ckpt.index" "${BASE_URL}/${MODEL_NAME}/model.ckpt.index"
curl -fLo "${CKPT_DIR}/model.ckpt.meta" "${BASE_URL}/${MODEL_NAME}/model.ckpt.meta"

echo "Downloaded ${MODEL_NAME} checkpoint to ${CKPT_DIR}"
