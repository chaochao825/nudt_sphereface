#!/bin/bash

# Docker run scripts for SPHEREFACE face recognition testing
# These scripts demonstrate how to run the container with different configurations

IMAGE_NAME="nudt_sphereface:latest"
INPUT_PATH="/path/to/input"
OUTPUT_PATH="/path/to/output"

echo "========================================"
echo "SPHEREFACE Docker Run Scripts"
echo "========================================"

# 1. Adversarial Sample Generation with BIM
echo ""
echo "1. BIM Attack (ADV)"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e DATA=lfw \
  -e NUM_CLASSES=1000 \
  -e ATTACK_METHOD=bim \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOFSCRIPT

# 2. DIM Attack
echo ""
echo "2. DIM Attack"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e ATTACK_METHOD=dim \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOFSCRIPT

# 3. TIM Attack
echo ""
echo "3. TIM Attack"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e ATTACK_METHOD=tim \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOFSCRIPT

# 4. PGD Attack
echo ""
echo "4. PGD Attack"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOFSCRIPT

# 5. C&W Attack
echo ""
echo "5. C&W Attack"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e ATTACK_METHOD=cw \
  -e MAX_ITERATIONS=100 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOFSCRIPT

# 6. DeepFool Attack
echo ""
echo "6. DeepFool Attack"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e ATTACK_METHOD=deepfool \
  -e MAX_ITERATIONS=50 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOFSCRIPT

# 7. HGD Defense
echo ""
echo "7. HGD Defense"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=sphereface \
  -e DEFEND_METHOD=hgd \
  ${IMAGE_NAME}
EOFSCRIPT

# 8. TVM Defense
echo ""
echo "8. TVM Defense"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=sphereface \
  -e DEFEND_METHOD=tvm \
  ${IMAGE_NAME}
EOFSCRIPT

# 9. Liveness Detection Defense
echo ""
echo "9. Liveness Detection Defense"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=sphereface \
  -e DEFEND_METHOD=livenessdetection \
  ${IMAGE_NAME}
EOFSCRIPT

# 10. Feature Space Purification Defense
echo ""
echo "10. Feature Space Purification Defense"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=sphereface \
  -e DEFEND_METHOD=featurespacepurification \
  ${IMAGE_NAME}
EOFSCRIPT

# 11. Ensemble Defense
echo ""
echo "11. Ensemble Defense"
echo "----------------------------------------"
cat << 'EOFSCRIPT'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=sphereface \
  -e DEFEND_METHOD=ensembledefense \
  ${IMAGE_NAME}
EOFSCRIPT

echo ""
echo "========================================"
echo "Note: Replace INPUT_PATH and OUTPUT_PATH with actual paths"
echo "========================================"
