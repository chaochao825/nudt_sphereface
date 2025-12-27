#!/bin/bash

# Docker run scripts for face recognition testing
# These scripts demonstrate how to run the container with different configurations

PROJECT_NAME=$(basename $(pwd) | sed 's/nudt_//')
IMAGE_NAME="nudt_${PROJECT_NAME}:latest"
INPUT_PATH="/path/to/data" # Should contain subfolders like person1, person2 or dataset folders
OUTPUT_PATH="/path/to/output"
DATA_NAME="lfw" # Options: lfw, webface, yaleb, celeba, megaface, vggface2
DEVICE="-1" # Use -1 for CPU, 0 for GPU 0

echo "========================================"
echo "${PROJECT_NAME^^} Docker Run Scripts"
echo "========================================"

# 1. Dataset Sampling
echo ""
echo "1. Dataset Sampling"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=dataset_sampling \
  -e DATA=\${DATA_NAME} \
  -e SAMPLE_COUNT=100 \
  \${IMAGE_NAME}
EOFSCRIPT

# 2. Model Training
echo ""
echo "2. Model Training"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=train \
  -e MODEL=\${PROJECT_NAME} \
  -e DATA=\${DATA_NAME} \
  -e EPOCHS=30 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 3. 人脸验证 (1:1)
# Note: INPUT_PATH should contain person1/ and person2/ subfolders with images
echo ""
echo "3. 人脸验证 (1:1)"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=inference_1_1 \
  -e MODEL=\${PROJECT_NAME} \
  -e DATA=\${DATA_NAME} \
  -e THRESHOLD=0.55 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 4. 人脸识别验证 (1:N)
echo ""
echo "4. 人脸识别验证 (1:N)"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=inference_1_n \
  -e MODEL=\${PROJECT_NAME} \
  -e DATA=\${DATA_NAME} \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 5. Attack and Defense Evaluation (BIM + HGD)
echo ""
echo "5. Attack and Defense Evaluation"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=attack_defense_eval \
  -e MODEL=\${PROJECT_NAME} \
  -e DATA=\${DATA_NAME} \
  -e ATTACK_METHOD=bim \
  -e DEFEND_METHOD=hgd \
  -e EPSILON=0.031 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 6. BIM Attack (ADV)
echo ""
echo "6. BIM Attack (ADV)"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=\${PROJECT_NAME} \
  -e ATTACK_METHOD=bim \
  -e EPSILON=0.031 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 7. DIM Attack
echo ""
echo "7. DIM Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=\${PROJECT_NAME} \
  -e ATTACK_METHOD=dim \
  -e EPSILON=0.031 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 8. TIM Attack
echo ""
echo "8. TIM Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=\${PROJECT_NAME} \
  -e ATTACK_METHOD=tim \
  -e EPSILON=0.031 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 9. PGD Attack
echo ""
echo "9. PGD Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=\${PROJECT_NAME} \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.031 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 10. C&W Attack
echo ""
echo "10. C&W Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=\${PROJECT_NAME} \
  -e ATTACK_METHOD=cw \
  -e MAX_ITERATIONS=100 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 11. DeepFool Attack
echo ""
echo "11. DeepFool Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=\${PROJECT_NAME} \
  -e ATTACK_METHOD=deepfool \
  -e MAX_ITERATIONS=50 \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 12. HGD Defense
echo ""
echo "12. HGD Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=\${PROJECT_NAME} \
  -e DEFEND_METHOD=hgd \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 13. TVM Defense
echo ""
echo "13. TVM Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=\${PROJECT_NAME} \
  -e DEFEND_METHOD=tvm \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 14. Liveness Detection Defense
echo ""
echo "14. Liveness Detection Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=\${PROJECT_NAME} \
  -e DEFEND_METHOD=livenessdetection \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 15. Feature Space Purification Defense
echo ""
echo "15. Feature Space Purification Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=\${PROJECT_NAME} \
  -e DEFEND_METHOD=featurespacepurification \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 16. Ensemble Defense
echo ""
echo "16. Ensemble Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input/data:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=\${PROJECT_NAME} \
  -e DEFEND_METHOD=ensembledefense \
  -e DEVICE=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

echo ""
echo "========================================"
echo "Note: Replace INPUT_PATH and OUTPUT_PATH with actual paths"
echo "========================================"
