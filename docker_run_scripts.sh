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
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=dataset_sampling \
  -e data=\${DATA_NAME} \
  -e sample_count=100 \
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
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=train \
  -e model=\${PROJECT_NAME} \
  -e data=\${DATA_NAME} \
  -e epochs=30 \
  -e device=\${DEVICE} \
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
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=inference_1_1 \
  -e model=\${PROJECT_NAME} \
  -e data=\${DATA_NAME} \
  -e threshold=0.55 \
  -e device=\${DEVICE} \
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
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=inference_1_n \
  -e model=\${PROJECT_NAME} \
  -e data=\${DATA_NAME} \
  -e device=\${DEVICE} \
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
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=attack_defense_eval \
  -e model=\${PROJECT_NAME} \
  -e data=\${DATA_NAME} \
  -e attack_method=bim \
  -e defend_method=hgd \
  -e epsilon=0.031 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 6. BIM Attack (ADV)
echo ""
echo "6. BIM Attack (ADV)"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=adv \
  -e model=\${PROJECT_NAME} \
  -e attack_method=bim \
  -e epsilon=0.031 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 7. DIM Attack
echo ""
echo "7. DIM Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=adv \
  -e model=\${PROJECT_NAME} \
  -e attack_method=dim \
  -e epsilon=0.031 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 8. TIM Attack
echo ""
echo "8. TIM Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=adv \
  -e model=\${PROJECT_NAME} \
  -e attack_method=tim \
  -e epsilon=0.031 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 9. PGD Attack
echo ""
echo "9. PGD Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=adv \
  -e model=\${PROJECT_NAME} \
  -e attack_method=pgd \
  -e epsilon=0.031 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 10. C&W Attack
echo ""
echo "10. C&W Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=adv \
  -e model=\${PROJECT_NAME} \
  -e attack_method=cw \
  -e max_iterations=100 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 11. DeepFool Attack
echo ""
echo "11. DeepFool Attack"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm --gpus all \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=adv \
  -e model=\${PROJECT_NAME} \
  -e attack_method=deepfool \
  -e max_iterations=50 \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 12. HGD Defense
echo ""
echo "12. HGD Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=defend \
  -e model=\${PROJECT_NAME} \
  -e defend_method=hgd \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 13. TVM Defense
echo ""
echo "13. TVM Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=defend \
  -e model=\${PROJECT_NAME} \
  -e defend_method=tvm \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 14. Liveness Detection Defense
echo ""
echo "14. Liveness Detection Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=defend \
  -e model=\${PROJECT_NAME} \
  -e defend_method=livenessdetection \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 15. Feature Space Purification Defense
echo ""
echo "15. Feature Space Purification Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=defend \
  -e model=\${PROJECT_NAME} \
  -e defend_method=featurespacepurification \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

# 16. Ensemble Defense
echo ""
echo "16. Ensemble Defense"
echo "----------------------------------------"
cat << EOFSCRIPT
docker run --rm \
  -v \${INPUT_PATH}:/project/input:ro \
  -v \${OUTPUT_PATH}:/project/output:rw \
  -e INPUT_PATH=/project/input \
  -e OUTPUT_PATH=/project/output \
  -e process=defend \
  -e model=\${PROJECT_NAME} \
  -e defend_method=ensembledefense \
  -e device=\${DEVICE} \
  \${IMAGE_NAME}
EOFSCRIPT

echo ""
echo "========================================"
echo "Note: Replace INPUT_PATH and OUTPUT_PATH with actual paths"
echo "========================================"
