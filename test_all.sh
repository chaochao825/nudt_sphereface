#!/bin/bash
# test_all.sh - Final comprehensive test script for all face recognition tasks

set -e

PROJECT_DIR="/data6/user23215430/nudt_sphereface"
PROJECT_NAME="nudt_sphereface"
IMAGE_NAME="${PROJECT_NAME}:latest"
BUILD_DIR="${HOME}/${PROJECT_NAME}_test_$(date +%Y%m%d_%H%M%S)"

echo ">>> [1/5] Preparing build environment in ${BUILD_DIR}..."
mkdir -p "${BUILD_DIR}"
# Copy only the project contents, excluding any large unrelated folders if any
cp -r "${PROJECT_DIR}/." "${BUILD_DIR}/"

cd "${BUILD_DIR}"

# Prepare real test data for valid verification
echo ">>> [2/5] Preparing sample images from system datasets..."
mkdir -p test_data/person1
mkdir -p test_data/person2

# Try to find images in standard locations
LFW_DIR="/data6/user23215430/nudt/input/data/LFW/lfw"
if [ -d "$LFW_DIR" ]; then
    # Pick two different people for 1:1 verification
    people_dirs=($(ls -d "$LFW_DIR"/*/ | shuf -n 2))
    cp $(ls "${people_dirs[0]}"*.jpg | head -n 1) test_data/person1/
    cp $(ls "${people_dirs[1]}"*.jpg | head -n 1) test_data/person2/
    echo "Prepared person1 from ${people_dirs[0]} and person2 from ${people_dirs[1]}"
else
    # Fallback dummy data if LFW not found
    touch test_data/person1/dummy1.jpg
    touch test_data/person2/dummy2.jpg
fi

echo ">>> [3/5] Building Docker image: ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .

TEST_INPUT="${BUILD_DIR}/test_data"
TEST_OUTPUT="${BUILD_DIR}/test_results"
mkdir -p "${TEST_OUTPUT}"

function run_task() {
    local task_name=$1
    local process_type=$2
    local extra_env=$3
    echo ""
    echo "============================================================"
    echo " TASK: ${task_name}"
    echo "============================================================"
    docker run --rm \
      -v "${TEST_INPUT}:/project/input/data:ro" \
      -v "${TEST_OUTPUT}/${process_type}:/project/output:rw" \
      -e process="${process_type}" \
      -e device=-1 \
      -e data="lfw" \
      ${extra_env} \
      "${IMAGE_NAME}"
}

echo ">>> [4/5] Running automated tests..."

# 1. Dataset Sampling
run_task "数据集采样" "dataset_sampling" "-e sample_count=5"

# 2. Training (Simulated)
run_task "模型训练" "train" "-e epochs=1"

# 3. Face Verification (1:1)
run_task "人脸验证" "inference_1_1" ""

# 4. Face Identification (1:N)
run_task "人脸识别验证" "inference_1_n" ""

# 5. Attack (ADV) - BIM
run_task "对抗样本生成 (BIM)" "adv" "-e attack_method=bim -e selected_samples=2"

# 6. Attack (ADV) - PGD
run_task "对抗样本生成 (PGD)" "adv" "-e attack_method=pgd -e selected_samples=2"

# 7. Attack (ADV) - DeepFool
run_task "对抗样本生成 (DeepFool)" "adv" "-e attack_method=deepfool -e selected_samples=2"

# 8. Defense - HGD
run_task "防御处理 (HGD)" "defend" "-e defend_method=hgd -e selected_samples=2"

# 9. Defense - TVM
run_task "防御处理 (TVM)" "defend" "-e defend_method=tvm -e selected_samples=2"

# 10. Evaluation (Attack & Defense)
run_task "全流程安全性评估 (BIM+HGD)" "attack_defense_eval" "-e attack_method=bim -e defend_method=hgd -e selected_samples=2"

echo ""
echo ">>> [5/5] Tests completed successfully!"
echo ">>> Results summary in: ${TEST_OUTPUT}"
echo ">>> Reports generated:"
find "${TEST_OUTPUT}" -name "*report.json"

# Cleanup
echo ">>> Cleaning up build directory..."
# Use a slightly safer cleanup
rm -rf "${BUILD_DIR}/test_data"
# Keep the results for a moment if needed, but per requirements we should clean up if we want
# rm -rf "${BUILD_DIR}"
echo ">>> DONE."

