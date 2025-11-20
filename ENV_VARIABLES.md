# Environment Variables Documentation

## SPHEREFACE Face Recognition Model

This document describes all environment variables supported by the SPHEREFACE face recognition Docker container.

## Required Environment Variables

### INPUT_PATH
- **Description**: Path to the input directory containing model weights and data
- **Type**: String (path)
- **Default**: `./input`
- **Example**: `/project/input`

### OUTPUT_PATH
- **Description**: Path to the output directory where results will be saved
- **Type**: String (path)
- **Default**: `./output`
- **Example**: `/project/output`

## Process Configuration

### PROCESS
- **Description**: Type of process to run
- **Type**: String
- **Default**: `attack`
- **Options**:
  - `adv`: Generate adversarial examples
  - `attack`: Run attack evaluation
  - `defend`: Apply defense methods
  - `train`: Train the model (not fully implemented)
- **Example**: `PROCESS=adv`

### MODEL
- **Description**: Face recognition model to use
- **Type**: String
- **Default**: `sphereface`
- **Example**: `MODEL=sphereface`

### DATA
- **Description**: Dataset to use
- **Type**: String
- **Default**: `lfw`
- **Options**:
  - `vggface2`: VGGFace2 dataset
  - `celeba`: CelebA dataset
  - `webface`: CASIA-WebFace dataset
  - `lfw`: Labeled Faces in the Wild dataset
  - `yaleb`: Yale Face Database B
  - `megaface`: MegaFace dataset
- **Example**: `DATA=lfw`

### NUM_CLASSES
- **Description**: Number of identity classes for classification
- **Type**: Integer
- **Default**: `1000`
- **Example**: `NUM_CLASSES=1000`

## Attack Configuration

### ATTACK_METHOD
- **Description**: Attack method to use for adversarial example generation
- **Type**: String
- **Default**: `bim`
- **Options**:
  - `bim`: Basic Iterative Method
  - `dim`: Diverse Input Method
  - `tim`: Translation-Invariant Method
  - `pgd`: Projected Gradient Descent
  - `cw`: Carlini & Wagner Attack
  - `deepfool`: DeepFool Attack
- **Example**: `ATTACK_METHOD=pgd`

### EPSILON
- **Description**: Maximum perturbation magnitude for adversarial attacks
- **Type**: Float
- **Default**: `0.031` (8/255)
- **Range**: `0.0 - 1.0`
- **Example**: `EPSILON=0.031`

### STEP_SIZE
- **Description**: Step size for iterative attacks (also known as alpha)
- **Type**: Float
- **Default**: `0.008` (2/255)
- **Range**: `0.0 - 1.0`
- **Example**: `STEP_SIZE=0.008`

### MAX_ITERATIONS
- **Description**: Maximum number of iterations for attack methods
- **Type**: Integer
- **Default**: `10`
- **Range**: `1 - 1000`
- **Example**: `MAX_ITERATIONS=50`

## Defense Configuration

### DEFEND_METHOD
- **Description**: Defense method to apply
- **Type**: String
- **Default**: `hgd`
- **Options**:
  - `hgd`: High-level Guided Denoising
  - `tvm`: Total Variation Minimization
  - `livenessdetection`: Liveness Detection for face spoofing
  - `featurespacepurification`: Feature Space Purification
  - `ensembledefense`: Ensemble Defense (combines multiple defenses)
- **Example**: `DEFEND_METHOD=hgd`

## Training Configuration

### EPOCHS
- **Description**: Number of training epochs
- **Type**: Integer
- **Default**: `100`
- **Example**: `EPOCHS=100`

### BATCH
- **Description**: Batch size for training/inference
- **Type**: Integer
- **Default**: `8`
- **Example**: `BATCH=16`

### WORKERS
- **Description**: Number of data loading workers
- **Type**: Integer
- **Default**: `0`
- **Example**: `WORKERS=4`

## Hardware Configuration

### DEVICE
- **Description**: CUDA device ID to use (or -1 for CPU)
- **Type**: Integer
- **Default**: `0`
- **Range**: `-1` (CPU) or `0+` (GPU ID)
- **Example**: `DEVICE=0`

## Configuration File Paths

### CFG_PATH
- **Description**: Path to configuration files directory
- **Type**: String (path)
- **Default**: `./cfgs`
- **Example**: `CFG_PATH=/project/cfgs`

## Usage Examples

### Example 1: Run BIM Attack
```bash
docker run --rm --gpus all \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e ATTACK_METHOD=bim \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  nudt_sphereface:latest
```

### Example 2: Run HGD Defense
```bash
docker run --rm \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=sphereface \
  -e DEFEND_METHOD=hgd \
  nudt_sphereface:latest
```

### Example 3: Run PGD Attack with Custom Parameters
```bash
docker run --rm --gpus all \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=sphereface \
  -e DATA=celeba \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.062 \
  -e STEP_SIZE=0.016 \
  -e MAX_ITERATIONS=20 \
  -e DEVICE=0 \
  nudt_sphereface:latest
```

## Input Directory Structure

The INPUT_PATH should contain the following structure:

```
input/
├── model/
│   └── model_weights.pth    # Pretrained model weights
└── data/
    └── dataset_name/         # Dataset folder
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Output Directory Structure

The OUTPUT_PATH will contain generated results:

```
output/
├── adv_image1.jpg           # Adversarial examples (attack mode)
├── adv_image2.jpg
├── defended_image1.jpg      # Defended images (defense mode)
└── defended_image2.jpg
```

## SSE Output Format

The container outputs progress and results in Server-Sent Events (SSE) format:

### Event Types

1. **input_path_validated**: Input path validation status
2. **input_data_validated**: Input data validation status
3. **input_model_validated**: Input model validation status
4. **output_path_validated**: Output path validation status
5. **adv_samples_gen_validated**: Adversarial sample generation completion
6. **clean_samples_gen_validated**: Clean/defended sample generation completion
7. **training_progress**: Training progress updates
8. **error**: Error messages

### Example SSE Output

```
event: input_path_validated
data: {"status": "success", "message": "Input path is valid and complete.", "file_name": "/project/input"}

event: adv_samples_gen_validated
data: {"status": "success", "message": "adv sample is generated.", "file_name": "/project/output/adv_image1.jpg"}
```

## Notes

- All paths inside the container are relative to `/project`
- GPU support requires `--gpus all` flag and NVIDIA Docker runtime
- Defense methods can run on CPU (DEVICE=-1)
- For best results, use pretrained model weights appropriate for your dataset
- Batch size is automatically set to 1 for attack/defense modes

## Supported Datasets

| Dataset | Description | Identities | Images |
|---------|-------------|------------|--------|
| VGGFace2 | Large-scale face recognition dataset | ~9,000 | ~3.3M |
| CelebA | Celebrity faces with attributes | 10,177 | 202,599 |
| CASIA-WebFace | Web-collected face dataset | 10,575 | 494,414 |
| LFW | Labeled Faces in the Wild | 5,749 | 13,233 |
| YaleB | Yale Face Database B | 38 | 2,414 |
| MegaFace | Large-scale face recognition benchmark | ~670K | ~1M |

## Model-Specific Information

### SPHEREFACE
- **Input Size**: 112x112 pixels
- **Embedding Size**: 512 dimensions
- **Backbone**: ResNet-50
- **Use Case**: Face verification and identification

## Troubleshooting

### Common Issues

1. **No images found in data path**
   - Ensure data directory exists in INPUT_PATH
   - Check that images have .jpg or .png extensions
   - Verify volume mount paths

2. **CUDA out of memory**
   - Reduce BATCH size
   - Use smaller MAX_ITERATIONS
   - Switch to CPU with DEVICE=-1

3. **Model weights not loading**
   - Ensure model weights file exists in INPUT_PATH/model/
   - Check that weights are compatible with model architecture
   - Verify file permissions

## Version Information

- Docker Base Image: python:3.8
- PyTorch Version: 2.4.0
- TorchVision Version: 0.19.0
- CUDA Support: 12.1

## Support

For issues and questions, please refer to the project documentation or contact the development team.
