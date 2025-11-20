# Use Python 3.8 image as base
FROM python:3.8

# Set working directory
WORKDIR /project

# Copy requirements
COPY requirements.txt ./

# Install PyTorch and dependencies
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . ./

# Copy pretrained model weights
COPY sphereface_weights.pth /project/sphereface_weights.pth

# Create necessary directories
RUN mkdir -p /project/input /project/output /project/cfgs

# Run main script
CMD ["python", "main.py"]
