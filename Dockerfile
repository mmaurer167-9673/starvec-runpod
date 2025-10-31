FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git vim build-essential \
    libcairo2 cuda-compiler-12-4 libaio-dev \
    pkg-config libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
WORKDIR /app
COPY app.py .

# Install dependencies
RUN pip install --upgrade pip

# Install flash_attn with specific pre-built wheel for CUDA 12.4 + PyTorch 2.5
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install other dependencies
RUN pip install torch==2.5.1 torchvision==0.20.1 transformers==4.49.0 accelerate
RUN pip install pillow fastapi uvicorn
RUN pip install svgpathtools cairosvg beautifulsoup4 webcolors
RUN pip install open-clip-torch datasets scikit-image
RUN pip install sentence-transformers reportlab svglib

# Clone and install star-vector (flash_attn already installed)
RUN git clone https://github.com/joanrod/star-vector.git /tmp/star-vector
RUN pip install /tmp/star-vector
RUN rm -rf /tmp/star-vector

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
