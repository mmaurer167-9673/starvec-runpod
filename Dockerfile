FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 FIRST - this is critical!
RUN pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Now install star-vector from GitHub
RUN git clone https://github.com/joanrod/star-vector.git /tmp/star-vector
RUN pip install /tmp/star-vector
RUN rm -rf /tmp/star-vector

# Install web framework
RUN pip install fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6

# Copy your app.py
COPY app.py /app/app.py

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "app.py"]
