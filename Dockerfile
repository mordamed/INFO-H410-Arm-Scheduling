FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Dépendances système — minimal, sans QEMU ni gcc-arm
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch cu128 en premier — couche lourde isolée
# Si requirements.txt change, PyTorch n'est pas réinstallé
RUN pip3 install --no-cache-dir \
    --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Autres dépendances
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Code du projet — en dernier pour profiter du cache
COPY . .
RUN pip3 install --no-cache-dir -e .

RUN echo "alias arm-gdb='gdb-multiarch -q'" >> ~/.bashrc

CMD ["python3", "experiments/run_all.py"]