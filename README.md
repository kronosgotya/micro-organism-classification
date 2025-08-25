# Micro-Organism Classification (GPU • Docker + PyTorch)

### Image classification with PyTorch accelerated by GPU inside Docker. Includes a reproducible JupyterLab environment, project structure, and training/inference commands.
---

## ✨ Features

- Isolated, reproducible environment (Docker)
- GPU acceleration (NVIDIA)
- JupyterLab exposed only on 127.0.0.1
- Standard data structure (train/, val/)
- Starter scripts train.py / infer.py (optional)
---

##  Dataset
The dataset has been taken from Kaggle (see link below):
`https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification/data`
---

## 📦 Requirements

- NVIDIA GPU + up-to-date drivers
- Docker Desktop with GPU support enabled
    - Windows: Settings → Resources → GPU ✅
    - Windows: WSL2 enabled
- Quick check:
#### Host o WSL2
`nvidia-smi`
---

## 📁 Project Structure

```micro-organism-classification/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .dockerignore
├─ .gitignore
├─ notebooks/                 # working notebooks
├─ data/                      
├─ models/                     
├─ api
```
---

## 🚀 Quick Start: docker compose
### 1) Initial Build 
`docker compose up --build`

Open <http://127.0.0.1:8888>

Copy the token shown in the terminal logs.

Next sessions: `docker compose up`

### 2) Verify GPU in a notebook
```import torch
torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)
```
It should show `True` and your NVIDIA GPU name.

### 🧪 Training (optional)

Typical variables (adjust to dataset)
`DATA_DIR=data BATCH_SIZE=16 EPOCHS=12 IMG_SIZE=320 python train.py`
---

## 🔐 Security (public repo)

- Do not disable the Jupyter token in production commands.
- `docker-compose.yml` exposes Jupyter only on loopback: `127.0.0.1:8888.`
- Do not commit secrets/credentials to the repo.

