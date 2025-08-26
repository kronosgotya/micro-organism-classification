# Micro-Organism Classification (GPU • Docker • PyTorch)

End-to-end image classification for microscopic organisms: **reproducible training** (Docker + Jupyter), **evaluation & interpretability** (confusion matrix, **Grad-CAM**), and **production-style inference** via FastAPI (tested with Postman). Best reference run reached **~0.76 val accuracy** with ResNet-50 fine-tuning.

---

## Highlights

- **Reproducible environment:** Dockerized PyTorch with optional NVIDIA GPU.
- **Notebooks & scripts:** Clean structure for experiments and CLI training.
- **Training goodies:** AMP (mixed precision), cosine LR, early stopping, label smoothing, optional MixUp/TTA.
- **Evaluation:** Per-class confusion matrix, curves; **Grad-CAM** heatmaps.
- **Serving:** FastAPI with `/health`, `/labels`, `/predict`, `/predict-json`.
- **Postman ready:** Collection + (optional) local environment file.

---

## Requirements

- NVIDIA GPU + recent drivers (optional but recommended)
- Docker Desktop with GPU enabled
  - Windows: **Settings → Resources → GPU** 
  - Windows: **WSL2** enabled
- Quick check:
  ```bash
  nvidia-smi
  ```

---

## Repository layout

```
micro-organism-classification/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .dockerignore
├─ .gitignore
├─ notebooks/                   # exploratory notebooks (training/eval/Grad-CAM)
├─ data/                        # ImageFolder dataset 
├─ models/                      # (best_resnet50_finetuned.pt)
└─ api/
   ├─ server.py                 # FastAPI app
   ├─ Dockerfile                # API image
   ├─ requirements.txt          # API dependencies
   └─ postman/
      ├─ Micro-Organism-Classifier.postman_collection.json
      └─ Local.postman_environment.json (optional)
```

**Dataset (torchvision `ImageFolder`)**
Extracted on Kaggle (see link below):

<https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification/data>

```
data/
└─ Micro_Organism/
  	├─ classA/
 	├─ classB/
 	└─ ...

```

---

## Jupyter environment (Docker + GPU)

`docker-compose.yml` (root) provides a JupyterLab service `imgcls` on **127.0.0.1:8888**:

```yaml
services:
  imgcls:
    image: pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]     # enable GPU if available
    ports:
      - "127.0.0.1:8888:8888"
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: >
      bash -lc "
        pip install --upgrade pip &&
        pip install jupyterlab torchvision torchmetrics timm albumentations scikit-learn matplotlib &&
        jupyter lab --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/workspace
      "
```

Run Jupyter:
```bash
docker compose up --build
# open http://127.0.0.1:8888 and use the token printed in logs
# next time: docker compose up
```

Verify GPU in a notebook:
```python
import torch
torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)
```

> If you hit a NumPy ABI warning inside Jupyter:  
> `pip install "numpy<2"` (then restart the kernel).

---

## Training & evaluation

### CLI (optional)
```bash
#### typical hyper-params (adjust)
DATA_DIR=data IMG_SIZE=224 BATCH_SIZE=64 EPOCHS=12 python train.py
```

**Training pipeline**
- **Backbone:** ResNet-50 (torchvision)
- **Head:** linear or `Sequential(Dropout, Linear)` depending on config
- **Optimizer:** AdamW (often higher LR for `fc`)
- **Scheduler:** CosineAnnealingLR
- **Regularization:** label smoothing; optional **MixUp**
- **Precision:** **AMP** (mixed precision) for speed & VRAM
- **Early stopping:** on `val_acc` with patience

**Artifacts saved** to `models/best_resnet50_finetuned.pt`, including:
- `model` (state dict)
- `classes` (list of class names)
- `img_size` (int)

### Evaluation & plots
- Accuracy/precision/recall and loss curves
- **Confusion matrix** per class  
  The plotting cell auto-adjusts figure size and tick rotation to avoid overlap.

### Grad-CAM
- Notebook cell uses hooks on `layer4` to generate heatmaps.
- Visualizes: raw image, heatmap, overlay; useful on misclassifications.

**Reference result:** best **val_acc ≈ 0.7604** (early stop roughly between epochs 9–12, depending on seed/aug).

---

## Serving (FastAPI + Docker)

### API code
`api/server.py`:
- Loads `MODEL_PATH` (env) and reconstructs the ResNet-50 **head** that matches your checkpoint:
  - If state dict contains `fc.1.*` → builds `Sequential(Dropout, Linear)`
  - Otherwise → uses single `Linear`
- Reads `classes` and `img_size` from checkpoint.
- Inference transforms → Resize → CenterCrop → ToTensor → Normalize(ImageNet)
- Optional **TTA** at inference time (query/env).

**Endpoints**
- `GET /health` → status/device/info
- `GET /labels` → class list
- `POST /predict` → multipart **file** upload
- `POST /predict-json` → base64 payload

### API Dockerfile
`api/Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
EXPOSE 8000
CMD ["uvicorn","api.server:app","--host","0.0.0.0","--port","8000"]
```

`api/requirements.txt`:
```
fastapi==0.110.0
uvicorn[standard]==0.29.0
pillow==10.2.0
python-multipart==0.0.9
```

### API docker-compose (run from `api/` folder)
`api/docker-compose.yml`:
```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - MODEL_PATH=/app/models/best_resnet50_finetuned.pt
      - DEVICE=cpu          # or 'cuda' if using GPU
      - ENABLE_TTA=false
      - TOPK=5
    volumes:
      - ..:/app             # mount repo into /app
    ports:
      - "127.0.0.1:8000:8000"
    command: uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Run the API:
```bash
cd api
docker compose up --build -d
docker compose logs -f
#### open http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs
```

---

## Calling the API

### Postman
- Import:
  - `api/postman/Micro-Organism-Classifier.postman_collection.json`
  - `api/postman/Local.postman_environment.json` (sets `{{baseUrl}} = http://127.0.0.1:8000`)
- Requests:
  - `GET {{baseUrl}}/health`
  - `GET {{baseUrl}}/labels`
  - `POST {{baseUrl}}/predict?tta=false&topk=5`  
    Body → **form-data** → key **file** (type **File**) → select image
  - `POST {{baseUrl}}/predict-json`  
    Body → **raw JSON**:
    { "image_base64":"<BASE64>", "tta": false, "topk": 5 }

### cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict?tta=false&topk=5"      -H "Accept: application/json"      -F "file=@/path/to/image.jpg"
```

---

## Security notes

- Jupyter exposed on **127.0.0.1:8888** (loopback) and **token enabled**.
- Do **not** commit `data/`, `models/`, or secrets.
- If exposing the API externally: use TLS, auth, and configure CORS for your front-end.

---

## Troubleshooting

**NumPy ABI error in notebooks**  
> “A module compiled with NumPy 1.x cannot run in NumPy 2.x”  
Pin NumPy inside the Jupyter container:  
`pip install "numpy<2"` (restart kernel).

**`Could not import module "server"` in API logs**  
You’re mounting the repo at `/app`, so the module path is **`api.server:app`**. Ensure the compose `command`/`CMD` uses it.

**`Missing key(s) "fc.weight"... Unexpected key(s) "fc.1.weight"`**  
Model head mismatch. The server **auto-detects** and builds the correct head before loading weights. Make sure you’re running the updated `server.py`.

**`POST /predict` returns 422**  
Use **Body → form-data**; the key must be exactly **`file`** (type **File**). Don’t set `Content-Type` manually—Postman will.

**Model file not found**  
Ensure the checkpoint exists on host at `models/best_resnet50_finetuned.pt` and that your compose mounts the repo to `/app`. Env `MODEL_PATH` must be `/app/models/best_resnet50_finetuned.pt`.

**Port already in use**  
Change port mapping in compose, e.g. "127.0.0.1:8010:8000".

**GPU VRAM errors**  
Stop the Jupyter service `imgcls` when serving on GPU, or switch API to CPU (`DEVICE=cpu`) for testing.

---

## Env vars (API)

- `MODEL_PATH` (str): absolute path inside container, e.g. `/app/models/best_resnet50_finetuned.pt`
- `DEVICE` (`cpu`|`cuda`): inference device
- `ENABLE_TTA` (`true`|`false`): default TTA toggle
- `TOPK` (int): number of top probabilities to return

---
