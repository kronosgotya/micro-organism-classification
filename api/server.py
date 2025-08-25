import io, os, base64, json
from typing import List, Optional
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF
from torchvision.models import resnet50
from torch import nn


app = FastAPI(title="Micro-Organism Classifier", version="1.0")

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_resnet50_finetuned.pt")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TOPK = int(os.getenv("TOPK", "5"))
ENABLE_TTA = os.getenv("ENABLE_TTA", "false").lower() == "true"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------- Carga modelo ----------
def build_model(num_classes: int, state_keys=None):
    m = resnet50(weights=None)
    in_features = m.fc.in_features

    # Si el checkpoint trae claves "fc.1.*" => cabeza Sequential(Dropout, Linear)
    if state_keys is not None and any(k.startswith("fc.1.") for k in state_keys):
        m.fc = nn.Sequential(
            nn.Dropout(0.3),                 # el valor del dropout no afecta al load_state_dict
            nn.Linear(in_features, num_classes)
        )
    else:
        m.fc = nn.Linear(in_features, num_classes)
    return m

def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("El checkpoint no contiene 'classes'.")
    img_size = int(ckpt.get("img_size", 224))

    state = ckpt["model"]
    model = build_model(len(classes), state_keys=state.keys())

    # Cargamos (tolerante por si hay buffers sin usar)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("load_state_dict warnings:",
              {"missing": missing, "unexpected": unexpected})

    model.eval().to(device)
    return model, classes, img_size

model, CLASSES, IMG_SIZE = load_checkpoint(MODEL_PATH, DEVICE)

infer_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------- Utilidades ----------
def tensor_from_pil(img: Image.Image) -> torch.Tensor:
    return infer_tfms(img.convert("RGB")).unsqueeze(0)

def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)

# TTA simple
AUGS = [
    lambda x: x,
    TF.hflip,
    lambda x: TF.rotate(x, 90),
    lambda x: TF.rotate(x, 270)
]

def predict_tensor(xb: torch.Tensor, tta: bool = False) -> torch.Tensor:
    xb = xb.to(DEVICE, non_blocking=True)
    with torch.no_grad(), autocast(enabled=(DEVICE == "cuda")):
        if tta:
            p = 0
            for aug in AUGS:
                logits = model(aug(xb))
                p += probs_from_logits(logits)
            p = p / len(AUGS)
        else:
            logits = model(xb)
            p = probs_from_logits(logits)
    return p.detach().cpu()

def topk_from_probs(p: torch.Tensor, k: int):
    vals, idxs = torch.topk(p, k=min(k, p.shape[1]), dim=1)
    return vals[0].tolist(), idxs[0].tolist()

# ---------- Schemas ----------
class PredictJson(BaseModel):
    image_base64: str
    tta: Optional[bool] = None
    topk: Optional[int] = None

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "img_size": IMG_SIZE, "num_classes": len(CLASSES)}

@app.get("/labels")
def labels():
    return {"classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...), tta: bool = Query(default=ENABLE_TTA), topk: int = Query(default=TOPK)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    xb = tensor_from_pil(img)
    p = predict_tensor(xb, tta=tta)
    vals, idxs = topk_from_probs(p, topk)
    top_probs = [{"class": CLASSES[i], "index": int(i), "prob": float(v)} for v, i in zip(vals, idxs)]

    pred_idx = int(torch.argmax(p, dim=1).item())
    return {
        "pred_idx": pred_idx,
        "pred_class": CLASSES[pred_idx],
        "topk": top_probs
    }

@app.post("/predict-json")
def predict_json(payload: PredictJson):
    try:
        img_bytes = base64.b64decode(payload.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 inválido")

    xb = tensor_from_pil(img)
    p = predict_tensor(xb, tta=payload.tta if payload.tta is not None else ENABLE_TTA)
    vals, idxs = topk_from_probs(p, payload.topk or TOPK)
    top_probs = [{"class": CLASSES[i], "index": int(i), "prob": float(v)} for v, i in zip(vals, idxs)]
    pred_idx = int(torch.argmax(p, dim=1).item())

    return {
        "pred_idx": pred_idx,
        "pred_class": CLASSES[pred_idx],
        "topk": top_probs
    }
