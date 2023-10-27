import logging
from typing import Dict
import torch
from fastapi import FastAPI
import base64
import io
import numpy as np
import cv2
from PIL import Image

from nike_detector.nike_detector.service.service import init_nike_model, predict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
app = FastAPI()
models = {}


@app.get("/")
async def main():
    return "It's an entry point of Text vectorization service"


@app.on_event("startup")
def load_models():
    models['nike_model'] = init_nike_model()
    logger.info("Text model is successfully loaded on: {device}".format(device=device))


@app.post("/vectorize_batch", status_code=200)
def image_vectorization_batch(imgs: Dict):
    imm = base64.b64decode(imgs['bin'])
    img_ = Image.open(io.BytesIO(imm)).convert('RGB')
    img = cv2.cvtColor(np.array(img_), cv2.COLOR_RGB2BGR)
    result = predict(img)

    if "boxes" in result:
        im_file = io.BytesIO()
        result["image"].save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        enc = base64.b64encode(im_bytes).decode()
        return {
            "boxes": result["boxes"],
            "encoded_image": enc,
        }
    else:
        return None
