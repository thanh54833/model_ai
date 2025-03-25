import io
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from yolo.yolo_router import yolo_router

app = FastAPI()
app.include_router(yolo_router, prefix="/yolo", tags=["yolo"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

headers = {
    # Your headers here
}

model = SentenceTransformer('clip-ViT-B-32')


# od_pipe = pipeline(task="object-detection", model="facebook/detr-resnet-50", threshold=0.1)


async def load_image(image_input):
    if isinstance(image_input, str):
        response = await requests.get(image_input, headers=headers)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(io.BytesIO(image_input.file.read()))

    return image.convert("RGBA") if image.mode == "P" and "transparency" in image.info else image


def get_image_embedding(img):
    return model.encode(img, batch_size=128, convert_to_tensor=True, show_progress_bar=False).tolist()


@app.post("/image_http_to_vector/")
async def image_to_vector(image_url: str):
    image = await load_image(image_url)
    img_embedding = get_image_embedding(image)
    return img_embedding


@app.post("/image_file_to_vector/")
async def image_to_vector(file: UploadFile = File(...)):
    image = await load_image(file)
    img_embedding = get_image_embedding(image)
    return img_embedding


@app.post("/search_by_images/")
async def search_by_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    vector1 = await image_to_vector(file1)
    vector2 = await image_to_vector(file2)

    merged_vector = np.concatenate((vector1, vector2))

    # 1024
    print(f"merged_vector {len(merged_vector)}")

    return JSONResponse(content={"merged_vector": merged_vector.tolist()})
