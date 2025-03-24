import base64
import io
import numpy as np
import requests
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image

app = FastAPI()

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
od_pipe = pipeline(task="object-detection", model="facebook/detr-resnet-50", threshold=0.1)


async def load_image(image_input):
    if isinstance(image_input, str):
        response = await requests.get(image_input, headers=headers)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(io.BytesIO(image_input.file.read()))

    return image.convert("RGBA") if image.mode == "P" and "transparency" in image.info else image


def get_image_embedding(img):
    return model.encode(img, batch_size=128, convert_to_tensor=True, show_progress_bar=False).tolist()


def get_cropped_objects(image, results):
    return [
        (
            image.crop((r['box']['xmin'], r['box']['ymin'], r['box']['xmax'], r['box']['ymax'])),
            r['score']
        )
        for r in results
    ]


def non_max_suppression(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if indices.size == 1:
            break
        ious = np.array([iou(boxes[current], boxes[i]) for i in indices[1:]])
        indices = indices[1:][ious <= iou_threshold]
    return keep


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


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


@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    image = await load_image(file)
    pipeline_output = od_pipe(image)
    boxes = [(r['box']['xmin'], r['box']['ymin'], r['box']['xmax'], r['box']['ymax']) for r in pipeline_output]
    scores = [r['score'] for r in pipeline_output]
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
    filtered_results = [pipeline_output[i] for i in keep_indices]
    cropped_objects = get_cropped_objects(image, filtered_results)

    if not cropped_objects:
        return JSONResponse(content=[])

    # # Select top 3 objects by score
    # top_3_objects = sorted(cropped_objects, key=lambda x: x[1], reverse=True)[:3]
    #
    # # Sort the top 3 objects by area (width * height)
    # sorted_objects = sorted(top_3_objects, key=lambda x: (x[0].width * x[0].height), reverse=True)
    #
    result = []
    for img, score in cropped_objects:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result.append({"score": score, "image": img_str})

    return JSONResponse(content=result)
