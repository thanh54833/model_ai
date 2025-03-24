import base64
import io
from io import BytesIO

import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from transformers import pipeline

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
    raw_image = image.resize((600, 400))

    pipeline_output = od_pipe(raw_image)

    # Calculate the size of each bounding box and sort by size (width * height)
    for result in pipeline_output:
        box = result['box']
        result['size'] = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])

    sorted_score_output = sorted(pipeline_output, key=lambda x: x['score'], reverse=True)
    top_2_output = sorted_score_output[:2]
    sorted_output = sorted(top_2_output, key=lambda x: x['size'], reverse=True)

    # Crop images and convert to base64 strings
    for result in sorted_output:
        box = result['box']
        cropped_image = raw_image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        buffered = BytesIO()
        cropped_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result["image"] = img_str

    return JSONResponse(content=sorted_output)
