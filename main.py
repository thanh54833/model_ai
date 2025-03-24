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


# JSONResponse(content=sorted_score) i want detect object full visible not 1/2 , 1/3 object
@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    image = await load_image(file)
    aspect_ratio = image.width / image.height

    # Check if the image is from a mobile device or a specific camera
    print(f"aspect_ratio : ${image.width} ${image.height} ", aspect_ratio)
    # if aspect_ratio < 1:  # Taller image, likely from a mobile device
    #     new_height = 800
    #     new_width = int(new_height * aspect_ratio)
    # else:
    #     new_width = 800
    #     new_height = int(new_width / aspect_ratio)

    raw_image = image.resize((600 , 400))

    pipeline_output = od_pipe(raw_image)
    image_width, image_height = raw_image.size

    # Filter out objects that are not fully visible
    # fully_visible_objects = [
    #     result for result in pipeline_output
    #     if result['box']['xmin'] > 0 and result['box']['ymin'] > 0
    #        and result['box']['xmax'] <= image_width and result['box']['ymax'] <= image_height
    # ]

    # Calculate the size of each bounding box and sort by size (width * height)
    for result in pipeline_output:
        box = result['box']
        result['size'] = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])

    sorted_size = sorted(pipeline_output, key=lambda x: x['size'], reverse=True)
    # sort_top_2 = sorted_size[:3]
    # sorted_score = sorted(sort_top_2, key=lambda x: x['score'], reverse=True)

    # Crop images and convert to base64 strings
    for result in sorted_size:
        box = result['box']
        cropped_image = raw_image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        buffered = BytesIO()
        cropped_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result["image"] = img_str

    return JSONResponse(content=sorted_size)
