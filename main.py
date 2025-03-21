import io

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer('clip-ViT-B-32')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/image-to-vector/")
async def image_to_vector(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize image to the input size expected by the model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    vector = model.encode(image_array)
    return {"embedding": vector.tolist()}
