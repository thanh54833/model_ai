from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
from PIL import Image
app = FastAPI()

model = SentenceTransformer('clip-ViT-B-32')

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
