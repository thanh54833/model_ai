import io

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer('clip-ViT-B-32')


def load_image(file: UploadFile):
    image = Image.open(io.BytesIO(file.file.read()))
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image


def get_image_embedding(img):
    img_embeddings = (
        model.encode(
            img,
            batch_size=128,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).tolist()
        if model is not None
        else []
    )
    return img_embeddings


@app.post("/image-to-vector/")
async def image_to_vector(file: UploadFile = File(...)):
    image = load_image(file)
    img_embedding = get_image_embedding(image)
    return {"embedding": img_embedding}
