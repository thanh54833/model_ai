import io
from io import BytesIO

import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer

app = FastAPI()

headers = {
    "authority": "cdn1.concung.com",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "cookie": "_gcl_au=1.1.1467091803.1703572922; _ga=GA1.1.380043844.1703572922; __admUTMtime=1703572922; _tt_enable_cookie=1; _ttp=WZWQ7LJvoLrl3y4MJri8ueA9m2v; __iid=; __su=0; __zi=3000.SSZzejyD6z4bYUkZra0Dq2-5fExB3WF8RDcsvDGSGTfwolIxaXn2bJ3Bzh6H7m-SCiYxk98U3vS-mRcz.1; _aff_network=accesstrade; _aff_sid=USDnhZI3I7yC27Gk9EOm1OpNYIJx8elurPN2PQwQpGs8aXRg; __utm=source%3Daccesstrade; dtdz=-1; __iid=; __iid=; __su=0; __su=0; 6f1eb01ca7fb61e4f6882c1dc816f22d=T%2FEqzjRRd5g%3D9wbPAi8i%2BPE%3D1Ci65WLYCYs%3DKsUJe1nSDl4%3DH9DwywDLCIw%3Da7NDiPDjkp8%3DBMNH2%2FPz1Ww%3DjFPr4PEbB58%3DD94ivb5Cw3c%3Dr1OchLBIGPo%3DXm3ctRf7oxM%3D9alt4piEgqQ%3DQ7x721%2FEaGg%3DznsRoJyh1cc%3DHZRFAWrCSGY%3DMBubZ79mL2c%3DsV7ckP9MEc4%3D0GV8B70dOvE%3D%2F1zaJQDvRRU%3Dp8QfFZQtGuc%3DV%2FZ3d8vjS%2BU%3DlWKnWytgm20%3DI0wdsiDtNSY%3Dn9M%2BXI5In%2Fg%3DvD%2BX42uOwWs%3Dk%2BzPD7NDXT4%3D; _gcl_aw=GCL.1704450004.CjwKCAiA7t6sBhAiEiwAsaieYq1Fe41nqEQbPyBuHkj2YKBXHp1OxqMeGIy5X-w-oHKHDGz_YUWS3RoCteUQAvD_BwE; __utma=65249340.246610421.1703572922.1704446042.1704450004.7; __utmz=65249340.1704450004.7.5.utmcsr=accesstrade|utmgclid=CjwKCAiA7t6sBhAiEiwAsaieYq1Fe41nqEQbPyBuHkj2YKBXHp1OxqMeGIy5X-w-oHKHDGz_YUWS3RoCteUQAvD_BwE|utmccn=(not%20set)|utmcmd=(not%20set)|utmctr=(not%20provided); _gac_UA-36329013-1=1.1704445376.CjwKCAiA7t6sBhAiEiwAsaieYq1Fe41nqEQbPyBuHkj2YKBXHp1OxqMeGIy5X-w-oHKHDGz_YUWS3RoCteUQAvD_BwE; _ga_DFG3FWNPBM=GS1.1.1704461153.8.0.1704461153.60.0.0; _ga_BBD6001M29=GS1.1.1704461153.8.0.1704461153.60.0.0; PHPSESSID=0fvsb7tbhb1eqa9ndobu760hdq; Srv=cc205|ZZohh|ZZocU; Srv=cc205|ZZohq|ZZoeR",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
}

model = SentenceTransformer('clip-ViT-B-32')


def load_image(image_url):
    if isinstance(image_url, str):
        response = requests.get(image_url, headers=headers)
        image = Image.open(BytesIO(response.content))
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        return image
    else:
        image = Image.open(io.BytesIO(image_url.file.read()))
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


@app.post("/image_http_to_vector/")
async def image_to_vector(image_url: str):
    image = load_image(image_url)
    img_embedding = get_image_embedding(image)
    return img_embedding


@app.post("/image_file_to_vector/")
async def image_to_vector(file: UploadFile = File(...)):
    image = load_image(file)
    img_embedding = get_image_embedding(image)
    return img_embedding
