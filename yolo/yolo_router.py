import base64
import io
import os

from PIL import Image
from fastapi import File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from ultralytics import YOLO

yolo_router = APIRouter()

model = YOLO("yolov8x.pt")  # Use "yolov8n.pt" for smaller model, "yolov8x.pt" for larger

def detect_and_crop_objects(image: Image.Image, margin=10):
    img_width, img_height = image.size

    # Perform object detection
    results = model.predict(image)

    # Extract bounding boxes, labels, and confidence scores
    boxes = results[0].boxes.xyxy.tolist()  # Bounding box coordinates (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.tolist()  # Confidence scores
    class_ids = results[0].boxes.cls.tolist()  # Class IDs (e.g., 0 for "person", 2 for "car")

    # Create a directory to save cropped objects
    os.makedirs("cropped_objects", exist_ok=True)

    # Crop and save detected objects
    cropped_images = []
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        # Add margin and ensure coordinates are within image bounds
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_width, x2 + margin)
        y2 = min(img_height, y2 + margin)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image.save(f"cropped_objects/object_{i}.jpg")
        cropped_images.append(cropped_image)
        print(f"Cropped Object {i}: Class ID {cls_id}, Confidence {conf:.2f}")

    print(f"Cropped {len(boxes)} objects!")

    return cropped_images

@yolo_router.post("/detect-and-crop/")
async def detect_and_crop(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    cropped_images = detect_and_crop_objects(image)

    # Convert cropped images to base64 strings
    cropped_images_base64 = []
    for cropped_image in cropped_images:
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        cropped_images_base64.append(img_str)

    return JSONResponse(content={"cropped_images": cropped_images_base64})

# To run the FastAPI app, use the following command:
# uvicorn yolo_router:app --reload