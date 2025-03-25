import base64
import io
import os

from PIL import Image, ExifTags
from fastapi import File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from ultralytics import YOLO

yolo_router = APIRouter()

model = YOLO("yolo11n.pt")

# Class names mapping
class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair dryer",
    79: "toothbrush",
}


def detect_and_crop_objects(image: Image.Image, margin=0):
    img_width, img_height = image.size

    # Perform object detection
    results = model.predict(image, conf=0.25)

    # Extract bounding boxes, labels, and confidence scores
    boxes = results[0].boxes.xyxy.tolist()  # Bounding box coordinates (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.tolist()  # Confidence scores
    class_ids = results[0].boxes.cls.tolist()  # Class IDs

    # Create a directory to save cropped objects
    os.makedirs("cropped_objects", exist_ok=True)

    # Crop and save detected objects
    cropped_images = []
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        # Exclude humans (class ID 0)
        if cls_id == 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        # Add margin and ensure coordinates are within image bounds
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_width, x2 + margin)
        y2 = min(img_height, y2 + margin)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image.save(f"cropped_objects/object_{i}.jpg")
        cropped_images.append((cropped_image, cls_id, conf))
        print(f"Cropped Object {i}: Class ID {cls_id}, Confidence {conf:.2f}")

    print(f"Cropped {len(boxes)} objects!")
    return cropped_images


def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    # Rotate the image based on the EXIF orientation
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
    return image


@yolo_router.post("/detect-and-crop/")
async def detect_and_crop(file: UploadFile = File(...)):
    image_ = Image.open(io.BytesIO(await file.read()))
    # Correct the orientation of the image
    image_ = correct_image_orientation(image_)

    image_.save("image_input.jpg")

    width, height = image_.size
    image = image_.resize((600, int(height * 600 / width)))

    cropped_images = detect_and_crop_objects(image)
    # Convert cropped images to base64 strings and include confidence scores and class names
    cropped_images_base64 = []
    for cropped_image, cls_id, conf in cropped_images:
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        class_name = class_names.get(cls_id, "unknown")
        cropped_images_base64.append({"image": img_str, "score": conf, "label": class_name})

    return JSONResponse(content=cropped_images_base64)
