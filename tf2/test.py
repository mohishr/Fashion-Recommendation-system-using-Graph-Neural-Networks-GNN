# file: test_random_api.py

import numpy as np
import base64
import requests
from PIL import Image
import io
import json

BASE_URL = "http://localhost:5000"


def random_image_base64(size=(256, 256)):
    img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)

    pil_img = Image.fromarray(img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def test_predict():

    img1 = random_image_base64()
    img2 = random_image_base64()

    payload = {
        "items": [
            {"image_base64": img1},
            {"image_base64": img2}
        ]
    }

    r = requests.post(
        f"{BASE_URL}/api/predict",
        json=payload
    )

    print("\n===== /api/predict =====")
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))


def test_recommend():

    img1 = random_image_base64()
    img2 = random_image_base64()

    payload = {
        "target_category": "shoes",
        "partial_outfit": [
            {"image_base64": img1},
            {"image_base64": img2}
        ]
    }

    r = requests.post(
        f"{BASE_URL}/api/recommend_item",
        json=payload
    )

    print("\n===== /api/recommend_item =====")
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))


def test_health():

    r = requests.get(f"{BASE_URL}/api/health")

    print("\n===== /api/health =====")
    print("Status:", r.status_code)
    print(r.json())


def main():

    print("Testing Fashion API with Random Images")

    test_health()
    test_predict()
    test_recommend()


if __name__ == "__main__":
    main()