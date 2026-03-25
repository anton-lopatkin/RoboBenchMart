from typing import Any, Dict, List, Optional

import base64
import cv2
from gymnasium import Env
import numpy as np


def prepare_observations(env: Env, obs: Dict[str, Any]) -> Dict[str, Any]:
    camera_data = obs["sensor_data"]["right_base_camera_link"]
    image = camera_data["rgb"][0].cpu().numpy() [:, :, ::-1]
    segmentation = camera_data["segmentation"][0].cpu().numpy()[..., 0]

    scale = 3
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
    segmentation = cv2.resize(segmentation, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    products = get_products(env)
    scene_objects = extract_scene_objects(segmentation, products)
    annotated_image = annotate_image(image, scene_objects)

    cv2.imwrite('outputs/original.png', image)
    cv2.imwrite('outputs/annotated.png', annotated_image)

    return {
        "image": image_to_base64(image),
        "annotated_image": image_to_base64(annotated_image),
        "scene_objects": scene_objects,
    }


def image_to_base64(image: np.ndarray) -> Optional[str]:
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return None
    encoded_bytes = encoded_image.tobytes()
    return base64.b64encode(encoded_bytes).decode("utf-8")


def get_products(env: Env) -> Dict[int, str]:
    return {
        product.per_scene_id[0].item(): product.name
        for product in env.unwrapped.actors["products"].values()
    }


def extract_scene_objects(segmentation: np.ndarray, products: Dict[int, str]) -> List[Dict[str, Any]]:
    scene_objects: List[Dict[str, Any]] = []

    height, width = segmentation.shape
    padding = 3

    for product_id, product_name in products.items():
        if not product_name.endswith("0"):
            continue

        mask = segmentation == product_id
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        xmin = max(0, int(xs.min()) - padding)
        ymin = max(0, int(ys.min()) - padding)
        xmax = min(width - 1, int(xs.max()) + padding)
        ymax = min(height - 1, int(ys.max()) + padding)

        scene_objects.append(
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "product_id": product_id,
                "product_name": product_name.split("_", 1)[1].split(":")[0],
            }
        )

    return scene_objects


def annotate_image(image: np.ndarray, scene_objects: List[Dict[str, Any]]) -> np.ndarray:
    output = image.copy()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    bg_color = (255, 255, 255)

    palette = build_palette(scene_objects)

    for obj in scene_objects:
        x_min, y_min, x_max, y_max = obj["bbox"]
        color = tuple(map(int, palette[obj["product_id"]]))

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, 1)

        label = str(obj["product_id"])
        (label_width, label_height), baseline = cv2.getTextSize(label, font_face, font_scale, font_thickness)
        label_x, label_y = x_min, max(y_min, label_height)

        cv2.rectangle(output, (label_x, label_y - label_height - baseline), (label_x + label_width, label_y - baseline), bg_color, -1)
        cv2.putText(output, label, (label_x, label_y - baseline), font_face, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    return output


def build_palette(scene_objects: List[Dict[str, Any]], seed: int = 42) -> np.ndarray:
    max_product_id = max(obj["product_id"] for obj in scene_objects)
    palette_size = max(128, max_product_id + 1)

    rng = np.random.RandomState(seed)
    palette = rng.randint(20, 235, size=(palette_size, 3), dtype=np.uint8)

    return palette

