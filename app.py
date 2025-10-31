import cv2
import numpy as np
from PIL import Image

def drape_saree(model_image, saree_image):
    # Convert PIL to NumPy RGB
    model = np.array(model_image.convert("RGB"))
    saree = np.array(saree_image.convert("RGB"))

    # Resize saree to model width
    model_h, model_w = model.shape[:2]
    aspect_ratio = saree.shape[0] / saree.shape[1]
    new_height = int(model_w * aspect_ratio)
    saree_resized = cv2.resize(saree, (model_w, new_height))

    # Convert to LAB for better tone matching
    model_lab = cv2.cvtColor(model, cv2.COLOR_RGB2LAB).astype(np.float32)
    saree_lab = cv2.cvtColor(saree_resized, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Match color tone between model and saree
    l_mean_m, a_mean_m, b_mean_m = np.mean(model_lab, axis=(0, 1))
    l_mean_s, a_mean_s, b_mean_s = np.mean(saree_lab, axis=(0, 1))

    saree_lab[..., 0] += (l_mean_m - l_mean_s)
    saree_lab[..., 1] += (a_mean_m - a_mean_s)
    saree_lab[..., 2] += (b_mean_m - b_mean_s)
    saree_lab = np.clip(saree_lab, 0, 255).astype(np.uint8)

    saree_matched = cv2.cvtColor(saree_lab, cv2.COLOR_LAB2RGB)

    # Blend from waist down
    start_y = int(model_h * 0.55)
    end_y = min(start_y + saree_matched.shape[0], model_h)
    alpha = 0.5

    # Ensure blending region sizes match
    region_model = model[start_y:end_y, :]
    region_saree = saree_matched[:end_y - start_y, :region_model.shape[1]]

    blended_region = cv2.addWeighted(region_model, 1 - alpha, region_saree, alpha, 0)
    blended = model.copy()
    blended[start_y:end_y, :] = blended_region

    return Image.fromarray(blended)
