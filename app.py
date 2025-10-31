import cv2
import numpy as np
from PIL import Image

def drape_saree(model_image, saree_image):
    # Convert PIL → OpenCV RGB arrays
    model = np.array(model_image.convert("RGB"))
    saree = np.array(saree_image.convert("RGB"))

    # Resize saree to match model width
    saree_resized = cv2.resize(
        saree, (model.shape[1], int(model.shape[1] * saree.shape[0] / saree.shape[1]))
    )

    # Convert to float for safe math operations
    saree_lab = cv2.cvtColor(saree_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    model_lab = cv2.cvtColor(model, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Simple color balance to blend tones
    l_mean, a_mean, b_mean = np.mean(model_lab, axis=(0, 1))
    l_s, a_s, b_s = np.mean(saree_lab, axis=(0, 1))
    saree_lab[..., 0] += (l_mean - l_s)
    saree_lab[..., 1] += (a_mean - a_s)
    saree_lab[..., 2] += (b_mean - b_s)

    # Clip and convert back to uint8
    saree_lab = np.clip(saree_lab, 0, 255).astype(np.uint8)
    saree_resized = cv2.cvtColor(saree_lab, cv2.COLOR_LAB2RGB)

    # Create overlay from waist down (simulation)
    overlay = model.copy()
    h = model.shape[0]
    start_y = int(h * 0.55)
    end_y = min(start_y + saree_resized.shape[0], h)
    alpha = 0.5

    overlay[start_y:end_y, :] = cv2.addWeighted(
        model[start_y:end_y, :],
        1 - alpha,
        saree_resized[: end_y - start_y, :],
        alpha,
        0,
    )

    blended = Image.fromarray(overlay)
    return blended
