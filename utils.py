from torch import Tensor
import numpy as np
from PIL import Image, ImageOps
import cv2
import numpy as np
from PIL import Image

def parse_data(dataset: Tensor) -> np.array:
    arr = dataset.data.numpy()
    arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
    return arr

def parse_labels(dataset: Tensor) -> np.array:
    return dataset.targets.numpy()

def normalize_image(image: Image) -> Image:
    should_invert = True
    if should_invert:
        return ImageOps.invert(image.convert('L')).resize((28, 28)).point(lambda x: 255 if x > 30 else 0)
    else:
        return image.convert('L').resize((28, 28)).point(lambda x: 255 if x > 30 else 0)
    
def center_image(image: Image) -> Image:
    # Convert image to grayscale
    img = np.array(image.convert('L'))

    # Get shape
    hh, ww = img.shape

    # Get contours (presumably just one around the nonzero pixels)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)

    # Recenter
    startx = (ww - w) // 2
    starty = (hh - h) // 2
    result = np.zeros_like(img)
    result[starty:starty+h, startx:startx+w] = img[y:y+h, x:x+w]

    # Convert result back to PIL Image
    centered_image = Image.fromarray(result)

    return centered_image

def convert_from_image(image: Image) -> np.array:
    return np.array(center_image(normalize_image(image))).flatten()

def convert_to_image(arr: np.array) -> Image:
    return Image.fromarray(arr.reshape((28, 28)))