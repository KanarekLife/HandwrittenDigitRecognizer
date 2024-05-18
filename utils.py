from torch import Tensor
import numpy as np
from PIL import Image, ImageOps

def parse_data(dataset: Tensor) -> np.array:
    arr = dataset.data.numpy()
    arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
    return arr

def parse_labels(dataset: Tensor) -> np.array:
    return dataset.targets.numpy()

def normalize_image(image: Image) -> Image:
    should_invert = True
    if should_invert:
        return ImageOps.invert(image.convert('L')).resize((28, 28))
    else:
        return image.convert('L').resize((28, 28))

def convert_from_image(image: Image) -> np.array:
    return np.array(normalize_image(image)).flatten()

def convert_to_image(arr: np.array) -> Image:
    return Image.fromarray(arr.reshape((28, 28)))