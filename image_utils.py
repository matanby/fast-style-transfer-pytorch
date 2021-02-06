from glob import glob
from typing import Union, Tuple, List

import numpy as np
from PIL import Image


def load(path: str, as_float32: bool = True, channels_first: bool = False) -> np.ndarray:
    image_pil = Image.open(path).convert("RGB")
    image_np = np.array(image_pil)

    if as_float32:
        image_np = image_np.astype(np.float32) / 255.0

    if channels_first:
        image_np = np.transpose(image_np, (2, 0, 1))

    return image_np


def save(image: Union[np.ndarray, Image.Image], path: str) -> None:
    to_pil(image).save(path)


def show(image: Union[np.ndarray, Image.Image, str]) -> None:
    if isinstance(image, str):
        image = load(image)

    to_pil(image).show()


def resize(image: Union[np.ndarray, Image.Image], size: Tuple[int, int]) -> np.ndarray:
    image_pil = to_pil(image)
    image_pil_resized = image_pil.resize(size, Image.ANTIALIAS)

    if isinstance(image, Image.Image):
        result = image_pil_resized
    else:
        as_float32 = image.dtype == np.float32
        result = from_pil(image_pil_resized, as_float32)

    return result


def to_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float32:
        image = image.astype('float32') / 255.0

    return image


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).astype('uint8')

    return image


def to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    image_uint8 = to_uint8(image)
    image_pil = Image.fromarray(image_uint8)
    return image_pil


def from_pil(image: Union[np.ndarray, Image.Image], as_float32: bool = True) -> np.ndarray:
    image_uint8 = np.array(image)
    if as_float32:
        return to_float32(image_uint8)
    else:
        return image_uint8


def is_image(path: str) -> bool:
    image_extensions = ('.jpg', '.jpeg', '.png')
    return path.lower().endswith(image_extensions)


def list_images(path: str) -> List[str]:
    all_files = glob(path + '/*.*')
    image_paths = sorted(filter(is_image, all_files))
    return image_paths
