"""
Data pipeline module for loading, preprocessing, and augmenting image datasets.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_images_from_folder(folder_path: str, recursive: bool = True) -> List[np.ndarray]:
    """
    Load images from a folder recursively.

    Args:
        folder_path (str): Path to the folder containing images.
        recursive (bool): Whether to scan folders recursively.

    Returns:
        List[np.ndarray]: List of loaded grayscale images.
    """
    images = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    logger.info(f"Loaded image: {img_path}")
    else:
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                logger.info(f"Loaded image: {img_path}")
    return images

def preprocess_image(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resize and normalize an image.

    Args:
        image (np.ndarray): Input grayscale image.
        size (Tuple[int, int]): Desired output size (width, height).

    Returns:
        np.ndarray: Preprocessed image.
    """
    resized = cv2.resize(image, size)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

# Additional augmentation functions can be added here
