"""
Segmentation module for adaptive thresholding and morphological operations.
"""

import cv2
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to an image.

    Args:
        image (np.ndarray): Input grayscale image.
        block_size (int): Size of a pixel neighborhood used to calculate threshold.
        c (int): Constant subtracted from the mean.

    Returns:
        np.ndarray: Binary image after thresholding.
    """
    thresh = cv2.adaptiveThreshold((image * 255).astype(np.uint8), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, c)
    logger.info("Applied adaptive thresholding.")
    return thresh

def morphological_cleaning(binary_image: np.ndarray, kernel_size: int = 3,
                          iterations: int = 1) -> np.ndarray:
    """
    Perform morphological cleaning (erosion followed by dilation).

    Args:
        binary_image (np.ndarray): Binary image to clean.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times erosion and dilation are applied.

    Returns:
        np.ndarray: Cleaned binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    logger.info("Performed morphological cleaning.")
    return cleaned

def find_contours(binary_image: np.ndarray) -> List[np.ndarray]:
    """
    Find contours in a binary image.

    Args:
        binary_image (np.ndarray): Binary image.

    Returns:
        List[np.ndarray]: List of contours.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Found {len(contours)} contours.")
    return contours
