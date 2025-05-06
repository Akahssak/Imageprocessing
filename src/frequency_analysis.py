"""
Frequency analysis module for FFT and low-pass filtering.
"""

import numpy as np
import cv2
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Compute the 2D FFT of an image.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: FFT of the image.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    logger.info("Computed FFT of the image.")
    return fshift

def create_circular_lowpass_mask(shape: Tuple[int, int], radius: int) -> np.ndarray:
    """
    Create a circular low-pass filter mask.

    Args:
        shape (Tuple[int, int]): Shape of the mask (height, width).
        radius (int): Radius of the low-pass filter.

    Returns:
        np.ndarray: Circular mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    logger.info(f"Created circular low-pass mask with radius {radius}.")
    return mask

def create_gaussian_lowpass_mask(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    """
    Create a Gaussian low-pass filter mask.

    Args:
        shape (Tuple[int, int]): Shape of the mask (height, width).
        sigma (float): Standard deviation for Gaussian.

    Returns:
        np.ndarray: Gaussian mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    x, y = np.meshgrid(x, y)
    gauss = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * sigma ** 2))
    logger.info(f"Created Gaussian low-pass mask with sigma {sigma}.")
    return gauss

def apply_mask_and_reconstruct(fshift: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to the FFT and reconstruct the image using inverse FFT.

    Args:
        fshift (np.ndarray): Shifted FFT of the image.
        mask (np.ndarray): Mask to apply.

    Returns:
        np.ndarray: Reconstructed image.
    """
    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    logger.info("Reconstructed image from masked FFT.")
    return img_back
