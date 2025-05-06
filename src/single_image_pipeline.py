"""
Simple script to process a single image through the pipeline steps:
preprocessing, frequency analysis, segmentation, and saving outputs.
"""

import os
import cv2
import numpy as np
from src import data_pipeline, frequency_analysis, segmentation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_image(input_path: str, output_dir: str, size: tuple = (256, 256),
                         filter_type: str = 'circular', radius: int = 30, sigma: float = 10.0) -> None:
    """
    Process a single image through preprocessing, frequency analysis, and segmentation.

    Args:
        input_path (str): Path to the input image file.
        output_dir (str): Directory to save output images.
        size (tuple): Resize dimensions (width, height).
        filter_type (str): 'circular' or 'gaussian' low-pass filter.
        radius (int): Radius for circular filter.
        sigma (float): Sigma for gaussian filter.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to load image: {input_path}")
        return
    logger.info(f"Loaded image: {input_path}")

    # Preprocess
    preprocessed = data_pipeline.preprocess_image(img, size)
    preprocessed_path = os.path.join(output_dir, "preprocessed.png")
    cv2.imwrite(preprocessed_path, (preprocessed * 255).astype('uint8'))
    logger.info(f"Saved preprocessed image to {preprocessed_path}")

    # Frequency analysis
    fshift = frequency_analysis.compute_fft(preprocessed)
    if filter_type == 'circular':
        mask = frequency_analysis.create_circular_lowpass_mask(preprocessed.shape, radius)
    else:
        mask = frequency_analysis.create_gaussian_lowpass_mask(preprocessed.shape, sigma)
    filtered_img = frequency_analysis.apply_mask_and_reconstruct(fshift, mask)
    filtered_path = os.path.join(output_dir, "frequency_filtered.png")
    cv2.imwrite(filtered_path, np.clip(filtered_img, 0, 255).astype('uint8'))
    logger.info(f"Saved frequency filtered image to {filtered_path}")

    # Segmentation
    thresh = segmentation.adaptive_threshold(filtered_img)
    cleaned = segmentation.morphological_cleaning(thresh)
    contours = segmentation.find_contours(cleaned)
    mask = np.zeros(filtered_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    mask_path = os.path.join(output_dir, "segmented_mask.png")
    cv2.imwrite(mask_path, mask)
    logger.info(f"Saved segmented mask to {mask_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a single image through the low-frequency object detection pipeline.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256], help='Resize dimensions (width height)')
    parser.add_argument('--filter', type=str, choices=['circular', 'gaussian'], default='circular', help='Type of low-pass filter')
    parser.add_argument('--radius', type=int, default=30, help='Radius for circular filter')
    parser.add_argument('--sigma', type=float, default=10.0, help='Sigma for gaussian filter')

    args = parser.parse_args()

    process_single_image(args.input_path, args.output_dir, tuple(args.size), args.filter, args.radius, args.sigma)
