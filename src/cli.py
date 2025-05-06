"""
Command-line interface for the low-frequency object detection pipeline.
"""

import argparse
import logging
import os
import json
import cv2
import numpy as np
from src import data_pipeline, frequency_analysis, segmentation, evaluate, model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Low-Frequency Object Detection Pipeline CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Run data preprocessing')
    preprocess_parser.add_argument('--input_dir', type=str, required=True, help='Input image folder')
    preprocess_parser.add_argument('--output_dir', type=str, required=True, help='Output folder for preprocessed images')
    preprocess_parser.add_argument('--size', type=int, nargs=2, default=[256, 256], help='Resize dimensions (width height)')

    # Frequency analysis command
    freq_parser = subparsers.add_parser('frequency', help='Run frequency analysis')
    freq_parser.add_argument('--input_dir', type=str, required=True, help='Input folder of preprocessed images')
    freq_parser.add_argument('--output_dir', type=str, required=True, help='Output folder for frequency filtered images')
    freq_parser.add_argument('--filter', type=str, choices=['circular', 'gaussian'], default='circular', help='Type of low-pass filter')
    freq_parser.add_argument('--radius', type=int, default=30, help='Radius for circular filter')
    freq_parser.add_argument('--sigma', type=float, default=10.0, help='Sigma for gaussian filter')

    # Segmentation command
    seg_parser = subparsers.add_parser('segment', help='Run segmentation')
    seg_parser.add_argument('--input_dir', type=str, required=True, help='Input folder of frequency filtered images')
    seg_parser.add_argument('--output_dir', type=str, required=True, help='Output folder for segmented masks')

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--pred_dir', type=str, required=True, help='Predicted masks folder')
    eval_parser.add_argument('--gt_dir', type=str, required=True, help='Ground truth masks folder')
    eval_parser.add_argument('--output_report', type=str, required=True, help='Output JSON report path')
    eval_parser.add_argument('--output_plot', type=str, required=True, help='Output plot image path')

    # Optional model training command placeholder
    model_parser = subparsers.add_parser('train_model', help='Train CNN model on segmented patches')
    model_parser.add_argument('--data_dir', type=str, required=True, help='Directory with labeled patches')
    model_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    args = parser.parse_args()

    if args.command == 'preprocess':
        os.makedirs(args.output_dir, exist_ok=True)
        images = data_pipeline.load_images_from_folder(args.input_dir)
        for i, img in enumerate(images):
            preprocessed = data_pipeline.preprocess_image(img, tuple(args.size))
            out_path = os.path.join(args.output_dir, f"img_{i:04d}.png")
            cv2.imwrite(out_path, (preprocessed * 255).astype('uint8'))
            logger.info(f"Saved preprocessed image to {out_path}")

    elif args.command == 'frequency':
        os.makedirs(args.output_dir, exist_ok=True)
        images = data_pipeline.load_images_from_folder(args.input_dir)
        for i, img in enumerate(images):
            fshift = frequency_analysis.compute_fft(img)
            if args.filter == 'circular':
                mask = frequency_analysis.create_circular_lowpass_mask(img.shape, args.radius)
            else:
                mask = frequency_analysis.create_gaussian_lowpass_mask(img.shape, args.sigma)
            filtered_img = frequency_analysis.apply_mask_and_reconstruct(fshift, mask)
            out_path = os.path.join(args.output_dir, f"freq_{i:04d}.png")
            cv2.imwrite(out_path, np.clip(filtered_img, 0, 255).astype('uint8'))
            logger.info(f"Saved frequency filtered image to {out_path}")

    elif args.command == 'segment':
        os.makedirs(args.output_dir, exist_ok=True)
        images = data_pipeline.load_images_from_folder(args.input_dir)
        for i, img in enumerate(images):
            thresh = segmentation.adaptive_threshold(img)
            cleaned = segmentation.morphological_cleaning(thresh)
            contours = segmentation.find_contours(cleaned)
            mask = np.zeros_like(img, dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            out_path = os.path.join(args.output_dir, f"mask_{i:04d}.png")
            cv2.imwrite(out_path, mask)
            logger.info(f"Saved segmented mask to {out_path}")

    elif args.command == 'evaluate':
        pred_images = data_pipeline.load_images_from_folder(args.pred_dir)
        gt_images = data_pipeline.load_images_from_folder(args.gt_dir)
        precision, recall, f1 = evaluate.precision_recall_f1(gt_images, pred_images)
        iou_score = np.mean([evaluate.iou(gt, pred) for gt, pred in zip(gt_images, pred_images)])
        report = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "iou": iou_score
        }
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=4)
        evaluate.plot_metrics(precision, recall, f1, iou_score, args.output_plot)
        logger.info(f"Saved evaluation report to {args.output_report}")

    elif args.command == 'train_model':
        logger.info("Model training is not yet implemented.")
        # Placeholder for model training implementation

if __name__ == "__main__":
    main()
