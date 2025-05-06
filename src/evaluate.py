"""
Evaluation module for computing precision, recall, F1-score, and IoU.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precision_recall_f1(true_masks: List[np.ndarray], pred_masks: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score for predicted masks.

    Args:
        true_masks (List[np.ndarray]): List of ground truth binary masks.
        pred_masks (List[np.ndarray]): List of predicted binary masks.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1-score.
    """
    tp = 0
    fp = 0
    fn = 0
    for true_mask, pred_mask in zip(true_masks, pred_masks):
        tp += np.logical_and(true_mask, pred_mask).sum()
        fp += np.logical_and(np.logical_not(true_mask), pred_mask).sum()
        fn += np.logical_and(true_mask, np.logical_not(pred_mask)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    logger.info(f"Computed precision: {precision:.4f}, recall: {recall:.4f}, F1-score: {f1:.4f}")
    return precision, recall, f1

def iou(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) for a single mask pair.

    Args:
        true_mask (np.ndarray): Ground truth binary mask.
        pred_mask (np.ndarray): Predicted binary mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    iou_score = intersection / (union + 1e-8)
    logger.info(f"Computed IoU: {iou_score:.4f}")
    return iou_score

def plot_metrics(precision: float, recall: float, f1: float, iou_score: float, output_path: str) -> None:
    """
    Plot evaluation metrics and save the figure.

    Args:
        precision (float): Precision score.
        recall (float): Recall score.
        f1 (float): F1-score.
        iou_score (float): IoU score.
        output_path (str): Path to save the plot.
    """
    metrics = {'Precision': precision, 'Recall': recall, 'F1-score': f1, 'IoU': iou_score}
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylim(0, 1)
    plt.title('Segmentation Evaluation Metrics')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved evaluation metrics plot to {output_path}")
