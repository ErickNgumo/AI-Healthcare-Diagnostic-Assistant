"""Report-ready plots created from model evaluation outputs."""

from pathlib import Path
from typing import Mapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_labels: Sequence[str], *,
                          title: str = 'Confusion Matrix',
                          output_dir: Union[str, Path] = 'reports',
                          filename: str = 'confusion_matrix.png') -> Path:
    """Save a labelled matrix of actual test labels against actual predictions."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels)))
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=class_labels, yticklabels=class_labels, ax=axis)
    axis.set(title=title, xlabel='Predicted', ylabel='Actual')
    plt.setp(axis.get_xticklabels(), rotation=45, ha='right')
    figure.tight_layout()
    path = destination / filename
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return path


def plot_model_comparison(results: Mapping[str, Mapping[str, float]], *,
                          output_dir: Union[str, Path] = 'reports',
                          filename: str = 'model_comparison.png') -> Path:
    """Save a grouped comparison chart from actual, already-calculated model metrics."""
    metrics = ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted')
    names = list(results)
    x_values = np.arange(len(names))
    width = 0.15
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(metrics):
        values = [results[name].get(metric) for name in names]
        values = [np.nan if value is None else value for value in values]
        axis.bar(x_values + (index - 2) * width, values, width, label=metric.replace('_weighted', ''))
    axis.set(title='Diagnostic Model Performance', ylabel='Score', ylim=(0, 1.05))
    axis.set_xticks(x_values, names, rotation=20, ha='right')
    axis.legend()
    axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return path
