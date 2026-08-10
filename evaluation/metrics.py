"""Metrics for multiclass diagnostic models, calculated from genuine predictions."""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)


def calculate_metrics(y_true, y_pred, probabilities=None,
                      class_labels: Optional[Sequence[str]] = None) -> Dict:
    """Return per-class, macro, weighted, and applicable multiclass ROC-AUC metrics."""
    labels = np.arange(len(class_labels)) if class_labels is not None else None
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_labels,
        output_dict=True, zero_division=0,
    )
    result = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'per_class': report,
        'roc_auc_ovr_weighted': None,
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(y_true):
            raise ValueError('probabilities must have one row per true label.')
        try:
            result['roc_auc_ovr_weighted'] = float(roc_auc_score(
                y_true, probabilities, multi_class='ovr', average='weighted', labels=labels,
            ))
        except ValueError:
            # ROC-AUC is undefined when a test split lacks a class or valid scores.
            result['roc_auc_ovr_weighted'] = None
    return result


def evaluate_model(model, X_test, y_test, class_labels: Optional[Sequence[str]] = None) -> Dict:
    """Evaluate a scikit-learn estimator or Keras model on supplied held-out data."""
    raw_predictions = np.asarray(model.predict(X_test, verbose=0) if hasattr(model, 'optimizer')
                                 else model.predict(X_test))
    probabilities = raw_predictions if raw_predictions.ndim == 2 else None
    predictions = np.argmax(raw_predictions, axis=1) if probabilities is not None else raw_predictions
    if probabilities is None and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)
    return calculate_metrics(y_test, predictions, probabilities, class_labels)


def compare_models(model_results: Mapping[str, Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    """Normalise actual model metric dictionaries for tables/charts; never invent values."""
    fields = ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted')
    return {name: {field: metrics.get(field) for field in fields}
            for name, metrics in model_results.items()}
