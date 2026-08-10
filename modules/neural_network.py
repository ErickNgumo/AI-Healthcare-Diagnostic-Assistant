# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks)
# ============================================================

"""TensorFlow/Keras diagnostic classifier using the project's symptom schema."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models


class NeuralDiagnosticModel:
    """Deep neural-network specialist used by :class:`HealthcareDiagnosticAgent`.

    The model deliberately shares Module 4's generated patient records and fixed
    18-symptom vocabulary.  Records are split into independent train, validation,
    and test partitions before fitting, so test data is never used by Keras.
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache', 'body_aches',
        'loss_of_smell', 'chest_pain', 'rash', 'joint_pain',
        'shortness_of_breath', 'sweating', 'frequent_urination',
        'excessive_thirst', 'blurred_vision', 'night_sweats',
        'weight_loss', 'stiff_neck', 'light_sensitivity',
    ]
    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event', 'diabetes',
        'common_cold', 'tuberculosis', 'meningitis',
    ]
    # Kept in sync with Module 4's synthetic data-generation probabilities.
    DISEASE_PROFILES = {
        'flu': {'fever': .90, 'cough': .85, 'fatigue': .88, 'headache': .70,
                'body_aches': .80, 'loss_of_smell': .20},
        'covid19': {'fever': .88, 'cough': .80, 'fatigue': .90,
                    'loss_of_smell': .85, 'headache': .65, 'body_aches': .60},
        'dengue': {'fever': .98, 'rash': .75, 'joint_pain': .85,
                   'headache': .90, 'fatigue': .80, 'body_aches': .88},
        'cardiac_event': {'chest_pain': .92, 'shortness_of_breath': .88,
                          'fatigue': .70, 'sweating': .75, 'headache': .30},
        'diabetes': {'fatigue': .82, 'frequent_urination': .95,
                     'excessive_thirst': .92, 'blurred_vision': .70,
                     'weight_loss': .50},
        'common_cold': {'cough': .90, 'fever': .50, 'headache': .60,
                        'fatigue': .55, 'body_aches': .50},
        'tuberculosis': {'cough': .95, 'weight_loss': .85, 'night_sweats': .80,
                         'fatigue': .88, 'fever': .70},
        'meningitis': {'headache': .95, 'stiff_neck': .90, 'fever': .92,
                      'light_sensitivity': .85, 'fatigue': .80},
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.is_trained = False
        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the 18 → 128 → 64 → 32 → 8 lab-specified MLP."""
        tf.keras.utils.set_random_seed(self.random_state)
        self.model = models.Sequential([
            layers.Input(shape=(len(self.SYMPTOM_FEATURES),)),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.20),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(len(self.DISEASE_LABELS), activation='softmax'),
        ], name='MedicalDNN')
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy', metrics=['accuracy'],
        )

    def _generate_data(self, n: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate records with Module 4's established feature conventions."""
        if n < len(self.DISEASE_LABELS):
            raise ValueError(f"n must be at least {len(self.DISEASE_LABELS)}.")
        rng = np.random.default_rng(self.random_state)
        samples_per_class = n // len(self.DISEASE_LABELS)
        rows, labels = [], []
        for label_index, disease in enumerate(self.DISEASE_LABELS):
            probabilities = self.DISEASE_PROFILES[disease]
            for _ in range(samples_per_class):
                row = np.asarray([
                    float(rng.random() < probabilities.get(feature, .05))
                    for feature in self.SYMPTOM_FEATURES
                ], dtype=np.float32)
                rows.append(row)
                labels.append(label_index)
        indices = rng.permutation(len(rows))
        X = np.asarray(rows, dtype=np.float32)[indices]
        y = np.asarray(labels, dtype=np.int64)[indices]
        return X, y

    def prepare_data(self, n_samples: int = 3000) -> Tuple[np.ndarray, ...]:
        """Create stratified 60%/20%/20% train/validation/test partitions."""
        X, y = self._generate_data(n_samples)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=0.40, random_state=self.random_state, stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout, test_size=0.50,
            random_state=self.random_state, stratify=y_holdout,
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, epochs: int = 50, verbose: int = 1,
              n_samples: int = 3000, batch_size: int = 64) -> Dict:
        """Fit with validation-only callbacks; retain the untouched test partition."""
        if epochs < 1:
            raise ValueError('epochs must be at least 1.')
        self.prepare_data(n_samples)
        callback_list = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=5, min_lr=1e-6),
        ]
        self.history = self.model.fit(
            self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callback_list,
            verbose=verbose,
        )
        self.is_trained = True
        return {
            'val_accuracy': float(max(self.history.history['val_accuracy'])),
            'epochs_trained': len(self.history.history['loss']),
            'train_size': len(self.y_train), 'validation_size': len(self.y_val),
            'test_size': len(self.y_test),
        }

    @staticmethod
    def _normalise_symptom(symptom: object) -> str:
        return str(symptom).strip().lower().replace(' ', '_').replace('-', '_')

    def _symptoms_to_features(self, symptoms: Sequence[str]) -> np.ndarray:
        if isinstance(symptoms, (str, bytes)) or not isinstance(symptoms, Sequence):
            raise TypeError('symptoms must be a sequence of symptom strings.')
        supplied = {self._normalise_symptom(symptom) for symptom in symptoms}
        return np.asarray(
            [[float(feature in supplied) for feature in self.SYMPTOM_FEATURES]],
            dtype=np.float32,
        )

    def predict(self, symptoms: Sequence[str]) -> Dict:
        """Return the softmax diagnosis and its genuine probability distribution."""
        if not self.is_trained:
            self.train(verbose=0)
        features = self._symptoms_to_features(symptoms)
        probabilities = self.model.predict(features, verbose=0)[0]
        prediction_index = int(np.argmax(probabilities))
        probability_map = {
            label: float(probability)
            for label, probability in zip(self.DISEASE_LABELS, probabilities)
        }
        return {
            'diagnosis': self.DISEASE_LABELS[prediction_index],
            'confidence': float(probabilities[prediction_index]),
            'all_probs': probability_map,
            'symptom_vector': features[0].tolist(),
        }

    def analyze(self, patient_percept) -> Dict:
        """Agent-compatible diagnostic entry point accepting ``PatientPercept``."""
        if not hasattr(patient_percept, 'symptoms'):
            raise TypeError('analyze() expects an object with a symptoms attribute.')
        result = self.predict(patient_percept.symptoms)
        result['summary'] = f"DNN: {result['diagnosis']} ({result['confidence']:.2%})"
        return result

    def evaluate(self, output_dir: Optional[Union[str, Path]] = None,
                 generate_plots: bool = False) -> Dict:
        """Calculate held-out test metrics from real predictions and softmax scores."""
        if not self.is_trained:
            self.train(verbose=0)
        probabilities = self.model.predict(self.X_test, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        from evaluation.metrics import calculate_metrics

        result = calculate_metrics(
            self.y_test, predictions, probabilities, self.DISEASE_LABELS,
        )
        if generate_plots:
            self.plot_evaluation(output_dir=output_dir)
        return result

    def plot_training(self, output_dir: Union[str, Path] = 'reports') -> Path:
        """Save actual training/validation accuracy and loss curves."""
        if self.history is None:
            raise RuntimeError('Train the model before plotting its history.')
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, metric, title in (
            (axes[0], 'accuracy', 'Accuracy'), (axes[1], 'loss', 'Loss'),
        ):
            axis.plot(self.history.history[metric], label='Training')
            axis.plot(self.history.history[f'val_{metric}'], label='Validation')
            axis.set(title=f'Model {title}', xlabel='Epoch', ylabel=title)
            axis.grid(alpha=0.3)
            axis.legend()
        figure.suptitle('Neural Network Training Curves')
        figure.tight_layout()
        path = destination / 'nn_training.png'
        figure.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(figure)
        return path

    def plot_evaluation(self, output_dir: Union[str, Path] = 'reports') -> Path:
        """Save the confusion matrix calculated from held-out test predictions."""
        if not self.is_trained:
            self.train(verbose=0)
        probabilities = self.model.predict(self.X_test, verbose=0)
        from evaluation.visualizations import plot_confusion_matrix

        return plot_confusion_matrix(
            self.y_test, np.argmax(probabilities, axis=1), self.DISEASE_LABELS,
            title='Deep Neural Network Confusion Matrix', output_dir=output_dir,
            filename='nn_confusion_matrix.png',
        )
