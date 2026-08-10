# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks)
# ============================================================

"""Naïve-Bayes medical diagnostic demonstration used by the agent."""

import math
from typing import Dict, Iterable, List


class SimpleBayesianDiagnostics:
    """Compute disease posteriors from symptom evidence in log space."""

    UNKNOWN_SYMPTOM_LIKELIHOOD = 0.01

    def __init__(self):
        # Priors P(disease), as supplied in the capstone lab manual.
        self.priors = {
            "flu": 0.15, "covid19": 0.08, "dengue": 0.05,
            "cardiac": 0.04, "diabetes": 0.10, "common_cold": 0.30,
            "healthy": 0.28,
        }
        # Likelihoods P(symptom | disease).
        self.likelihoods = {
            "flu": {"fever": 0.90, "cough": 0.85, "fatigue": 0.88, "headache": 0.70, "body_aches": 0.80, "loss_of_smell": 0.20, "chest_pain": 0.05, "rash": 0.05, "joint_pain": 0.40},
            "covid19": {"fever": 0.88, "cough": 0.80, "fatigue": 0.90, "loss_of_smell": 0.85, "headache": 0.65, "body_aches": 0.60, "chest_pain": 0.20, "rash": 0.05, "joint_pain": 0.20},
            "dengue": {"fever": 0.98, "rash": 0.75, "joint_pain": 0.85, "headache": 0.90, "fatigue": 0.80, "cough": 0.15, "loss_of_smell": 0.05, "chest_pain": 0.05, "body_aches": 0.88},
            "cardiac": {"chest_pain": 0.92, "shortness_of_breath": 0.88, "fatigue": 0.70, "sweating": 0.75, "fever": 0.10, "cough": 0.15, "rash": 0.02, "joint_pain": 0.10, "headache": 0.30},
            "diabetes": {"fatigue": 0.82, "frequent_urination": 0.95, "excessive_thirst": 0.92, "blurred_vision": 0.70, "fever": 0.10, "cough": 0.05, "rash": 0.08, "headache": 0.40, "joint_pain": 0.20},
            "common_cold": {"cough": 0.90, "fever": 0.50, "headache": 0.60, "fatigue": 0.55, "body_aches": 0.50, "loss_of_smell": 0.30, "rash": 0.02, "chest_pain": 0.05, "joint_pain": 0.15},
            "healthy": {"fever": 0.02, "cough": 0.05, "fatigue": 0.10, "headache": 0.08, "rash": 0.01, "chest_pain": 0.01, "joint_pain": 0.05, "loss_of_smell": 0.01, "body_aches": 0.05},
        }

    @staticmethod
    def _normalise_symptoms(symptoms: Iterable[str]) -> List[str]:
        if symptoms is None:
            raise TypeError("Symptoms must be an iterable of strings.")
        normalised = []
        for symptom in symptoms:
            if not isinstance(symptom, str):
                raise TypeError("Symptoms must be strings.")
            clean = "_".join(symptom.strip().lower().replace("-", " ").split())
            if clean and clean not in normalised:
                normalised.append(clean)
        return normalised

    def compute_posterior(self, symptoms: Iterable[str]) -> Dict[str, float]:
        """Return normalized ``P(disease | symptoms)`` probabilities.

        The lab specification uses present symptoms as evidence and a small
        likelihood floor for unlisted symptoms.  Log probabilities avoid numeric
        underflow when several symptoms are supplied.
        """
        symptoms_clean = self._normalise_symptoms(symptoms)
        log_scores: Dict[str, float] = {}
        for disease, prior in self.priors.items():
            log_score = math.log(prior)
            for symptom in symptoms_clean:
                likelihood = self.likelihoods[disease].get(
                    symptom, self.UNKNOWN_SYMPTOM_LIKELIHOOD
                )
                log_score += math.log(max(likelihood, self.UNKNOWN_SYMPTOM_LIKELIHOOD))
            log_scores[disease] = log_score

        max_log_score = max(log_scores.values())
        weights = {
            disease: math.exp(score - max_log_score)
            for disease, score in log_scores.items()
        }
        total = sum(weights.values())
        return {disease: probability / total for disease, probability in weights.items()}

    def analyze(self, percept) -> Dict:
        """Provide the standard diagnostic result expected by the agent."""
        if not hasattr(percept, "symptoms"):
            raise TypeError("analyze() expects a patient percept with symptoms.")
        posteriors = self.compute_posterior(percept.symptoms)
        ranked = sorted(posteriors.items(), key=lambda item: item[1], reverse=True)
        diagnosis, confidence = ranked[0]
        rounded_posteriors = {disease: round(probability, 4) for disease, probability in posteriors.items()}

        return {
            "summary": f"Top: {diagnosis} ({confidence:.2%})",
            "diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "all_posteriors": rounded_posteriors,
            "ranked_diagnoses": [(disease, round(probability, 4)) for disease, probability in ranked[:5]],
        }

    def explain(self, disease: str, symptoms: Iterable[str]) -> str:
        """Show the likelihood terms used for a disease's posterior score."""
        if disease not in self.priors:
            raise ValueError(f"Unknown disease: {disease}")
        evidence = [
            f"P({symptom}|{disease})={self.likelihoods[disease].get(symptom, self.UNKNOWN_SYMPTOM_LIKELIHOOD):.2f}"
            for symptom in self._normalise_symptoms(symptoms)
        ]
        return f"P({disease}) = {self.priors[disease]:.2f}" + (
            " × " + " × ".join(evidence) if evidence else ""
        )
