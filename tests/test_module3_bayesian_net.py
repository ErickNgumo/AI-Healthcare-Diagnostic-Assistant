from types import SimpleNamespace

import pytest

from modules.bayesian_net import SimpleBayesianDiagnostics


def test_posteriors_are_normalized_and_covid_is_top_for_lab_example():
    diagnostics = SimpleBayesianDiagnostics()

    posteriors = diagnostics.compute_posterior(
        ["Fever", "cough", "loss of smell", "fatigue"]
    )

    assert sum(posteriors.values()) == pytest.approx(1.0)
    assert max(posteriors, key=posteriors.get) == "covid19"
    assert posteriors["covid19"] > posteriors["flu"]


def test_unknown_symptom_uses_shared_likelihood_floor():
    diagnostics = SimpleBayesianDiagnostics()

    posteriors = diagnostics.compute_posterior(["unlisted symptom"])

    for disease, prior in diagnostics.priors.items():
        assert posteriors[disease] == pytest.approx(prior)


def test_analyze_returns_ranked_agent_compatible_result():
    diagnostics = SimpleBayesianDiagnostics()
    patient = SimpleNamespace(
        symptoms=["frequent urination", "excessive thirst", "blurred vision"]
    )

    result = diagnostics.analyze(patient)

    assert result["diagnosis"] == "diabetes"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["ranked_diagnoses"][0][0] == result["diagnosis"]
    assert len(result["ranked_diagnoses"]) == 5
