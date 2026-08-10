from types import SimpleNamespace

import pytest

from modules.knowledge_base import MedicalKnowledgeBase


def test_forward_chain_derives_covid_and_follow_up_action():
    kb = MedicalKnowledgeBase()
    kb.load_patient_symptoms(
        ["Fever", "cough", "loss of smell", "fatigue", "positive PCR"]
    )

    inferred = kb.forward_chain()

    assert inferred["covid19_suspected"] == pytest.approx(0.85)
    assert inferred["covid19_confirmed"] == pytest.approx(0.8415)
    assert inferred["isolate_and_treat"] == pytest.approx(0.8331)


def test_backward_chain_proves_goal_with_rule_certainty():
    kb = MedicalKnowledgeBase()
    kb.load_patient_symptoms(["fever", "cough", "loss_of_smell", "fatigue"])

    proved, certainty = kb.backward_chain("covid19_suspected")

    assert proved is True
    assert certainty == pytest.approx(0.85)
    assert "covid19_suspected" not in kb.facts


def test_analyze_uses_vital_fever_and_clears_previous_patient_facts():
    kb = MedicalKnowledgeBase()
    flu_patient = SimpleNamespace(
        symptoms=["cough", "fatigue"], temperature=38.5, heart_rate=80
    )
    healthy_patient = SimpleNamespace(symptoms=[], temperature=36.8, heart_rate=72)

    flu_result = kb.analyze(flu_patient)
    healthy_result = kb.analyze(healthy_patient)

    assert flu_result["diagnosis"] == "flu_suspected"
    assert flu_result["confidence"] == pytest.approx(0.375)
    assert healthy_result["diagnosis"] == "Unknown"
    assert "flu_suspected" not in kb.facts


def test_rule_and_fact_validation():
    kb = MedicalKnowledgeBase()

    with pytest.raises(ValueError):
        kb.add_rule([], "invalid")
    with pytest.raises(ValueError):
        kb.add_fact("fever", 1.1)
