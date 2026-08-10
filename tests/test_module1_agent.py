import pytest

from modules.agent import AgentState, HealthcareDiagnosticAgent, PatientPercept
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.knowledge_base import MedicalKnowledgeBase


@pytest.fixture
def sample_patient():
    return PatientPercept(
        patient_id='P001',
        symptoms=['fever', 'cough', 'fatigue', 'loss_of_smell'],
        age=34,
        temperature=38.9,
        heart_rate=98,
        blood_pressure='120/80',
    )


def test_perceive_stores_patient_and_updates_state(sample_patient):
    agent = HealthcareDiagnosticAgent()

    result = agent.perceive(sample_patient)

    assert result is agent
    assert agent.memory.current_patient == sample_patient
    assert len(agent.memory.patient_history) == 1
    assert agent.memory.patient_history[0]['id'] == 'P001'
    assert agent.memory.patient_history[0]['time']
    assert agent.state == AgentState.COLLECTING


def test_think_calls_registered_module_analyze(sample_patient):
    class DummyModule:
        def __init__(self, name):
            self.name = name
            self.calls = []

        def analyze(self, patient_data):
            self.calls.append(patient_data.patient_id)
            return {'summary': f'{self.name} summary', 'diagnosis': self.name, 'confidence': 0.6}

    agent = HealthcareDiagnosticAgent()
    first = DummyModule('alpha')
    second = DummyModule('beta')
    agent.register_module('Alpha', first)
    agent.register_module('Beta', second)
    agent.perceive(sample_patient)

    results = agent.think()

    assert results['Alpha']['diagnosis'] == 'alpha'
    assert results['Beta']['diagnosis'] == 'beta'
    assert first.calls == ['P001']
    assert second.calls == ['P001']
    assert agent.state == AgentState.RECOMMENDING


def test_act_generates_structured_report(sample_patient):
    agent = HealthcareDiagnosticAgent()
    agent.register_module('KnowledgeBase', MedicalKnowledgeBase())
    agent.register_module('BayesianNet', SimpleBayesianDiagnostics())
    agent.register_module('Severity', FuzzySeverityAssessor())
    agent.perceive(sample_patient)
    results = agent.think()
    report = agent.act(results)

    assert 'patient_id' in report
    assert report['patient_id'] == 'P001'
    assert isinstance(report['diagnosis'], str)
    assert 0.0 <= report['confidence'] <= 1.0
    assert report['urgency'] in {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}
    assert isinstance(report['recommendations'], list)
    assert report['recommendations']
    assert 'next_action' in report


def test_full_agent_flow_with_lab_example(sample_patient):
    agent = HealthcareDiagnosticAgent()
    agent.register_module('KnowledgeBase', MedicalKnowledgeBase())
    agent.register_module('BayesianNet', SimpleBayesianDiagnostics())
    agent.register_module('Severity', FuzzySeverityAssessor())

    report = agent.run(sample_patient)

    assert report['patient_id'] == 'P001'
    assert report['urgency'] in {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}
    assert report['diagnosis']
    assert report['confidence'] >= 0.0
    assert report['recommendations']
    assert agent.state == AgentState.DONE
