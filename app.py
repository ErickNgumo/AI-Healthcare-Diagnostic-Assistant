# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================

import json
import warnings

warnings.filterwarnings('ignore')

# Import all modules
from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.knowledge_base import MedicalKnowledgeBase
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.planner import TreatmentPlanner


class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def banner():
    print(f"""
{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════════╗
║          INTELLIGENT HEALTHCARE DIAGNOSTIC AI            ║
║         Introduction to AI — Capstone Project            ║
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║
╚══════════════════════════════════════════════════════════╝
{C.END}""")


def section(title: str):
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")


def build_system() -> HealthcareDiagnosticAgent:
    """Instantiate and wire all AI modules to the intelligent agent."""
    section("Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()
    print("\n  Initializing modules...")

    module_registry = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet': SimpleBayesianDiagnostics(),
        'MLClassifier': MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'FuzzySeverity': FuzzySeverityAssessor(),
        'TreatmentPlanner': TreatmentPlanner(),
    }

    for name, module in module_registry.items():
        agent.register_module(name, module)

    return agent


def demo_patient() -> PatientPercept:
    return PatientPercept(
        patient_id='P001',
        symptoms=['fever', 'cough', 'fatigue', 'loss_of_smell'],
        age=34,
        temperature=38.9,
        heart_rate=98,
        blood_pressure='120/80',
    )


def main():
    banner()
    agent = build_system()
    patient = demo_patient()

    report = agent.run(patient)
    print(json.dumps(report, indent=2))
    agent.print_log()


if __name__ == '__main__':
    main()
