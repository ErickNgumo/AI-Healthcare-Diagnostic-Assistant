# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from collections import Counter
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import datetime


class AgentState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting_symptoms"
    DIAGNOSING = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING = "planning_treatment"
    DONE = "done"


@dataclass
class PatientPercept:
    """What the agent perceives from the environment."""
    patient_id: str
    symptoms: List[str]
    age: int
    temperature: float
    heart_rate: int
    blood_pressure: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


@dataclass
class AgentMemory:
    """Internal model — makes this a model-based agent."""
    patient_history: List[Dict] = field(default_factory=list)
    current_patient: Optional[PatientPercept] = None
    diagnosis_history: List[Dict] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)


class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    ─────────────────────────────────────────────────
    Performance : Diagnostic accuracy, patient safety,
                  recommendation quality, response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan,
                  referral recommendation, alerts
    Sensors     : Symptom input, vitals, lab results,
                  patient history
    ─────────────────────────────────────────────────
    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.performance_score = 0
        self._modules = {}
        self.last_results = {}
        self.last_report = {}

    def register_module(self, name: str, module):
        """Plug in AI sub-modules (KB, Bayes, ML, etc.)"""
        self._modules[name] = module
        print(f"  Module registered: [{name}]")

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment."""
        if not isinstance(percept, PatientPercept):
            raise TypeError(
                "perceive() expects a PatientPercept instance, not a dictionary."
            )
        if not percept.patient_id:
            raise ValueError("Patient record is missing a patient_id.")

        self.memory.current_patient = percept
        self.memory.patient_history.append({
            'id': percept.patient_id,
            'symptoms': list(percept.symptoms),
            'age': percept.age,
            'temperature': percept.temperature,
            'heart_rate': percept.heart_rate,
            'blood_pressure': percept.blood_pressure,
            'time': percept.timestamp,
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} with {len(percept.symptoms)} symptoms")
        return self

    def think(self):
        """Step 2: Process and reason across all registered modules."""
        if self.memory.current_patient is None:
            raise ValueError("think() called before perceive().")

        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}
        for module_name, module in self._modules.items():
            if module is None:
                continue
            if not hasattr(module, 'analyze'):
                self._log(f"  [{module_name}] skipped: no analyze() method")
                continue
            try:
                raw_result = module.analyze(self.memory.current_patient)
                if raw_result is None:
                    raise ValueError("analyze() returned None")
                if not isinstance(raw_result, dict):
                    raw_result = {
                        'diagnosis': str(raw_result),
                        'confidence': 0.0,
                        'summary': 'Unexpected module result format',
                    }
                results[module_name] = raw_result
                summary = raw_result.get('summary') or raw_result.get('diagnosis') or 'completed'
                self._log(f"  [{module_name}] → {summary}")
            except Exception as exc:
                self._log(f"  [{module_name}] failed: {exc}")
                results[module_name] = {
                    'diagnosis': 'Unavailable',
                    'confidence': 0.0,
                    'summary': f'{module_name} module error',
                    'error': str(exc),
                }

        if not results:
            raise RuntimeError("No registered modules available for diagnosis.")

        self.last_results = results
        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Optional[Dict] = None) -> Dict:
        """Step 3: Generate a structured action report."""
        if self.memory.current_patient is None:
            raise ValueError("act() called before perceive().")
        if diagnosis_results is None:
            if not self.last_results:
                raise ValueError("act() called before think().")
            diagnosis_results = self.last_results

        if not diagnosis_results:
            raise ValueError("No diagnosis results were produced to act on.")

        self.state = AgentState.PLANNING
        patient = self.memory.current_patient
        avg_confidence = self._average_confidence(diagnosis_results)
        urgency = self._assess_urgency(patient, avg_confidence, diagnosis_results)
        diagnosis = self._aggregate_diagnosis(diagnosis_results)
        recommendations = self._generate_recommendations(urgency, diagnosis_results)

        report = {
            'patient_id': patient.patient_id,
            'timestamp': patient.timestamp,
            'symptoms': patient.symptoms,
            'diagnosis': diagnosis,
            'confidence': round(avg_confidence, 3),
            'urgency': urgency,
            'recommendations': recommendations,
            'next_action': self._decide_next_action(urgency),
        }

        self.performance_score += 10 if avg_confidence > 0.7 else 5
        self.state = AgentState.DONE
        self.last_report = report
        self._log(f"Action generated: {urgency} urgency")
        return report

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle: Perceive → Think → Act."""
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    def _average_confidence(self, diagnosis_results: Dict) -> float:
        confidences = []
        for value in diagnosis_results.values():
            if not isinstance(value, dict):
                continue
            score = value.get('confidence')
            if score is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            confidences.append(max(0.0, min(1.0, score)))
        if not confidences:
            return 0.5
        return sum(confidences) / len(confidences)

    def _assess_urgency(self, patient: PatientPercept, confidence: float, diagnosis_results: Dict) -> str:
        diagnosis_text = ' '.join(
            str(v.get('diagnosis', '')).lower()
            for v in diagnosis_results.values()
            if isinstance(v, dict)
        )

        if patient.temperature >= 39.5 or patient.heart_rate >= 120 or 'emergency' in diagnosis_text:
            return "CRITICAL"
        if patient.temperature >= 38.5 or patient.heart_rate >= 100 or confidence >= 0.8:
            return "HIGH"
        if patient.temperature >= 37.5 or len(patient.symptoms) >= 4:
            return "MEDIUM"
        return "LOW"

    def _aggregate_diagnosis(self, results: Dict) -> str:
        diagnoses = [
            v.get('diagnosis', 'Unknown')
            for v in results.values()
            if isinstance(v, dict) and v.get('diagnosis') not in (None, 'Unavailable', '')
        ]
        if not diagnoses:
            return "Insufficient data"
        return Counter(diagnoses).most_common(1)[0][0]

    def _generate_recommendations(self, urgency: str, results: Dict) -> List[str]:
        if urgency == "CRITICAL":
            recommendations = [
                "Immediate emergency consultation required",
                "Alert the attending physician immediately",
                "Transfer to emergency or intensive monitoring if needed",
            ]
        elif urgency == "HIGH":
            recommendations = [
                "Schedule an urgent medical review within 24 hours",
                "Order relevant diagnostic testing and monitoring",
                "Continue close observation of vital signs",
            ]
        elif urgency == "MEDIUM":
            recommendations = [
                "Schedule a follow-up within 3 days",
                "Monitor temperature and symptoms daily",
                "Maintain hydration and rest",
            ]
        else:
            recommendations = [
                "Home rest and hydration are appropriate",
                "Follow up if symptoms worsen or new signs develop",
                "Continue general wellness monitoring",
            ]

        planner = self._modules.get('TreatmentPlanner')
        if planner is not None and hasattr(planner, 'create_treatment_plan'):
            diagnosis = self._aggregate_diagnosis(results)
            plan = planner.create_treatment_plan(diagnosis, urgency)
            if isinstance(plan, dict) and 'plan' in plan and plan['plan']:
                plan_steps = [
                    f"Step {step['step']}: {step['action']}"
                    for step in plan['plan'][:3]
                ]
                recommendations.extend(plan_steps)
        return recommendations

    def _decide_next_action(self, urgency: str) -> str:
        actions = {
            "CRITICAL": "EMERGENCY_REFERRAL",
            "HIGH": "URGENT_APPOINTMENT",
            "MEDIUM": "SCHEDULE_FOLLOWUP",
            "LOW": "MONITOR_AT_HOME",
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message: str):
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)

    def print_log(self):
        print("\nAgent Action Log:")
        print("─" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self):
        return {
            'total_patients': len(self.memory.patient_history),
            'performance_score': self.performance_score,
            'diagnoses_made': len(self.memory.diagnosis_history),
        }
