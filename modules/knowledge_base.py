# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

"""Rule-based medical reasoning used by the diagnostic agent.

This module is an educational demonstration of first-order-style rules and
certainty factors.  Its results are decision support, not a clinical diagnosis.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple


Rule = Tuple[List[str], str, float]


class MedicalKnowledgeBase:
    """Medical facts, rules, and forward/backward inference methods."""

    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Rule] = []
        self.certainty_factors: Dict[str, float] = {}
        self._load_medical_knowledge()

    @staticmethod
    def _normalise_fact(fact: str) -> str:
        """Convert user-facing symptom text to the rule vocabulary."""
        if not isinstance(fact, str):
            raise TypeError("Facts and symptoms must be strings.")
        return "_".join(fact.strip().lower().replace("-", " ").split())

    @staticmethod
    def _validate_certainty(certainty: float) -> float:
        try:
            certainty = float(certainty)
        except (TypeError, ValueError) as exc:
            raise TypeError("Certainty factors must be numeric.") from exc
        if not 0.0 <= certainty <= 1.0:
            raise ValueError("Certainty factors must be between 0.0 and 1.0.")
        return certainty

    def _load_medical_knowledge(self) -> None:
        """Load the domain rules supplied in the capstone lab manual."""
        disease_rules: List[Rule] = [
            (["fever", "cough", "fatigue"], "flu_suspected", 0.75),
            (["fever", "cough", "loss_of_smell", "fatigue"], "covid19_suspected", 0.85),
            (["fever", "rash", "joint_pain"], "dengue_suspected", 0.80),
            (["chest_pain", "shortness_of_breath", "sweating"], "cardiac_event_suspected", 0.90),
            (["headache", "stiff_neck", "high_fever", "light_sensitivity"], "meningitis_suspected", 0.88),
            (["cough", "weight_loss", "night_sweats", "fatigue"], "tuberculosis_suspected", 0.82),
            (["frequent_urination", "excessive_thirst", "blurred_vision"], "diabetes_suspected", 0.78),
            (["flu_suspected", "high_fever"], "flu_confirmed", 0.85),
            (["covid19_suspected", "positive_pcr"], "covid19_confirmed", 0.99),
            (["cardiac_event_suspected", "elevated_troponin"], "myocardial_infarction", 0.95),
            (["myocardial_infarction"], "emergency", 1.00),
            (["meningitis_suspected"], "emergency", 0.95),
            (["covid19_confirmed"], "isolate_and_treat", 0.99),
            (["flu_confirmed"], "rest_and_medicate", 0.90),
        ]
        for conditions, conclusion, certainty in disease_rules:
            self.add_rule(conditions, conclusion, certainty)

    def add_fact(self, fact: str, certainty: float = 1.0) -> None:
        """Store a fact, retaining the strongest available certainty factor."""
        fact = self._normalise_fact(fact)
        certainty = self._validate_certainty(certainty)
        self.facts.add(fact)
        self.certainty_factors[fact] = max(
            certainty, self.certainty_factors.get(fact, 0.0)
        )

    def add_rule(
        self, conditions: Iterable[str], conclusion: str, certainty: float = 1.0
    ) -> None:
        """Store a rule as ``conditions -> conclusion`` with a certainty factor."""
        conditions = [self._normalise_fact(condition) for condition in conditions]
        if not conditions:
            raise ValueError("A rule must have at least one condition.")
        self.rules.append(
            (conditions, self._normalise_fact(conclusion), self._validate_certainty(certainty))
        )

    def load_patient_symptoms(self, symptoms: Iterable[str]) -> None:
        """Load observed symptoms as certain facts after normalising their names."""
        if symptoms is None:
            raise TypeError("Symptoms must be an iterable of strings.")
        for symptom in symptoms:
            self.add_fact(symptom)

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """Infer all reachable conclusions using certainty-factor propagation.

        A rule fires when all of its conditions are known.  The conclusion CF is
        ``rule CF × min(condition CFs)``, exactly as specified in the lab manual.
        Conclusions become facts so that multi-step rules can fire in later loops.
        """
        inferred: Dict[str, float] = {}
        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1
            for conditions, conclusion, rule_cf in self.rules:
                if not all(condition in self.facts for condition in conditions):
                    continue

                combined_cf = round(
                    rule_cf * min(self.certainty_factors[condition] for condition in conditions),
                    4,
                )
                previous_cf = self.certainty_factors.get(conclusion, 0.0)
                if conclusion in self.facts and combined_cf <= previous_cf:
                    continue

                self.add_fact(conclusion, combined_cf)
                inferred[conclusion] = self.certainty_factors[conclusion]
                if verbose:
                    print(
                        f"  Iter {iteration}: {' ∧ '.join(conditions)} → "
                        f"{conclusion} (CF={combined_cf:.3f})"
                    )
                changed = True
        return inferred

    def backward_chain(
        self, goal: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> Tuple[bool, float]:
        """Recursively prove a goal and return its best supporting certainty."""
        del depth  # Kept for compatibility with the lab's recursive signature.
        goal = self._normalise_fact(goal)
        visited = set() if visited is None else set(visited)

        if goal in self.facts:
            return True, self.certainty_factors.get(goal, 1.0)
        if goal in visited:
            return False, 0.0
        visited.add(goal)

        best_cf = 0.0
        for conditions, conclusion, rule_cf in self.rules:
            if conclusion != goal:
                continue
            # Each condition receives its own visited set.  A fact explored on
            # one branch must not prevent an independent sibling branch from
            # proving that same fact.
            results = [
                self.backward_chain(condition, visited.copy())
                for condition in conditions
            ]
            if all(proved for proved, _ in results):
                best_cf = max(best_cf, round(rule_cf * min(cf for _, cf in results), 4))
        return (best_cf > 0.0), best_cf

    def analyze(self, percept) -> Dict:
        """Apply the knowledge base to the agent's ``PatientPercept`` interface."""
        if not hasattr(percept, "symptoms"):
            raise TypeError("analyze() expects a patient percept with symptoms.")

        self.facts.clear()
        self.certainty_factors.clear()
        self.load_patient_symptoms(percept.symptoms)

        temperature = getattr(percept, "temperature", None)
        if temperature is not None and temperature > 38.0:
            self.add_fact("fever", min(1.0, (float(temperature) - 37.0) / 3.0))
        if temperature is not None and temperature > 39.5:
            self.add_fact("high_fever")

        heart_rate = getattr(percept, "heart_rate", None)
        if heart_rate is not None and heart_rate > 100:
            self.add_fact("tachycardia")

        inferred = self.forward_chain()
        diagnoses = {
            fact: certainty
            for fact, certainty in inferred.items()
            if fact.endswith("_suspected") or fact.endswith("_confirmed")
        }
        diagnosis = max(diagnoses, key=diagnoses.get) if diagnoses else "Unknown"

        return {
            "summary": f"Inferred {len(inferred)} conclusions",
            "diagnosis": diagnosis,
            "confidence": diagnoses.get(diagnosis, 0.5),
            "all_inferred": inferred,
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Return the rule used to derive a diagnosis, if one exists."""
        diagnosis = self._normalise_fact(diagnosis)
        for conditions, conclusion, certainty in self.rules:
            if conclusion == diagnosis:
                return f"'{diagnosis}' derived from: {' + '.join(conditions)} (CF={certainty})"
        return f"'{diagnosis}' is a base fact"
