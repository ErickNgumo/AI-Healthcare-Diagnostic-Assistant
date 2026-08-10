# Intelligent Healthcare Diagnostic Assistant

This project is a student AI capstone system for healthcare diagnosis support. It combines multiple artificial intelligence techniques to analyze patient symptoms and vital signs, estimate diagnostic likelihood, assess severity, and generate a treatment-oriented recommendation report.

The system is designed as an academic demonstration and is not a replacement for professional medical diagnosis or treatment.

## Project overview

The Intelligent Healthcare Diagnostic Assistant takes a patient record and passes it through a model-based intelligent agent. The agent perceives the patient, invokes multiple AI modules, combines the outputs, and produces a final diagnostic report with an urgency score and recommendations.

The workflow is:

Patient Data
     ↓
Intelligent Agent
     ↓
Perception
     ↓
AI Diagnostic Modules
     ↓
Reasoning / Prediction
     ↓
Diagnosis + Confidence
     ↓
Severity / Urgency
     ↓
Treatment Planning
     ↓
Final Report

The project demonstrates how multiple AI approaches can work together in a healthcare decision-support setting:

- a rule-based medical knowledge base
- a Bayesian probabilistic diagnosis model
- a supervised machine learning classifier
- a neural network model
- fuzzy logic for severity assessment
- treatment planning logic

## AI modules

### 1. Intelligent Agent
- Purpose: main decision-making coordinator
- Input: `PatientPercept` object with patient information and symptoms
- Processing: stores patient data, calls registered modules, aggregates results, and decides urgency and next action
- Output: final diagnostic report and recommendation summary

### 2. Medical Knowledge Base
- Purpose: infer disease likelihood from symptoms and vital signs using rules and certainty factors
- Input: symptom list and key vitals
- Processing: load symptom facts, apply forward chaining, infer likely disease conditions
- Output: diagnosis candidate and confidence score

### 3. Bayesian Network
- Purpose: produce a probabilistic diagnosis using symptom likelihoods
- Input: symptom list
- Processing: computes posterior probabilities across disease classes
- Output: disease ranking and confidence values

### 4. Machine Learning Classifier
- Purpose: classify diseases using supervised learning
- Input: symptom vector
- Processing: trains or uses an ensemble of decision tree-based models and predicts the top disease
- Output: predicted diagnosis, confidence, and ranked alternatives

### 5. Deep Neural Network
- Purpose: apply a neural diagnostic model to patient symptoms
- Input: binary symptom vector
- Processing: passes the input through a trained neural network and estimates class probabilities
- Output: predicted diagnosis and probability distribution

### 6. Fuzzy Logic Severity Assessment
- Purpose: measure patient emergency severity using temperature, heart rate, and symptom burden
- Input: temperature, heart rate, number of symptoms
- Processing: fuzzifies inputs, evaluates fuzzy rules, and defuzzifies to severity score
- Output: urgency label and severity score

### 7. Treatment Planner
- Purpose: create a treatment plan based on the selected diagnosis and urgency
- Input: diagnosis and urgency
- Processing: uses a STRIPS-style action library to generate a plan
- Output: steps, order, and planning summary

## System architecture

The repository uses a modular pipeline. The agent is the central orchestrator:

1. A patient record is created as a `PatientPercept`
2. The agent stores the record and marks the state as `COLLECTING`
3. Each registered AI module receives the same patient data through `.analyze()`
4. The agent aggregates module results
5. The diagnosis and confidence are combined
6. Severity and urgency are assessed using temperature, pulse, and symptoms
7. Recommendations and treatment suggestions are generated
8. A final structured report is returned

## Technologies

This project uses the following libraries and tools that are present in the repository:

- Python 3
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow / Keras
- pgmpy
- scikit-fuzzy
- NLTK
- NetworkX
- Gymnasium
- SciPy
- Streamlit (used for the final UI)

## Project structure

```text
AI-Healthcare-Diagnostic-Assistant/
├── app.py
├── README.md
├── requirements.txt
├── modules/
│   ├── agent.py
│   ├── bayesian_net.py
│   ├── fuzzy_controller.py
│   ├── knowledge_base.py
│   ├── ml_classifier.py
│   ├── neural_network.py
│   └── planner.py
├── tests/
│   ├── test_module1_agent.py
│   ├── test_module2_knowledge_base.py
│   ├── test_module3_bayesian_net.py
│   └── test_neural_network.py
├── evaluation/
│   ├── metrics.py
│   └── visualizations.py
├── lab_kMeans/
│   ├── AI lab_assessment_results.csv
│   ├── result_C026-01-0752_2024.csv
│   └── result_C026-01-0761_2024.csv
├── .venv/   (local virtual environment, not usually committed)
└── maybe generated artifacts such as training plots or logs
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ErickNgumo/AI-Healthcare-Diagnostic-Assistant.git
cd AI-Healthcare-Diagnostic-Assistant
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to run the tests, install pytest as well:

```bash
python -m pip install pytest
```

## Running the project

### Command-line version

The project can be run from the command line with:

```bash
python3 app.py
```

This creates an agent, registers the available AI modules, runs a sample patient through the full cycle, and prints the final report as JSON.

### Streamlit UI version

The project also includes a lightweight web interface for demonstration:

```bash
python3 -m streamlit run streamlit_app.py
```

The UI allows the user to enter:

- patient ID
- age
- symptoms
- temperature
- heart rate
- blood pressure

When the user clicks the run button, the system runs the agent and displays:

- patient information
- diagnosis
- confidence
- urgency level
- recommendations
- module analysis summary

## Testing

The repository uses pytest for unit and integration-style module tests.

Run the tests with:

```bash
python3 -m pytest -v
```

The tests verify:

- patient perception stores data correctly
- the state changes properly
- registered modules are called during thinking
- act() produces a structured report
- the full perceive → think → act flow works
- knowledge-base inference and certainty-factor validation
- Bayesian posterior normalization and ranking
- neural-network data splitting, prediction, evaluation, and plot generation

### End-to-end check

Start the Streamlit app, submit the sample patient record from the demonstration procedure, and confirm that the patient information, diagnosis report, urgency, recommendations, and six module-analysis panels are displayed. The command-line sample in `app.py` exercises the same agent pipeline without the web interface.

## User interface

The UI is intentionally simple and suitable for a classroom demonstration.

It includes:

- patient identification input
- age and vital sign controls
- symptom selection
- validation messages for invalid data
- a run diagnosis button
- a results panel with diagnosis, severity, confidence, and recommendations

### Validation rules

The UI validates:

- missing patient ID
- invalid age
- invalid temperature
- invalid heart rate
- invalid blood pressure
- no symptoms selected
- invalid input types

If validation fails, the app shows a clear error instead of crashing.

## Medical disclaimer

This application is an academic AI demonstration project created for educational purposes. It is not a certified medical device and must not be used as a replacement for professional clinical judgment, diagnosis, or treatment.

## Documentation of AI results

The system combines outputs from multiple modules as follows:

- the agent stores the current patient and their symptoms
- each module runs `.analyze(patient_data)`
- each registered module returns a dictionary with a diagnosis and confidence value; the fuzzy module also returns a severity score and label
- the agent averages every available module confidence after constraining each score to the range 0–1
- the final diagnosis is the most frequent available module diagnosis; it is not a weighted clinical consensus
- urgency is assigned by the agent's explicit thresholds: critical for a temperature of at least 39.5°C, heart rate of at least 120 BPM, or an emergency result; high for temperature of at least 38.5°C, heart rate of at least 100 BPM, or average confidence of at least 0.8; otherwise medium or low according to temperature and symptom count
- recommendations are generated from the urgency level, then the planner is asked to add up to three STRIPS-style plan steps for the selected diagnosis and urgency

The final result is not a claim of medical certainty. It is an AI-assisted decision-support output designed for demonstration and learning.

## Final demonstration procedure

A 10-minute final demo can be run as follows:

1. Start the application with `python app.py` or `streamlit run streamlit_app.py`
2. Enter a sample patient record, such as:
   - Patient ID: `P001`
   - Age: `34`
   - Symptoms: `fever`, `cough`, `fatigue`, `loss_of_smell`
   - Temperature: `38.9`
   - Heart Rate: `98`
   - Blood Pressure: `120/80`
3. Submit the form
4. Show the agent processing
5. Display the final diagnosis and confidence
6. Highlight the urgency/severity result
7. Show the recommended actions
8. Explain which AI modules contributed to the decision
9. Summarize the system workflow and the academic nature of the project

## Final checks

Before submission, confirm:

- the project installs with `pip install -r requirements.txt`
- the app starts successfully
- the UI loads and accepts patient data
- diagnosis generation works
- the tests pass
- the README instructions are accurate

## Notes

This repository already contains a working academic architecture for intelligent healthcare diagnosis. The goal is to preserve that design and provide a clean, demonstrable final presentation without rewriting working AI modules unnecessarily.
