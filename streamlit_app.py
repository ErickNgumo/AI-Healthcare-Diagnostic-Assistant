import re

import streamlit as st

from app import build_system
from modules.agent import PatientPercept


SYMPTOM_OPTIONS = [
    'fever',
    'cough',
    'fatigue',
    'loss_of_smell',
    'headache',
    'body_aches',
    'rash',
    'joint_pain',
    'chest_pain',
    'shortness_of_breath',
    'sweating',
    'frequent_urination',
    'excessive_thirst',
    'blurred_vision',
    'night_sweats',
    'weight_loss',
    'stiff_neck',
    'light_sensitivity',
]


def validate_bp(value: str) -> bool:
    """Validate a realistic systolic/diastolic blood-pressure reading."""
    match = re.fullmatch(r"(\d{2,3})\s*/\s*(\d{2,3})", value.strip())
    if not match:
        return False
    systolic, diastolic = (int(reading) for reading in match.groups())
    return 70 <= systolic <= 250 and 40 <= diastolic <= 150 and systolic > diastolic


def validate_inputs(patient_id: str, age: int, temperature: float, heart_rate: int, blood_pressure: str, symptoms):
    if not patient_id or not patient_id.strip():
        raise ValueError('Patient ID is required.')
    if not isinstance(age, (int, float)) or age <= 0 or age > 150:
        raise ValueError('Age must be a valid number between 1 and 150.')
    if temperature is None or temperature <= 0 or temperature > 45:
        raise ValueError('Temperature must be a valid value between 0 and 45 °C.')
    if heart_rate is None or heart_rate <= 0 or heart_rate > 250:
        raise ValueError('Heart rate must be a valid value between 1 and 250 BPM.')
    if not blood_pressure or not validate_bp(str(blood_pressure)):
        raise ValueError('Blood pressure must use a valid systolic/diastolic format, such as 120/80.')
    if not symptoms:
        raise ValueError('Please select at least one symptom.')


def render_report(report, agent):
    st.header('Diagnosis report')
    diagnosis_column, confidence_column, urgency_column = st.columns(3)
    diagnosis_column.metric('Final diagnosis', report['diagnosis'])
    confidence_column.metric('Average module confidence', f"{report['confidence'] * 100:.1f}%")
    urgency_column.metric('Urgency', report['urgency'])

    st.subheader('Recommendations')
    for item in report['recommendations']:
        st.write('-', item)

    st.subheader('AI Module Analysis')
    if hasattr(agent, 'last_results') and agent.last_results:
        for name, result in agent.last_results.items():
            diagnosis = result.get('diagnosis', 'Unavailable')
            confidence = result.get('confidence', 0)
            with st.expander(name, expanded=False):
                st.write(f"Result: {diagnosis}")
                st.write(f"Confidence: {float(confidence) * 100:.1f}%")
                if name == 'FuzzySeverity':
                    st.write(f"Severity score: {result.get('severity_score', 'Unavailable')} / 100")
                    st.write(f"Severity level: {result.get('severity_label', 'Unavailable')}")
                if 'summary' in result:
                    st.caption(result['summary'])
    else:
        st.info('No module results were produced.')

    st.subheader('Patient Information')
    patient = agent.memory.current_patient
    st.write({
        'Patient ID': patient.patient_id,
        'Age': patient.age,
        'Symptoms': patient.symptoms,
        'Temperature (°C)': patient.temperature,
        'Heart Rate (BPM)': patient.heart_rate,
        'Blood Pressure': patient.blood_pressure,
    })


st.set_page_config(page_title='Healthcare Diagnostic Assistant', layout='wide')
st.title('Intelligent Healthcare Diagnostic Assistant')
st.warning('Academic AI demonstration only. This system is not a medical device and does not replace professional medical diagnosis or treatment.')

with st.form('patient_form'):
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input('Patient ID', value='P001')
        age = st.number_input('Age', min_value=1, max_value=150, value=34, step=1)
        temperature = st.number_input('Temperature (°C)', min_value=30.0, max_value=45.0, value=38.9, step=0.1)
    with col2:
        heart_rate = st.number_input('Heart Rate (BPM)', min_value=1, max_value=250, value=98, step=1)
        blood_pressure = st.text_input('Blood Pressure', value='120/80')
        symptoms = st.multiselect('Symptoms', options=SYMPTOM_OPTIONS, default=['fever', 'cough', 'fatigue', 'loss_of_smell'])

    submitted = st.form_submit_button('Run Diagnosis')

if submitted:
    try:
        validate_inputs(patient_id, age, temperature, heart_rate, blood_pressure, symptoms)
        patient = PatientPercept(
            patient_id=patient_id.strip(),
            symptoms=list(symptoms),
            age=int(age),
            temperature=float(temperature),
            heart_rate=int(heart_rate),
            blood_pressure=str(blood_pressure).strip(),
        )
        with st.spinner('Running the diagnostic modules...'):
            agent = build_system()
            report = agent.run(patient)
        render_report(report, agent)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f'Diagnosis could not be completed: {exc}')
