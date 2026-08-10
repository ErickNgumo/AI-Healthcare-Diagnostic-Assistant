import pytest


tf = pytest.importorskip('tensorflow')
pytest.importorskip('sklearn')

from modules.agent import PatientPercept
from modules.neural_network import NeuralDiagnosticModel


@pytest.fixture(scope='module')
def trained_model():
    model = NeuralDiagnosticModel()
    model.train(epochs=1, verbose=0, n_samples=80, batch_size=16)
    return model


def test_model_has_expected_input_and_output_dimensions():
    model = NeuralDiagnosticModel()
    assert model.model.input_shape == (None, 18)
    assert model.model.output_shape == (None, 8)


def test_training_uses_disjoint_train_validation_and_test_sets(trained_model):
    assert len(trained_model.y_train) == 48
    assert len(trained_model.y_val) == 16
    assert len(trained_model.y_test) == 16


def test_prediction_and_agent_interface_return_softmax_confidence(trained_model):
    result = trained_model.predict(['fever', 'cough', 'fatigue'])
    assert result['diagnosis'] in trained_model.DISEASE_LABELS
    assert 0 <= result['confidence'] <= 1
    assert pytest.approx(sum(result['all_probs'].values()), abs=1e-5) == 1
    percept = PatientPercept('P1', ['fever', 'cough'], 30, 38.0, 90, '120/80')
    assert trained_model.analyze(percept)['summary'].startswith('DNN:')


def test_evaluation_and_plots_are_generated_from_test_results(trained_model, tmp_path):
    metrics = trained_model.evaluate(output_dir=tmp_path, generate_plots=True)
    assert 0 <= metrics['accuracy'] <= 1
    assert (tmp_path / 'nn_confusion_matrix.png').is_file()
    assert trained_model.plot_training(tmp_path).is_file()


def test_invalid_symptoms_are_rejected(trained_model):
    with pytest.raises(TypeError):
        trained_model.predict('fever')
