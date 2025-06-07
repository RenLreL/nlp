import pytest
from pathlib import Path
from src.backend.text_classification import Classifier # Assuming your Classifier is here

# Define a fixture for the Classifier (assuming a test model is available)
@pytest.fixture(scope="session")
def classifier_instance():
    # You might want to point to a minimal test model or mock it for speed
    # For a real integration test, point to your actual model dir
    project_root = Path(__file__).resolve().parent.parent.parent
    test_model_path = project_root / "modell_klassen_notebooks" / "bert_news_classifier"
    # Or for speed/isolation in unit tests, consider mocking parts of it if model loading is too heavy
    return Classifier(model_dir=test_model_path)

def test_classifier_initialization_success(classifier_instance):
    # Just checking if initialization completes without error
    assert classifier_instance.tok is not None
    assert classifier_instance.model is not None
    assert classifier_instance.id2label is not None
    assert isinstance(classifier_instance.id2label, dict)

def test_classify_basic_text(classifier_instance):
    text = "This is a test sentence."
    label = classifier_instance.classify(text, return_probs=False)
    assert isinstance(label, str)
    # You might assert that the label is one of your expected labels
    # assert label in ['left', 'center', 'right'] # Example

def test_classify_with_probabilities(classifier_instance):
    text = "Another test text for probabilities."
    label, probs = classifier_instance.classify(text, return_probs=True)
    assert isinstance(label, str)
    assert isinstance(probs, dict)
    assert all(isinstance(k, str) for k in probs.keys())
    assert all(isinstance(v, float) for v in probs.values()) # Crucial for JSON serialization!
    assert all(0.0 <= v <= 1.0 for v in probs.values()) # Probabilities should be between 0 and 1
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3) # Sum of probs is ~1
