# test_score.py

import pytest
from score import score
import pickle

# Load the pre-trained model
def load_model():
    model_path = "best_model.pkl"
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

# Perform a smoke test to check if the score function runs without errors
def test_smoke_test():
    model = load_model()
    try:
        score("Example", model, 0.5)
    except Exception as e:
        pytest.fail(f"score function raised an exception: {e} (Smoke test failed)")

# Check if the score function returns the expected output format
def test_format_test():
    model = load_model()
    text = "Example"
    threshold = 0.7
    prediction, probability = score(text, model, threshold)
    assert isinstance(prediction, int), "Expected prediction to be an integer"
    assert isinstance(probability, float), "Expected probability to be a float"

# Test if the prediction values are either 0 or 1
def test_prediction_0_or_1():
    model = load_model()
    text = "Example"
    threshold = 0.7
    prediction, _ = score(text, model, threshold)
    assert prediction in (0, 1), "Prediction should be 0 or 1"

# Check if the propensity score is within the valid range [0, 1]
def test_propensity_between_0_and_1():
    model = load_model()
    text = "Example"
    threshold = 0.7
    _, propensity = score(text, model, threshold)
    assert 0 <= propensity <= 1, "Propensity score should be between 0 and 1"

# Test when threshold is 0, prediction should always be 1
def test_when_threshold_0_prediction_always_1():
    model = load_model()
    text_1 = "Be there tonight"
    threshold = 0
    prediction, _ = score(text_1, model, threshold)
    assert prediction == 1, "Prediction should be 1 for threshold 0"

    text_2 = "Get a chance to go on a vacation to Hawaii"
    prediction, _ = score(text_2, model, threshold)
    assert prediction == 1, "Prediction should be 1 for threshold 0"

# Test when threshold is 1, prediction should always be 0
def test_when_threshold_1_prediction_always_0():
    model = load_model()
    text_1 = "Be there tonight"
    threshold = 1
    prediction, _ = score(text_1, model, threshold)
    assert prediction == 0, "Prediction should be 0 for threshold 1"

    text_2 = "Get a chance to go on a vacation to Hawaii"
    prediction, _ = score(text_2, model, threshold)
    assert prediction == 0, "Prediction should be 0 for threshold 1"
