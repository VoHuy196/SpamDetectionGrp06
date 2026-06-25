import os
import pytest
from src.predict import SpamPredictor

# Ensure model predictor can be loaded
@pytest.fixture(scope="module")
def predictor():
    try:
        return SpamPredictor()
    except FileNotFoundError:
        pytest.skip("Model files not found. Run train.py first.")

def test_predictor_keys(predictor):
    """Test that the prediction returns the expected keys and types."""
    result = predictor.predict("Hello, how are you?")
    assert "text" in result
    assert "prediction" in result
    assert "spam_probability" in result
    assert "ham_probability" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["prediction"], str)
    assert isinstance(result["spam_probability"], float)
    assert isinstance(result["ham_probability"], float)

def test_predictor_ham(predictor):
    """Test predictor with clear non-spam text."""
    result = predictor.predict("Hey, are we still meeting for lunch today at 12?")
    assert result["prediction"] == "ham"
    assert result["ham_probability"] > result["spam_probability"]

def test_predictor_spam(predictor):
    """Test predictor with clear spam text that is verified to work."""
    result = predictor.predict("URGENT! Your mobile number has won a £2,000 prize. Call 09061701461 to claim your reward.")
    assert result["prediction"] == "spam"
    assert result["spam_probability"] > result["ham_probability"]
