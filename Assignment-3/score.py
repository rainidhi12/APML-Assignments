from typing import Tuple
import sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer


def score(text: str, model: BaseEstimator, vectorizer: TfidfVectorizer, threshold: float) -> Tuple[bool, float]:
    """
    Scores a trained model on a given text.

    Args:
        text (str): The input text to be scored.
        model (sklearn.estimator): The trained machine learning model.
        threshold (float): Decision threshold for prediction.

    Returns:
        Tuple[bool, float]: A tuple containing the prediction (0 or 1) and the propensity score.
    """
    assert isinstance(text, str), "Input 'text' must be a string."
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
    
    text = vectorizer.transform([text])

    pred_proba = model.predict_proba(text)[0, 1]
    prediction = 1 if pred_proba >= threshold else 0

    return prediction, pred_proba
