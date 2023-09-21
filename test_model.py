import pytest
import numpy as np
from ml.model import train_model, compute_model_metrics, inference

# Define fixtures for testing
@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    return X_train, y_train

# Test train_model function
def test_train_model(sample_data):
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    assert hasattr(model, 'predict')

# Test compute_model_metrics function
def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_preds = np.array([0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

# Test inference function
def test_inference(sample_data):
    X_train, _ = sample_data
    model = train_model(X_train, np.array([0, 1, 0]))  # Modify labels according to your data
    X_test = X_train  # Modify test data accordingly
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)
