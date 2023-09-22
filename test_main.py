# test_main.py
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app instance


client = TestClient(app)

# a sample data for testing
sample_data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}


# Test the GET request for the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message":
                               "Welcome to the Income Prediction API!"}


# Test the POST request for model inference
def test_predict_income():
    response = client.post("/inference/", json=sample_data)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data


# Test a specific prediction using POST
def test_specific_prediction():
    specific_sample_data = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 13,
        "native_country": "United-States",
    }
    response = client.post("/inference/", json=specific_sample_data)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
