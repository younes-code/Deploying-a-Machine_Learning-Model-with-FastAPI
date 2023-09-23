# test_main.py
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app instance
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


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


sample_data_above_50k = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 280464,
    "education": "Some-college",
    "education_num": 10,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 80,
    "native_country": "United-States",
}


sample_data_below_50k = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "India",
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


# Test the POST Request for Model Inference (Above 50K)
def test_predict_income_above_50k():
    response = client.post("/inference/", json=sample_data_above_50k)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert response_data["prediction"] == ">50K"


# Test the POST Request for Model Inference (Below 50K)
def test_predict_income_below_50k():
    response = client.post("/inference/", json=sample_data_below_50k)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert response_data["prediction"] == "<=50K"
