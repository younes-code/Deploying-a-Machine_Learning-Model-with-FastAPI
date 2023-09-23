import requests
import json

# URL of your Heroku-deployed FastAPI instance
url = "https://income-predictor-app-75a78bb3abb1.herokuapp.com/inference"

# Sample data for inference
sample = {
    "age": 35,
    "workclass": "Private",
    "fnlgt": 176756,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 45,
    "native_country": "United-States",
}

# Convert sample data to JSON
data = json.dumps(sample)

# Send a POST request to the Heroku-deployed FastAPI endpoint
response = requests.post(url, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse and display the response JSON
    response_data = response.json()
    print("Response status code:", response.status_code)
    print(
        "Model prediction:",
        response_data.get("prediction", "Prediction not available")
    )
else:
    print("Failed to make a POST request. Status code:",
          response.status_code)
