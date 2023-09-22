from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml.data import process_data
import pickle

# instantiate a FastAPI app
app = FastAPI()

# Declare the data object with its components and their type.


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# Define a root endpoint with a welcome message


@app.get("/")
async def welcome():
    return {"message": "Welcome to the Income Prediction API!"}


# Define the POST endpoint for model inference
# This allows sending of data (our InferenceSample) via POST to the API.


@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {
        "age": inference.age,
        "workclass": inference.workclass,
        "fnlgt": inference.fnlgt,
        "education": inference.education,
        "education-num": inference.education_num,
        "marital-status": inference.marital_status,
        "occupation": inference.occupation,
        "relationship": inference.relationship,
        "race": inference.race,
        "sex": inference.sex,
        "capital-gain": inference.capital_gain,
        "capital-loss": inference.capital_loss,
        "hours-per-week": inference.hours_per_week,
        "native-country": inference.native_country,
    }

    # prepare the sample for inference as a DataFrame
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Load the model and encoders using pickle
    with open("model/trained_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("model/label_encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    with open("model/one_hot_encoder.pkl", "rb") as lb_file:
        lb = pickle.load(lb_file)

    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Get model prediction
    prediction = model.predict(sample)

    # Convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction = ">50K"
    else:
        prediction = "<=50K"

    data["prediction"] = prediction

    return data
