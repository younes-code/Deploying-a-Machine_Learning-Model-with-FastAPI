from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml.data import process_data
import pickle
import uvicorn

# instantiate a FastAPI app
app = FastAPI()

# Declare the data object with its components and their type.


class InputData(BaseModel):
    age: int = 35
    workclass: str = "Private"
    fnlgt: int = 176756
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Married-civ-spouse"
    occupation: str = "Exec-managerial"
    relationship: str = "Husband"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = 5000
    capital_loss: int = 0
    hours_per_week: int = 45
    native_country: str = "United-States"


@app.get("/")
async def welcome():
    return {"message": "Welcome to the Income Prediction API!"}


# Define the POST endpoint for model inference
# This allows sending of data (our InferenceSample) via POST to the API.


# Define the POST endpoint for model inference
@app.post("/inference/")
async def ingest_data(inference: InputData = None):
    if inference is None:
        inference = InputData(
            age=35,
            workclass="Private",
            fnlgt=176756,
            education="Bachelors",
            education_num=13,
            marital_status="Married-civ-spouse",
            occupation="Exec-managerial",
            relationship="Husband",
            race="White",
            sex="Male",
            capital_gain=5000,
            capital_loss=0,
            hours_per_week=45,
            native_country="United-States",
        )

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
