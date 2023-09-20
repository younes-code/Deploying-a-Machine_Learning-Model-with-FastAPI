# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving the trained model
import pandas as pd
from data import process_data
from model import train_model,inference
import os, pickle
# Add the necessary imports for the starter code.

# Add code to load in the data.
script_directory = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(script_directory, "../data/census.csv")

data = pd.read_csv(datapath)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder,lb=lb
)


# Train and save a model.
model = train_model(X_train, y_train)


# Save the model as a .pkl file
model_filename = "../model/trained_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(model, model_file)

# Save label encoder as .pkl file
encoder_filename = "../model/label_encoder.pkl"
with open(encoder_filename, "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

# Save one-hot encoder as .pkl file
lb_filename = "../model/one_hot_encoder.pkl"
with open(lb_filename, "wb") as lb_file:
    pickle.dump(lb, lb_file)