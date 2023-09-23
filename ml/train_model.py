# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from data import process_data
from model import train_model, inference, compute_model_metrics, compute_slices
import pickle
import logging
from contextlib import redirect_stdout


datapath = "./data/census.csv"
data = pd.read_csv(datapath)

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
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)


# Train and save a model.
model = train_model(X_train, y_train)

# Save the model as a .pkl file
model_filename = "./model/trained_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(model, model_file)

# Save label encoder as .pkl file
encoder_filename = "./model/label_encoder.pkl"
with open(encoder_filename, "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

# Save one-hot encoder as .pkl file
lb_filename = "./model/one_hot_encoder.pkl"
with open(lb_filename, "wb") as lb_file:
    pickle.dump(lb, lb_file)


# Make predictions on the training data
y_pred = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)


# Log computed metrics

logging.basicConfig(
    filename="ml/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Precision: %f", precision)
logging.info("Recall: %f", recall)
logging.info("F-beta: %f", fbeta)

logging.shutdown()

# Compute and log performance on slices of categorical features.
slices_df = compute_slices(
    test, feature=cat_features[0], y=y_test, preds=inference(model, X_test)
)
# Log the performance DataFrame for slices.
print("Performance on Slices for Feature:", cat_features[0])
print(slices_df)

with open("./slice_output.txt", "w") as output_file:
    with redirect_stdout(output_file):
        print("Performance on Slices for Feature:", cat_features[0])
        print(slices_df)
