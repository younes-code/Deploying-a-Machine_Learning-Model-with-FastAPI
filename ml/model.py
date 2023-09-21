from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds



def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed

    Parameters
    ----------
    df : pd.DataFrame
        Test dataframe pre-processed with features as columns used for slices.
    feature : str
        Feature on which to perform the slices.
    y : np.array
        Corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'feature': the feature name
        - 'feature_value': the unique value of the feature
        - 'n_samples': number of data samples in the slice
        - 'precision': precision score
        - 'recall': recall score
        - 'fbeta': F1 score
    """

    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(columns=['feature', 'feature_value', 'n_samples', 'precision', 'recall', 'fbeta'])

    for option in slice_options:
        slice_mask = df[feature] == option
        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]

        precision = precision_score(slice_y, slice_preds, zero_division=1)
        recall = recall_score(slice_y, slice_preds, zero_division=1)
        fbeta = fbeta_score(slice_y, slice_preds, beta=1, zero_division=1)

        slice_perf_df = pd.DataFrame({
            'feature': [feature],
            'feature_value': [option],
            'n_samples': [len(slice_y)],
            'precision': [precision],
            'recall': [recall],
            'fbeta': [fbeta]
        })

        perf_df = pd.concat([perf_df, slice_perf_df], ignore_index=True)

    return perf_df
