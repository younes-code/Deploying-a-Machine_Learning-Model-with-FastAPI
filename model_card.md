# Model Card

For additional information see the Model Card paper: https: // arxiv.org/pdf/1810.03993.pdf

# Model Details
- Model Name: Census Income Prediction Model
- Model Version: 1.0
- Date: September 20, 2023
- Author: Younes Kebour
- License: [Specify the License, if applicable]
# Intended Use
This machine learning model is intended for predicting the income level of individuals based on demographic and employment-related features. It is designed to assist in identifying individuals with higher income levels for targeted marketing or other applications.
# Training Data
- Data Source: https: // archive.ics.uci.edu/ml/datasets/census+income
- Data Description: The model was trained on a dataset obtained from the UCI Machine Learning Repository, which contains information about individuals'. The data was preprocessed to handle missing values and encode categorical features.
# Evaluation Data
20 % of the dataset was used for model evaluation.
# Metrics
The model was evaluated using the following metrics:
- precision: 0.741
- recall: 0.60
- fbeta: 0.66
# Ethical Considerations
When developing and deploying this model, ethical considerations were taken into account. These considerations include:
- Privacy: User privacy is protected as the model does not require or use personally identifiable information.
- Fairness: We performed fairness audits to identify and mitigate potential biases in the model's predictions.

# Caveats and Recommendations
- Caveats:
    - The model may not perform well on individuals from underrepresented demographic groups
- Recommendations:
    - Continuously monitor and evaluate the model's performance and retrain it with updated data to improve accuracy and fairness.
