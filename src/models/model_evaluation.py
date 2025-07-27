from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pandas as pd
import pickle  # For loading the trained model
import json  # For saving evaluation metrics

# Load the trained Random Forest model from the saved pickle file
model = pickle.load(open("models/random_forest_model.pkl", "rb"))

# Load the test data from the interim processed file
test_data = pd.read_csv("data/interim/test_bow.csv")

# Separate features (X_test) and labels (y_test) from the test data
X_test = test_data.drop(columns=['label']).values  # Features for testing
y_test = test_data['label'].values  # Labels for testing

# Use the trained model to predict labels for the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics for the model
metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),  # Accuracy of the model
    "precision": precision_score(y_test, y_pred),  # Precision of the model
    "recall": recall_score(y_test, y_pred),  # Recall of the model
    "roc_auc": roc_auc_score(y_test, y_pred)  # ROC-AUC score of the model
}

# Save the evaluation metrics to a JSON file for reporting
with open("reports/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)  # Save metrics with indentation for readability