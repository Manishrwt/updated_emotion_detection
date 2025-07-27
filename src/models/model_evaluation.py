import os
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(model_path: str):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully from %s", model_path)
        return model
    except FileNotFoundError:
        logging.error("The model file was not found at %s", model_path)
        raise
    except Exception as e:
        logging.error("Error occurred while loading the model: %s", str(e))
        raise

def load_test_data(test_data_path: str) -> pd.DataFrame:
    """Load the test data from a CSV file."""
    try:
        test_data = pd.read_csv(test_data_path)
        logging.info("Test data loaded successfully from %s", test_data_path)
        return test_data
    except FileNotFoundError:
        logging.error("The test data file was not found at %s", test_data_path)
        raise
    except Exception as e:
        logging.error("Error occurred while loading test data: %s", str(e))
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and calculate metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed successfully.")
        return metrics
    except Exception as e:
        logging.error("Error occurred during model evaluation: %s", str(e))
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved successfully to %s", output_path)
    except Exception as e:
        logging.error("Error occurred while saving metrics: %s", str(e))
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        # Paths
        model_path = "models/random_forest_model.pkl"
        test_data_path = "data/interim/test_bow.csv"
        metrics_output_path = "reports/metrics.json"

        # Load model and test data
        model = load_model(model_path)
        test_data = load_test_data(test_data_path)

        # Separate features and labels
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics, metrics_output_path)
    except Exception as e:
        logging.error("Model evaluation pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()