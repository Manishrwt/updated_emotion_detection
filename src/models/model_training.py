import os
import pandas as pd
import pickle
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("The params.yaml file was not found at %s", params_path)
        raise
    except Exception as e:
        logging.error("Error occurred while loading parameters: %s", str(e))
        raise

def load_training_data(train_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load training data and separate features and labels."""
    try:
        train_data = pd.read_csv(train_path)
        logging.info("Training data loaded successfully from %s", train_path)

        x_train = train_data.drop(columns=['label']).values  # Features for training
        y_train = train_data['label'].values  # Labels for training
        logging.info("Features and labels separated successfully.")
        return x_train, y_train
    except FileNotFoundError as e:
        logging.error("File not found: %s", str(e))
        raise
    except Exception as e:
        logging.error("Error occurred while loading training data: %s", str(e))
        raise

def train_model(x_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a Random Forest Classifier."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error occurred while training the model: %s", str(e))
        raise

def save_model(model: RandomForestClassifier, output_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Model saved successfully to %s", output_path)
    except Exception as e:
        logging.error("Error occurred while saving the model: %s", str(e))
        raise

def main() -> None:
    """Main function to orchestrate model training."""
    try:
        # Load parameters
        params = load_params("params.yaml")
        n_estimators = params['model_training']['n_estimators']
        max_depth = params['model_training']['max_depth']

        # Load training data
        x_train, y_train = load_training_data("data/interim/train_bow.csv")

        # Train the model
        model = train_model(x_train, y_train, n_estimators, max_depth)

        # Save the trained model
        save_model(model, "models/random_forest_model.pkl")
    except Exception as e:
        logging.error("Model training pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()