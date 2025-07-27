import os
import pandas as pd
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer
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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Train and test data loaded successfully.")
        return train_data, test_data
    except FileNotFoundError as e:
        logging.error("File not found: %s", str(e))
        raise
    except Exception as e:
        logging.error("Error occurred while loading data: %s", str(e))
        raise

def extract_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features using Bag of Words (CountVectorizer)."""
    try:
        # Extract features (content) and labels (sentiment)
        X_train = train_data['content'].fillna("")  # Replace NaN with an empty string
        y_train = train_data['sentiment']

        X_test = test_data['content'].fillna("")  # Replace NaN with an empty string
        y_test = test_data['sentiment']

        # Apply Bag of Words (CountVectorizer) for feature extraction
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit the vectorizer on the training data and transform it to feature vectors
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform the test data using the same vectorizer (do not fit again to avoid data leakage)
        X_test_bow = vectorizer.transform(X_test)

        # Convert the feature vectors to DataFrames for easier handling
        train_df = pd.DataFrame(X_train_bow.toarray())  # Convert training feature vectors to DataFrame
        train_df['label'] = y_train  # Add the labels to the DataFrame

        test_df = pd.DataFrame(X_test_bow.toarray())  # Convert testing feature vectors to DataFrame
        test_df['label'] = y_test  # Add the labels to the DataFrame

        logging.info("Feature extraction completed successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error("Error occurred during feature extraction: %s", str(e))
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save the processed feature data to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_bow.csv"), index=False)
        logging.info("Processed feature data saved successfully to '%s'.", output_dir)
    except Exception as e:
        logging.error("Error occurred while saving data: %s", str(e))
        raise

def main() -> None:
    """Main function to orchestrate feature engineering."""
    try:
        # Load parameters
        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']

        # Load processed train and test data
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")

        # Extract features
        train_df, test_df = extract_features(train_data, test_data, max_features)

        # Save processed feature data
        save_data(train_df, test_df, output_dir="data/interim")
    except Exception as e:
        logging.error("Feature engineering pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()