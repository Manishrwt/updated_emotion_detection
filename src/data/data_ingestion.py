import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str) -> Dict:
    """Load parameters from a YAML file.

    Args:
        params_path (str): Path to the params.yaml file.

    Returns:
        Dict: Dictionary containing parameters.
    """
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

def load_dataset(url: str) -> pd.DataFrame:
    """Load dataset from a given URL.

    Args:
        url (str): URL to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully from %s", url)
        return df
    except Exception as e:
        logging.error("Error occurred while loading dataset: %s", str(e))
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by filtering and encoding.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    try:
        # Remove unnecessary columns
        df.drop(columns=['tweet_id'], inplace=True)
        logging.info("'tweet_id' column removed.")

        # Filter for specific sentiments
        filtered_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        logging.info("Filtered dataset to include only 'happiness' and 'sadness' sentiments.")

        # Encode sentiments
        filtered_df['sentiment'] = filtered_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Sentiments encoded successfully.")
        return filtered_df
    except KeyError as e:
        logging.error("KeyError during preprocessing: %s", str(e))
        raise
    except Exception as e:
        logging.error("Error occurred during preprocessing: %s", str(e))
        raise

def split_and_save_data(
    df: pd.DataFrame, test_size: float, output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test sets and save them to CSV files.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        output_dir (str): Directory to save the train and test datasets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Directory '%s' created or already exists.", output_dir)

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        logging.info("Train and test datasets saved successfully in '%s'.", output_dir)
        return train_data, test_data
    except Exception as e:
        logging.error("Error occurred while splitting and saving data: %s", str(e))
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        # Load parameters
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']

        # Load dataset
        dataset_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_dataset(dataset_url)

        # Preprocess dataset
        processed_df = preprocess_dataset(df)

        # Split and save data
        split_and_save_data(processed_df, test_size, output_dir='data/raw')
    except Exception as e:
        logging.error("Data ingestion pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()