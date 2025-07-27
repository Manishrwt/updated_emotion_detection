import os
import re
import numpy as np
import pandas as pd
import nltk
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logging.error("Error while removing punctuations: %s", str(e))
        raise

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Set text to NaN if sentence has fewer than 3 words."""
    try:
        df['content'] = df['content'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
        logging.info("Removed small sentences with fewer than 3 words.")
        return df
    except Exception as e:
        logging.error("Error while removing small sentences: %s", str(e))
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df['content'] = df['content'].fillna("").apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error("Error during text normalization: %s", str(e))
        raise

def preprocess_and_save_data(input_dir: str, output_dir: str) -> None:
    """Load, preprocess, and save train and test datasets."""
    try:
        # Load raw train and test data
        train_data = pd.read_csv(os.path.join(input_dir, "train.csv"))
        test_data = pd.read_csv(os.path.join(input_dir, "test.csv"))
        logging.info("Loaded raw train and test data.")

        # Normalize train and test data
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        # Save processed data to CSV files
        os.makedirs(output_dir, exist_ok=True)
        train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        logging.info("Processed train and test data saved to '%s'.", output_dir)
    except FileNotFoundError as e:
        logging.error("File not found: %s", str(e))
        raise
    except Exception as e:
        logging.error("Error during preprocessing and saving data: %s", str(e))
        raise

def main() -> None:
    """Main function to orchestrate data preprocessing."""
    try:
        input_dir = "data/raw"
        output_dir = "data/processed"
        preprocess_and_save_data(input_dir, output_dir)
    except Exception as e:
        logging.error("Data preprocessing pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()