# filepath: /Users/manishrawat/Desktop/MLOPS/emotion_detection/src/features/feature_eng.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os

# Load processed train and test data
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

# Extract features (content) and labels (sentiment) from train and test data
X_train = train_data['content'].fillna("")  # Replace NaN with an empty string
y_train = train_data['sentiment']

X_test = test_data['content'].fillna("")  # Replace NaN with an empty string
y_test = test_data['sentiment']

# Apply Bag of Words (CountVectorizer) for feature extraction
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it to feature vectors
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer (do not fit again to avoid data leakage)
X_test_bow = vectorizer.transform(X_test)

# Convert the feature vectors to DataFrames for easier handling
train_df = pd.DataFrame(X_train_bow.toarray())  # Convert training feature vectors to DataFrame
train_df['label'] = y_train  # Add the labels to the DataFrame

test_df = pd.DataFrame(X_test_bow.toarray())  # Convert testing feature vectors to DataFrame
test_df['label'] = y_test  # Add the labels to the DataFrame

# Save the processed feature data to CSV files
os.makedirs("data/interim", exist_ok=True)  # Ensure the directory exists
train_df.to_csv("data/interim/train_bow.csv", index=False)  # Save training data
test_df.to_csv("data/interim/test_bow.csv", index=False)  # Save testing data