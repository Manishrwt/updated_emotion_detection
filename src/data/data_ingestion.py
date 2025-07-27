import os  # For directory operations
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the provided URL
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Remove the 'tweet_id' column as it's not needed for modeling
df.drop(columns=['tweet_id'], inplace=True)

# Filter the dataset to only include 'happiness' and 'sadness' sentiments
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()  # .copy() avoids SettingWithCopyWarning

# Replace 'happiness' with 1 and 'sadness' with 0 explicitly
final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

# Create a 'data/raw' directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# Save the splits to CSV files
train_data.to_csv('data/raw/train.csv', index=False)  # Save training data
test_data.to_csv('data/raw/test.csv', index=False)  # Save testing data