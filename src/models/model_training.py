import pandas as pd
import numpy as np
import pickle  # For saving the trained model

from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier

# Load the training data from the interim processed file
train_data = pd.read_csv("data/interim/train_bow.csv")

# Separate features (x_train) and labels (y_train) from the training data
x_train = train_data.drop(columns=['label']).values  # Features for training
y_train = train_data['label'].values  # Labels for training

# Initialize the Random Forest Classifier with 100 estimators and a fixed random state
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to a file using pickle
pickle.dump(model, open("models/random_forest_model.pkl", "wb"))