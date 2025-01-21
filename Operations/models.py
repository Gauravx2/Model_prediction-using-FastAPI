
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Global variables for storing the dataset and model
data = None
model = None

def generate_synthetic_data():
    np.random.seed(42)
    data = {
        "Machine_ID": np.arange(1, 10001),
        "Temperature": np.random.uniform(50, 100, 10000),
        "Run_Time": np.random.uniform(10, 500, 10000),
        "Downtime_Flag": np.random.choice([0, 1], size=10000, p=[0.7, 0.3])
    }
    return pd.DataFrame(data)

def train_model(data):
    global model
    # Split data into features and target
    X = data.drop(columns=["Downtime_Flag","Machine_ID"])
    y = data["Downtime_Flag"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1

def predict(input_data):
    global model
    if model is None:
        raise ValueError("Model not trained")
    prediction = model.predict(input_data)
    confidence = max(model.predict_proba(input_data)[0])
    return prediction, confidence
