import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the water potability dataset."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the dataset by handling missing values."""
    data = data.dropna()  # Drop rows with missing values
    return data

def preprocess_data(file_path):
    """Load and preprocess the dataset."""
    data = load_data(file_path)
    data = clean_data(data)
    
    # Separate features and target variable
    X = data.drop('potability', axis=1)
    y = data['potability']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y