import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
import json

def train_and_save_model():
    # Load the dataset
    data = pd.read_csv('../data/water_potability.csv')

    #data = data.dropna()  # Drop rows with missing values for simplicity
    
    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Features and target
    # X = data.drop('Potability', axis=1)
   
    # Select specified features
    selected_features = ['ph', 'Sulfate', 'Hardness', 'Solids', 'Turbidity']
    X = data[selected_features]
    y = data['Potability']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')

    # Load existing metrics if the file exists
    try:
        with open('metrics.json', 'r') as f:
            all_metrics = json.load(f)
    except FileNotFoundError:
        all_metrics = {}

    # Save metrics for the Random Forest model
    all_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'f1_score': float(f1),
        'mean_squared_error': float(mse),
        'true_positives': int(tp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'false_positives': int(fp)
    }

    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    train_and_save_model()