import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
try:
    data = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Error: 'diabetes.csv' file not found.")
    exit()

# Check for missing values in the dataset
if data.isnull().any().any():
    print("Warning: The dataset contains missing values. Consider handling them.")
    data = data.fillna(data.mean())  # Simple strategy to fill missing values with column mean

# Features and target variable
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Insulin', 'Age', 'DiabetesPedigreeFunction', 'SkinThickness']
X = data[features]
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'diabetes_model.pkl')
print("Model saved as 'diabetes_model.pkl'")
