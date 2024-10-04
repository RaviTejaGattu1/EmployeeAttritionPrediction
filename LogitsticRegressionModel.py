# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Data Collection
# HR dataset with features such as 'Age', 'JobRole', 'Salary', 'PerformanceRating', etc.
# and a target variable 'Attrition' where 1 means the employee left and 0 means they stayed.

data = pd.read_csv('employee_data.csv')  # Load your HR dataset

# Step 2: Data Preprocessing
# Handling missing values, categorical encoding, etc.

# Convert categorical columns to dummy variables (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
X = data.drop('Attrition', axis=1)  # Features
y = data['Attrition']  # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Model Building (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Predictive Analysis
#  to predict the probability of attrition for new employees:
probabilities = model.predict_proba(X_test)[:, 1]  # Predicting probability of attrition (class 1)
high_risk_employees = np.where(probabilities > 0.8)[0]  # Employees with high probability of leaving (threshold of 80%)

print("High-risk employees index in test set: ", high_risk_employees)
