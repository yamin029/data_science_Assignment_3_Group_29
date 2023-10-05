# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt  # Added import for data visualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('po2_data.csv')

# Handle missing values if any
data.fillna(0, inplace=True)  # You may want to choose a more appropriate strategy

# Split the data into features (X) and target variables (y)
X = data.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor = data['motor_updrs']
y_total = data['total_updrs']

# Split the data into training and testing sets
X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(
    X, y_motor, y_total, test_size=0.2, random_state=42)

# Initialize and train the linear regression model for motor UPDRS
motor_model = LinearRegression()
motor_model.fit(X_train, y_motor_train)

# Make predictions on the test set for motor UPDRS
y_motor_pred = motor_model.predict(X_test)

# Initialize and train the linear regression model for total UPDRS
total_model = LinearRegression()
total_model.fit(X_train, y_total_train)

# Make predictions on the test set for total UPDRS
y_total_pred = total_model.predict(X_test)

# Create side-by-side subplots
plt.figure(figsize=(12, 5))

# Subplot for motor UPDRS
plt.subplot(1, 2, 1)
plt.scatter(y_motor_test, y_motor_pred)
plt.title('Motor UPDRS Model: True vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')

# Subplot for total UPDRS
plt.subplot(1, 2, 2)
plt.scatter(y_total_test, y_total_pred)
plt.title('Total UPDRS Model: True vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')

# Adjust layout for better visualization
plt.tight_layout()

# Show the plots
plt.show()

# Evaluate the motor UPDRS model
mse_motor = mean_squared_error(y_motor_test, y_motor_pred)
r2_motor = r2_score(y_motor_test, y_motor_pred)

print("Motor UPDRS Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_motor}")
print(f"R-squared (R2) Score: {r2_motor}")

# Evaluate the total UPDRS model
mse_total = mean_squared_error(y_total_test, y_total_pred)
r2_total = r2_score(y_total_test, y_total_pred)

print("\nTotal UPDRS Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_total}")
print(f"R-squared (R2) Score: {r2_total}")