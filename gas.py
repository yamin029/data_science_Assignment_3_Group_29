# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Loading the data
data = pd.read_csv('po2_data.csv')
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
null_mask = df.isnull().any(axis=1)
null_rows = df[null_mask]

# Checking for null values
print(df.isnull().sum())
print(f"There are {len(null_rows)} empty rows in the DataFrame")

# Step 3.1: Split the data into features (X) and target variables (y)
X = data.drop(columns=['motor_updrs', 'total_updrs'])
y_motor = data['motor_updrs']
y_total = data['total_updrs']

# Step 3.2: Split the data into training and test sets
X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(
    X, y_motor, y_total, test_size=0.2, random_state=0)

# Step 3.3: Select a Model (Multiple Linear Regression)
motor_model = LinearRegression()
total_model = LinearRegression()

# Step 3.4: Train the Models
motor_model.fit(X_train, y_motor_train)
total_model.fit(X_train, y_total_train)

# Step 4.1: Print the intercept and coefficient learned by the linear regression model
print("The Intercept for motor updrs:", motor_model.intercept_)
print("The coefficient for motor updrs:", motor_model.coef_)
print("The Intercept for total updrs:", total_model.intercept_)
print("The coefficient for total updrs:", total_model.coef_)

# Step 5.1: Make Predictions
motor_predictions = motor_model.predict(X_test)
total_predictions = total_model.predict(X_test)

# Step 5.2 Evaluate the Models
motor_mae = mean_absolute_error(y_motor_test, motor_predictions)
total_mae = mean_absolute_error(y_total_test, total_predictions)
motor_mse = mean_squared_error(y_motor_test, motor_predictions)
total_mse = mean_squared_error(y_total_test, total_predictions)
motor_r2 = r2_score(y_motor_test, motor_predictions)
total_r2 = r2_score(y_total_test, total_predictions)

# Print evaluation metrics
print(f"Motor UPDRS MAE: {motor_mae}")
print(f"Total UPDRS MAE: {total_mae}")
print("\nMotor UPDRS MSE:", motor_mse)
print("Total UPDRS MSE:", total_mse)
print("\nMotor UPDRS R-squared:", motor_r2)
print("Total UPDRS R-squared:", total_r2)

# Calculate Root Mean Square Error (RMSE)
motor_rmse = np.sqrt(motor_mse)
total_rmse = np.sqrt(total_mse)

# Calculate Normalized Root Mean Square Error (NRMSE)
motor_range = y_motor_test.max() - y_motor_test.min()
total_range = y_total_test.max() - y_total_test.min()
motor_nrmse = motor_rmse / motor_range
total_nrmse = total_rmse / total_range

# Print RMSE and NRMSE
print(f"Motor UPDRS RMSE: {motor_rmse}")
print(f"Total UPDRS RMSE: {total_rmse}")
print("\nMotor UPDRS NRMSE:", motor_nrmse)
print("Total UPDRS NRMSE:", total_nrmse)

# Print Predictions
print("Motor UPDRS Predictions:", motor_predictions)
print("Total UPDRS Predictions:", total_predictions)

# Scatter plot for motor UPDRS predictions
regression_model = LinearRegression()
y_motor_test = np.array(y_motor_test).reshape(-1, 1)
motor_predictions = np.array(motor_predictions).reshape(-1, 1)
regression_model.fit(y_motor_test, motor_predictions)
predictions = regression_model.predict(y_motor_test)

plt.scatter(y_motor_test, motor_predictions, label='Data Points')
plt.plot(y_motor_test, predictions, color='red', label='Best Fit Line')
plt.xlabel("Actual result")
plt.ylabel("Predicted result")
plt.title("Actual vs. Predicted Motor Results")
plt.legend()
plt.show()

# Scatter plot for total UPDRS predictions
r2 = r2_score(y_total_test, total_predictions)
y_total_test = np.array(y_total_test).reshape(-1, 1)
total_predictions = np.array(total_predictions).reshape(-1, 1)
regression_model.fit(y_total_test, total_predictions)
predictions = regression_model.predict(y_total_test)

plt.scatter(y_total_test, total_predictions, label='Data Points')
plt.plot(y_total_test, predictions, color='red', label='Best Fit Line')
plt.xlabel("Actual result")
plt.ylabel("Predicted result")
plt.title("Actual vs. Predicted Total UPDRS Results")
plt.legend()
plt.show()

# Step 6: Visualizations and Analysis

# Correlation matrix heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[['motor_updrs', 'total_updrs']], annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Summary and statistics using statsmodels
X = sm.add_constant(X)
model_motor = sm.OLS(y_motor, X).fit()
model_total = sm.OLS(y_total, X).fit()

print("Summary for motor_updrs:")
print(model_motor.summary())
print("\nSummary for total_updrs:")
print(model_total.summary())

# Box plots for sex vs. motor_updrs and total_updrs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.boxplot([df['motor_updrs'][df['sex'] == 0], df['motor_updrs'][df['sex'] == 1]], labels=['Male', 'Female'])
plt.xlabel('Sex')
plt.ylabel('Motor UPDRS')
plt.title('Box Plot: Sex vs. Motor UPDRS')

plt.subplot(1, 2, 2)
plt.boxplot([df['total_updrs'][df['sex'] == 0], df['total_updrs'][df['sex'] == 1]], labels=['Male', 'Female'])
plt.xlabel('Sex')
plt.ylabel('Total UPDRS')
plt.title('Box Plot: Sex vs. Total UPDRS')

plt.tight_layout()
plt.show()

# Box plots for selected independent variables vs. motor_updrs and total_updrs
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.boxplot([df['age'][df['sex'] == 0], df['age'][df['sex'] == 1]], labels=['Male', 'Female'])
plt.xlabel('Sex')
plt.ylabel('Age')
plt.title('Age vs. Motor UPDRS by Sex')

plt.subplot(2, 3, 2)
plt.boxplot([df['age'][df['sex'] == 0], df['age'][df['sex'] == 1]], labels=['Male', 'Female'])
plt.xlabel('Sex')
plt.ylabel('Age')
plt.title('Age vs. Total UPDRS by Sex')

plt.tight_layout()
plt.show()

# Histograms for test_time vs. motor_updrs and jitter(%) vs. total_updrs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['test_time'], bins=20, alpha=0.5)
plt.xlabel('Test time')
plt.ylabel('Number of days')
plt.title('Histogram: Test Time vs. Motor UPDRS')

plt.subplot(1, 2, 2)
plt.hist(df['jitter(%)'], bins=20, alpha=0.5)
plt.xlabel('Jitter(%)')
plt.ylabel('Number of days')
plt.title('Histogram: Jitter(%) vs. Total UPDRS')

plt.tight_layout()
plt.show()

# Scatter plots for selected independent variables vs. motor_updrs and total_updrs
independent_vars = [
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'nhr',
    'hnr', 'rpde', 'dfa', 'ppe'
]

for var in independent_vars:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    if df[var].dtype == 'object':
        sns.boxplot(x=var, y='motor_updrs', data=df)
    else:
        plt.scatter(df[var], df['motor_updrs'], alpha=0.5)
    plt.xlabel(var)
    plt.ylabel('Motor UPDRS')
    plt.title(f'Scatter Plot: {var} vs. Motor UPDRS')

    plt.subplot(1, 2, 2)
    if df[var].dtype == 'object':
        sns.boxplot(x=var, y='total_updrs', data=df)
    else:
        plt.scatter(df[var], df['total_updrs'], alpha=0.5)
    plt.xlabel(var)
    plt.ylabel('Total UPDRS')
    plt.title(f'Scatter Plot: {var} vs. Total UPDRS')

    plt.tight_layout()
    plt.show()
