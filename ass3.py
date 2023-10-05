import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Load datasets
df_po1 = pd.read_csv('po1_data.txt', header=None, names=[
    'subject', 'jitter%', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
    'shimmer%', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
    'harmonicity_autocorrelation', 'harmonicity_nhr', 'harmonicity_hnr',
    'pitch_median', 'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
    'pulse_number', 'pulse_periods', 'pulse_mean', 'pulse_std',
    'voice_unvoiced_frames', 'voice_num_breaks', 'voice_degree_breaks', 'updrs', 'pd_indicator'
])
df_po2 = pd.read_csv('po2_data.csv')

# Data Exploration
print("Dataset 1:")
print(df_po1.head())
print(df_po1.info())
print(df_po1.describe())

print("\nDataset 2:")
print(df_po2.head())
print(df_po2.info())
print(df_po2.describe())

# Combining Datasets
df_combined = pd.merge(df_po1, df_po2, left_on='subject', right_on='subject#', how='inner')
print("combined data--->", df_combined.columns)


# Splitting the Data
X_combined = df_combined[['age', 'sex', 'test_time', 'jitter%', 'shimmer%', 'harmonicity_nhr', 'pitch_mean', 'pulse_mean', 'voice_num_breaks']]
y_motor_combined = df_combined['motor_updrs']
y_total_combined = df_combined['total_updrs']

X_train_combined, X_test_combined, y_motor_train_combined, y_motor_test_combined, y_total_train_combined, y_total_test_combined = train_test_split(
    X_combined, y_motor_combined, y_total_combined, test_size=0.2, random_state=42
)

# Linear Regression Model for Motor UPDRS
model_motor_combined = LinearRegression()
model_motor_combined.fit(X_train_combined, y_motor_train_combined)

# Predictions
y_motor_pred_combined = model_motor_combined.predict(X_test_combined)

# Model Evaluation
print("\nMotor UPDRS Model Evaluation (Combined Datasets):")
print("Mean Squared Error:", mean_squared_error(y_motor_test_combined, y_motor_pred_combined))
print("R-squared:", r2_score(y_motor_test_combined, y_motor_pred_combined))

# Linear Regression Model for Total UPDRS
model_total_combined = LinearRegression()
model_total_combined.fit(X_train_combined, y_total_train_combined)

# Predictions
y_total_pred_combined = model_total_combined.predict(X_test_combined)

# Model Evaluation
print("\nTotal UPDRS Model Evaluation (Combined Datasets):")
print("Mean Squared Error:", mean_squared_error(y_total_test_combined, y_total_pred_combined))
print("R-squared:", r2_score(y_total_test_combined, y_total_pred_combined))

# Summary Stats with Statsmodels
X_train_combined_sm = sm.add_constant(X_train_combined)
model_motor_combined_sm = sm.OLS(y_motor_train_combined, X_train_combined_sm).fit()
print("\nMotor UPDRS Model Summary (Combined Datasets):")
print(model_motor_combined_sm.summary())

model_total_combined_sm = sm.OLS(y_total_train_combined, X_train_combined_sm).fit()
print("\nTotal UPDRS Model Summary (Combined Datasets):")
print(model_total_combined_sm.summary())

# Data Visualization
# Scatter plot for Motor UPDRS
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_motor_test_combined, y_motor_pred_combined, alpha=0.5)
plt.title('Motor UPDRS: Actual vs Predicted')
plt.xlabel('Actual Motor UPDRS')
plt.ylabel('Predicted Motor UPDRS')
plt.grid(True)

# Scatter plot for Total UPDRS
plt.subplot(1, 2, 2)
plt.scatter(y_total_test_combined, y_total_pred_combined, alpha=0.5)
plt.title('Total UPDRS: Actual vs Predicted')
plt.xlabel('Actual Total UPDRS')
plt.ylabel('Predicted Total UPDRS')
plt.grid(True)

plt.tight_layout()
plt.show()


# Display plots if desired
plt.show()
