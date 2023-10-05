import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('/Users/yaminhossain/Desktop/CDU/Data_Science/Assignment3/po2_data.csv')

# Data Exploration
print("printing head\n",df.head())
print("printing info\n",df.info())
print("printing describe\n",df.describe())

# Data Visualization
sns.pairplot(df[['age', 'sex', 'test_time', 'motor_updrs', 'total_updrs']])
plt.show()

# Correlation Matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# Splitting the Data
X = df[['age', 'sex', 'test_time']]
y_motor = df['motor_updrs']
y_total = df['total_updrs']
X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(
    X, y_motor, y_total, test_size=0.2, random_state=42
)

# Linear Regression Model for Motor UPDRS
model_motor = LinearRegression()
model_motor.fit(X_train, y_motor_train)

# Predictions
y_motor_pred = model_motor.predict(X_test)

# Model Evaluation
print("Motor UPDRS Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_motor_test, y_motor_pred))
print("R-squared:", r2_score(y_motor_test, y_motor_pred))

# Linear Regression Model for Total UPDRS
model_total = LinearRegression()
model_total.fit(X_train, y_total_train)

# Predictions
y_total_pred = model_total.predict(X_test)

# Model Evaluation
print("\nTotal UPDRS Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_total_test, y_total_pred))
print("R-squared:", r2_score(y_total_test, y_total_pred))

# Summary Stats with Statsmodels
X_train_sm = sm.add_constant(X_train)
model_motor_sm = sm.OLS(y_motor_train, X_train_sm).fit()
print("\nMotor UPDRS Model Summary:")
print(model_motor_sm.summary())

model_total_sm = sm.OLS(y_total_train, X_train_sm).fit()
print("\nTotal UPDRS Model Summary:")
print(model_total_sm.summary())


# # Load datasets
# df_po1 = pd.read_csv('po1_data.txt', header=None, names=[
#     'subject', 'jitter%', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
#     'shimmer%', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
#     'harmonicity_autocorrelation', 'harmonicity_nhr', 'harmonicity_hnr',
#     'pitch_median', 'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
#     'pulse_number', 'pulse_periods', 'pulse_mean', 'pulse_std',
#     'voice_unvoiced_frames', 'voice_num_breaks', 'voice_degree_breaks', 'updrs', 'pd_indicator'
# ])