from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
# Load the combined dataset
combined_df = pd.read_csv('./dataset/INFY.NS_historical_data.csv')

# Drop rows with missing values
combined_df.dropna(inplace=True)

# Convert relevant columns to numeric
numeric_columns = ['Open', 'High', 'Low', 'Volume']
for col in numeric_columns:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

# Drop rows with any remaining missing values after conversion
combined_df.dropna(inplace=True)

# Define features and target
X = combined_df[numeric_columns]

y = combined_df['Close']  # Predicting the 'Close' price

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

filename = './models/INFY_RF_ST_model.pkl'
with open(filename,'wb') as f:
    pickle.dump(scaler,f)
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)
filename = './models/INFY_RF_model.pkl'
with open(filename,'wb') as f:
    pickle.dump(model,f)
# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

import matplotlib.pyplot as plt

# Make predictions on the training set
y_train_pred = model.predict(X_train_scaled)

# Plot the actual vs predicted values for the training set
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(y_train)), y_train, color='blue', label='Actual Training Data')
plt.plot(np.arange(len(y_train)), y_train_pred, color='red', linestyle='--', label='Predicted Training Data')
plt.title('Training Set: Actual vs Predicted Close Prices')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Plot the actual vs predicted values for the testing set
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(y_test)), y_test, color='blue', label='Actual Testing Data')
plt.plot(np.arange(len(y_test)), y_pred, color='red', linestyle='--', label='Predicted Testing Data')
plt.title('Testing Set: Actual vs Predicted Close Prices')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()



