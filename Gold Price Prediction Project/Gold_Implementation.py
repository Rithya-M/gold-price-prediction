# Importing required libraries
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit title and header
st.title("Gold Price Prediction Using Random Forest Regressor")
st.header("Predicting future gold prices based on historical data")

# Fetch historical gold price data using yfinance
ticker = 'GC=F'  # Gold futures data from Yahoo Finance
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # 5 years back

# Download gold price data
gold_data = yf.download(ticker, start=start_date, end=end_date)

# Show the first few rows of the dataset
#st.write("First few rows of the gold data:")
#st.write(gold_data.head())

# Resetting index to make sure 'Date' is not considered during correlation
gold_data.reset_index(inplace=True)

# Select only numeric columns for correlation calculation
numeric_data = gold_data.select_dtypes(include=[float, int])

# Calculate the correlation matrix for numeric columns only
correlation = numeric_data.corr()

# Display the correlation matrix
#st.write("Correlation Matrix:")
#st.write(correlation)

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Gold Price Data")
#st.pyplot(plt)

# Feature engineering: Using 'Date' column to predict the gold price
# Extract year, month, day, and day of the week from the 'Date' column
gold_data['Year'] = gold_data['Date'].dt.year
gold_data['Month'] = gold_data['Date'].dt.month
gold_data['Day'] = gold_data['Date'].dt.day
gold_data['DayOfWeek'] = gold_data['Date'].dt.dayofweek

# Drop the original 'Date' column and any columns that are not useful for prediction
gold_data = gold_data.drop(columns=['Date', 'Adj Close'])

# Feature columns (X) and target column (y)
X = gold_data.drop(columns='Close')  # Features (everything except 'Close' price)
y = gold_data['Close']  # Target (the closing price of gold)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the evaluation metrics
#st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot the predicted vs actual gold prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Prices")
plt.plot(y_pred, label="Predicted Prices", alpha=0.7)
plt.title("Actual vs Predicted Gold Prices")
plt.xlabel("Test Data Index")
plt.ylabel("Gold Price")
plt.legend()
#st.pyplot(plt)

# Allow the user to input features for prediction
st.header("Input Your Data to Predict Future Gold Price")

def user_input():
    # Inputs for SPX, USO, SLV, EUR/USD
    spx = st.number_input("SPX (S&P 500)", value=4000.0)
    uso = st.number_input("USO (Oil ETF)", value=70.0)
    slv = st.number_input("SLV (Silver ETF)", value=22.0)
    eur_usd = st.number_input("EUR/USD Exchange Rate", value=1.1)

    # Inputs for the gold data features (year, month, day, day of week)
    year = st.slider("Year", int(gold_data['Year'].min()), int(gold_data['Year'].max()), int(gold_data['Year'].mean()))
    month = st.slider("Month", 1, 12, int(gold_data['Month'].mean()))
    day = st.slider("Day", 1, 31, int(gold_data['Day'].mean()))
    day_of_week = st.slider("Day of the Week", 0, 6, int(gold_data['DayOfWeek'].mean()))
    
    input_data = {
        'Year': year, 
        'Month': month, 
        'Day': day, 
        'DayOfWeek': day_of_week,
        'SPX': spx, 
        'USO': uso, 
        'SLV': slv, 
        'EUR/USD': eur_usd
    }
    return pd.DataFrame(input_data, index=[0])

# Get user input
user_data = user_input()

# Button to trigger prediction
predict_button = st.button("Predict Gold Price")

if predict_button:
    # Ensure the model gets the same number of features
    if user_data.shape[1] == X_train.shape[1]:
        # Predict the future gold price using the trained Random Forest model
        user_prediction = rf_model.predict(user_data)

        # Display the predicted result
        st.write(f"Predicted Gold Price: ${user_prediction[0]:.2f}")
    else:
        st.error("Input features do not match the model's expected number of features.")
