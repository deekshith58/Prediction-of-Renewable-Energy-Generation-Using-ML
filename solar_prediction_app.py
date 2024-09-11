import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import streamlit as st

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Feature Engineering
def feature_engineering(data):
    data['hour'] = data.index.hour
    data['day'] = data.index.dayofyear
    data['day_of_week'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month

    for i in range(1, 6):
        data[f'lag_{i}'] = data['AC_POWER'].shift(i)

    return data

# Generation Prediction
def generation_prediction(data):
    st.subheader("Generation Prediction")
    st.write('give a data')

    uploaded_file = st.file_uploader('', type=["csv"])

    if uploaded_file is not None:
        # Load and preprocess data
        data = load_data(uploaded_file)
        data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
        data.set_index('DATE_TIME', inplace=True)
        data = feature_engineering(data)
        # Train-Test Split
        target = 'AC_POWER'
        features = ['hour', 'day', 'day_of_week', 'quarter', 'month', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        train_size = int(0.8 * len(data))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        train_data_imputed = imputer.fit_transform(train_data[features])
        test_data_imputed = imputer.transform(test_data[features])

        # Initialize the model
        model = HistGradientBoostingRegressor(random_state=42)

        # Fit the model
        model.fit(train_data_imputed, train_data[target])

        # Make predictions
        predictions = model.predict(test_data_imputed)

        # Calculate Mean Squared Error
        mse = mean_squared_error(test_data[target], predictions)

        # Create a DataFrame for results
        output_data = pd.DataFrame({
            'Date': test_data.index,
            'Actual': test_data[target],
            'Predicted': predictions
        })

        # Display predicted graph and results
        plt.figure(figsize=(10, 6))
        plt.plot(test_data.index, test_data[target], label='Actual')
        plt.plot(test_data.index, predictions, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('AC Power')
        plt.title('Actual vs. Predicted Solar Power Generation')
        plt.legend()
        st.pyplot(plt)  # Display the generated plot
        st.write(output_data)  # Display the DataFrame


        

# Consumption Prediction
def consumption_prediction(data):
    st.subheader("Consumption Prediction")
    st.write('give a data')
    uploaded_file = st.file_uploader('', type=["csv"])

    if uploaded_file is not None:
        # Load and preprocess data
        data = load_data(uploaded_file)
        data['TimeReviewed'] = pd.to_datetime(data['TimeReviewed'])
        data.set_index('TimeReviewed', inplace=True)
        # Define features and target variable
        features = ['Solar_Power_Consumption(Kw)', 'Temp( C)', 'EmployeeCount']
        target = 'Solar_Power_Consumption(Kw)'

        # Split the dataset
        train_size = int(0.8 * len(data))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Handle missing values
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # Define features and target for training
        X_train = train_data[features]
        y_train = train_data[target]

        # Initialize the model
        model = RandomForestRegressor(random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Define features for testing
        X_test = test_data[features]

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(test_data[target], predictions)

        # Create a DataFrame for results
        results_df = pd.DataFrame({
            'Actual': test_data[target],
            'Predicted': predictions
        })

        # Display predicted graph and results
        plt.figure(figsize=(10, 6))
        plt.plot(test_data.index, test_data[target], label='Actual')
        plt.plot(test_data.index, predictions, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Solar Power Consumption (kW)')
        plt.title('Actual vs. Predicted Solar Power Consumption')
        plt.legend()
        st.pyplot(plt)  # Display the generated plot
        st.write(results_df)  # Display the DataFrame


    

# Main Streamlit app
def main():
    st.title("Solar Power Prediction App")

    # Sidebar menu
    selected_option = st.sidebar.selectbox("Select Prediction Type", ["Generation", "Consumption"])

    if selected_option == "Generation":
        generation_prediction(pd.DataFrame)  # Provide your generation dataset file path
    elif selected_option == "Consumption":
        consumption_prediction(pd.DataFrame)  # Provide your consumption dataset file path

if __name__ == "__main__":
    main()
