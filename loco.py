import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('Loco Data.csv')

# Separate Features and Target Variables
X = data[['LOCO_TYPE', 'LOCO_NUMBER', 'YEAR']]
y = data[['Availability_Days', 'Train_kms', 'Train_km_per_day', 'Reliabilty', 'Days_before_failure']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the Random Forest Regression Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Streamlit app
st.title("Locomotive Performance Prediction System")

# Sidebar sections
sections = ["Performance Predictions", "Performance Analysis"]
selected_section = st.sidebar.radio("Select Section", sections)

# Performance Predictions section
if selected_section == "Performance Predictions":
    st.header("Performance Predictions")
    
    # Input fields
    loco_type = st.selectbox("LOCO_TYPE", (1, 2), format_func=lambda x: "NRZ FLEET" if x == 1 else "HIRED FLEET")
    loco_number = st.number_input("LOCO_NUMBER", placeholder='Enter value')
    year = st.number_input("YEAR", min_value=1900, value=2021)
    # Prediction button
    predict_button = st.button("Predict Performance")

    # Display predictions only when the button is clicked
    if predict_button:
        # Predict performance
        user_input = [loco_type, loco_number, year]
        user_input_scaled = scaler.transform([user_input])
        performance_prediction = rf_model.predict(user_input_scaled)[0]

    # Predict performance
    user_input = [loco_type, loco_number, year]
    user_input_scaled = scaler.transform([user_input])
    performance_prediction = rf_model.predict(user_input_scaled)[0]

    # Round AVAILABILITY_Days and Days_before_failure to nearest whole number
    performance_prediction[0] = round(performance_prediction[0])
    performance_prediction[4] = round(performance_prediction[4])

    # Calculate Average Performance as a Percentage
    reliability = performance_prediction[3]
    train_km_per_day = performance_prediction[2]
    average_performance = (train_km_per_day/reliability ) * 100

    # Create a DataFrame from the predictions
    prediction_df = pd.DataFrame({
        'Variable': ['AVAILABILITY_Days', 'Train_kms', 'Train_km_per_day', 'Reliability', 'Days_before_failure'],
        'Predicted Value': performance_prediction
    })

    # Add Average Performance to the DataFrame
    prediction_df.loc[len(prediction_df)] = ['Average Performance (%)', average_performance]

    # Display prediction
    st.dataframe(prediction_df)
    # Recommendation section
    #st.header("Recommendations")
    fleet_name = "HIRED FLEET" if loco_type == 2 else "NRZ FLEET"
    recommendation = ""
    if average_performance >= 6:
       recommendation = f"Based on the predicted average performance of {average_performance:.2f}%, you should consider using this {fleet_name}."
    else:
       recommendation = f"Based on the predicted average performance of {average_performance:.2f}%, you can consider an alternative fleet."

    st.header("Recommendations")
    st.success(recommendation)

    # Plot the prediction
    fig = go.Figure(data=go.Scatter(
        x=prediction_df['Variable'],
        y=prediction_df['Predicted Value'],
        mode='lines+markers',
        marker=dict(color='skyblue', size=10),
        line=dict(color='skyblue', width=3)
    ))

    fig.update_layout(
        title="Predicted Locomotive Performance",
        xaxis_title="Performance Metrics",
        yaxis_title="Predicted Values",
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig)

# Performance Analysis section
elif selected_section == "Performance Analysis":
    st.header("Performance Analysis")

    # Create a scatter plot of actual performance vs. predicted performance
    fig = go.Figure(data=go.Scatter(
        x=y_test['Train_km_per_day'],
        y=rf_model.predict(X_test)[:, 2],
        mode='markers',
        marker=dict(color='blue', size=8)
    ))

    fig.update_layout(
        title="Actual vs. Predicted Train_km_per_day",
        xaxis_title="Actual Train_km_per_day",
        yaxis_title="Predicted Train_km_per_day"
    )
   
    st.plotly_chart(fig)

