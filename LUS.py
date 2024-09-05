#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# In[102]:


data = pd.read_csv('Loco Data.csv')
data


# In[104]:


# Separate Features and Target Variables
X = data[['LOCO_TYPE', 'LOCO_NUMBER', 'YEAR']]
y = data[['Availability_Days', 'Train_kms', 'Train_km_per_day', 'Reliabilty', 'Days_before_failure']]


# In[106]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[108]:


# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[110]:


# Create the Random Forest Regression Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[112]:


# Train the model
rf_model.fit(X_train, y_train)


# In[114]:


# Make predictions
y_pred = rf_model.predict(X_test)


# In[116]:


# Evaluate the model (using mean squared error and r-squared)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[118]:


print('Mean squared error', mse)
print('R_squared', r2)


# In[134]:


# User Input and Prediction (with Recommendation)
def predict_loco_performance():
    loco_type = int(input("Enter LOCO_TYPE (1 for NRZ FLEET, 2 for HIRED FLEET): ")) 
    loco_number = int(input("Enter LOCO_NUMBER: "))
    year = int(input("Enter YEAR: "))

    # Prepare input for prediction (as a list)
    user_input = [loco_type, loco_number, year]

    # Scale user input 
    user_input_scaled = scaler.transform([user_input])  

    # Make predictions
    performance_prediction = rf_model.predict(user_input_scaled)[0]

    # Round AVAILABILITY_Days and Days_before_failure to nearest whole number
    performance_prediction[0] = round(performance_prediction[0])
    performance_prediction[4] = round(performance_prediction[4])

    # Calculate Average Performance as a Percentage
    reliability = performance_prediction[3]  # Reliability (from the prediction)
    train_km_per_day = performance_prediction[2]  # Train_km_per_day (from the prediction)
    average_performance = (train_km_per_day/reliability ) * 100  # Multiply by 100 for percentage

    # Create a DataFrame from the predictions
    prediction_df = pd.DataFrame({
        'Variable': ['AVAILABILITY_Days', 'Train_kms', 'Train_km_per_day', 'Reliability', 'Days_before_failure'],
        'Predicted Value': performance_prediction
    })

    # Add Average Performance to the DataFrame
    prediction_df.loc[len(prediction_df)] = ['Average Performance (%)', average_performance]

    # Call the recommendation function
    recommendation = generate_recommendation(loco_type, average_performance)

    print("\nPredicted Locomotive Performance:")
    print(prediction_df)
    print(f"\n{recommendation}")
    # Plot the prediction output using Plotly
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
        xaxis_tickangle=-45  # Rotate x-axis labels for better readability
    )

    fig.show()

    

# Recommendation Function
def generate_recommendation(loco_type, average_performance):
    if loco_type == 2:  # User entered HIRED FLEET (LOCO_TYPE 1)
        fleet_name = "HIRED FLEET"
    else:
        fleet_name = "NRZ FLEET"

    if average_performance >= 5:
        recommendation = f"Recommendation: Based on the predicted average performance of {average_performance:.2f}%, you should consider using this {fleet_name}."
    else:
        recommendation = f"Recommendation: Based on the predicted average performance of {average_performance:.2f}%, you can consider an alternative fleet."

    return recommendation

# Call the function to make a prediction
predict_loco_performance()


# In[ ]:




