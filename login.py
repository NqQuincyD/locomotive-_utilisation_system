import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
#import weasyprint
import io
import base64

conn = sqlite3.connect("admin.db", timeout=10.0)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS admins (
            username TEXT,
            password TEXT,
            role TEXT
            )""")
conn.commit()

c.execute("INSERT INTO admins VALUES ('admin', 'admin', 'super_admin')")
conn.commit()

def login(username, password, role):
    c.execute("SELECT role FROM admins WHERE username=? AND password=? AND role=?", (username, password, role))
    row = c.fetchone()
    if row:
        return row[0]
    else:
        return None

def create_admin(username, password, role):
    c.execute("INSERT INTO admins VALUES (?, ?, ?)", (username, password, role))
    conn.commit()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

st.header(":green[Login]")
username = st.text_input(":blue[Username]",placeholder="Enter your Username")
password = st.text_input(":blue[Password]", placeholder="Enter your password", type="password")
login_as = st.radio(":blue[Login as]", ["Super Admin", "User"])

if st.button("Login"):
    if login_as == "Super Admin":
        role = "super_admin"
    else:
        role = "user"
    logged_in_role = login(username, password, role)
    if logged_in_role:
        st.session_state.logged_in = True
        st.session_state.role = logged_in_role
        st.success("Logged in as " + logged_in_role)
    else:
        st.error("Invalid username or password")

if st.session_state.logged_in:
    if st.session_state.role == "super_admin":
        st.header(":green[Create Users]")
        new_username = st.text_input(":blue[New Username]", placeholder="Enter your Username")
        new_password = st.text_input(":blue[New Password]", placeholder="Enter your password", type="password")
        new_role = st.selectbox(":blue[New Role]", ["super_admin", "admin", "user"])

        if st.button("Create"):
            create_admin(new_username, new_password, new_role)
            st.success("New admin created successfully")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()
    else:
        st.header("User Dashboard")
        
        # Load data
        data = pd.read_csv('Loco Data.csv')

        # Separate Features and Target Variables
        X = data[['LOCO_TYPE', 'LOCO_NUMBER', 'YEAR']]
        y = data[['Availability_Days', 'Train_kms', 'Train_km_per_day', 'Reliabilty', 'Days_before_failure']]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create the Random Forest Regression Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        rf_model.fit(X_train, y_train)
        # Create a database connection
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()

        # Create a table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                loco_type INTEGER,
                loco_number INTEGER,
                year INTEGER,
                availability_days REAL,
                train_kms REAL,
                train_km_per_day REAL,
                reliability REAL,
                days_before_failure REAL,
                average_performance REAL
            )
        ''')
        conn.commit()

        # Streamlit app
        st.sidebar.title("Locomotive Utilization System")
        st.markdown('<div style="position: fixed; bottom: 0; width: 100%; text-align: center;"><p><a href="https://nrz.co.zw/">Contact Details @ NqQuincyD.serv 2024</a></p></div>', unsafe_allow_html=True)

        st.markdown(
                """
                <style>
                .stButton>button {
                    color: blue;
                }
                .css-1fcmnwj h1 {
                    color: blue;
                }
                </style>
                """,
            unsafe_allow_html=True
            )

        # Sidebar sections
        sections = st.sidebar.selectbox("Go to",["Home","Performance Predictions", "Performance Analysis","Prediction Report"])
        #selected_section = st.sidebar.radio("Select Section", sections)

        # Global variables to store predictions and average performance
        performance_prediction = None
        average_performance = None
        show_predictions = False
        # Database cleared flag
        database_cleared = False

        # Create a map centered around Zimbabwe
        zimbabwe_map = folium.Map(location=[-19.0154, 29.1549], zoom_start=6)
        if sections == "Home":
            st.markdown("<h1 style='text-align: center;'>Welcome to Zimbabwe Railway Traffic Route Analysis.</h1>", unsafe_allow_html=True)
                # Create a map centered around Zimbabwe
            zimbabwe_map = folium.Map(location=[-19.0154, 29.1549], zoom_start=6)

            # Add markers for key railway locations
            locations = {
                "Bulawayo": [-20.1561, 28.5834],
                "Harare": [-17.8292, 31.0522],
                "Gweru": [-19.4500, 29.8200],
                "Victoria Falls": [-17.9244, 25.8567],
                "Maputo (Mozambique)": [-25.9667, 32.5833]
            }

            for location, coords in locations.items():
                folium.Marker(location=coords, popup=location).add_to(zimbabwe_map)

            # Display the map in Streamlit
            folium_static(zimbabwe_map)
        #loco_number=data['LOCO_NUMBER'].unique()
        # Performance Predictions section
        if sections == "Performance Predictions":
            st.header("Performance Predictions")

            # Input fields
            loco_type = st.selectbox("SELECT 'LOCO_TYPE' ", (1, 2), format_func=lambda x: "NRZ FLEET" if x == 1 else "HIRED FLEET")
            #loco_number = st.selectbox("Select Loco Number",data['LOCO_NUMBER'].unique())
            loco_number = st.number_input("LOCO_NUMBER", min_value=0, value=0)
            year = st.number_input("YEAR", min_value=1900, value=2024)
            
            # Prediction button and Hide button
            col1, col2 = st.columns(2)
            with col1:
                predict_button = st.button("Click to Predict")
            with col2:
                hide_button = st.button("Hide Predictions")

            # Display predictions only when the button is clicked
            if predict_button:
                show_predictions = True

            if hide_button:
                show_predictions = False

            if show_predictions:
                # Check if the entered LOCO_NUMBER exists in the dataset
                if loco_number not in data['LOCO_NUMBER'].values:
                    st.error("The entered loco number is invalid.Please enter a valid loco number")
                # Predict performance
                else:
                    user_input = [loco_type, loco_number, year]
                    user_input_scaled = scaler.transform([user_input])
                    performance_prediction = rf_model.predict(user_input_scaled)[0]


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
                        title="Locomotive Performance Chart",
                        xaxis_title="Performance Metrics",
                        yaxis_title="Predicted Values",
                        xaxis_tickangle=-45
                    )

                    st.plotly_chart(fig)
                # Insert prediction into database
                    cursor.execute('''
                    INSERT INTO predictions (loco_type, loco_number, year, availability_days, train_kms, train_km_per_day, reliability, days_before_failure, average_performance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (loco_type, loco_number, year, performance_prediction[0], performance_prediction[1], performance_prediction[2], performance_prediction[3], performance_prediction[4], average_performance))
                    conn.commit()

        # Performance Analysis section
        elif sections == "Performance Analysis":
            st.header("Performance Train km Metrics")

            # Create a scatter plot of actual performance vs. predicted performance
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test['Train_km_per_day'],
                y=rf_model.predict(X_test)[:, 2],
                mode='markers',
                marker=dict(color='blue', size=8),
                name='Actual vs. Predicted'
            ))

            # Create the line graph of predictions
            if performance_prediction is not None:
                prediction_df = pd.DataFrame({
                    'Variable': ['AVAILABILITY_Days', 'Train_kms', 'Train_km_per_day', 'Reliability', 'Days_before_failure'],
                    'Predicted Value': performance_prediction
                })
                prediction_df.loc[len(prediction_df)] = ['Average Performance (%)', average_performance]
                # Transpose the dataframe
                prediction_df = prediction_df.set_index('Variable').T
                # Display prediction (full width)
                st.dataframe(prediction_df, use_container_width=True)
                fig.add_trace(go.Scatter(
                    x=prediction_df['Variable'],
                    y=prediction_df['Predicted Value'],
                    mode='lines+markers',
                    marker=dict(color='red', size=10),
                    line=dict(color='red', width=3),
                    name='Predicted Performance'
                ))

            fig.update_layout(
                title="Train km metrics",
                xaxis_title="Performance Metrics",
                yaxis_title="Predicted Values",
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig)
        # Prediction Database section
        elif sections == "Prediction Report":
            st.header("Report")
            # Fetch predictions from the database
            cursor.execute("SELECT * FROM predictions")
            predictions = cursor.fetchall()
            # Create a DataFrame from the fetched data
            prediction_df = pd.DataFrame(predictions, columns=['id', 'loco_type', 'loco_number', 'year', 'availability_days', 'train_kms', 'train_km_per_day', 'reliability', 'days_before_failure', 'average_performance'])
            # Display the DataFrame
            st.dataframe(prediction_df)
            # Save button
            file_format = st.selectbox("Select File Format", ["Excel", "CSV", "PDF"])
            save_button = st.button("Save Predictions")

            if save_button:
                # Create a buffer for the file
                buffer = io.BytesIO()

                # Save the DataFrame to the buffer based on the selected format
                if file_format == "Excel":
                    prediction_df.to_excel(buffer, index=False)
                    file_name = "predictions.xlsx"
                elif file_format == "CSV":
                    prediction_df.to_csv(buffer, index=False)
                    file_name = "predictions.csv"
                elif file_format == "PDF":
                    # Create a buffer for the PDF
                    buffer = io.BytesIO()
                    c = canvas.Canvas(buffer, pagesize=letter)

                    # Set font and starting position
                    c.setFont("Helvetica", 12)
                    x = 0.5 * inch
                    y = 10.5 * inch

                    # Add DataFrame header
                    for col in prediction_df.columns:
                        c.drawString(x, y, col)
                        x += 2 * inch

                    # Add DataFrame rows
                    y -= 0.5 * inch
                    for row in prediction_df.values:
                        x = 0.5 * inch
                        for cell in row:
                            c.drawString(x, y, str(cell))
                            x += 2 * inch
                        y -= 0.5 * inch

                    # Save the PDF to the buffer
                    c.save()

                    # Create a download link
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    file_name = "predictions.xlsx"
                    file_name = "predictions.csv"
                    
                    href = f'<a href="data:application/pdf;base64,{b64}" download="predictions.pdf">Download Predictions</a>'
                    #href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download Predictions</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Clear database button
            clear_database_button = st.button("Clear Database")
            if clear_database_button:
                cursor.execute("DELETE FROM predictions")
                conn.commit()
                st.success("Database cleared successfully!")
                database_cleared = True
                    # Convert DataFrame to HTML
            # Refresh button
            if database_cleared:
                refresh_button = st.button("Refresh")
                if refresh_button:
                    # Fetch predictions from the database
                    cursor.execute("SELECT availability_days, train_kms, train_km_per_day, reliability, days_before_failure, average_performance FROM predictions")
                    predictions = cursor.fetchall()
                    # Create a DataFrame from the fetched data
                    prediction_df = pd.DataFrame(predictions, columns=['Availability_Days', 'Train_kms', 'Train_km_per_day', 'Reliability', 'Days_before_failure', 'Average Performance (%)'])
                    # Display the DataFrame
                    st.dataframe(prediction_df)
                    database_cleared = False
                
        # Close the database connection
        conn.close()

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()
