import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="Revenue Forecasting Agent", page_icon="📈", layout="wide")

st.title("📈 Revenue Forecasting Agent")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the Excel file
    try:
        df = pd.read_excel(uploaded_file)

        # Display the first few rows of the file
        st.subheader("📋 Uploaded Data Preview")
        st.write(df.head())

        # Check if the required columns are present
        if 'Date' in df.columns and 'Revenue' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[['Date', 'Revenue']]
            df.columns = ['ds', 'y']  # Prophet expects columns named 'ds' and 'y'

            # Initialize and fit the Prophet model
            model = Prophet()
            model.fit(df)

            # Make future predictions
            future = model.make_future_dataframe(periods=30)  # Forecasting 30 days into the future
            forecast = model.predict(future)

            # Display the forecasted data
            st.subheader("📈 Forecasted Data")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

            # Plotting the forecast
            st.subheader("📊 Forecast Plot")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Plotting forecast components
            st.subheader("📊 Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
        else:
            st.error('The uploaded file must contain "Date" and "Revenue" columns.')
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info('Please upload an Excel file with columns Date and Revenue.')
