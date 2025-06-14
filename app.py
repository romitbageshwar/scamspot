import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("🔍 ScamSpot — Spot Fake Job Postings")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:", data.head())

    # Make sure 'title' and 'description' exist
    if 'title' in data.columns and 'description' in data.columns:
        # Combine title and description
        text = data['title'].fillna('') + ' ' + data['description'].fillna('')
        
        # Predict fraud probability
        pred_proba = model.predict_proba(text)[:, 1]
        pred = model.predict(text)

        # Add to DataFrame
        data['fraud_probability'] = pred_proba
        data['prediction'] = pred

        # Show results table
        st.subheader("✅ Predictions")
        st.write(data[['title', 'fraud_probability', 'prediction']])

        # Pie Chart: Genuine vs Fraudulent
        pie = data['prediction'].value_counts().rename_axis('label').reset_index(name='value')
        pie['label'] = pie['label'].map({0: 'Genuine', 1: 'Fraudulent'})
        fig_pie = px.pie(pie, values='value', names='label', title="Genuine vs Fraudulent Jobs")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Histogram: Fraud Probabilities
        fig_hist = px.histogram(data, x='fraud_probability', nbins=20, title="Distribution of Fraud Probability")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Top 10 suspicious listings
        top10 = data.sort_values(by='fraud_probability', ascending=False).head(10)
        st.subheader("🚨 Top 10 Most Suspicious Listings")
        st.write(top10[['title', 'fraud_probability']])

    else:
        st.error("Uploaded CSV must have 'title' and 'description' columns.")
