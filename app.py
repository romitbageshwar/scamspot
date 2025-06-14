import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="ScamSpot", layout="wide")
st.title("🔍 ScamSpot — Spot Fake Job Postings")

# Load model
model = joblib.load('model.pkl')

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['title'] = data['title'].fillna('')
    data['description'] = data['description'].fillna('')
    data['text'] = data['title'] + ' ' + data['description']
    data['fraud_probability'] = model.predict_proba(data['text'])[:, 1]
    data['prediction'] = (data['fraud_probability'] > 0.5).astype(int)

    st.subheader("✅ Predictions")
    st.dataframe(data[['title', 'location', 'fraud_probability', 'prediction']])

    # Pie chart
    pie = data['prediction'].value_counts().rename({0: 'Genuine', 1: 'Fraudulent'}).reset_index()
    fig_pie = px.pie(pie, values='prediction', names='index', title="Genuine vs Fraudulent Jobs")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Histogram
    fig_hist = px.histogram(data, x='fraud_probability', nbins=20, title='Fraud Probability Distribution')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Top 10 suspicious
    st.subheader("⚠️ Top 10 Suspicious Jobs")
    top10 = data.sort_values('fraud_probability', ascending=False).head(10)
    st.dataframe(top10[['title', 'location', 'fraud_probability']])
