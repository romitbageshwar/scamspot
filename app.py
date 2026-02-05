import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(
    page_title="ScamSpot - Fake Job Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ScamSpot — Spot Fake Job Postings")
st.write("Upload a CSV file containing job listings to detect fraudulent postings.")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

uploaded_file = st.file_uploader(
    "📂 Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "title" not in data.columns or "description" not in data.columns:
            st.error("CSV must contain 'title' and 'description' columns.")
            st.stop()

        data["title"] = data["title"].fillna("")
        data["description"] = data["description"].fillna("")

        data["text"] = data["title"] + " " + data["description"]

        predictions = model.predict(data["text"])
        probabilities = model.predict_proba(data["text"])[:, 1]

        data["prediction"] = predictions
        data["fraud_probability"] = probabilities

        data["prediction_label"] = data["prediction"].map({
            0: "Genuine",
            1: "Fraudulent"
        })

        st.success("✅ Predictions generated successfully!")

        st.subheader("📋 Prediction Results")
        st.dataframe(
            data[["title", "prediction_label", "fraud_probability"]]
            .sort_values("fraud_probability", ascending=False),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            pie_data = (
                data["prediction_label"]
                .value_counts()
                .reset_index()
            )
            pie_data.columns = ["label", "count"]

            fig_pie = px.pie(
                pie_data,
                names="label",
                values="count",
                title="Genuine vs Fraudulent Jobs"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_hist = px.histogram(
                data,
                x="fraud_probability",
                nbins=20,
                title="Fraud Probability Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("🚨 Top 10 Most Suspicious Job Listings")
        top10 = data.sort_values(
            "fraud_probability", ascending=False
        ).head(10)

        st.dataframe(
            top10[["title", "fraud_probability"]],
            use_container_width=True
        )

        st.subheader("📥 Download Predictions")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="scamspot_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ An error occurred while processing the file.")
        st.exception(e)
