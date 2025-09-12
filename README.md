🔍 ScamSpot — Spot Fake Job Postings

ScamSpot is an intelligent machine learning tool that detects fraudulent job listings BEFORE users apply — helping job seekers stay safe and saving them time and money.

-----------------------------------------------------------
🚀 Key Features

✔️ Upload any CSV file with job postings
✔️ Predict if each posting is Genuine or Fraudulent
✔️ See fraud probabilities for every listing
✔️ Interactive dashboard:
   - Table with predictions
   - Histogram of fraud probabilities
   - Pie chart of Genuine vs Fraudulent jobs
   - Top 10 most suspicious listings
✔️ Download predictions as CSV

-----------------------------------------------------------
📊 Model Performance

Metric   : F1-Score
Value    : 0.6080

*Evaluated on the provided training data with class imbalance handling.*

-----------------------------------------------------------
⚙️ Tech Stack

- Python
- Scikit-Learn for model training
- Streamlit for the interactive dashboard
- Pandas & Plotly for data handling and charts

-----------------------------------------------------------
🗂️ Project Structure

scamspot/
 ├── app.py              -> Streamlit dashboard
 ├── model.pkl           -> Trained binary classifier
 ├── test_sample.csv     -> Example CSV for testing
 ├── requirements.txt    -> Python dependencies
 └── README.txt          -> This file

-----------------------------------------------------------
🏁 How to Run Locally

1️⃣ Clone this repository

   git clone https://github.com/romitbageshwar/scamspot.git
   cd scamspot

2️⃣ Install dependencies

   pip install -r requirements.txt

3️⃣ Run the dashboard

   streamlit run app.py

4️⃣ Use it

   - Click 'Browse files' to upload any job listings CSV
   - Get predictions and download results
   - Explore fraud insights via the charts

-----------------------------------------------------------
🌐 Live Demo

If deployed on Streamlit Cloud, you can access it directly:
👉 https://scamspot-8h5cywahpaxkdkfkeyxqh5.streamlit.app/
-----------------------------------------------------------
📁 Example Test File

Use 'test_sample.csv' to test the app.
It’s included in this repo for your convenience.

-----------------------------------------------------------
🧭 Future Work

✔️ Improve fraud detection using advanced NLP embeddings
✔️ Add real-time API scanning for job boards
✔️ Auto-retraining pipeline for continuous learning
✔️ Email or push alerts for high-risk jobs
✔️ More explainable AI features (e.g., SHAP plots)

-----------------------------------------------------------
📹 Project Video

Watch the full demo & walkthrough here:
👉https://drive.google.com/file/d/1lsui0ikvtllTyEeqOMLnoePK0TcxwCb8/view?usp=sharing

-----------------------------------------------------------
📜 License

This project is for educational purposes only.

✨ Let ScamSpot fight frauds, so you don’t have to!
