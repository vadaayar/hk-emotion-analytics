import streamlit as st
import pandas as pd
import sqlite3
import pickle
import os
import smtplib
from email.message import EmailMessage
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.pdfgen import canvas
import fitz  # PyMuPDF
from datetime import datetime

# Load model and vectorizer
with open("emotion_classifier.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# SQLite setup
def init_db():
    conn = sqlite3.connect("emotion_app_data.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS emotion_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            predicted_emotion TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Save prediction to DB
def save_prediction_to_db(text, emotion):
    conn = sqlite3.connect("emotion_app_data.db")
    c = conn.cursor()
    c.execute("INSERT INTO emotion_predictions (input_text, predicted_emotion, timestamp) VALUES (?, ?, ?)",
              (text, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Predict function
def predict_emotion(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

# Email alert simulation
def send_email_alert(email, text, emotion):
    st.info(f"📧 Simulating alert to {email}...\nText: {text}\nPredicted Emotion: {emotion}")

# Export results to PDF
def export_results_to_pdf(df):
    file_path = "emotion_results.pdf"
    c = canvas.Canvas(file_path)
    c.setFont("Helvetica", 12)
    c.drawString(100, 800, "Emotion Classification Results")
    y = 780
    for index, row in df.iterrows():
        c.drawString(100, y, f"{index+1}. {row['Text']} -> {row['Emotion']}")
        y -= 20
    c.save()
    return file_path

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.split('\n')

# Dark theme
st.set_page_config(page_title="AI Emotion Classifier", layout="wide", page_icon="🧠")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📄 Real-time", "📁 CSV Upload", "📑 PDF Upload", "📊 Visualize", "📬 Email Alert", "🙏 Thanks"])

# HOME PAGE
if page == "🏠 Home":
    st.markdown("## 🧠 AI Emotion Classifier")
    st.markdown("Welcome to the **AI Emotion Classifier** app. You can classify emotions from text input, CSV, or PDF files. Results are stored in a database, and you can export them as PDF or simulate email alerts.")
    st.image("banner.png", width=700)
    st.success("Get started from the left menu! 🚀")

# REAL-TIME
elif page == "📄 Real-time":
    st.subheader("💬 Enter Text to Predict Emotion")
    text_input = st.text_area("Your text here:")
    if st.button("Classify"):
        if text_input:
            emotion = predict_emotion(text_input)
            save_prediction_to_db(text_input, emotion)
            st.success(f"Predicted Emotion: **{emotion}**")
        else:
            st.warning("Please enter some text.")

# CSV Upload
elif page == "📁 CSV Upload":
    st.subheader("📂 Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            df["Emotion"] = df["text"].apply(lambda x: predict_emotion(x))
            for idx, row in df.iterrows():
                save_prediction_to_db(row["text"], row["Emotion"])
            st.dataframe(df)
            if st.button("Export as PDF"):
                path = export_results_to_pdf(df[["text", "Emotion"]].rename(columns={"text": "Text"}))
                with open(path, "rb") as file:
                    st.download_button(label="📥 Download PDF", data=file, file_name="emotion_results.pdf", mime="application/pdf")

# PDF Upload
elif page == "📑 PDF Upload":
    st.subheader("📑 Upload a PDF file")
    pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if pdf_file:
        lines = extract_text_from_pdf(pdf_file)
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        results = [{"Text": line, "Emotion": predict_emotion(line)} for line in cleaned_lines]
        for res in results:
            save_prediction_to_db(res["Text"], res["Emotion"])
        df = pd.DataFrame(results)
        st.dataframe(df)
        if st.button("Export as PDF"):
            path = export_results_to_pdf(df)
            with open(path, "rb") as file:
                st.download_button(label="📥 Download PDF", data=file, file_name="emotion_results.pdf", mime="application/pdf")

# VISUALIZATION
elif page == "📊 Visualize":
    st.subheader("📊 View Recent Predictions")
    conn = sqlite3.connect("emotion_app_data.db")
    df = pd.read_sql_query("SELECT * FROM emotion_predictions ORDER BY id DESC LIMIT 100", conn)
    st.dataframe(df)
    conn.close()

# EMAIL
elif page == "📬 Email Alert":
    st.subheader("📧 Simulate Email Alert")
    email = st.text_input("Receiver Email")
    text = st.text_area("Enter text")
    if st.button("Send Alert"):
        if email and text:
            emotion = predict_emotion(text)
            send_email_alert(email, text, emotion)
            save_prediction_to_db(text, emotion)
        else:
            st.warning("Please fill all fields.")

# THANK YOU
elif page == "🙏 Thanks":
    st.success("🎉 Thank you for using the AI Emotion Classifier!")
    st.markdown("💡 **Built with Python, Streamlit, and ML**")
    st.markdown("📫 Contact us at `emotion@classifier.ai`")






