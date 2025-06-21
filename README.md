# 💡 HK Emotion Analytics

An AI-powered web app to classify emotions from text using a machine learning model (MultinomialNB). Built with **Streamlit**, this project allows users to analyze emotions from real-time text, CSV uploads, and even PDF documents.

---

## 🚀 Features

- 🧠 **ML Model** trained on labeled emotion data (joy, sadness, anger, fear, surprise)
- 💬 Real-time text input for emotion detection
- 📂 Upload **CSV** or **PDF** files for bulk classification
- 🌘 Elegant **Dark Mode UI**
- 🧾 **Export results to PDF** (professional format)
- 📬 **Email alert** simulation for flagged emotions (like anger)
- 🗄️ **SQLite database** logging all inputs and results

---

## 🖼️ Screenshot

![Screenshot](banner.png)

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn** (ML)
- **Streamlit** (Frontend)
- **ReportLab** (PDF Export)
- **SQLite** (Data Storage)
- **smtplib** (Email simulation)

---

## 📁 Project Structure

hk-emotion-analytics/
├── app.py # Main Streamlit app
├── train_model.py # Model training script
├── emotion_classifier.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── database.db # SQLite DB file
├── banner.png # App banner image
├── requirements.txt # All dependencies
├── training.csv # Sample training data

---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/vadaayar/hk-emotion-analytics.git
cd hk-emotion-analytics

2. Install Required Packages
pip install -r requirements.txt
python train_model.py
streamlit run app.py

📬 Email Alert (Simulated)
The app checks for high-alert emotions (e.g., "anger") and prints a simulated email alert to the console. You can configure real alerts via SMTP.
📥 Export to PDF
Each analysis result can be exported as a downloadable PDF using the ReportLab library.
💾 Database Logging
All predictions are stored in an SQLite database (database.db) with timestamp, input, and predicted emotion.
✅ Requirements
Python 3.7+

Streamlit

Scikit-learn

Pandas

ReportLab

sqlite3

(see requirements.txt for full list)
🧠 Future Improvements
Real email integration (SMTP)

User authentication & dashboard

Deploy on Streamlit Cloud / Hugging Face Spaces
👨‍💻 Author
Harish Kummara
GitHub: @vadaayar
