# =========================================
# SENTIMENT ANALYSIS DASHBOARD (STREAMLIT)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="centered"
)

st.title("🎬 Sentiment Analysis Dashboard")
st.write("Movie Review Sentiment Classification using Machine Learning")

# -----------------------------------------
# LOAD & TRAIN MODEL (CACHE)
# -----------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("IMDB_Dataset.csv")

    # No cleaning needed since dataset is already clean
    df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    return model, acc, cm

model, accuracy, cm = train_model()

# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------
st.subheader("📊 Model Performance")
st.write(f"Accuracy: **{accuracy*100:.2f}%**")

fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm)
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_xticks([0,1])
ax_cm.set_yticks([0,1])
ax_cm.set_xticklabels(["Negative","Positive"])
ax_cm.set_yticklabels(["Negative","Positive"])

for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i,j], ha="center", va="center")

st.pyplot(fig_cm)

# -----------------------------------------
# SINGLE REVIEW PREDICTION
# -----------------------------------------
st.subheader("🔍 Predict Single Review")

user_text = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter text.")
    else:
        result = model.predict([user_text])[0]

        if result == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😠")

# -----------------------------------------
# CSV BATCH PREDICTION
# -----------------------------------------
st.subheader("📁 Upload CSV File")

file = st.file_uploader("CSV file must contain column name: review")

if file:
    data = pd.read_csv(file)

    if "review" not in data.columns:
        st.error("Column 'review' not found.")
    else:
        data["prediction"] = model.predict(data["review"])
        data["sentiment"] = data["prediction"].map({1:"Positive",0:"Negative"})

        st.write(data.head())

        counts = data["sentiment"].value_counts()

        # Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
        ax1.set_title("Sentiment Distribution")
        st.pyplot(fig1)

        # Bar Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(counts.index, counts.values)
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Count")
        ax2.set_title("Sentiment Count")
        st.pyplot(fig2)

        st.download_button(
            "Download Result CSV",
            data.to_csv(index=False),
            file_name="prediction_result.csv",
            mime="text/csv"
        )

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")
st.write("Developed for NLP Sentiment Analysis Project")
