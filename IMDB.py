# =====================================================
# SENTIMENT ANALYSIS DASHBOARD (STREAMLIT)
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="centered"
)

st.title("Sentiment Analysis Dashboard")
st.write("Customer & Movie Review Sentiment Classification")

# =====================================================
# LOAD DATA & TRAIN MODEL (CACHED)
# =====================================================
@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("IMDB_Dataset.csv")

    # Encode sentiment labels
    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    X = df["review"]
    y = df["sentiment"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Build ML pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(max_iter=200))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm


model, accuracy, cm = load_and_train_model()

# =====================================================
# MODEL PERFORMANCE
# =====================================================
st.subheader("Model Performance")

st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

fig_cm, ax = plt.subplots(figsize=(3, 3))
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negative", "Positive"])
ax.set_yticklabels(["Negative", "Positive"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig_cm)

# =====================================================
# SINGLE REVIEW PREDICTION
# =====================================================
st.subheader("Single Review Prediction")

user_input = st.text_area("Enter a review text:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review text.")
    else:
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.success("Prediction Result: Positive Review")
        else:
            st.error("Prediction Result: Negative Review")

# =====================================================
# BATCH PREDICTION (CSV UPLOAD)
# =====================================================
st.subheader("Batch Prediction (CSV File)")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain column: review)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "review" not in data.columns:
        st.error("CSV file must contain a column named 'review'.")
    else:
        # Predict sentiments
        data["prediction"] = model.predict(data["review"])
        data["sentiment"] = data["prediction"].map({
            1: "Positive",
            0: "Negative"
        })

        st.write("Preview of prediction results:")
        st.dataframe(data.head())

        sentiment_counts = data["sentiment"].value_counts()

        # Pie chart
        fig1, ax1 = plt.subplots()
        ax1.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%"
        )
        ax1.set_title("Sentiment Distribution")
        st.pyplot(fig1)

        # Bar chart
        fig2, ax2 = plt.subplots()
        ax2.bar(
            sentiment_counts.index,
            sentiment_counts.values
        )
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Number of Reviews")
        ax2.set_title("Sentiment Count")
        st.pyplot(fig2)

        # Download result
        st.download_button(
            label="Download Prediction Result",
            data=data.to_csv(index=False),
            file_name="sentiment_prediction.csv",
            mime="text/csv"
        )


