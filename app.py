import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (if not already downloaded)
#nltk.download('stopwords')
#nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing Data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http[s]?://\S+|www\.\S+|@[\w_]+|#[\w_]+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

# Streamlit UI
st.title("🧠 Social Media Sentiment Analyzer")
st.write("Enter a social media post or tweet to analyze its sentiment.")

user_input = st.text_area("Your Text", height=150)

if st.button("Analyze"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        probs = model.predict_proba(vectorized)[0]
        result = model.classes_[probs.argmax()]
        st.success(f"Predicted Sentiment: **{result.capitalize()}**")
    else:
        st.warning("Please enter some text before analyzing.")
