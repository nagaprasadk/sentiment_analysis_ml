import streamlit as st
import pickle
import os
import openai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model & vectorizer
@st.cache_resource  # Cache the model to avoid reloading on each interaction
def load_model():
    with open("gridsearch_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Custom CSS for UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    body { font-family: 'Roboto', sans-serif; }
    .chat-container { max-width: 600px; margin: auto; }
    .user-msg { background-color: #DCF8C6; text-align: right; padding: 12px; border-radius: 12px; margin: 5px 0; float: right; clear: both; max-width: 70%; }
    .bot-msg { background-color: #E5E5EA; text-align: left; padding: 12px; border-radius: 12px; margin: 5px 0; float: left; clear: both; max-width: 70%; }
    .sentiment-box { display: block; padding: 6px; border-radius: 6px; font-weight: bold; margin: 10px auto; max-width: 200px; text-align: center; }
    .positive { background-color: #D4EDDA; color: #155724; }
    .neutral { background-color: #FFF3CD; color: #856404; }
    .negative { background-color: #F8D7DA; color: #721C24; }
    .clear { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("💬Customer Service for E-commerce ")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="{css_class}">{msg["content"]}</div><div class="clear"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Get user input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    st.markdown(f'<div class="user-msg">{user_input}</div><div class="clear"></div>', unsafe_allow_html=True)

    # === Sentiment Analysis Using Your Model ===
    user_input_tfidf = vectorizer.transform([user_input])  # Convert input to TF-IDF features
    sentiment_prediction = model.predict(user_input_tfidf)[0]  # Predict sentiment
    

    # Sentiment Mapping
    sentiment_map = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}  # Adjust based on your model labels
    sentiment_text = sentiment_map.get(sentiment_prediction, "Neutral")  # Default to Neutral if unknown

    sentiment_emojis = {"Positive": "🟢😊", "Neutral": "🟡😐", "Negative": "🔴😠"}
    sentiment_classes = {"Positive": "positive", "Neutral": "neutral", "Negative": "negative"}
    
    sentiment_emoji = sentiment_emojis.get(sentiment_text, "⚪🤖")
    sentiment_class = sentiment_classes.get(sentiment_text, "neutral")

    # Display sentiment result
    st.markdown(f'<div class="sentiment-box {sentiment_class}">{sentiment_emoji} {sentiment_text}</div>', unsafe_allow_html=True)


       # **Generate AI response**
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages
    )
    bot_reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Display AI response
    st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)
