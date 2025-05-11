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

    .user-msg, .bot-msg {
        font-size: 14px;
        padding: 10px 14px;
        border-radius: 18px;
        margin: 4px 0;
        max-width: 75%;
        word-wrap: break-word;
    }

    .user-msg {
        background-color: #DCF8C6;
        float: right;
        clear: both;
        text-align: left;
    }

    .bot-msg {
        background-color: #E5E5EA;
        float: left;
        clear: both;
        text-align: left;
    }

    .sentiment-box {
        font-size: 13px;
        display: inline-block;
        margin: 4px 0 12px;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 500;
    }

    .positive {
        background-color: #d4edda;
        color: #155724;
    }

    .neutral {
        background-color: #fff3cd;
        color: #856404;
    }

    .negative {
        background-color: #f8d7da;
        color: #721c24;
    }

    .clear {
        clear: both;
    }

    .block-container {
        padding-top: 0 !important;
    }

    h1 {
        margin-top: 20px;
        padding-top: 0;
        font-size: 28px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)





st.title("💬Customer Service for E-commerce ")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    # If user message, show sentiment too
    if msg["role"] == "user" and "sentiment" in msg:
        st.markdown(
            f'<div class="sentiment-box {msg["sentiment_class"]}">{msg["emoji"]} {msg["sentiment"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="clear"></div>', unsafe_allow_html=True)

# Get user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # st.session_state.messages.append({"role": "user", "content": user_input})

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

    if sentiment_text == "Negative":
        st.session_state.negative_count += 1
    else:
        st.session_state.negative_count = 0

    
    # Display sentiment result
    st.markdown(f'<div class="sentiment-box {sentiment_class}">{sentiment_emoji} {sentiment_text}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "sentiment": sentiment_text,
        "emoji": sentiment_emoji,
        "sentiment_class": sentiment_class
    })

    if st.session_state.negative_count >= 2:
        bot_reply = "🔔 It looks like you're experiencing issues. I've escalated this to a human assistant who will help you shortly."
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)
    else:
        # Call OpenAI if not escalated
        response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=50
        )
        bot_reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)


    
       # **Generate AI response**
    # response = openai.chat.completions.create(
    #     model="gpt-4-turbo",
    #     messages=st.session_state.messages
    # )
    # bot_reply = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # # Display AI response
    # st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)
