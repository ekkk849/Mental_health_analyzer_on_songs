#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lyricsgenius
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load tokenizer and model
model = DistilBertForSequenceClassification.from_pretrained("emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("emotion_model")

label_map = {0: "sad", 1: "happy", 2: "angry", 3: "relaxed"}

def predict_emotion(text, model):
    st.write("### Emotion Classification Result")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    label = label_map.get(predicted_class_id, str(predicted_class_id))
    st.info(f"Predicted Emotion: **{label.upper()}**")
    emotion_gifs = {
        "happy": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWswdm1kd2x1dmF4Z3Mydm1tZ2N3ZTVpeGp0YmNhd3JhcXk3b3BndCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4iHEuZAX9gSPXnPozr/giphy.gif",
        "sad": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3owY2V5bDB5eDMyeTIyeno3cTgycTA5ZDV5NXdhM3B0anF2aXRtciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dUO9Bji1QqmkDN8EBn/giphy.gif",
        "angry": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjZqb2E2ejFxYXYyZHRweXJ2dGszMzhxaW5qM2JiZjJ5djl6dTVjaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qMfogifwTduBHLGCAU/giphy.gif",
        "relaxed": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWNreGU1dmZpYTBwdHFsYmFybjJldGU4bjcwZGtqbHpwaGZyOXAxcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/epIKaVafVEmPoyhykE/giphy.gif"
    }
    gif_url = emotion_gifs.get(label.lower())
    if gif_url:
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="{gif_url}" width="500">
                <p>Emotion: {label}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    classes = [label_map.get(i, str(i)) for i in range(len(probs))]
    proba_df = pd.DataFrame({'Emotion': classes, 'Probability': probs})
    fig, ax = plt.subplots()
    sns.barplot(data=proba_df, x='Emotion', y='Probability', palette='pastel', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Emotion Probabilities")
    st.pyplot(fig)

GENIUS_API_KEY = os.getenv("GENIUS_API_KEY", "QtsQEIbM2Mxz9Wg1rwnf6C4v8clysmPL2fDjOhzi9_6zD_1lQhJ5wJ-9qbVTz6G3")
genius = lyricsgenius.Genius(GENIUS_API_KEY)
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True

def fetch_lyrics(artist, title):
    try:
        song = genius.search_song(title, artist)
        return song.lyrics if song else ''
    except Exception as e:
        st.error(f"Error fetching lyrics: {e} !!!")
        return ''

def analyze_depression(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    st.write("### VADER Sentiment Breakdown")
    fig, ax = plt.subplots()
    labels = ['Positive', 'Negative', 'Neutral']
    values = [scores['pos'], scores['neg'], scores['neu']]
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)
    st.write(f"VADER Compound Score: `{compound}`")
    if compound <= -0.5:
        st.error("Potential signs of depression!!!")
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src="https://media4.giphy.com/media/x5MYQtds9nTNK/giphy.gif" width="500">
                <p>Stay strong 💙</p>
            </div>
            """, unsafe_allow_html=True)
    elif compound < 0:
        st.warning("Slightly negative tone:(((")
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src="https://media3.giphy.com/media/frHrYiqGmFvSo/giphy.gif" width="500">
                <p>Cheer up!</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("Positive or neutral tone :))))")
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src="https://media1.giphy.com/media/IejVI3NqwKEG78skNG/giphy.gif" width="500">
                <p>Keep smiling!</p>
            </div>
            """, unsafe_allow_html=True)

st.set_page_config(page_title="Lyrics Analyzer", layout="centered")
st.title("Lyrics Mental Health & Emotion Analyzer")
input_type = st.radio("Select Input Method", ["Manual Entry", "Upload .txt File", "Search by Song"])
user_input = ""
if input_type == "Manual Entry":
    user_input = st.text_area("Enter your lyrics below:", height=200)
elif input_type == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file with lyrics", type="txt")
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")
elif input_type == "Search by Song":
    artist = st.text_input("Artist Name")
    title = st.text_input("Song Title")
    fetch = st.button("Fetch Lyrics")
    if artist and title and fetch:
        fetched_lyrics = fetch_lyrics(artist, title)
        if fetched_lyrics:
            st.session_state['lyrics'] = fetched_lyrics
    user_input = st.session_state.get('lyrics', '')
    st.text_area("Fetched Lyrics", user_input, height=200, key="lyrics_display")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter or upload some lyrics.")
    else:
        st.subheader("Depression Analysis (VADER)")
        analyze_depression(user_input)
        st.subheader("Emotion Classification")
        predict_emotion(user_input, model)
