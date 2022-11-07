import streamlit as st
import pandas as pd
from utils.predict import get_predictions
from utils.helpers import *

DISPLAY_RESULTS = False

st.title("Sentiment Predictor")

col1, col2, = st.columns([6,4])
with col1:
    user_txt = st.text_input("Type in a phrase, sentence, anything!", placeholder = "Try 'It just had to be you, didnt it?'")
    if st.button("Predict ✨", type='primary'):
        if user_txt != "":
            DISPLAY_RESULTS = True
            results = get_predictions(user_txt)
            with col2: display_sentiment(results)
        else:
            st.error("Please enter some text into the box")

st.markdown('---')

if DISPLAY_RESULTS: 
    st.header("Per-word breakdown")
    col3, col4 = st.columns([2,1])
    with col3: 
        data = display_word_barchart(results)
    with col4: st.dataframe(data.style.highlight_max(axis=1, color='brown'))
    st.markdown('---')

with st.expander('FAQ'):
    st.subheader('What is this? 🙋‍♀️')
    st.write('''This is a machine learning model that will detect the
                    sentiment / emotion of whatever you type into the box.
                    Scores closer to 1 are more positive, while scores closer to 0 are more negative.''')
    st.subheader('How does this work? 🤔')
    deepnote_url = "https://deepnote.com/@garreth-lees-workspace/Sentiment-Analysis-c0d9e62f-663b-4307-98f2-d779cedc3b2d"
    st.markdown(f"""This demo is part of an end-to-end 
                        sentiment analysis project I completed. 
                        You can check it out [here]({deepnote_url})""")

        
    
    




