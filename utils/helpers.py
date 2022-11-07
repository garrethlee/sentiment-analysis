import streamlit as st
import pandas as pd


def display_sentiment(results):
    vals = list(results.values())
    neg_score = vals[-1][0]
    if neg_score > 0.5:
        sentiment = "Negative 🥲"
        label = f"{neg_score*100:.2f}% more negative than neutral"
        delta_color = 'inverse'
    else:
        sentiment = "Positive 😀"
        delta_color = 'normal'
        label = f"{(1 - neg_score)*100:.2f}% more positive than neutral"
    
    st.metric(
            label = "Sentiment", 
            value = sentiment, 
            delta = label,
            delta_color = delta_color,
            help = f"This value is the probability calculated by the model that your sentence is {sentiment.lower()}")

    return True

def display_word_barchart(results):
    data = pd.DataFrame(results, index = ['Negative', 'Positive']).T
    data['Negative'] = data['Negative'] * -1
    if len(data) > 1:
        data = data.drop(data.tail(1).index)
    st.bar_chart(data)
    data['Negative'] = data['Negative'] * -1
    return data
