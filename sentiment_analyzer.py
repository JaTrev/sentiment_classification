from nltk.sentiment.vader import SentimentIntensityAnalyzer
from enum import Enum
import streamlit as st


class Sentiment(Enum):
    Positive = "positive"
    Negative = "negative"
    Neutral = "neutral"


@st.cache
def get_model(model_type: str):
    assert model_type in ["VADER"]

    if model_type == "VADER":
        return SentimentIntensityAnalyzer()


@st.cache
def get_sentiment(model, model_type: str, user_input: str):
    if model_type == "VADER":

        output_dict = model.polarity_scores(user_input)
        sentiment_dic = {Sentiment.Positive: output_dict["pos"], Sentiment.Negative: output_dict["neg"],
                         Sentiment.Neutral: output_dict["neu"]}

        return max(sentiment_dic, key=sentiment_dic.get).value

    else:
        assert model_type == "VADER"
