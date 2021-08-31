from nltk.sentiment.vader import SentimentIntensityAnalyzer
from enum import Enum
import streamlit as st


class Sentiment(Enum):
    Positive = "positive"
    Negative = "negative"
    Neutral = "neutral"


class SentimentModel:

    def __init__(self, model_type: str):
        assert model_type in ["VADER"]
        self.model_type = model_type

        if model_type == "VADER":
            self.model = SentimentIntensityAnalyzer()

    def get_sentiment(self, user_input: str):
        if self.model_type == "VADER":
            output_dict = self.model.polarity_scores(user_input)

            sentiment_dic = {Sentiment.Positive: output_dict["pos"], Sentiment.Negative: output_dict["neg"],
                             Sentiment.Neutral: output_dict["neu"]}

            return max(sentiment_dic, key=sentiment_dic.get).value

        else:
            assert self.model_type == "VADER"
