from nltk.sentiment.vader import SentimentIntensityAnalyzer
from enum import Enum
import streamlit as st
import tensorflow as tf
from transformers import TFAutoModel, BertTokenizer


class Sentiment(Enum):
    Positive = "positive"
    Negative = "negative"
    Neutral = "neutral"


def define_model(model, max_seq_length: int = 55):
    input_ids = tf.keras.Input(shape=(max_seq_length,), dtype='int32')
    attention_mask = tf.keras.Input(shape=(max_seq_length,), dtype='int32')

    outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask},
                    training=False)
    pooler_output = outputs["pooler_output"]

    # Model Head
    h1 = tf.keras.layers.Dense(128, activation='relu')(pooler_output)
    dropout = tf.keras.layers.Dropout(0.2)(h1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

    new_model = tf.keras.models.Model(inputs=[input_ids, attention_mask],
                                      outputs=output)

    new_model.compile(tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return new_model


@st.cache
def get_model(model_type: str, transformers_model_name: str = "bert-base-uncased"):
    assert model_type in ["VADER", "Own Model"]

    if model_type == "VADER":
        return SentimentIntensityAnalyzer()

    elif model_type == "Own Model":
        # load own bert model (saved with '.h5')

        bert_model = TFAutoModel.from_pretrained(transformers_model_name, output_hidden_states=True)
        own_model = define_model(bert_model)
        own_model.load_weights('data/own_weights.h5')

        return own_model


def get_tokenizer(transformers_model_name: str = "bert-base-uncased"):
    return BertTokenizer.from_pretrained(transformers_model_name)


@st.cache
def get_sentiment(model, user_input: str):
    output_dict = model.polarity_scores(user_input)
    sentiment_dic = {Sentiment.Positive: output_dict["pos"], Sentiment.Negative: output_dict["neg"],
                     Sentiment.Neutral: output_dict["neu"]}

    return max(sentiment_dic, key=sentiment_dic.get).value


def tokenization(data: list, tokenizer: BertTokenizer, max_seq_length: int = 55):
    return tokenizer(data, padding='max_length', max_length=max_seq_length, truncation=True, return_tensors="tf")


def get_sentiment_bert(user_input: str, tokenizer: BertTokenizer, model: tf.keras.models.Model) -> str:
    model_input = tokenization([user_input], tokenizer=tokenizer)
    prediction = model.predict([model_input.input_ids, model_input.attention_mask])[0]

    if prediction > 0.5:
        return Sentiment.Positive.value
    else:
        return Sentiment.Negative.value
