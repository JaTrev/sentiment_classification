import streamlit as st
from enum import Enum


class TextTypes(Enum):
    Text = "text"
    Chapter = "chapter"
    Subchapter = "subchapter"
    Warning = "warning"
    Title = "title"


def markdown_text(text: str, text_class: TextTypes = TextTypes.Text):
    """
    markdown_text is used to create text on the streamlit page

    :param text: string that should appear on the page
    :param text_class: type of text presentation
    :return:
    """
    assert text_class in TextTypes

    st.markdown(f'<p class="{text_class.value}"> {text} </p>', unsafe_allow_html=True)


def page_configuration():
    st.set_page_config("Sentiment Analysis", layout="centered", initial_sidebar_state="expanded")

    text_chapter = """<style>.{chapter} { font-size:40px ;}  </style> """
    st.markdown(text_chapter, unsafe_allow_html=True)

    text_subchapter = """<style>.subchapter {font-size:30px ;} </style> """
    st.markdown(text_subchapter, unsafe_allow_html=True)

    title_font = """ <style>.title {font-size:50px ;} </style> """
    st.markdown(title_font, unsafe_allow_html=True)

    text_font = """<style>.text {} </style> """
    st.markdown(text_font, unsafe_allow_html=True)

    error_font = """<style>.warning { color: red;} </style> """
    st.markdown(error_font, unsafe_allow_html=True)

    page_style = """
            <style>
            /* This is to hide hamburger menu completely */
            MainMenu {visibility: hidden;}

            /* This is to hide Streamlit footer */
            footer {visibility: hidden;}
            """
    st.markdown(page_style, unsafe_allow_html=True)


def introduction():
    markdown_text("NLP Sentiment Classification", TextTypes.Title)

    _, col_mid, _ = st.columns([1, 6, 1])

    with col_mid:
        intro_text = 'Sentiment analysis is the process of identifying and classifying text based on its emotion. ' \
                     'It is often used in social media mentoring to give qualitative insights into a product or topic.'
        markdown_text(intro_text, TextTypes.Text)

        intro_text = 'This project looks at three different sentiment classifiers: ' \
                     'VADER, a pre-trained BERTSequenceClassifier, ' \
                     'and a self-created classifier based on a pre-trained BERT language model.'
        markdown_text(intro_text, TextTypes.Text)

        intro_text = 'All models are tested on the Sentiment140 data set. ' \
                     'The data set includes 1.6 million tweets, each tweet is labeled by its sentiment ' \
                     '(negative, neutral, positive).'
        markdown_text(intro_text, TextTypes.Text)


def training_results():
    with st.expander("VADER"):
        markdown_text("VADER", text_class=TextTypes.Title)

        body = "VADER (Valence Aware Dictionary and Sentiment Reasoner) " \
               "is a common sentiment analysis tool that is based on rules and a lexicon. " \
               "It works best with social media text."
        markdown_text(body, text_class=TextTypes.Text)

        body = 'This model was able to predict <b> 50 % </b> of the tweet sentiment correctly'
        markdown_text(body, text_class=TextTypes.Text)

    with st.expander("Specialized BERT"):
        title = "Specialized BERT"
        body = "text"

        markdown_text(title, text_class=TextTypes.Title)
        markdown_text(body, text_class=TextTypes.Text)

    with st.expander("Own Model"):
        title = "Own Model"
        body = "has no neutral... acc = 0.83"
        markdown_text(title, text_class=TextTypes.Title)
        markdown_text(body, text_class=TextTypes.Text)

        st.image("images/train.jpg", caption="Figure shows the accuracy and loss scores over training time.")


def interactive_section():
    user_input = st.text_input("Type in input for sentiment classification", value="I love data.", max_chars=50, )

    model_type = st.radio("What model should be used?", options=["VADER", "Specialized BERT", "Own Model"])

    return user_input, model_type


def show_result(model_name: str, sentiment: str):
    st.markdown(f""" <div align="center"> {model_name} classifies the input to be <b> {sentiment} </b>. </div> """,
                unsafe_allow_html=True)
