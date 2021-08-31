from app import page_configuration, introduction, training_results, interactive_section, show_result
from sentiment_analyzer import SentimentModel

if __name__ == '__main__':
    page_configuration()

    introduction()

    training_results()

    user_input, model_type = interactive_section()

    model = SentimentModel(model_type=model_type)

    result = model.get_sentiment(user_input=user_input)

    show_result(model_type, user_input, result)
