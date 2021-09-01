from app import page_configuration, introduction, training_results, interactive_section, show_result
from sentiment_analyzer import get_model, get_sentiment

if __name__ == '__main__':
    page_configuration()

    introduction()

    training_results()

    user_input, model_type = interactive_section()

    model = get_model(model_type=model_type)

    result = get_sentiment(model, model_type=model_type, user_input=user_input)

    show_result(model_type, user_input, result)
