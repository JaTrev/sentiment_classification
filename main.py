from app import page_configuration, introduction, training_results, interactive_section, show_result
from sentiment_analyzer import get_model, get_sentiment, get_sentiment_bert, get_tokenizer

if __name__ == '__main__':
    page_configuration()

    tokenizer = get_tokenizer()

    introduction()

    training_results()

    user_input, model_type = interactive_section()

    model = get_model(model_type=model_type)

    if model_type == "Own Model":
        result = get_sentiment_bert(user_input=user_input, model=model, tokenizer=tokenizer)

    else:
        assert model_type == "VADER"
        result = get_sentiment(model, user_input=user_input)

    show_result(model_type, result)
