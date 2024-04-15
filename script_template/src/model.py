from custom_arguments import ModelArguments

from transformers import AutoModelForSequenceClassification, PreTrainedModel


def get_model(model_arguments: ModelArguments) -> PreTrainedModel:
    """
    Prepare the model to be used.
    :param model_arguments: Various configuration options to be provided for the model.
    :return: A HuggingFace Trainer compatible model which can be used for training.
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_arguments.model_name_or_path, num_labels=2)
    return model
