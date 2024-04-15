from custom_arguments import DataArguments

from typing import Tuple, Dict, Optional
from transformers import PreTrainedTokenizer
from datasets import Dataset


def get_datasets(
    data_arguments: DataArguments, tokenizer: PreTrainedTokenizer
) -> Tuple[Optional[Dataset], Dict[str, Dataset]]:
    """
    Prepare the tokenized dataset to be used for training.
    :param data_arguments: Various configuration options provided for the data.
    :param tokenizer: Tokenizer to be used for tokenization.
    :return: The final, tokenized Training, Validation and Test datasets.
    """

    all_data_paths = [
        data_path
        for data_path in [
            data_arguments.train_data_path,
            data_arguments.valid_data_path,
            data_arguments.test_data_path,
        ]
        if data_path is not None
    ]
    if len(all_data_paths) == 0:
        return None, {}

    all_data_paths_end_in_csv = all([data_path.endswith("csv") for data_path in all_data_paths])
    dataset_constructor = Dataset.from_csv
    if not all_data_paths_end_in_csv:
        all_data_paths_end_in_json = all([data_path.endswith("json") for data_path in all_data_paths])
        assert all_data_paths_end_in_json, f"Error: all data files must be either CSV or JSON, both kind of files found in the paths: {all_data_paths}."
        dataset_constructor = Dataset.from_json

    train_dataset, valid_dataset, test_dataset = None, None, None

    def tokenize(examples):
        return tokenizer(
            examples[data_arguments.text_header],
            max_length=data_arguments.max_sequence_length,
            padding="max_length" if data_arguments.pad_to_max_sequence_length else True,
            truncation=True,
        )

    if data_arguments.train_data_path is not None:
        train_dataset = dataset_constructor(data_arguments.train_data_path).map(tokenize, batched=True)
    if data_arguments.valid_data_path is not None:
        valid_dataset = dataset_constructor(data_arguments.valid_data_path).map(tokenize, batched=True)
    if data_arguments.test_data_path is not None:
        test_dataset = dataset_constructor(data_arguments.test_data_path).map(tokenize, batched=True)

    return (train_dataset, valid_dataset, test_dataset)
