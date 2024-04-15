from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the training dataset."})
    valid_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the validation dataset."})
    test_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the test dataset."})
    text_header: str = field(
        default="text",
        metadata={"help": "The heading (CSV) or key (JSON) containing the text in the dataset"},
    )
    max_sequence_length: int = field(
        default=512, metadata={"help": "Maximum sequence length to use during tokenization."}
    )
    pad_to_max_sequence_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to use static padding (length `max_sequence_length`) or dynamic padding. Static padding can be used to check if a batch haivng particular fits inside the GPU so that it doesn't cause CUDA OOMs with dynamic padding."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Model name as present in the HuggingFace Hub or the path to a saved model checkpoint."
        },
    )


@dataclass
class ExperimentArguments:
    pass
