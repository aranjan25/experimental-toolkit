from custom_arguments import DataArguments, ModelArguments, ExperimentArguments
from model import get_model
from data import get_datasets
from metrics import get_metric, get_compute_metrics_function, preprocess_logits_for_metrics

import os
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)


def main():
    # parse all arguments
    argument_parser = HfArgumentParser(
        (TrainingArguments, DataArguments, ModelArguments, ExperimentArguments)
    )
    training_arguments, data_arguments, model_arguments, experiment_arguemnts = (
        argument_parser.parse_args_into_dataclasses()
    )

    # sanity checks
    if training_arguments.do_train:
        assert (
            data_arguments.train_data_path is not None
        ), "Error: please provide the training dataset in training mode (`--do_train`)"
    if training_arguments.do_eval:
        assert (
            data_arguments.valid_data_path is not None
        ), "Error: please provide the validation dataset in eval mode (`--do_eval`)"
    if training_arguments.do_predict:
        assert (
            data_arguments.test_data_path is not None
        ), "Error: please provide the test dataset in predict mode (`--do_predict`)"

    # set seed
    set_seed(training_arguments.seed)

    # prepare data (including tokenization)
    tokenizer = AutoTokenizer.from_pretrained(model_arguments.model_name_or_path)
    train_dataset, valid_dataset, test_dataset = get_datasets(
        data_arguments=data_arguments, tokenizer=tokenizer
    )
    data_collator = (
        default_data_collator
        if data_arguments.pad_to_max_sequence_length
        else DataCollatorWithPadding(tokenizer=tokenizer)
    )  # dynamic vs static padding

    # prepare model
    model = get_model(model_arguments)

    # prepare metrics
    metric = get_metric()

    # train
    trainer = Trainer(
        model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=train_dataset if training_arguments.do_train else None,
        eval_dataset=valid_dataset if training_arguments.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=get_compute_metrics_function(metric),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # print stuff before proceeding
    for name, dataset in zip(["train", "valid", "test"], [train_dataset, valid_dataset, test_dataset]):
        trainer.accelerator.print(f"{name} dataset:\n{dataset}")

    if training_arguments.do_train:
        train_metrics = trainer.train().metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_model()
        trainer.save_state()

    # evaluate on validation dataset
    if training_arguments.do_eval:
        valid_metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("valid", valid_metrics)
        trainer.save_metrics("valid", valid_metrics)

    # predict on test set
    if training_arguments.do_predict:
        test_output = trainer.predict(test_dataset, metric_key_prefix="test")
        test_metrics = test_output.metrics
        test_predictions = test_output.predictions
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        if trainer.is_world_process_zero():
            with open(
                os.path.join(training_arguments.output_dir, "predictions.csv"), "w"
            ) as prediction_output_file:
                prediction_output_file.write("text,label\n")
                for index, label in enumerate(test_predictions):
                    text = test_dataset[index][data_arguments.text_header]
                    prediction_output_file.write(f"{text},{label}\n")


if __name__ == "__main__":
    main()
