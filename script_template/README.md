# Template script

This is a template script for training/finetuning pretrained or custom models which uses the [HuggingFace Trainer API](https://huggingface.co/docs/transformers/en/main_classes/trainer). It is meant to be modified flexibly and on-the-go to have a flexible training pipeline.

To suit the script for your own needs, one can modify the provided functions such as `get_datasets`, `get_model`, etc.

To install all dependencies, run `pip3 install -r requirements.txt`

To run the script, run the following:
```bash
python3 src/main.py --all --arguments --to --the --script
```

Given the large number of arguments generally provided to the trainer, it might be easier to write the command in another shell script (like `run.sh`) and run it (`./run.sh`).

## Tip for training on Nvidia GPUs

Experiment with batch sizes while keeping the `--pad_to_maximum_sequence_length` argument to find out a large enough batch size that can be run without running out of GPU VRAM. Then, run the script formally without the flag to train with **dynamic padding** which can be faster.
