"""
Fine-tuning the library models for sequence to sequence.
"""



# Model will load the csv file for 512 ner tokens, preprocess it
    
    # Do the same for eval/test file if true in passed parameter


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk 
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets import load_dataset, load_metric
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode

import pdb

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # ner_model_name_or_path: str = field(
    #     metadata={"help": "Path to pretrained ner model or model identifier from huggingface.co/models"}
    # )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    do_ner: Optional[bool] = field(
        default=False,
        metadata={"help": "Add the ner attention ?"},
    )
    ner_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before ner text"}
    )
    ner_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the ner string (for summarization)."},
    )
    k_fold_cnt: int = field(
        default=5,
        metadata={
            "help": (
                "Number of k-fold testing to be done"
            )
        },
    )

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError("Need dataset name")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."

def main():
    # See all possible arguments in src/transformers/training_args.py

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model args {model_args}")
    logger.info(f"Data args {data_args}")


    # Set seed before initializing model.
    set_seed(training_args.seed)

    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    ner_prefix = data_args.ner_prefix if data_args.ner_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval`")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang
        
        # pdb.set_trace()

    # Get the column names for input/target.
    text_column = data_args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
        )
    summary_column = data_args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )
    if data_args.do_ner:
        ner_column = data_args.ner_column
        if ner_column not in column_names:
            raise ValueError(
                f"--ner_column' value '{data_args.ner_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # This is for when the input is in a batch
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                if data_args.do_ner:
                    inp_str = f"{ner_prefix} {examples[ner_column][i]} {prefix} {examples[text_column][i]}"
                else:
                    inp_str = examples[text_column][i]
                    
                # print(inp_str)
                    
                inputs.append(inp_str)
                targets.append(examples[summary_column][i])
                
            
        # pdb.set_trace()
        # inputs_pref = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.]
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # pdb.set_trace()
        return model_inputs

    # pdb.set_trace()

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            
    # pdb.set_trace()

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    
    # pdb.set_trace()

    # Initialize our Trainer
    
    main_output_dir = training_args.output_dir
    kf = KFold(n_splits=data_args.k_fold_cnt, shuffle = True, random_state=training_args.seed)
    
    for i, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        print(f"Fold: {i}")
        
        training_args.output_dir = main_output_dir + f"fold-{i}/"
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        
        model.resize_token_embeddings(len(tokenizer))
        

        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
            # raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    "Increasing the model's number of position embedding vectors from"
                    f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                    f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                    f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                    " model's position encodings by passing `--resize_position_embeddings`."
                )

        if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
            forced_bos_token_id = (
                tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
            )
            model.config.forced_bos_token_id = forced_bos_token_id
            
        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )
            
        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        
        # Detecting last checkpoint. (If there is output dir, we are training and overwrite output is false)
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
                
        # pdb.set_trace()
        
        if training_args.do_train:
            train_dataset_tmp = train_dataset.select(train_index)
        if training_args.do_eval:
            eval_dataset_tmp = train_dataset.select(val_index)
            
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tmp if training_args.do_train else None,
            eval_dataset=eval_dataset_tmp if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            
            # pdb.set_trace()
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        # results = {}
        # if training_args.do_eval:
        #     logger.info("*** Evaluate ***")
        #     if isinstance(eval_dataset_tmp, dict):
        #         metrics = {}
        #         for eval_ds_name, eval_ds in eval_dataset_tmp.items():
        #             dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
        #             metrics.update(dataset_metrics)
        #     else:
        #         metrics = trainer.evaluate(metric_key_prefix="eval")
        #     metrics["eval_samples"] = len(eval_dataset_tmp)

        #     trainer.log_metrics("eval", metrics)
        #     trainer.save_metrics("eval", metrics)
            
        del model, trainer
        
        if training_args.do_predict:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
            metrics = predict_results.metrics
            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    output_prediction_file = os.path.join(training_args.output_dir, f"generated_predictions_{data_args.pred_type}.csv")
                    df = pd.DataFrame({"Summary": predictions}) 
                    df.to_csv(output_prediction_file, index=False)
                    # with open(output_prediction_file, "w") as writer:
                    #     writer.write("\n".join(predictions))

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    # if data_args.lang is not None:
    #     kwargs["language"] = data_args.lang

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    # return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()