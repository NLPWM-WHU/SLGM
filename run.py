import sys
import os
import logging
import time

import ipdb
import pickle
from typing import Optional
from dataclasses import dataclass, field

import torch
import transformers
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoConfig,
    T5Config,
    T5Tokenizer,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from models import T5ForConditionalGeneration, MarkerT5
from aste_dataset import ASTEDataset
from marker_trainer import MarkerSeq2SeqTrainer
from utils.eval_utils import parse_and_score

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="../../PretrainModel/t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: str = field(
        default='laptop14', metadata={"help": "laptop14/rest14/rest15/rest16 "}
    )
    data_format: str = field(
        default="A", metadata={"help": "A/O/AO"}
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of train data"}
    )
    valid_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of valid data"},
    )
    test_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of test data"},
    )
    max_length: int = field(
        default=256, metadata={"help": "The max padding length of source text and target text"}
    )
    num_beams: int = field(
        default=1, metadata={"help": "greedy search"}
    )
    shot_ratio_index: Optional[str] = field(
        default="-1[+]-1[+]1", metadata={"help": "1[+]-1[+]1->1-shot"}
    )

    def __post_init__(self):
        if self.dataset is None and self.train_path is None and self.valid_path is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        shot, ratio, index = self.shot_ratio_index.split("[+]")
        shot, ratio, index = int(shot), float(ratio), int(index)
        assert shot in [-1, 1, 5, 10] and ratio in [-1, 0.01, 0.05, 0.1]
        self.full_supervise = True if shot == -1 and ratio == -1 else False
        name_mapping = {"laptop14": "14lap", "rest14": "14res", "rest15": "15res", "rest16": "16res"}
        if shot != -1:
            self.train_path = f'./data/ASTE/{name_mapping[self.dataset]}_shot/seed{index}/{shot}shot/train.json'
            self.valid_path = f'./data/ASTE/{name_mapping[self.dataset]}_shot/seed{index}/{shot}shot/val.json'
            self.test_path = f'./dataa/ASTE/{name_mapping[self.dataset]}_shot/seed{index}/{shot}shot/test.json'
        elif ratio != -1:
            self.train_path = f'./data/ASTE/{name_mapping[self.dataset]}_ratio/seed{index}/{ratio}/train.json'
            self.valid_path = f'./data/ASTE/{name_mapping[self.dataset]}_ratio/seed{index}/{ratio}/val.json'
            self.test_path = f'./data/ASTE/{name_mapping[self.dataset]}_ratio/seed{index}/{ratio}/test.json'
        else:
            self.train_path = f'./data/ASTE/{self.dataset}/train.txt'
            self.valid_path = f'./data/ASTE/{self.dataset}/dev.txt'
            self.test_path = f'./data/ASTE/{self.dataset}/test.txt'

        self.source_aspect_prefix = ["aspect", "first:"]
        self.source_opinion_prefix = ["opinion", "first:"]
        self.prefix_word_length = 2  # ["aspect", "first:"]
        self.prefix_token_length = 3  # ["aspect", "first", ":"]

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )


@dataclass
class MarkerSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments for our model in training procedure
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    use_marker: bool = field(default=False, metadata={"help": "Whether to use marker"})
    marker_type: Optional[str] = field(default='AO', metadata={"help": "marker_type A/O"})
    alpha: float = field(default=0.5, metadata={"help": "adjust the loss weight of ao_template and oa_template"})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MarkerSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print(model_args, data_args, training_args, sep='\n')

    if data_args.data_format == "AO":
        training_args.per_device_train_batch_size *= 2

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # ipdb.set_trace()
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # pass args
    config.max_length = data_args.max_length
    config.num_beams = data_args.num_beams

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.add_tokens(['[SSEP]'])

    pretrain_model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    pretrain_model.resize_token_embeddings(len(tokenizer))
    model = MarkerT5(
        args=data_args,
        tokenizer=tokenizer,
        t5_model=pretrain_model,
        t5_config=config,
        use_marker=training_args.use_marker,
        marker_type=training_args.marker_type
    )

    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    # print(get_parameter_number(model))

    if not training_args.do_train:
        logger.info("load checkpoint of Marker-T5 !")
        model.load_state_dict(torch.load(f"{training_args.output_dir}/checkpoint-1120/pytorch_model.bin"))

    train_dataset = ASTEDataset(tokenizer=tokenizer, data_path=data_args.train_path, opt=data_args)
    eval_dataset = ASTEDataset(tokenizer=tokenizer, data_path=data_args.valid_path, opt=data_args)
    test_dataset = ASTEDataset(tokenizer=tokenizer, data_path=data_args.test_path, opt=data_args)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = parse_and_score(preds, labels, data_args.data_format)
        return result

    trainer = MarkerSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_format=data_args.data_format,
        tokenizer=tokenizer,
        ignore_pad_token=data_args.ignore_pad_token_for_loss,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))

    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     eval_results = trainer.predict(
    #         eval_dataset,
    #         metric_key_prefix="eval",
    #         max_length=data_args.max_length,
    #         num_beams=data_args.num_beams,
    #         constraint_decoding=True
    #     )
    #
    #     output_eval_file = os.path.join(training_args.output_dir, "eval_results_seq2seq.txt")
    #     if trainer.is_world_process_zero():
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key, value in sorted(eval_results.metrics.items()):
    #                 logger.info(f"  {key} = {value}")
    #                 writer.write(f"{key} = {value}\n")

    if training_args.do_predict:
        logger.info(f"*** Test constraint decoding: {training_args.constraint_decoding}***")
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.max_length,
            num_beams=data_args.num_beams,
            constraint_decoding=training_args.constraint_decoding,
        )
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)
        output_test_result_file = os.path.join(training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"{key} = {value}")
                    writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
