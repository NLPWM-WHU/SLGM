from itertools import chain

import ipdb
import torch
import torch.nn as nn
from typing import Union, Any, Mapping, Dict, Optional, List, Tuple

from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_utils import PredictionOutput

from utils.data_utils import collate_func_train, collate_func_eval, ASTESampler, padded_stack


class MarkerSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, data_format=None, ignore_pad_token=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_format = data_format
        self.ignore_pad_token = ignore_pad_token
        self._constraint_decoding = self.args.constraint_decoding

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # ipdb.set_trace()
        if self.data_format == "A":
            outputs = model(**(inputs['ao_data']))
        elif self.data_format == "O":
            outputs = model(**(inputs['oa_data']))
        else:
            ao_outputs = model(**(inputs['ao_data']))
            oa_outputs = model(**(inputs['oa_data']))
            outputs = {"loss": self.args.alpha * ao_outputs["loss"] + (1 - self.args.alpha) * oa_outputs["loss"]}
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
            constraint_decoding: Optional[bool] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
        <Tip>
        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._constraint_decoding = constraint_decoding if constraint_decoding is not None else self.args.constraint_decoding
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        inputs = self.ignore_pad_token_for_training(inputs)
        return inputs

    def ignore_pad_token_for_training(self, inputs):
        if self.is_in_train and self.ignore_pad_token:
            if "A" in self.data_format and 'ao_data' in inputs:
                lm_labels = inputs['ao_data']['labels']
                lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                inputs['ao_data']['labels'] = lm_labels
            if "O" in self.data_format and 'oa_data' in inputs:
                lm_labels = inputs['oa_data']['labels']
                lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                inputs['oa_data']['labels'] = lm_labels
        return inputs

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None, ) -> Tuple[
        Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using obj:*inputs*.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        # ipdb.set_trace()
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else model.config.num_beams,
            "constraint_decoding": self._constraint_decoding if self._constraint_decoding is not None else self.args.constraint_decoding,
            "next_ids": inputs["next_ids"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "word_index": inputs["word_index"]
        }
        # ipdb.set_trace()
        generated_tokens = self.model.predict(**gen_kwargs)['pred']
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return loss, generated_tokens, labels

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = ASTESampler(self.train_dataset, self.data_format)
        drop_last = True if self.train_dataset.opt.full_supervise else False
        return DataLoader(self.train_dataset,
                          batch_size=self.args.train_batch_size,
                          sampler=train_sampler,
                          num_workers=4,
                          drop_last=drop_last,
                          collate_fn=collate_func_train,
                          pin_memory=True)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(eval_dataset,
                          batch_size=self.args.eval_batch_size,
                          num_workers=4,
                          collate_fn=collate_func_eval,
                          pin_memory=True)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset
        return DataLoader(test_dataset,
                          batch_size=self.args.eval_batch_size,
                          num_workers=4,
                          collate_fn=collate_func_eval,
                          pin_memory=True)
