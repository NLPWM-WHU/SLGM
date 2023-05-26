import torch
import numpy as np
from itertools import chain
from collections import defaultdict
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence


def get_input_label_position(words, target_indices, tokenizer):
    data = {}
    words = words + ["</s>"]
    s_to_t, cur_index = defaultdict(list), 0
    specific_tokens, specific_ids = [], []
    for i in range(len(words)):
        specific_token = tokenizer.tokenize(words[i])
        specific_id = tokenizer.convert_tokens_to_ids(specific_token)
        specific_tokens.append(specific_token)
        specific_ids.append(specific_id)
        s_to_t[i] = [c for c in range(cur_index, cur_index + len(specific_token))]
        cur_index += len(specific_token)
    lens = list(map(len, specific_tokens))
    _specific_tokens = list(chain(*specific_tokens))
    _specific_ids = tokenizer.convert_tokens_to_ids(_specific_tokens)
    assert _specific_ids == list(chain(*specific_ids))

    aspect_label, opinion_label = [], []
    cum_aspect_label = [0] * len(words)
    cum_opinion_label = [0] * len(words)
    # BIO tagging scheme
    for triplet in target_indices:
        # aspect
        a_st, a_ed = triplet[0][0], triplet[0][-1]

        cur_aspect_label = [0] * len(words)
        cur_aspect_label[a_st] = 2
        cum_aspect_label[a_st] = 2
        for i in range(a_st + 1, a_ed + 1):
            cur_aspect_label[i] = 1
            cum_aspect_label[i] = 1
        aspect_label.append(cur_aspect_label)
        # opinion
        o_st, o_ed = triplet[1][0], triplet[1][-1]

        cur_opinion_label = [0] * len(words)
        cur_opinion_label[o_st] = 2
        cum_opinion_label[o_st] = 2
        for i in range(o_st + 1, o_ed + 1):
            cur_opinion_label[i] = 1
            cum_opinion_label[i] = 1
        opinion_label.append(cur_opinion_label)

    data['pack_ids'] = specific_ids
    data['input_ids'] = torch.LongTensor(_specific_ids).unsqueeze(0)
    data['attention_mask'] = torch.LongTensor([1] * len(_specific_ids)).unsqueeze(0)
    data['aspect_label'] = torch.LongTensor(aspect_label)
    data['opinion_label'] = torch.LongTensor(opinion_label)

    word_matrix = []
    for i in range(len(words)):
        row = [0] * len(_specific_tokens)
        for j in s_to_t[i]:
            row[j] = 1
        word_matrix.append(row)
    data['word_index'] = torch.LongTensor(word_matrix)
    data['word_mask'] = torch.LongTensor([1] * len(words))
    return data


def get_target_marker_position(target_seq, tokenizer):
    data = tokenizer(target_seq, return_tensors='pt')
    target_seq_len = data['input_ids'].shape[-1]
    marker_position = torch.zeros((target_seq_len,), dtype=torch.long)
    marker_names = {'aspect': 1, 'opinion': 2, 'sentiment': 3}
    sep_seq = torch.tensor([10] * target_seq_len, dtype=torch.long)
    sep_t = data['input_ids'].eq(sep_seq).roll(-1, dims=1)
    for marker_name, val in marker_names.items():
        marker_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(marker_name))[0]
        marker_seq = torch.tensor([marker_id] * target_seq_len, dtype=torch.long)
        t = data['input_ids'].eq(marker_seq)
        marker_position = torch.where(t & sep_t, val, marker_position)
    data['marker_position'] = marker_position
    return data


def collate_func_train(batch):
    ao_data, oa_data = {}, {}
    ao_batch = [batch[i] for i in range(len(batch)) if batch[i]['marker_order'] == 'aspect']
    oa_batch = [batch[i] for i in range(len(batch)) if batch[i]['marker_order'] == 'opinion']

    pad_batch_data(ao_batch, ao_data)
    pad_batch_data(oa_batch, oa_data)

    return {"ao_data": ao_data, "oa_data": oa_data}


def collate_func_eval(batch):
    data = {}
    pad_batch_data(batch, data)
    return data


def pad_batch_data(cur_batch, cur_data):
    if len(cur_batch) == 0:
        return
    for k, v in cur_batch[0].items():
        if k in ['word_index']:
            cur_data[k] = padded_stack([s[k] for s in cur_batch])
            continue
        if isinstance(v, torch.Tensor):
            if len(v.shape) == 1:
                cur_data[k] = pad_sequence([x[k].squeeze(0) for x in cur_batch], batch_first=True)
            else:
                rows = [list(map(lambda c: c.squeeze(0), torch.split(x[k], 1, dim=0))) for x in cur_batch]
                cur_data[k] = pad_sequence(list(chain(*rows)), batch_first=True)
        else:
            cur_data[k] = [x[k] for x in cur_batch]


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


class ASTESampler(Sampler):
    def __init__(self, data_source, target_format):
        super().__init__(data_source)
        self.target_format = target_format
        self.data_source = data_source
        self.data_range = []
        if target_format == 'AO':
            length = len(data_source) // 2
            a = [c for c in range(length)]
            o = [c for c in range(length, 2 * length)]
            for i in range(length):
                self.data_range.append([a[i], o[i]])
        else:
            for i in range(len(data_source)):
                self.data_range.append(i)

    def __iter__(self):
        np.random.shuffle(self.data_range)
        if isinstance(self.data_range[0], list):
            self.data_range = list(chain(*self.data_range))
        return iter(self.data_range)

    def __len__(self):
        return len(self.data_source)
