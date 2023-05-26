import torch
from collections import defaultdict
from torch.utils.data import Dataset
from utils.data_utils import get_input_label_position, get_target_marker_position
from utils.format_utils import ASTE_format
from utils.read_utils import read_line_examples_from_file, read_shot_ratio_from_file


class ASTEDataset(Dataset):
    def __init__(self, tokenizer, data_path, opt):
        super(ASTEDataset, self).__init__()
        self.opt = opt
        self.tokenizer = tokenizer
        if opt.full_supervise:
            self.all_inputs, self.all_targets = read_line_examples_from_file(data_path)
        else:
            self.all_inputs, self.all_targets = read_shot_ratio_from_file(data_path)
        self.all_additions = {}
        for k, v in tokenizer.get_vocab().items():
            if k[-1] == ',' and len(k) >= 2:
                x = tokenizer.convert_tokens_to_ids(k[:-1])
                self.all_additions[x] = v
        if self.opt.data_format == 'A':
            self.marker_orders = ['aspect'] * len(self.all_inputs)
        if self.opt.data_format == 'O':
            self.marker_orders = ['opinion'] * len(self.all_inputs)
        if self.opt.data_format == 'AO':
            self.marker_orders = ['aspect'] * len(self.all_inputs) + ['opinion'] * len(self.all_inputs)
            self.all_inputs = self.all_inputs + self.all_inputs
            self.all_targets = self.all_targets + self.all_targets

    def __getitem__(self, index):
        input_seq, target_seq = self.all_inputs[index].copy(), self.all_targets[index].copy()
        marker_order = self.marker_orders[index]
        if marker_order == 'aspect':
            input_seq = self.opt.source_aspect_prefix + input_seq
            target_seq.sort(key=lambda x: (x[0][-1], x[1][-1]))
        if marker_order == 'opinion':
            input_seq = self.opt.source_opinion_prefix + input_seq
            target_seq.sort(key=lambda x: (x[1][-1], x[0][-1]))

        add_len = self.opt.prefix_word_length
        for i in range(len(target_seq)):
            a = [x + add_len for x in target_seq[i][0]]
            b = [x + add_len for x in target_seq[i][1]]
            s = target_seq[i][2]
            target_seq[i] = (a, b, s)
        target_copy = list(target_seq)
        target_seq = ASTE_format(input_seq, target_seq, marker_order)
        source = get_input_label_position(input_seq, target_copy, self.tokenizer)
        target = get_target_marker_position(target_seq, self.tokenizer)
        assert torch.sum(target['marker_position'].squeeze(0).eq(1)) == source['aspect_label'].shape[0]
        next_ids = defaultdict(list)
        input_ids = source['input_ids'].tolist()[0]
        next_ids[1] = []    # last pad token
        next_ids[-1] = []   # addition token
        next_ids[0] = []
        for i in range(self.opt.prefix_token_length - 1, len(input_ids)):
            cur = input_ids[i]
            ne = None if i == len(input_ids) - 1 else input_ids[i + 1]
            if ne in self.all_additions:
                next_ids[cur].append(self.all_additions[ne])
                next_ids[-1].append(self.all_additions[ne])
            if ne:
                next_ids[cur].append(ne)
        for cur_ids in source['pack_ids'][self.opt.prefix_word_length:]:
            if len(cur_ids) == 1:
                next_ids[cur_ids[0]].append(6)
            else:
                next_ids[cur_ids[-1]].append(6)
        next_ids = dict(next_ids)
        return {
            "index": index,
            "input_ids": source['input_ids'].squeeze(0),
            "attention_mask": source['attention_mask'].squeeze(0),
            "labels": target['input_ids'].squeeze(0),
            "decoder_attention_mask": target['attention_mask'].squeeze(0),
            "input_seq": input_seq, "target_seq": self.all_targets[index],
            "aspect_label": source['aspect_label'],
            "opinion_label": source['opinion_label'],
            "marker_position": target['marker_position'],
            "word_index": source["word_index"], "word_mask": source["word_mask"],
            "next_ids": next_ids, "marker_order": marker_order
        }

    def __len__(self):
        return len(self.all_inputs)



