import torch
import torch.nn as nn
import torch.nn.functional as F


def get_word_representation(matrix, states):
    length = matrix.size(1)
    min_value = torch.min(states).item()
    states = states.unsqueeze(1).expand(-1, length, -1, -1)
    states = torch.masked_fill(states, matrix.eq(0).unsqueeze(-1), min_value)
    word_reps, _ = torch.max(states, dim=2)
    # word_reps = torch.relu(F.dropout(word_reps, p=0.1, training=self.training))
    return word_reps


def get_marker_state(index, marker_position, decoder_hidden_state):
    assert index in [1, 2, 3]  # 1-aspect, 2-opinion, 3-sentiment
    aos_marker_position = marker_position.eq(index)
    aos_marker_len = torch.sum(aos_marker_position, dim=-1).unsqueeze(-1)
    aos_marker = decoder_hidden_state[aos_marker_position.bool()]
    return aos_marker, aos_marker_len


class MarkerT5(nn.Module):
    def __init__(self, args, tokenizer, t5_config, t5_model, use_marker, marker_type):
        super(MarkerT5, self).__init__()
        # ipdb.set_trace()
        self.args = args
        self.tokenizer = tokenizer
        self.config = t5_config
        self.t5 = t5_model
        self.use_marker = use_marker
        self.marker_type = marker_type

        self.prefix_len = self.args.prefix_word_length

        self.ignore_index = -100
        self.aspect_fc = nn.Sequential()
        self.opinion_fc = nn.Sequential()
        if self.use_marker:
            if "A" in self.marker_type:
                self.aspect_fc.add_module(
                    'aspect_projection',
                    nn.Linear(self.config.d_model, self.config.d_model, bias=True)
                )
            if "O" in self.marker_type:
                self.opinion_fc.add_module(
                    'opinion_projection',
                    nn.Linear(self.config.d_model, self.config.d_model, bias=True)
                )
            self.proj = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier = nn.Linear(self.config.d_model, 3, bias=True)

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask, **kwargs):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        loss = outputs.loss
        hidden_states = outputs.encoder_hidden_states[-1]
        hidden_states = get_word_representation(kwargs['word_index'], hidden_states)
        aspect_hidden_states = self.aspect_fc(hidden_states)
        opinion_hidden_states = self.opinion_fc(hidden_states)
        if self.use_marker:
            d_state = outputs.decoder_hidden_states[-1]
            decoder_marker_loss = self.marker_decoder_similarity(
                aspect_hidden_states, opinion_hidden_states,
                d_state,
                **kwargs
            )
            loss += decoder_marker_loss
        return {"loss": loss}

    def marker_decoder_similarity(self, aspect_hidden_state, opinion_hidden_state, decoder_hidden_state, **kwargs):
        loss = 0
        batch = decoder_hidden_state.shape[0]
        # aspect
        if "A" in self.marker_type:
            aspect_marker, aspect_len = get_marker_state(1, kwargs['marker_position'], decoder_hidden_state)
            max_marker_per_sample = torch.max(aspect_len)
            encoder_index = torch.lt(
                torch.arange(max_marker_per_sample.int()).repeat(batch, 1).to(aspect_marker),
                aspect_len.repeat(1, max_marker_per_sample)
            )
            aspect_hidden_state = aspect_hidden_state.unsqueeze(1).repeat(1, max_marker_per_sample, 1, 1)
            aspect_hidden_state = aspect_hidden_state[encoder_index]
            aspect_mask = kwargs['word_mask'].unsqueeze(1).repeat(1, max_marker_per_sample, 1)
            aspect_mask = aspect_mask[encoder_index]
            aspect_mask[:, :self.prefix_len] = 0
            aux_aspect_loss, aspect_probs = self.calculate_sim_loss(
                aspect_hidden_state, aspect_marker, kwargs['aspect_label'], aspect_mask
            )
            loss += aux_aspect_loss
        # opinion
        if "O" in self.marker_type:
            opinion_marker, opinion_len = get_marker_state(2, kwargs['marker_position'], decoder_hidden_state)
            max_marker_per_sample = torch.max(opinion_len)
            encoder_index = torch.lt(
                torch.arange(max_marker_per_sample.int()).repeat(batch, 1).to(opinion_marker),
                opinion_len.repeat(1, max_marker_per_sample)
            )
            opinion_hidden_state = opinion_hidden_state.unsqueeze(1).repeat(1, max_marker_per_sample, 1, 1)
            opinion_hidden_state = opinion_hidden_state[encoder_index]
            opinion_mask = kwargs['word_mask'].unsqueeze(1).repeat(1, max_marker_per_sample, 1)
            opinion_mask = opinion_mask[encoder_index]
            opinion_mask[:, :self.prefix_len] = 0
            aux_opinion_loss, opinion_probs = self.calculate_sim_loss(
                opinion_hidden_state, opinion_marker, kwargs['opinion_label'], opinion_mask
            )
            loss += aux_opinion_loss
        return loss

    def calculate_sim_loss(self, marker_encoder_state, marker, label, mask):
        marker = marker.unsqueeze(-2).repeat(1, marker_encoder_state.shape[1], 1)
        state = torch.selu(self.proj(torch.cat([marker, marker_encoder_state], dim=-1)))
        state = self.classifier(state).view(-1, 3)
        aux_marker_loss = F.cross_entropy(state, label.view(-1).long(), reduction='none')
        aux_marker_loss = torch.sum(aux_marker_loss * mask.reshape(-1)) / torch.sum(mask)
        return aux_marker_loss, state

    def predict(self, input_ids, attention_mask, **kwargs):
        out = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=kwargs["max_length"],
            num_beams=kwargs["num_beams"],
            output_hidden_states=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            **{'next_ids': kwargs['next_ids'], 'constraint_decoding': kwargs['constraint_decoding']}
        )

        ans = {"pred": out.sequences}
        return ans
