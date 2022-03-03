from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import XLMRobertaConfig, XLMRobertaModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaPreTrainedModel

CONCAT_STRATEGIES = {
    # u dot v
    "none": None,
    # [u, v]
    "uv": 2,
    # [|u - v|]
    "minus": 1,
    # [u * v]
    "dot": 1,
    # [|u - v|, u * v]
    "minus_dot": 2,
    # [u, v, u * v]
    "uv_dot": 3,
    # [u, v, |u - v|]
    "uv_minus": 3,
    # [u, v, |u - v|, u * v]
    "uv_minus_dot": 4,
}


@dataclass
class LabelEncoderModelOutput(ModelOutput):
    mlm_loss: Optional[torch.FloatTensor] = None
    le_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

    mlm_logits: Optional[torch.FloatTensor] = None
    le_logits: Optional[torch.FloatTensor] = None

    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class XLMRobertaLabelEncoder(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        if not hasattr(config, "use_rnn"):
            config.use_rnn = False

        if not hasattr(config, "cls_init"):
            config.cls_init = False

        if not hasattr(config, "concatenation_strategy"):
            config.concatenation_strategy = "none"

        if not hasattr(config, "add_cosine_loss"):
            config.add_cosine_loss = False

        super().__init__(config)
        self.num_labels = 1

        self.roberta = XLMRobertaModel(
            config, add_pooling_layer=self.config.cls_init & self.config.use_rnn
        )
        self.lm_head = RobertaLMHead(config)

        if self.config.use_rnn:
            self.encoder = nn.GRU(
                self.config.hidden_size, self.config.hidden_size, batch_first=True
            )

        self.concatenation = self.config.concatenation_strategy
        self.use_rnn = self.config.use_rnn
        self.cls_init = self.config.cls_init
        self.add_cosine_loss = self.config.add_cosine_loss

        proj_size = CONCAT_STRATEGIES[self.concatenation]
        if proj_size is not None:
            self.proj = nn.Linear(proj_size * self.config.hidden_size, self.num_labels)

        self.init_weights()

    def _reorder_cache(self, past, beam_idx):
        pass

    def _forward_label_encoder(
        self,
        encoder_hidden_state,
        mask_token_idx,
        labels,
        labels_mask,
        labels_weights,
        labels_input_ids,
        labels_attention_mask,
        return_dict,
    ):
        labels_encoded = self.roberta.embeddings.word_embeddings(
            labels_input_ids
        ) * labels_attention_mask.unsqueeze(-1)

        batch_size, num_options, num_tokens, hidden_size = labels_encoded.size()

        # Gather the hidden representations for the mask_token_idx, i.e., the position of the label [MASK]
        mask_token_hidden = torch.gather(
            encoder_hidden_state,
            1,
            mask_token_idx.view(batch_size, 1, 1).repeat([1, 1, hidden_size]),
        )
        labels_lengths = labels_attention_mask.sum(axis=2).clamp(min=1)

        if self.use_rnn:
            labels_encoded = labels_encoded.view((-1, num_tokens, hidden_size))
            packed = pack_padded_sequence(
                labels_encoded,
                labels_lengths.view(-1).detch().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            labels_encoded, labels_lengths = pad_packed_sequence(packed, batch_first=True)

            hx = None
            if self.cls_init:
                # [CLS], or <s> is the first token, we take only this from the hidden state from the Transformer Encoder
                hx = (
                    encoder_hidden_state[:, 0, :]
                    .unsqueeze(1)
                    .repeat([1, num_options, 1])
                    .view(1, -1, hidden_size)
                )

            outputs, labels_encoded = self.encoder(labels_encoded, hx=hx)
            labels_encoded = labels_encoded.view((batch_size, -1, hidden_size))
        else:
            labels_encoded = labels_encoded.sum(axis=2) / labels_lengths.unsqueeze(-1)

        if self.concatenation == "none":
            logits = labels_encoded @ mask_token_hidden.permute([0, 2, 1])
        else:
            expanded_labels = mask_token_hidden.repeat([1, labels_encoded.shape[1], 1])
            similarity_features = []

            if "uv" in self.concatenation:
                similarity_features += (expanded_labels, labels_encoded)
            if "minus" in self.concatenation:
                similarity_features += [torch.abs(labels_encoded - expanded_labels)]
            if "dot" in self.concatenation:
                similarity_features += [labels_encoded * mask_token_hidden]

            similarity_features = torch.cat(similarity_features, dim=-1)
            logits = self.proj(similarity_features)
        logits += (1 - labels_mask).unsqueeze(-1) * -10000

        loss = None
        if labels is not None:
            if labels_weights is not None:
                labels_weights = labels_weights.view(-1)
            loss_bce = BCEWithLogitsLoss(weight=labels_mask.view(-1), pos_weight=labels_weights)
            loss = loss_bce(logits.view(-1), labels.view(-1))

            if self.add_cosine_loss:
                loss_cosine = CosineEmbeddingLoss(margin=0.1, reduction="none")
                loss_cos = loss_cosine(
                    labels_encoded.view(-1, hidden_size),
                    mask_token_hidden.repeat([1, labels_encoded.shape[1], 1]).view(-1, hidden_size),
                    (2 * labels.view(-1) - 1).float() * labels_mask.view(-1),
                )
                loss += loss_cos.mean()

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def _forward_mlm(self, sequence_output, labels_mlm, return_dict):
        prediction_scores = self.lm_head(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=250001)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), labels_mlm.view(-1)
        )

        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_mask=None,
        labels_input_ids=None,
        labels_attention_mask=None,
        labels_weights=None,
        labels_mlm=None,
        mask_token_idx=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.roberta.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        loss = None
        le_loss, le_logits = None, None
        if labels_input_ids is not None:
            le_output = self._forward_label_encoder(
                outputs.last_hidden_state,
                mask_token_idx,
                labels,
                labels_mask,
                labels_weights,
                labels_input_ids,
                labels_attention_mask,
                return_dict=True,
            )
            le_loss, le_logits = le_output.loss, le_output.logits
            loss = (1 - self.config.lambda_mlm) * le_loss

        mlm_loss, mlm_logits = None, None
        if labels_mlm is not None:
            mlm_output = self._forward_mlm(outputs[0], labels_mlm, return_dict=True)
            mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits
            loss += self.config.lambda_mlm * mlm_loss

        return LabelEncoderModelOutput(
            mlm_loss=mlm_loss,
            le_loss=le_loss,
            loss=loss,
            mlm_logits=mlm_logits,
            le_logits=le_logits.squeeze(-1),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
