import warnings
from typing import Optional, Tuple, Union, List

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from transformers import MT5ForConditionalGeneration, MT5Config, T5Config, T5ForConditionalGeneration
from tqdm import tqdm
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class MT6(MT5ForConditionalGeneration):

    def __init__(self, config: Union[MT5Config, T5Config]):
        super().__init__(config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        __HEAD_MASK_WARNING_MSG = """
        The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
        `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
        If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
        num_heads)`.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # group_lab, transl_lab = self.create_groups(labels, attention_mask)

        # Encode if needed (training, first prediction pass)
        encoder_outputs, hidden_states = self.encoder_forward(attention_mask, encoder_outputs, head_mask, input_ids,
                                                              inputs_embeds, output_attentions, output_hidden_states,
                                                              return_dict)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        pnat_labels = False if labels is None else (labels.ndim == 3)
        decoder_iterations = labels.shape[1] if pnat_labels else 1

        lm_logits, decoder_outputs = None, None

        total_loss = None if (labels is None) else 0
        label = None
        for i in range(decoder_iterations):
            if labels is not None:
                label = labels[:, i, :].contiguous() if pnat_labels else labels.contiguous()
                decoder_attention_mask, decoder_input_ids, decoder_inputs_embeds = None, None, None
            attention_mask, decoder_attention_mask, decoder_input_ids, hidden_states = self.prepare_decoder(
                attention_mask,
                decoder_attention_mask,
                decoder_input_ids,
                decoder_inputs_embeds,
                hidden_states,
                label)

            decoder_outputs, lm_logits = self.decoder_forward(attention_mask, cross_attn_head_mask,
                                                              decoder_attention_mask,
                                                              decoder_head_mask, decoder_input_ids,
                                                              decoder_inputs_embeds,
                                                              hidden_states, output_attentions,
                                                              output_hidden_states,
                                                              past_key_values, return_dict, use_cache)

            loss = self.compute_loss(lm_logits, label)
            if not bool(torch.isnan(loss).any()):
                total_loss = total_loss + loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def encoder_forward(self, attention_mask, encoder_outputs, head_mask, input_ids, inputs_embeds, output_attentions,
                        output_hidden_states, return_dict):
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        return encoder_outputs, hidden_states

    def compute_loss(self, lm_logits, labels):
        loss = torch.tensor(float("nan"))
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss

    def decoder_forward(self, attention_mask, cross_attn_head_mask, decoder_attention_mask, decoder_head_mask,
                        decoder_input_ids, decoder_inputs_embeds, hidden_states, output_attentions,
                        output_hidden_states, past_key_values, return_dict, use_cache):
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        return decoder_outputs, lm_logits

    def prepare_decoder(self, attention_mask, decoder_attention_mask, decoder_input_ids, decoder_inputs_embeds,
                        hidden_states, labels):
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        return attention_mask, decoder_attention_mask, decoder_input_ids, hidden_states

