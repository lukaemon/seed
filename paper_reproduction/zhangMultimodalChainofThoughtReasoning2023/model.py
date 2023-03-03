"""
__init__():
    Add 3 new modules image_dense, attn and gate_dense, 3 lines

forward(): 
    add image_ids in signature to take in image features, 1 line
    between # --- model implant start --- and # --- model implant finish --- are gated fusion, 6 lines.
    The messy rest in the forward() are all copy paste from T5ForConditionalGeneration.

Don't have to understand every single line of original framework. 
Out of 170+ lines here, only 10 lines are relevant to this paper.

T5config: https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.d_model
T5ForConditionalGeneration: https://github.com/huggingface/transformers/blob/eb6c59bc7826529122e916fb0ac8b303b23e8d64/src/transformers/models/t5/modeling_t5.py#L1595
torch.nn.MultiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, d_image):
        super().__init__(config)

        self.image_dense = nn.Linear(d_image, config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=1, batch_first=True
        )
        self.gate_dense = nn.Linear(2 * config.d_model, config.d_model)

        self.post_init()  # for param init

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,  # taking image feature
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
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

        # --- model implant start ---
        # this is h_lang in pseudo code, fork from here to ingest gated fusion
        hidden_states = encoder_outputs[0]

        # forked
        h_lang = hidden_states  # (b, n, d_model)
        h_vis = self.image_dense(image_ids)  # (b, m, d_image) -> (b, m, d_model)
        h_attn_vis, _ = self.attn(h_lang, h_vis, h_vis)  # (b, n, d_model)

        # same idea in pooled qkv transform in attention module
        # merge 2 matmaul into bigger one. simple kernel fusion in a sense
        merged = torch.cat([h_lang, h_attn_vis], dim=-1)  # (b, n, 2*d_model)

        r = F.sigmoid(self.gate_dense(merged))  # (b, n, d_model)

        # this is the new hidden state with gated fusion
        h_fuss = (1 - r) * h_lang + r * h_attn_vis  # (b, n, d_model)

        # merge back to source
        hidden_states = h_fuss
        # --- model implant finish ---

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
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
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

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
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
