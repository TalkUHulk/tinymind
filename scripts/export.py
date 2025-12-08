from model.model_vlm import *
import torch
from PIL import Image
from torch import nn
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, MomiMindVLM, MomiMindVLM1, TinyMindVLM, TinyMindVLM_CompactExport, \
    TinyMindVLM_Compact
import open_clip
from trainer.trainer_utils import init_tiny_vlm_model

device = "cpu"

vlm_config = VLMConfig(ve_hidden_size=679, hidden_size=512, num_hidden_layers=8,
                       use_moe=False, per_image_token_num=49)

ckp = "sft_vlm_512_epoch4.pth"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'TinyCLIP-auto-ViT-63M-32-Text-31M', pretrained='TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt')

_model = TinyMindVLM_Compact(
    vlm_config,
    vision_model=clip_model._image_encoder, processor=preprocess
)
state_dict = torch.load(ckp, map_location=device)
_model.load_state_dict({k: v for k, v in state_dict.items()}, strict=True)

vision_model = VisionEncoderExport(vlm_config, clip_model._image_encoder)

vision_state_dict = {k: v for k, v in state_dict.items() if "vision" in k}
vision_model.load_state_dict(vision_state_dict, strict=True)

model = TinyMindVLM_CompactExport(vlm_config)
model_state_dict = {k: v for k, v in state_dict.items() if "vision" not in k}
model.load_state_dict(model_state_dict, strict=True)

past_num = 923
input_ids = torch.rand(1, 1, 512)
attention_mask = torch.ones(1, past_num + 1, dtype=torch.long)

cos_pe = torch.randn([1, vlm_config.hidden_size // vlm_config.num_attention_heads])
sin_pe = torch.randn([1, vlm_config.hidden_size // vlm_config.num_attention_heads])

past_keys = torch.randn([vlm_config.num_hidden_layers, past_num, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads])

past_values = torch.randn([vlm_config.num_hidden_layers, past_num, vlm_config.num_key_value_heads,
                           vlm_config.hidden_size // vlm_config.num_attention_heads])

torch.onnx.export(
    model,
    (input_ids, attention_mask, cos_pe, sin_pe, past_keys, past_values),
    "onnx_model/llm.onnx",
    input_names=["input_ids", "attention_mask", "cos_pe", "sin_pe", "past_keys", "past_values"],
    output_names=["logits", "hidden_states", "present_keys", "present_values"],
    dynamic_axes={"input_ids": {0: "batch", 1: 'token'},
                  "attention_mask": {0: "batch", 1: 'token'},
                  "cos_pe": {0: "batch", 1: 'token'},
                  "sin_pe": {0: "batch", 1: 'token'},
                  "past_keys": {1: "cache"},
                  "past_values": {1: "cache"}
                  },
    do_constant_folding=True,
    verbose=False,
    opset_version=15
)


class Embedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.emb = embedding

    def forward(self, x):
        return self.emb(x)


embedding = Embedding(_model.model.embed_tokens)

dummy_x = torch.randint(0, 6400, (1, 5), dtype=torch.long)

print(dummy_x.shape)

torch.onnx.export(
    embedding,
    dummy_x,
    "onnx_model/embed_tokens.onnx",
    input_names=["input_ids"],
    output_names=["embed_tokens"],
    dynamic_axes={"input_ids": {1: 'length'},
                  "embed_tokens": {1: 'length'}},
    do_constant_folding=True,
    verbose=False,
    opset_version=15
)

dummy_x = torch.randn([17, 3, 224, 224])
torch.onnx.export(
    vision_model,
    dummy_x,
    "onnx_model/vision_encoder.onnx",
    input_names=["inputs"],
    output_names=["deepstack_embeds"],
    dynamic_axes={"inputs": {0: 'batch'},
                  "deepstack_embeds": {0: 'batch'}},
    do_constant_folding=True,
    verbose=False,
    opset_version=15
)


###

def export_tokenizer(tokenizer_path, export_path, stop_ids=[2]):
    import base64
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    # TOKENIZER MAGIC NUMBER
    MAGIC_NUMBER = 430
    # TOKENIZER TYPE
    SENTENCEPIECE = 0
    TIKTOIKEN = 1
    BERT = 2
    HUGGINGFACE = 3

    def write_line(fp, *args):
        for arg in args:
            for token in arg:
                fp.write(str(token) + ' ')
        fp.write('\n')

    def write_header(fp, type, speicals, prefix=[]):
        fp.write(f'{MAGIC_NUMBER} {type}\n')
        fp.write(f'{len(speicals)} {len(stop_ids)} {len(prefix)}\n')
        write_line(fp, speicals, stop_ids, prefix)

    file_path = os.path.join(export_path, "tokenizer.txt")
    special_list = list(tokenizer.added_tokens_decoder.keys())
    if hasattr(tokenizer, 'special_tokens'):
        for k, v in tokenizer.special_tokens.items():
            special_list.append(v)
    if hasattr(tokenizer, 'all_special_ids'):  # gemma3
        special_list.extend(tokenizer.all_special_ids)
    if hasattr(tokenizer, 'gmask_token_id'):
        special_list.append(tokenizer.gmask_token_id)
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        generation_config = model.generation_config
        if hasattr(generation_config, 'user_token_id'):
            special_list.append(generation_config.user_token_id)
        if hasattr(generation_config, 'assistant_token_id'):
            special_list.append(generation_config.assistant_token_id)
    vocab_list = []
    prefix_list = []
    if hasattr(tokenizer, 'get_prefix_tokens'):
        prefix_list = tokenizer.get_prefix_tokens()

    # Simple prefix token detection
    if len(prefix_list) == 0:
        try:
            test_txt = 'A'
            ids = tokenizer.encode(test_txt)
            get_txt = tokenizer.decode(ids[-1])
            if len(ids) > 1 and get_txt == test_txt:
                prefix_list += ids[:-1]
        except Exception:
            pass

    # Determine tokenizer type based on tokenizer class and characteristics
    tokenizer_class_name = type(tokenizer).__name__.lower()
    vocab = tokenizer.get_vocab()

    # Check for SentencePiece-based tokenizers first
    if ('xlmroberta' in tokenizer_class_name or
            'roberta' in tokenizer_class_name or
            'sentencepiece' in tokenizer_class_name or
            hasattr(tokenizer, 'sp_model') or
            (hasattr(tokenizer, 'vocab_file') and
             tokenizer.vocab_file and 'sentencepiece' in tokenizer.vocab_file.lower()) or
            # Check if tokenizer uses SentencePiece patterns (▁ prefix)
            (len(vocab) > 0 and any('▁' in token for token in list(vocab.keys())[:100]))):
        tokenizer_type = SENTENCEPIECE
        print(f"Detected SentencePiece-based tokenizer: {tokenizer_class_name}")
    elif 'bert' in tokenizer_class_name:
        tokenizer_type = BERT
        print(f"Detected BERT tokenizer: {tokenizer_class_name}")
    else:
        tokenizer_type = TIKTOIKEN
        print(f"Detected TikToken tokenizer: {tokenizer_class_name}")

    vocab = tokenizer.get_vocab()

    if tokenizer_type == SENTENCEPIECE:
        # Handle SentencePiece tokenizer (like XLMRoberta)
        # Try to get SentencePiece model if available
        sp_model_path = None
        if hasattr(tokenizer, 'vocab_file') and tokenizer.vocab_file:
            sp_model_path = tokenizer.vocab_file
        elif hasattr(tokenizer, 'sp_model_kwargs'):
            sp_model_path = getattr(tokenizer, 'vocab_file', None)

        if sp_model_path and os.path.exists(sp_model_path):
            # Use existing SentencePiece export logic
            print(f"Found SentencePiece model file: {sp_model_path}")
            # This will be handled by the existing SentencePiece logic above
            # For now, fall back to vocab-based export
            pass

        # Export SentencePiece vocabulary in the correct format
        vocab_list = []
        NORMAL = 1  # SentencePiece piece type

        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            try:
                # SentencePiece tokens are typically already properly encoded
                token_bytes = token.encode('utf-8')
                token_b64 = base64.b64encode(token_bytes).decode('utf-8')
                # Format: token_base64 score piece_type
                vocab_list.append(f'{token_b64} 0.0 {NORMAL}\n')
            except Exception as e:
                print(f"Warning: Failed to encode SentencePiece token '{token}': {e}")
                # Use replacement character for problematic tokens
                token_b64 = base64.b64encode('▁'.encode('utf-8')).decode('utf-8')
                vocab_list.append(f'{token_b64} 0.0 {NORMAL}\n')

        with open(file_path, "w", encoding="utf8") as fp:
            write_header(fp, SENTENCEPIECE, special_list, prefix_list)
            fp.write(f'{len(vocab_list)}\n')
            for vocab_line in vocab_list:
                fp.write(vocab_line)
    else:
        # Handle BERT or TikToken tokenizer
        # bert tokenizer
        def unicode_to_byte(u: int):
            # Handle special unicode mappings for BERT tokenizers
            if u >= 256 and u <= 288:
                return u - 256
            if u >= 289 and u <= 322:
                return u - 162
            if u == 323:
                return 173
            return u

        vocab_list = ['<unk>' for i in range(len(vocab))]

        # Process vocabulary with better UTF-8 handling
        for k, v in vocab.items():
            if tokenizer_type == "BERT":
                try:
                    # For BERT tokenizers, preserve the original token format
                    # Most BERT models already have proper UTF-8 encoded tokens
                    vocab_list[int(v)] = k.encode('utf-8')
                except Exception as e:
                    # Fallback: try unicode_to_byte conversion for special cases
                    try:
                        vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                    except Exception as e2:
                        print(f"Warning: Failed to encode token '{k}' with id {v}: {e2}")
                        vocab_list[int(v)] = k.encode('utf-8', errors='replace')
            else:
                # Fallback: try unicode_to_byte conversion for special cases
                try:
                    vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                except Exception as e2:
                    print(f"Warning: Failed to encode token '{k}' with id {v}: {e2}")
                    vocab_list[int(v)] = k.encode('utf-8', errors='replace')

        special_list = list(tokenizer.added_tokens_decoder.keys())
        with open(file_path, "w", encoding="utf8") as fp:
            write_header(fp, tokenizer_type, special_list)
            fp.write(f'{len(vocab_list)}\n')
            for v in vocab_list:
                line = base64.b64encode(v).decode("utf8") + "\n"
                fp.write(line)
    return file_path


export_tokenizer("../custom_tokenizer", "./mnn_model/")