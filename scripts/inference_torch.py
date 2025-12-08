import torch
import open_clip
from PIL import Image
from torch import nn
from model.model_vlm import *
from transformers import AutoTokenizer
from trainer.trainer_utils import init_tiny_vlm_model
from dataset.lm_dataset import adaptive_square_split
from dataset.template import prompts_template
import random
from model.model_minimind import *
import numpy as np
class Embedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.emb = embedding

    def forward(self, x):
        return self.emb(x)

def find_indices(tokens):
    B, T = tokens.size()
    # <fake_token_around_image> <row_i_col_j> +  <fake_token_around_image> <global-img>
    image_ids = [[3, i] for i in range(6, 22)] + [[3, 4]]
    image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
    len_image_ids = len(image_ids[0])
    if len_image_ids > tokens.size(1):
        return None
    tokens_view = tokens.unfold(1, len_image_ids, 1)
    matches = []
    for image_id_tensor in image_ids_tensor:
        match = (tokens_view == image_id_tensor).all(dim=2)
        matches.append(match)
    results = {}
    for b in range(B):
        batch_res = {}
        for k, m in enumerate(matches):
            idxs = m[b].nonzero(as_tuple=True)[0]
            if len(idxs) > 0:
                batch_res[k] = [(i.item() + 2, i.item() + 50) for i in idxs]
        if batch_res:
            results[b] = batch_res
    return results or None


vlm_config = VLMConfig(ve_hidden_size=679, hidden_size=512, num_hidden_layers=8,
                       use_moe=False, per_image_token_num=49)

_, tokenizer, preprocess = init_tiny_vlm_model(
    vlm_config,
    device="cpu", tokenizer_path="../custom_tokenizer",
    vision_model_path='../TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt',
    is_split=True
)

ckp = "sft_vlm_512_epoch4.pth"
device = "cpu"

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

llm = TinyMindVLM_CompactExport(vlm_config)
model_state_dict = {k: v for k, v in state_dict.items() if "vision" not in k}
llm.load_state_dict(model_state_dict, strict=True)

embedding = Embedding(_model.model.embed_tokens)

query = "如何做一道番茄炒蛋"
messages = [
    {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
    {
        "role": "user",
        "content": query
    }
]

inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True)

attention_mask = inputs["attention_mask"]

input_ids = inputs["input_ids"]
print("input_ids", input_ids, input_ids.shape)
freqs_cos, freqs_sin = precompute_freqs_cis(dim=vlm_config.hidden_size // vlm_config.num_attention_heads,
                                            end=vlm_config.max_position_embeddings, rope_base=vlm_config.rope_theta,
                                            rope_scaling=vlm_config.rope_scaling)

seqlen = input_ids.shape[1]

hidden_states = embedding(input_ids)

print("!!!! hidden_states", hidden_states.shape)

past_keys = torch.randn([vlm_config.num_hidden_layers, 0, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads])

past_values = torch.randn([vlm_config.num_hidden_layers, 0, vlm_config.num_key_value_heads,
                           vlm_config.hidden_size // vlm_config.num_attention_heads])

cos_pe = freqs_cos[0: seqlen]
sin_pe = freqs_sin[0: seqlen]

logits, hidden_states, present_keys, present_values = llm(hidden_states,
                                                          attention_mask,
                                                          cos_pe, sin_pe,
                                                          past_keys, past_values)

print(logits.shape, hidden_states.shape, present_keys.shape, present_values.shape)


token_id = torch.argmax(logits[:, -1, :])
decoded_text = tokenizer.decode(
                    token_id,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
print(decoded_text)
max_new_token_num = 1
generate_token_num = 0
start_pos = seqlen


while generate_token_num < max_new_token_num:
    attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], -1)
    embed_tokens = embedding(torch.tensor([[token_id]]))
    cos_pe = freqs_cos[start_pos: start_pos + 1]
    sin_pe = freqs_sin[start_pos: start_pos + 1]

    logits, hidden_states, present_keys, present_values = llm(embed_tokens,
                                                              attention_mask,
                                                              cos_pe, sin_pe,
                                                              present_keys, present_values)
    token_id = torch.argmax(logits[:, -1, :])
    if token_id == 2:
        print("", end='\n')
        break
    start_pos += 1
    generate_token_num += 1

    decoded_text = tokenizer.decode(
                    token_id,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
    # print(decoded_text)
    print(decoded_text, end="", flush=True)