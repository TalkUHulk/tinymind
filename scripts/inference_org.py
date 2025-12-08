import torch
import open_clip
from PIL import Image
from torch import nn
from model.model_vlm import TinyMindVLM_Compact, VLMConfig
from transformers import AutoTokenizer
from trainer.trainer_utils import init_tiny_vlm_model
from dataset.lm_dataset import adaptive_square_split
from dataset.template import prompts_template
import random
from model.model_minimind import *

device = "cpu"
vlm_config = VLMConfig(ve_hidden_size=679, hidden_size=512, num_hidden_layers=8,
                       use_moe=False, per_image_token_num=49)


ckp = "sft_vlm_512_epoch4.pth"

model, tokenizer, preprocess = init_tiny_vlm_model(
    vlm_config,
    device=device, tokenizer_path="../custom_tokenizer",
    vision_model_path='../TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt',
    is_split=True
)
state_dict = torch.load(ckp, map_location=device)
model.load_state_dict({k: v for k, v in state_dict.items()}, strict=True)
print(f'VLM模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
print(f'clip模型参数: {sum(p.numel() for p in model.vision_encoder.parameters()) / 1e6:.2f} M(illion)')

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
print("input_ids", input_ids)
# prefill

past_key_values = None
seqlen = input_ids.shape[1]
print("======>>", input_ids.shape)

outputs = model(input_ids, attention_mask, past_key_values)
# self.OUT.__setitem__('last_hidden_state', hidden_states)
# self.OUT.__setitem__('logits', logits)
# self.OUT.__setitem__('aux_loss', aux_loss)
# self.OUT.__setitem__('past_key_values', presents)
token_id = torch.argmax(outputs["logits"][:, -1, :])

past_key_values = outputs["past_key_values"]

print("token_id:", token_id)
decoded_text = tokenizer.decode(
                    token_id,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
print(decoded_text)
max_new_token_num = 1
generate_token_num = 0
start_pos = seqlen

# print(past_key_values[0])

while generate_token_num < max_new_token_num:
    attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], -1)
    input_ids = torch.tensor([[token_id]])
    print("@", generate_token_num, input_ids)
    outputs = model(input_ids, attention_mask, past_key_values)

    token_id = torch.argmax(outputs["logits"][:, -1, :])
    past_key_values = outputs["past_key_values"]
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
    print(decoded_text, end="", flush=True)