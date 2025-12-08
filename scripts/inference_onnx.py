import torch
from PIL import Image
from model.model_vlm import TinyMindVLM_Compact, VLMConfig
from trainer.trainer_utils import init_tiny_vlm_model
from dataset.lm_dataset import adaptive_square_split
from dataset.template import prompts_template
import random
from model.model_minimind import *
import onnxruntime
import numpy as np

vlm_config = VLMConfig(ve_hidden_size=679, hidden_size=512, num_hidden_layers=8,
                       use_moe=False, per_image_token_num=49)

_, tokenizer, preprocess = init_tiny_vlm_model(
    vlm_config,
    device="cpu", tokenizer_path="../custom_tokenizer",
    vision_model_path='../TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt',
    is_split=True
)

"""
preprocess Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    <function _convert_to_rgb at 0x7fb9f48db040>
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
)
"""

special_token = {
    "<global-img>": tokenizer.convert_tokens_to_ids("<global-img>"),
    "<fake_token_around_image>": tokenizer.convert_tokens_to_ids("<fake_token_around_image>"),
    "<image>": tokenizer.convert_tokens_to_ids("<image>"),
}
for i in range(4):
    for j in range(4):
        special_token[f"<row_{i + 1}_col_{j + 1}>"] = tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>")

# {'<global-img>': 4, '<fake_token_around_image>': 3, '<image>': 5, '<row_1_col_1>': 6, '<row_1_col_2>': 7, '<row_1_col_3>': 8, '<row_1_col_4>': 9, '<row_2_col_1>': 10, '<row_2_col_2>': 11, '<row_2_col_3>': 12, '<row_2_col_4>': 13, '<row_3_col_1>': 14, '<row_3_col_2>': 15, '<row_3_col_3>': 16, '<row_3_col_4>': 17, '<row_4_col_1>': 18, '<row_4_col_2>': 19, '<row_4_col_3>': 20, '<row_4_col_4>': 21}
image_path = "/Users/hulk/Downloads/coco128/images/train2017/000000000165.jpg"  # "crop_0.jpg" #
image = Image.open(image_path).convert('RGB')

pixel_values = TinyMindVLM_Compact.image2tensor(image, preprocess)

blocks, rows, cols, block_size = adaptive_square_split(
    image_path=image_path,
    max_rows=4,
    max_cols=4
)

patch_num = len(blocks)
pad_num = 16 - patch_num
mask_token_id = []
if pad_num:
    patch_tensor = []
    for i in range(4):
        for j in range(4):
            if i >= rows or j >= cols:
                patch_tensor.append(torch.zeros_like(pixel_values))
                mask_token_id.append(special_token[f"<row_{i + 1}_col_{j + 1}>"])
            else:
                patch_tensor.append(TinyMindVLM_Compact.image2tensor(blocks[i * cols + j], preprocess))
else:
    patch_tensor = [TinyMindVLM_Compact.image2tensor(block, preprocess) for block in blocks]

assert len(patch_tensor) == 16

pixel_values = torch.stack(patch_tensor + [pixel_values], dim=0)
# import struct
# data_array = np.array(pixel_values.numpy().flatten(), dtype=np.float32)
# with open("pixel.bin", 'wb') as f:
#     # 先写入数据长度（4字节整数）
#     f.write(struct.pack('i', len(data_array)))
#     data_array.tofile(f)

image_place_holder = random.choice(["图片如下：", "如下所示的图片:", "请见下面这张图:", "如下图显示:", "参考下方图片:", "图示如下:"])
for row in range(4):
    for col in range(4):
        image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
        image_place_holder += "<image>" * 49

image_place_holder += f"<fake_token_around_image><global-img>{'<image>' * 49}<fake_token_around_image>"


# query = random.choice(prompts_template)
query = "图片中的人在做什么。"
messages = [
    {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
    {
        "role": "user",
        "content": query + image_place_holder
    }
]

inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True)

attention_mask = inputs["attention_mask"]


windows = inputs["input_ids"].unfold(-1, size=2, step=1)

for token in mask_token_id:
    pattern = torch.tensor([special_token["<fake_token_around_image>"], token])
    matches = (windows == pattern).all(dim=-1)
    indices = matches.nonzero(as_tuple=True)[0]
    attention_mask[indices[0] + 2: indices[0] + 2 + 49] = 0

input_ids = inputs["input_ids"]

# 64 32768 1000000.0 None
freqs_cos, freqs_sin = precompute_freqs_cis(dim=vlm_config.hidden_size // vlm_config.num_attention_heads,
                                            end=vlm_config.max_position_embeddings, rope_base=vlm_config.rope_theta,
                                            rope_scaling=vlm_config.rope_scaling)

# start_pos, seq_length
# 0 920
# 920 1
# 921 1
# freqs_cos[start_pos:start_pos + seq_length],

config = onnxruntime.SessionOptions()
cpu_num_thread = 2
config.intra_op_num_threads = cpu_num_thread
config.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

vision_session = onnxruntime.InferenceSession("../onnx_model/vision_encoder.onnx", sess_options=config)

ort_inputs = {"inputs": pixel_values.numpy()}

deepstack_embeds = vision_session.run(["deepstack_embeds"], ort_inputs)[0]  # (1, 17, 49, 512)

embed_tokens_session = onnxruntime.InferenceSession("../onnx_model/embed_tokens.onnx", sess_options=config)

seqlen = input_ids.shape[1]

ort_inputs = {"input_ids": input_ids.numpy()}
embed_tokens = embed_tokens_session.run(["embed_tokens"], ort_inputs)[0]  # (1, 922, 512)



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


image_batch_indices = find_indices(input_ids)

B, P, L, D = deepstack_embeds.shape  # (1, 17, 49, 512)
new_h = []

for i in range(B):
    h_i = embed_tokens[i]
    image_indices = image_batch_indices[i]
    for tki, index in image_indices.items():
        vision_proj_i = deepstack_embeds[i][tki]
        start_idx, end_idx = index[0]
        h_i = np.concatenate((h_i[:start_idx], vision_proj_i, h_i[end_idx + 1:]), axis=0)[:seqlen]

    new_h.append(h_i)

hidden_states = np.stack(new_h, axis=0)

# prefill
llm_session = onnxruntime.InferenceSession("../onnx_model/llm.onnx", sess_options=config)

past_keys = torch.randn([vlm_config.num_hidden_layers, 0, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads]).numpy()

past_values = torch.randn([vlm_config.num_hidden_layers, 0, vlm_config.num_key_value_heads,
                           vlm_config.hidden_size // vlm_config.num_attention_heads]).numpy()

# input_names = ["hidden_states", "attention_mask", "cos_pe", "sin_pe", "past_keys", "past_values"],
# output_names = ["logits", "hidden_states", "present_keys", "present_values"],
cos_pe = freqs_cos[0: seqlen].numpy()
sin_pe = freqs_sin[0: seqlen].numpy()

ort_inputs = {"input_ids": hidden_states, "attention_mask": attention_mask.numpy(),
              "cos_pe": cos_pe, "sin_pe": sin_pe, "past_keys": past_keys, "past_values": past_values}

logits, hidden_states, present_keys, present_values = llm_session.run(
    ["logits", "hidden_states", "present_keys", "present_values"], ort_inputs)  # (1, 17, 49, 512)

token_id = np.argmax(logits[:, -1, :])


max_new_token_num = 128
generate_token_num = 0
start_pos = seqlen
attention_mask = attention_mask.numpy()
while generate_token_num < max_new_token_num:
    decoded_text = tokenizer.decode(
        token_id,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    print(decoded_text, end="", flush=True)

    attention_mask = np.concatenate([attention_mask, np.array([[1]])], axis=1)
    embed_tokens = embed_tokens_session.run(["embed_tokens"], {"input_ids": np.array([[token_id]], dtype=np.int64)})[0]
    cos_pe = freqs_cos[start_pos: start_pos + 1].numpy()
    sin_pe = freqs_sin[start_pos: start_pos + 1].numpy()
    ort_inputs = {"input_ids": embed_tokens, "attention_mask": attention_mask,
                  "cos_pe": cos_pe, "sin_pe": sin_pe, "past_keys": present_keys, "past_values": present_values}
    logits, hidden_states, present_keys, present_values = llm_session.run(
        ["logits", "hidden_states", "present_keys", "present_values"], ort_inputs)  # (1, 17, 49, 512)
    token_id = np.argmax(logits[:, -1, :])
    if token_id == 2:
        print("【出来了】", end='\n')
        break
    start_pos += 1
    generate_token_num += 1
