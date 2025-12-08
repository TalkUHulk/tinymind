from PIL import Image
from model.model_vlm import TinyMindVLM_Compact, VLMConfig
from trainer.trainer_utils import init_tiny_vlm_model
from dataset.lm_dataset import adaptive_square_split
from dataset.template import prompts_template
import random
from model.model_minimind import *
import MNN
import numpy as np


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

special_token = {
    "<global-img>": tokenizer.convert_tokens_to_ids("<global-img>"),
    "<fake_token_around_image>": tokenizer.convert_tokens_to_ids("<fake_token_around_image>"),
    "<image>": tokenizer.convert_tokens_to_ids("<image>"),
}
for i in range(4):
    for j in range(4):
        special_token[f"<row_{i + 1}_col_{j + 1}>"] = tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>")

image_path = "/Users/hulk/Downloads/coco128/images/train2017/000000000241.jpg"  # "crop_0.jpg" #
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

image_place_holder = random.choice(["图片如下：", "如下所示的图片:", "请见下面这张图:", "如下图显示:", "参考下方图片:", "图示如下:"])
for row in range(4):
    for col in range(4):
        image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
        image_place_holder += "<image>" * 49

image_place_holder += f"<fake_token_around_image><global-img>{'<image>' * 49}<fake_token_around_image>"

query = random.choice(prompts_template)
# query = "图中是什么动物"
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
seqlen = input_ids.shape[1]
freqs_cos, freqs_sin = precompute_freqs_cis(dim=vlm_config.hidden_size // vlm_config.num_attention_heads,
                                            end=vlm_config.max_position_embeddings, rope_base=vlm_config.rope_theta,
                                            rope_scaling=vlm_config.rope_scaling)

config = {}
config['precision'] = 'low'
config['backend'] = 0  # CPU
config['numThread'] = 4  # 线程数

vision_interp = MNN.Interpreter('mnn_model/vision_encoder.mnn')
embed_interp = MNN.Interpreter('mnn_model/embed_tokens.mnn')
llm_interp = MNN.Interpreter('mnn_model/llm.mnn')

vision_session = vision_interp.createSession(config)
embed_session = embed_interp.createSession(config)
llm_session = llm_interp.createSession(config)

# prefill
# vision encoder
inputs_tensor = vision_interp.getSessionInput(vision_session)
deepstack_embeds_tensor = vision_interp.getSessionOutput(vision_session, "deepstack_embeds")

pixel_values_data = MNN.Tensor((17, 3, 224, 224), MNN.Halide_Type_Float,
                               pixel_values.numpy(),
                               MNN.Tensor_DimensionType_Caffe)

vision_interp.resizeTensor(inputs_tensor, (17, 3, 224, 224))
vision_interp.resizeTensor(deepstack_embeds_tensor, (1, 17, 49, 512))
vision_interp.resizeSession(vision_session)
inputs_tensor.copyFrom(pixel_values_data)

vision_interp.runSession(vision_session)

deepstack_embeds = MNN.Tensor((1, 17, 49, 512), MNN.Halide_Type_Float,
                              np.ones([1, 17, 49, 512]).astype(np.float32),
                              MNN.Tensor_DimensionType_Caffe)
deepstack_embeds_tensor.copyToHostTensor(deepstack_embeds)
deepstack_embeds = np.array(deepstack_embeds.getData()).reshape(deepstack_embeds.getShape())


# embed
input_ids_tensor = embed_interp.getSessionInput(embed_session)
embed_tokens_tensor = embed_interp.getSessionOutput(embed_session, "embed_tokens")

input_ids_data = MNN.Tensor((1, seqlen), MNN.Halide_Type_Int,
                            input_ids.numpy().astype(np.int32),
                            MNN.Tensor_DimensionType_Caffe)

embed_interp.resizeTensor(input_ids_tensor, (1, seqlen))
embed_interp.resizeTensor(embed_tokens_tensor, (1, seqlen, 512))
embed_interp.resizeSession(embed_session)
input_ids_tensor.copyFrom(input_ids_data)

embed_interp.runSession(embed_session)

embed_tokens = MNN.Tensor((1, seqlen, 512), MNN.Halide_Type_Float,
                          np.ones([1, seqlen, 512]).astype(np.float32),
                          MNN.Tensor_DimensionType_Caffe)
embed_tokens_tensor.copyToHostTensor(embed_tokens)
embed_tokens = np.array(embed_tokens.getData()).reshape(embed_tokens.getShape())
B, T, D = embed_tokens.shape
# llm
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


hidden_states_tensor = llm_interp.getSessionInput(llm_session, "input_ids")
attention_mask_tensor = llm_interp.getSessionInput(llm_session, "attention_mask")
cos_pe_tensor = llm_interp.getSessionInput(llm_session, "cos_pe")
sin_pe_tensor = llm_interp.getSessionInput(llm_session, "sin_pe")
past_keys_tensor = llm_interp.getSessionInput(llm_session, "past_keys")
past_values_tensor = llm_interp.getSessionInput(llm_session, "past_values")

logits_tensor = llm_interp.getSessionOutput(llm_session, "logits")
present_keys_tensor = llm_interp.getSessionOutput(llm_session, "present_keys")
present_values_tensor = llm_interp.getSessionOutput(llm_session, "present_values")

llm_interp.resizeTensor(hidden_states_tensor,
                        (1, T, 512))
llm_interp.resizeTensor(attention_mask_tensor,
                        (1, T))
llm_interp.resizeTensor(cos_pe_tensor,
                        (T, 64))
llm_interp.resizeTensor(sin_pe_tensor,
                        (T, 64))
llm_interp.resizeTensor(past_keys_tensor,
                        (vlm_config.num_hidden_layers,
                         0, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads))
llm_interp.resizeTensor(past_values_tensor,
                        (vlm_config.num_hidden_layers,
                         0, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads))

llm_interp.resizeTensor(logits_tensor,
                        (1, T, 6400))
llm_interp.resizeTensor(present_keys_tensor,
                        (vlm_config.num_hidden_layers,
                         T, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads))
llm_interp.resizeTensor(present_values_tensor,
                        (vlm_config.num_hidden_layers,
                         T, vlm_config.num_key_value_heads,
                         vlm_config.hidden_size // vlm_config.num_attention_heads))

llm_interp.resizeSession(llm_session)

hidden_states_data = MNN.Tensor((1, T, 512), MNN.Halide_Type_Float,
                                hidden_states.astype(np.float32),
                                MNN.Tensor_DimensionType_Caffe)
hidden_states_tensor.copyFrom(hidden_states_data)

attention_mask_data = MNN.Tensor((1, T), MNN.Halide_Type_Int,
                                 attention_mask.numpy().astype(np.int32),
                                 MNN.Tensor_DimensionType_Caffe)
attention_mask_tensor.copyFrom(attention_mask_data)

cos_pe = freqs_cos[0: seqlen].numpy()
sin_pe = freqs_sin[0: seqlen].numpy()

cos_pe_data = MNN.Tensor((T, 64), MNN.Halide_Type_Float,
                         cos_pe,
                         MNN.Tensor_DimensionType_Caffe)
cos_pe_tensor.copyFrom(cos_pe_data)

sin_pe_data = MNN.Tensor((T, 64), MNN.Halide_Type_Float,
                         sin_pe,
                         MNN.Tensor_DimensionType_Caffe)
sin_pe_tensor.copyFrom(sin_pe_data)

llm_interp.runSession(llm_session)

logits = MNN.Tensor((1, T, 6400), MNN.Halide_Type_Float,
                    np.ones([1, T, 6400]).astype(np.float32),
                    MNN.Tensor_DimensionType_Caffe)
logits_tensor.copyToHostTensor(logits)
logits = np.array(logits.getData()).reshape(logits.getShape())

present_keys = MNN.Tensor((vlm_config.num_hidden_layers,
                           T, vlm_config.num_key_value_heads,
                           vlm_config.hidden_size // vlm_config.num_attention_heads),
                          MNN.Halide_Type_Float,
                          np.ones([vlm_config.num_hidden_layers,
                                   T, vlm_config.num_key_value_heads,
                                   vlm_config.hidden_size // vlm_config.num_attention_heads]).astype(np.float32),
                          MNN.Tensor_DimensionType_Caffe)
present_keys_tensor.copyToHostTensor(present_keys)
present_keys = np.array(present_keys.getData()).reshape(present_keys.getShape())

present_values = MNN.Tensor((vlm_config.num_hidden_layers,
                             T, vlm_config.num_key_value_heads,
                             vlm_config.hidden_size // vlm_config.num_attention_heads),
                            MNN.Halide_Type_Float,
                            np.ones([vlm_config.num_hidden_layers,
                                     T, vlm_config.num_key_value_heads,
                                     vlm_config.hidden_size // vlm_config.num_attention_heads]).astype(np.float32),
                            MNN.Tensor_DimensionType_Caffe)
present_values_tensor.copyToHostTensor(present_values)
present_values = np.array(present_values.getData()).reshape(present_values.getShape())

token_id = np.argmax(logits[:, -1, :])

## 循环

max_new_token_num = 32
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

    input_ids_data = MNN.Tensor((1, 1), MNN.Halide_Type_Int,
                                np.array([[token_id]], dtype=np.int32),
                                MNN.Tensor_DimensionType_Caffe)

    embed_interp.resizeTensor(input_ids_tensor, (1, 1))
    embed_interp.resizeTensor(embed_tokens_tensor, (1, 1, 512))
    embed_interp.resizeSession(embed_session)
    input_ids_tensor.copyFrom(input_ids_data)

    embed_interp.runSession(embed_session)

    embed_tokens = MNN.Tensor((1, 1, 512), MNN.Halide_Type_Float,
                              np.ones([1, 1, 512]).astype(np.float32),
                              MNN.Tensor_DimensionType_Caffe)
    embed_tokens_tensor.copyToHostTensor(embed_tokens)
    hidden_states = np.array(embed_tokens.getData()).reshape(embed_tokens.getShape())

    cos_pe = freqs_cos[start_pos: start_pos + 1].numpy()
    sin_pe = freqs_sin[start_pos: start_pos + 1].numpy()

    # LLM

    llm_interp.resizeTensor(hidden_states_tensor,
                            (1, 1, 512))
    llm_interp.resizeTensor(attention_mask_tensor,
                            attention_mask.shape)
    llm_interp.resizeTensor(cos_pe_tensor,
                            (1, 64))
    llm_interp.resizeTensor(sin_pe_tensor,
                            (1, 64))
    llm_interp.resizeTensor(past_keys_tensor,
                            present_keys.shape)
    llm_interp.resizeTensor(past_values_tensor,
                            present_values.shape)

    llm_interp.resizeTensor(logits_tensor,
                            (1, 1, 6400))
    llm_interp.resizeTensor(present_keys_tensor,
                            (
                                present_keys.shape[0], present_keys.shape[1] + 1, present_keys.shape[2],
                                present_keys.shape[3]))
    llm_interp.resizeTensor(present_values_tensor,
                            (present_values.shape[0], present_values.shape[1] + 1, present_values.shape[2],
                             present_values.shape[3]))

    llm_interp.resizeSession(llm_session)

    hidden_states_data = MNN.Tensor((1, 1, 512), MNN.Halide_Type_Float,
                                    hidden_states.astype(np.float32),
                                    MNN.Tensor_DimensionType_Caffe)
    hidden_states_tensor.copyFrom(hidden_states_data)

    attention_mask_data = MNN.Tensor(attention_mask.shape, MNN.Halide_Type_Int,
                                     attention_mask.astype(np.int32),
                                     MNN.Tensor_DimensionType_Caffe)
    attention_mask_tensor.copyFrom(attention_mask_data)

    cos_pe_data = MNN.Tensor((1, 64), MNN.Halide_Type_Float,
                             cos_pe,
                             MNN.Tensor_DimensionType_Caffe)
    cos_pe_tensor.copyFrom(cos_pe_data)

    sin_pe_data = MNN.Tensor((1, 64), MNN.Halide_Type_Float,
                             sin_pe,
                             MNN.Tensor_DimensionType_Caffe)
    sin_pe_tensor.copyFrom(sin_pe_data)

    past_keys_data = MNN.Tensor(present_keys.shape, MNN.Halide_Type_Float,
                                present_keys.astype(np.float32),
                                MNN.Tensor_DimensionType_Caffe)
    past_keys_tensor.copyFrom(past_keys_data)

    past_values_data = MNN.Tensor(present_values.shape, MNN.Halide_Type_Float,
                                  present_values.astype(np.float32),
                                  MNN.Tensor_DimensionType_Caffe)
    past_values_tensor.copyFrom(past_values_data)

    llm_interp.runSession(llm_session)

    logits = MNN.Tensor((1, 1, 6400), MNN.Halide_Type_Float,
                        np.ones([1, 1, 6400]).astype(np.float32),
                        MNN.Tensor_DimensionType_Caffe)
    logits_tensor.copyToHostTensor(logits)
    logits = np.array(logits.getData()).reshape(logits.getShape())

    present_keys = MNN.Tensor((present_keys.shape[0],
                               present_keys.shape[1] + 1,
                               present_keys.shape[2],
                               present_keys.shape[3]),
                              MNN.Halide_Type_Float,
                              np.ones([present_keys.shape[0],
                                       present_keys.shape[1] + 1,
                                       present_keys.shape[2],
                                       present_keys.shape[3]]).astype(np.float32),
                              MNN.Tensor_DimensionType_Caffe)
    present_keys_tensor.copyToHostTensor(present_keys)
    present_keys = np.array(present_keys.getData()).reshape(present_keys.getShape())

    present_values = MNN.Tensor((present_values.shape[0],
                                 present_values.shape[1] + 1,
                                 present_values.shape[2],
                                 present_values.shape[3]),
                                MNN.Halide_Type_Float,
                                np.ones([present_values.shape[0],
                                         present_values.shape[1] + 1,
                                         present_values.shape[2],
                                         present_values.shape[3]]).astype(np.float32),
                                MNN.Tensor_DimensionType_Caffe)
    present_values_tensor.copyToHostTensor(present_values)
    present_values = np.array(present_values.getData()).reshape(present_values.getShape())

    token_id = np.argmax(logits[:, -1, :])

    if token_id == 2:
        print("【出来了】", end='\n')
        break
    start_pos += 1
    generate_token_num += 1
