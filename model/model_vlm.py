import os
import sys
import torch
from torch import nn
import warnings

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root)
from model.model_minimind import *
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * (64 * 5 + 1),
            image_ids: List = [34] * (64 * 5 + 1),
            ve_hidden_size: int = 1024,
            max_seq_len: int = 2048,
            per_image_token_num: int = 49,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.ve_hidden_size = ve_hidden_size
        self.max_seq_len = max_seq_len
        self.per_image_token_num = per_image_token_num
        super().__init__(**kwargs)


class VisionProjOld(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=1024, hidden_size=512):
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(ve_hidden_size),
            nn.Linear(ve_hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, image_encoders):
        return self.vision_proj(image_encoders)


# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model(pixel_values=image_tensors)

        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                       batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                                   matches[batch_idx].nonzero(as_tuple=True)[0]]
                       for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
                   } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT


class MomiMindVLM1(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model=None, processor=None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder = vision_model
        self.processor = processor
        self.vision_proj = VisionProj(ve_hidden_size=params.ve_hidden_size, hidden_size=params.hidden_size)

    @staticmethod
    def build_2d_sincos_position_embedding(h, w, dim, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
        pos = torch.stack([x, y], dim=-1).float()  # [H, W, 2]
        omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
        omega = 1. / (10000 ** omega)
        out = torch.cat([
            torch.sin(pos[..., 0:1] * omega),
            torch.cos(pos[..., 0:1] * omega),
            torch.sin(pos[..., 1:2] * omega),
            torch.cos(pos[..., 1:2] * omega),
        ], dim=-1)
        return out.reshape(1, h * w, dim)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            vision_tensors = MomiMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            B, C, H, W = vision_tensors.shape
            vision_tensors = vision_tensors.flatten(2).transpose(1, 2)
            pos_emb = MomiMindVLM.build_2d_sincos_position_embedding(H, W, C, vision_tensors.device)
            vision_tensors = vision_tensors + pos_emb
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT


class MomiMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model=None, processor=None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder = vision_model
        self.processor = processor
        self.vision_proj = VisionProj(ve_hidden_size=params.ve_hidden_size, hidden_size=params.hidden_size)

    @staticmethod
    def build_2d_sincos_position_embedding(h, w, dim, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
        pos = torch.stack([x, y], dim=-1).float()  # [H, W, 2]
        omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
        omega = 1. / (10000 ** omega)
        out = torch.cat([
            torch.sin(pos[..., 0:1] * omega),
            torch.cos(pos[..., 0:1] * omega),
            torch.sin(pos[..., 1:2] * omega),
            torch.cos(pos[..., 1:2] * omega),
        ], dim=-1)
        return out.reshape(1, h * w, dim)

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False
        return module

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(image)
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model(image_tensors)
        return outputs

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=1524):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                       batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                                   matches[batch_idx].nonzero(as_tuple=True)[0]]
                       for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
                   } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)

            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    for start_idx, end_idx in image_indices[i]:
                        h_i = torch.cat((h_i[:start_idx], vision_proj[i], h_i[end_idx + 1:]), dim=0)[
                              :seqlen]
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            vision_tensors = MomiMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            B, C, H, W = vision_tensors.shape
            vision_tensors = vision_tensors.flatten(2).transpose(1, 2)
            pos_emb = MomiMindVLM.build_2d_sincos_position_embedding(H, W, C, vision_tensors.device)
            vision_tensors = vision_tensors + pos_emb
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT


class TinyMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model=None, processor=None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder = vision_model
        self.processor = processor
        self.vision_proj = VisionProj(ve_hidden_size=params.ve_hidden_size, hidden_size=params.hidden_size)

    @staticmethod
    def build_2d_sincos_position_embedding(h, w, dim, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
        pos = torch.stack([x, y], dim=-1).float()  # [H, W, 2]
        omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
        omega = 1. / (10000 ** omega)
        out = torch.cat([
            torch.sin(pos[..., 0:1] * omega),
            torch.cos(pos[..., 0:1] * omega),
            torch.sin(pos[..., 1:2] * omega),
            torch.cos(pos[..., 1:2] * omega),
        ], dim=-1)
        return out.reshape(1, h * w, dim)

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False
        return module

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(image)
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            last_hidden_state = vision_model.last_hidden_state(image_tensors)
        return last_hidden_state

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=1524):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                       batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                                   matches[batch_idx].nonzero(as_tuple=True)[0]]
                       for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
                   } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    for start_idx, end_idx in image_indices[i]:
                        h_i = torch.cat((h_i[:start_idx], vision_proj[i], h_i[end_idx + 1:]), dim=0)[
                              :seqlen]
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            last_hidden_state = TinyMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            N, L, D = last_hidden_state.shape

            # pos_emb = MomiMindVLM.build_2d_sincos_position_embedding(H, W, C, vision_tensors.device)
            # vision_tensors = vision_tensors + pos_emb
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=last_hidden_state,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT

# SmolVLM的一些设计准则
class TinyMindVLM_Compact(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model=None, processor=None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder = vision_model
        self.processor = processor
        self.vision_proj = VisionProj(ve_hidden_size=params.ve_hidden_size, hidden_size=params.hidden_size)

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False
        return module

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(image)
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            last_hidden_state = vision_model.last_hidden_state(image_tensors)
        return last_hidden_state

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=2047):
        def find_indices(tokens):
            """
            返回：
            {
                0: {
                    0: [(3,4)],     # pattern 0 出现的位置
                    1: [(20,22)],   # pattern 1 出现的位置
                    2: [(25,26)]      # pattern 2 出现的位置
                    ...
                },
                1: {
                   ....
                },
                .....
            }
            """
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
                        batch_res[k] = [(i.item() + 2, i.item() + vision_tensors.shape[-2] + 1) for i in idxs]
                if batch_res:
                    results[b] = batch_res
            return results or None

        image_batch_indices = find_indices(tokens)
        if vision_tensors is not None and image_batch_indices:
            vision_proj = self.vision_proj(vision_tensors)
            B, P, L, D = vision_tensors.shape
            new_h = []
            for i in range(B):
                h_i = h[i]
                image_indices = image_batch_indices[i]
                for tki, index in image_indices.items():
                    vision_proj_i = vision_proj[i][tki]
                    start_idx, end_idx = index[0]
                    h_i = torch.cat((h_i[:start_idx], vision_proj_i, h_i[end_idx + 1:]), dim=0)[:seqlen]

                new_h.append(h_i)
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = True,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):

        batch_size, seq_length = input_ids.shape

        # past_key： [batch, seq_len, num_key_value_heads, head_dim]，
        # head_dim = hidden_size / num_attention_heads = 512/8 = 64
        # 每个 KV-head 会被映射到多个 attention heads（这里每个 KV-head 对应 8/2 = 4 个 attention heads）

        # if past_key_values:
        #     for past_key_value in past_key_values:
        #         if past_key_value:
        #             print(len(past_key_value), past_key_value[0].shape, past_key_value[1].shape)
        #         else:
        #             print(past_key_value)

        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0


        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            B, P, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape([-1, C, H, W])
            last_hidden_state = TinyMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            N, L, D = last_hidden_state.shape
            assert B * P == N
            last_hidden_state = last_hidden_state.reshape(B, P, L, D)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=last_hidden_state)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )
        presents = []
        # past_key_value 8 layer, tokens num, 2, 64
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask if attention_mask.shape[1] == seq_length else None
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT


class VisionEncoderExport(nn.Module):
    def __init__(self, params: VLMConfig, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_encoder = vision_model
        self.vision_proj = VisionProj(ve_hidden_size=params.ve_hidden_size, hidden_size=params.hidden_size)

    def forward(self, pixel_values):
        last_hidden_state = self.vision_encoder.last_hidden_state(pixel_values)
        N, L, D = last_hidden_state.shape
        last_hidden_state = last_hidden_state.reshape(1, N, L, D)
        vision_proj = self.vision_proj(last_hidden_state)
        return vision_proj


class TinyMindVLM_CompactExport(MiniMindForCausalLMExport):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params

    # 'input_ids', 'attention_mask', 'position_ids', 'past_key_values'
    def forward(self,
                hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cos_position_embeddings: Optional[torch.Tensor] = None,
                sin_position_embeddings: Optional[torch.Tensor] = None,
                past_keys: Optional[torch.Tensor] = None,
                past_values: Optional[torch.Tensor] = None,
                use_cache: bool = True):

        use_cache = True
        present_keys = []
        present_values = []
        for layer_idx, layer in enumerate(self.model.layers):
            hidden_states, present_key, present_value = layer(
                hidden_states,
                cos_position_embeddings=cos_position_embeddings,
                sin_position_embeddings=sin_position_embeddings,
                past_key=past_keys[layer_idx].unsqueeze(0),
                past_value=past_values[layer_idx].unsqueeze(0),
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            present_keys.append(present_key)
            present_values.append(present_value)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, hidden_states, torch.cat(present_keys, 0), torch.cat(present_values, 0)



if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from dataset.lm_dataset import Objects365DatasetWithSplit
    import open_clip

    vlm_config = VLMConfig(ve_hidden_size=679, hidden_size=512, num_hidden_layers=8, use_moe=False,
                           per_image_token_num=49)

    tokenizer = AutoTokenizer.from_pretrained("../minimind-master/TinyMind/")
    vision_model_path = '../Cream-0ef394cd0c0f41b55fb073ab9abbb95acc13104e/TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'TinyCLIP-auto-ViT-63M-32-Text-31M', pretrained=vision_model_path)

    model = TinyMindVLM_Compact(vlm_config, vision_model=clip_model._image_encoder, processor=preprocess)

    train_ds = Objects365DatasetWithSplit(annotation_file="/objects365/val/zhiyuan_objv2_captions_x512_val.json",
                                          root_dir="/objects365/val/", tokenizer=tokenizer, preprocess=preprocess)

    dl = DataLoader(train_ds, batch_size=5, shuffle=True)
    for X, Y, loss_mask, image_tensor, attention_mask in dl:
        print(X.shape, attention_mask.shape, image_tensor.shape)
        y = model(X, attention_mask=attention_mask, pixel_values=image_tensor)
        break
