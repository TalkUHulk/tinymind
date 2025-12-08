import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root)
from model.model_vlm import MiniMindVLM, MomiMindVLM, TinyMindVLM, TinyMindVLM_Compact
import os
from pycocotools.coco import COCO
import ujson as json
import random
from dataset.template import *
from dataset.labels import object365_zh_with_synonyms
import lmdb
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def adaptive_square_split(image_path, max_rows=4, max_cols=4):
    """
    自适应切分图像为正方形patch，优先保证正方形，行列数可以少于最大值

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        max_rows: 最大行数
        max_cols: 最大列数
    """

    # 打开图像
    img = Image.open(image_path)
    original_width, original_height = img.size

    # 1. 计算最佳的正方形切分方案
    rows, cols, block_size = calculate_optimal_split_with_fixed_max(
        original_width, original_height, max_rows, max_cols
    )
   
    # 3. 执行切分
    blocks = []
    for i in range(rows):
        for j in range(cols):
            # 计算每个块的坐标
            left = j * block_size
            upper = i * block_size
            right = left + block_size
            lower = upper + block_size

            # 切割图像块
            block = img.crop((left, upper, right, lower))
            blocks.append(block)

    return blocks, rows, cols, block_size


def calculate_optimal_split_with_fixed_max(width, height, max_rows, max_cols):
    """
    计算最佳切分方案，保证行或列至少一个为最大值
    """

    best_rows = 1
    best_cols = 1
    best_block_size = 0
    best_coverage = 0

    # 方案1: 固定行数为4，自适应列数
    rows_fixed = max_rows
    for cols in range(1, max_cols + 1):
        block_width = width // cols
        block_height = height // rows_fixed
        square_size = min(block_width, block_height)

        if square_size > 0:
            coverage = (cols * square_size) * (rows_fixed * square_size) / (width * height)
            # 选择覆盖率高且正方形尺寸大的方案
            if coverage > best_coverage or (coverage == best_coverage and square_size > best_block_size):
                best_rows = rows_fixed
                best_cols = cols
                best_block_size = square_size
                best_coverage = coverage

    # 方案2: 固定列数为4，自适应行数
    cols_fixed = max_cols
    for rows in range(1, max_rows + 1):
        block_width = width // cols_fixed
        block_height = height // rows
        square_size = min(block_width, block_height)

        if square_size > 0:
            coverage = (cols_fixed * square_size) * (rows * square_size) / (width * height)
            if coverage > best_coverage or (coverage == best_coverage and square_size > best_block_size):
                best_rows = rows
                best_cols = cols_fixed
                best_block_size = square_size
                best_coverage = coverage

    # 方案3: 如果可能，行列都达到最大值
    block_width = width // max_cols
    block_height = height // max_rows
    square_size = min(block_width, block_height)

    if square_size > 0:
        coverage = (max_cols * square_size) * (max_rows * square_size) / (width * height)
        if coverage > best_coverage or (coverage == best_coverage and square_size > best_block_size):
            best_rows = max_rows
            best_cols = max_cols
            best_block_size = square_size
            best_coverage = coverage

    # 最终确定的正方形尺寸（向下取整到16的倍数）
    best_block_size = (best_block_size // 16) * 16
    if best_block_size == 0:
        best_block_size = 16  # 最小尺寸

    return best_rows, best_cols, best_block_size



class VLMDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return X, Y, loss_mask, image_tensors



class Objects365Dataset(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, preprocess=None, max_length=1024,
                 image_special_token='@' * 64):

        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.image_ids)

    def _create_chat_prompt(self, conversations):
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content":  self.image_token + conversations["q"] if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content": self.image_token + ":" + random.choice(prompts_template)
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        # assert os.path.isfile(image_path)
        if not os.path.isfile(image_path):
            image_id = self.image_ids[0]
            image_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.root_dir, image_info['file_name'])
        
        image = Image.open(image_path)
        image_tensor = TinyMindVLM.image2tensor(image, self.preprocess)

        caption = image_info["caption"]
        qa = image_info["qa"]

        msg = None
        if random.random() < 0.6:
            msg = random.choice(qa)
            if "q" not in msg or "a" not in msg:
                msg = caption
        else:
            msg = caption
        prompt = self._create_chat_prompt(msg)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask, image_tensor



class Objects365DatasetMB(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, preprocess=None, max_length=1024,
                 image_special_token='@' * 64):

        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.image_ids)

    def _create_chat_prompt(self, conversations):
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content":  self.image_token + conversations["q"] if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content": self.image_token + ":" + random.choice(prompts_template)
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        # assert os.path.isfile(image_path)
        if not os.path.isfile(image_path):
            image_id = self.image_ids[0]
            image_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.root_dir, image_info['file_name'])
        
        image = Image.open(image_path)
        image_tensor = MomiMindVLM.image2tensor(image, self.preprocess)

        caption = image_info["caption"]
        qa = image_info["qa"]

        msg = None
        if random.random() < 0.6:
            msg = random.choice(qa)
            if "q" not in msg or "a" not in msg:
                msg = caption
        else:
            msg = caption
        prompt = self._create_chat_prompt(msg)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask, image_tensor



class Objects365DatasetWithSplit(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, text_only_jsonl=None, preprocess=None, max_length=2048, per_image_token_num=49):

        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.conversations = []
        if text_only_jsonl:
            with open(text_only_jsonl, "r") as f:
                lines = f.readlines()
                for line in lines:
                    conversation = json.loads(line.strip())
                    self.conversations.append(conversation)
        self.datasets = self.image_ids + self.conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.per_image_token_num = per_image_token_num
        self.image_token = "<image>"
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        self.max_rows = 4
        self.max_cols = 4
        self.special_token = {
            "<global-img>": tokenizer.convert_tokens_to_ids("<global-img>"),
            "<fake_token_around_image>": tokenizer.convert_tokens_to_ids("<fake_token_around_image>"),
            "<image>": tokenizer.convert_tokens_to_ids("<image>"),
        }
        for i in range(self.max_rows):
            for j in range(self.max_cols):
                self.special_token[f"<row_{i + 1}_col_{j + 1}>"] = tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>")


    def __len__(self):
        return len(self.datasets)

    def _create_chat_prompt(self, conversations):
        
        image_place_holder = random.choice(["图片如下：", "如下所示的图片:", "请见下面这张图:", "如下图显示:", "参考下方图片:", "图示如下:"])
        for row in range(self.max_rows):
            for col in range(self.max_cols):
                image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
                image_place_holder += self.image_token * self.per_image_token_num

        image_place_holder += f"<fake_token_around_image><global-img>{self.image_token * self.per_image_token_num}<fake_token_around_image>"
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
                {
                    "role": "user",
                    "content":  conversations["q"] + image_place_holder if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
                {
                    "role": "user",
                    "content": random.choice(prompts_template) + image_place_holder
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _create_chat_prompt_with_fake_image(self, conversations):
        
        image_place_holder = ""
        for row in range(self.max_rows):
            for col in range(self.max_cols):
                image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
                image_place_holder += self.image_token * self.per_image_token_num

        image_place_holder += f"<fake_token_around_image><global-img>{self.image_token * self.per_image_token_num}<fake_token_around_image>"

        messages = [
            {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
            {
                "role": "user",
                "content":  conversations["instruction"] + image_place_holder
                
            },
            {
                "role": "assistant",
                "content":  conversations["output"]
            }
        ]
       
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        item = self.datasets[idx]
        mask_token_id = []
        try:
            if isinstance(item, int):
                image_id = item
                image_info = self.coco.imgs[image_id]
                image_path = os.path.join(self.root_dir, image_info['file_name'])
                if not os.path.isfile(image_path):
                    image_id = self.image_ids[0]
                    image_info = self.coco.imgs[image_id]
                    image_path = os.path.join(self.root_dir, image_info['file_name'])
                # print("image_path:", image_path)
                image = Image.open(image_path)
                image_tensor = TinyMindVLM_Compact.image2tensor(image, self.preprocess)

                blocks, rows, cols, block_size = adaptive_square_split(
                    image_path=image_path,
                    max_rows=self.max_rows,
                    max_cols=self.max_cols
                )
                # print("rows, cols", rows, cols)
                patch_num = len(blocks)
                pad_num = self.max_rows * self.max_cols - patch_num
                
                if pad_num:
                    patch_tensor = []
                    for i in range(self.max_rows):
                        for j in range(self.max_cols):
                            if i >= rows or j >= cols:
                                patch_tensor.append(torch.zeros_like(image_tensor))
                                mask_token_id.append(self.special_token[f"<row_{i + 1}_col_{j + 1}>"])
                                # print("mask:", f"<row_{i + 1}_col_{j + 1}>")
                            else:
                                patch_tensor.append(MomiMindVLM.image2tensor(blocks[i * cols + j], self.preprocess))
                else:
                    patch_tensor = [MomiMindVLM.image2tensor(block, self.preprocess) for block in blocks]

                assert len(patch_tensor) == self.max_cols * self.max_rows

                image_tensor = torch.stack(patch_tensor + [image_tensor], dim=0)
                
                caption = image_info["caption"]
                qa = image_info["qa"]

                msg = None
                if random.random() < 0.6:
                    msg = random.choice(qa)
                    if "q" not in msg or "a" not in msg:
                        msg = caption
                else:
                    msg = caption
                prompt_with_label = self._create_chat_prompt(msg)
                # input_ids = self.tokenizer(prompt_with_label).input_ids[:self.max_length]
                # pad_num = self.max_length - len(input_ids)
                # input_ids += [self.tokenizer.pad_token_id] * pad_num

                # loss_mask = self._generate_loss_mask(input_ids)
                # print(prompt_with_label)
                # During SFT, only train on completions.
                # print("input_ids:", input_ids)

                # X = torch.tensor(input_ids[:-1], dtype=torch.long)
                # print("X", X.numpy().tolist())
                # attention_mask = torch.ones([X.shape[-1]])
                # attention_mask[-(pad_num - 1):] = 0
                # windows = X.unfold(-1, size=2, step=1)
                # print("mask_token_id", mask_token_id)
                # print("windows", windows.numpy().tolist())
                # for token in mask_token_id:
                #     pattern = torch.tensor([self.special_token["<fake_token_around_image>"], token])
                #     matches = (windows == pattern).all(dim=1)
                #     indices = matches.nonzero(as_tuple=True)[0]
                #     # print(token, indices, pattern)
                #     attention_mask[indices[0]+2: indices[0]+2 + self.per_image_token_num] = 0
                # Y = torch.tensor(input_ids[1:], dtype=torch.long)
                # loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
            else:
                prompt_with_label = self._create_chat_prompt_with_fake_image(item)
                # input_ids = self.tokenizer(prompt_with_label).input_ids[:self.max_length]
                # pad_num = self.max_length - len(input_ids)
                # input_ids += [self.tokenizer.pad_token_id] * pad_num
                # X = torch.tensor(input_ids[:-1], dtype=torch.long)
                # attention_mask = torch.ones([X.shape[-1]])
                # attention_mask[-(pad_num - 1):] = 0
                image_tensor = torch.zeros([self.max_rows * self.max_cols + 1, 3, 224, 224], dtype=torch.float)
                for i in range(self.max_rows):
                    for j in range(self.max_cols):
                        mask_token_id.append(self.special_token[f"<row_{i + 1}_col_{j + 1}>"])
                mask_token_id.append(self.special_token["<global-img>"])  

            input_ids = self.tokenizer(prompt_with_label).input_ids[:self.max_length]
            pad_num = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_num
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            loss_mask = self._generate_loss_mask(input_ids)
            Y = torch.tensor(input_ids[1:], dtype=torch.long)
            loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)


            attention_mask = torch.ones([X.shape[-1]])
            attention_mask[-(pad_num - 1):] = 0
            windows = X.unfold(-1, size=2, step=1)
            for token in mask_token_id:
                pattern = torch.tensor([self.special_token["<fake_token_around_image>"], token])
                matches = (windows == pattern).all(dim=1)
                indices = matches.nonzero(as_tuple=True)[0]
                # print(token, indices, pattern)
                attention_mask[indices[0]+2: indices[0]+2 + self.per_image_token_num] = 0
            
            return X, Y, loss_mask, image_tensor, attention_mask
        except Exception as e:
            return self.__getitem__(0)



class Objects365DatasetWithSplitLMDB(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, text_only_jsonl=None, preprocess=None, max_length=2048, per_image_token_num=49):

        super().__init__()
        self.root_dir = root_dir
        # self.coco = COCO(annotation_file)
        # self.image_ids = list(self.coco.imgs.keys())
        self.env = lmdb.open(annotation_file, readonly=True, max_readers=512, lock=False)
        with self.env.begin() as txn:
            self.images = list(txn.cursor().iternext(values=False))

        self.conversations = []
        if text_only_jsonl:
            with open(text_only_jsonl, "r") as f:
                lines = f.readlines()
                for line in lines:
                    conversation = json.loads(line.strip())
                    self.conversations.append(conversation)
        self.datasets = self.images + self.conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.per_image_token_num = per_image_token_num
        self.image_token = "<image>"
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        self.max_rows = 4
        self.max_cols = 4
        self.special_token = {
            "<global-img>": tokenizer.convert_tokens_to_ids("<global-img>"),
            "<fake_token_around_image>": tokenizer.convert_tokens_to_ids("<fake_token_around_image>"),
            "<image>": tokenizer.convert_tokens_to_ids("<image>"),
        }
        for i in range(self.max_rows):
            for j in range(self.max_cols):
                self.special_token[f"<row_{i + 1}_col_{j + 1}>"] = tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>")

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()

    def __len__(self):
        return len(self.datasets)

    def _create_chat_prompt(self, conversations):
        
        image_place_holder = random.choice(["图片如下：", "如下所示的图片:", "请见下面这张图:", "如下图显示:", "参考下方图片:", "图示如下:"])
        for row in range(self.max_rows):
            for col in range(self.max_cols):
                image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
                image_place_holder += self.image_token * self.per_image_token_num

        image_place_holder += f"<fake_token_around_image><global-img>{self.image_token * self.per_image_token_num}<fake_token_around_image>"
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
                {
                    "role": "user",
                    "content":  conversations["q"] + image_place_holder if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
                {
                    "role": "user",
                    "content": random.choice(prompts_template) + image_place_holder
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _create_chat_prompt_with_fake_image(self, conversations):
        
        image_place_holder = ""
        for row in range(self.max_rows):
            for col in range(self.max_cols):
                image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
                image_place_holder += self.image_token * self.per_image_token_num

        image_place_holder += f"<fake_token_around_image><global-img>{self.image_token * self.per_image_token_num}<fake_token_around_image>"

        messages = [
            {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
            {
                "role": "user",
                "content":  conversations["instruction"] + image_place_holder
                
            },
            {
                "role": "assistant",
                "content":  conversations["output"]
            }
        ]
       
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        item = self.datasets[idx]
        mask_token_id = []
        try:
            if isinstance(item, bytes):
                file_name = item
                with self.env.begin() as txn:
                    byte_data = txn.get(file_name)
                    while byte_data is None:
                        idx = random.randint(0, self.__len__())
                        byte_data = txn.get(file_name)

                    data = pickle.loads(byte_data)
                caption = data["caption"]
                qa = data["qa"]
                
                image_path = os.path.join(self.root_dir, file_name.decode('utf-8'))

                if not os.path.isfile(image_path):
                    return self.__getitem__(0)
                # print("image_path:", image_path)
                image = Image.open(image_path)
                image_tensor = TinyMindVLM_Compact.image2tensor(image, self.preprocess)

                blocks, rows, cols, block_size = adaptive_square_split(
                    image_path=image_path,
                    max_rows=self.max_rows,
                    max_cols=self.max_cols
                )
                # print("rows, cols", rows, cols)
                patch_num = len(blocks)
                pad_num = self.max_rows * self.max_cols - patch_num
                
                if pad_num:
                    patch_tensor = []
                    for i in range(self.max_rows):
                        for j in range(self.max_cols):
                            if i >= rows or j >= cols:
                                patch_tensor.append(torch.zeros_like(image_tensor))
                                mask_token_id.append(self.special_token[f"<row_{i + 1}_col_{j + 1}>"])
                                # print("mask:", f"<row_{i + 1}_col_{j + 1}>")
                            else:
                                patch_tensor.append(MomiMindVLM.image2tensor(blocks[i * cols + j], self.preprocess))
                else:
                    patch_tensor = [MomiMindVLM.image2tensor(block, self.preprocess) for block in blocks]

                assert len(patch_tensor) == self.max_cols * self.max_rows

                image_tensor = torch.stack(patch_tensor + [image_tensor], dim=0)

                msg = None
                if random.random() < 0.6:
                    msg = random.choice(qa)
                    if "q" not in msg or "a" not in msg:
                        msg = caption
                else:
                    msg = caption
                prompt_with_label = self._create_chat_prompt(msg)
            else:
                prompt_with_label = self._create_chat_prompt_with_fake_image(item)
                image_tensor = torch.zeros([self.max_rows * self.max_cols + 1, 3, 224, 224], dtype=torch.float)
                for i in range(self.max_rows):
                    for j in range(self.max_cols):
                        mask_token_id.append(self.special_token[f"<row_{i + 1}_col_{j + 1}>"])
                mask_token_id.append(self.special_token["<global-img>"])  

            input_ids = self.tokenizer(prompt_with_label).input_ids[:self.max_length]
            pad_num = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_num
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            loss_mask = self._generate_loss_mask(input_ids)
            Y = torch.tensor(input_ids[1:], dtype=torch.long)
            loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)


            attention_mask = torch.ones([X.shape[-1]])
            attention_mask[-(pad_num - 1):] = 0
            windows = X.unfold(-1, size=2, step=1)
            for token in mask_token_id:
                pattern = torch.tensor([self.special_token["<fake_token_around_image>"], token])
                matches = (windows == pattern).all(dim=1)
                indices = matches.nonzero(as_tuple=True)[0]
                # print(token, indices, pattern)
                attention_mask[indices[0]+2: indices[0]+2 + self.per_image_token_num] = 0
            
            return X, Y, loss_mask, image_tensor, attention_mask
        except Exception as e:
            print("!!!!!!!!!!", e)
            return self.__getitem__(0)




class Objects365DatasetLMDB(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, preprocess=None, max_length=1024,
                 image_special_token='@' * 64):

        super().__init__()
        self.root_dir = root_dir

        self.env = lmdb.open(annotation_file, readonly=True)
        with self.env.begin() as txn:
            self.images = list(txn.cursor().iternext(values=False))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
       return len(self.images)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()

    def _create_chat_prompt(self, conversations):
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content":  self.image_token + conversations["q"] if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content": self.image_token+ ":" + random.choice(prompts_template)
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        file_name = self.images[idx]
        with self.env.begin() as txn:
            byte_data = txn.get(file_name)
            while byte_data is None:
                idx = random.randint(0, self.__len__())
                byte_data = txn.get(file_name)
            
            data = pickle.loads(byte_data) 

        caption = data["caption"]
        qa = data["qa"]

        image_path = os.path.join(self.root_dir, file_name.decode('utf-8'))
        assert os.path.isfile(image_path)
        
        image = Image.open(image_path)
        image_tensor = MomiMindVLM.image2tensor(image, self.preprocess)

        msg = None
        if random.random() < 0.6:
            msg = random.choice(qa)
        else:
            msg = caption
        prompt = self._create_chat_prompt(msg)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

    
        return X, Y, loss_mask, image_tensor
  
# /objects365/train/images/v2/patch16/objects365_v2_00908726.jpg


class Objects365DatasetCLIP(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, preprocess=None, max_length=1024,
                 image_special_token='@' * 64):

        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.image_ids)

    def _create_chat_prompt(self, conversations):
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content":  self.image_token + conversations["q"] if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content": self.image_token + ":" + random.choice(prompts_template)
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        # assert os.path.isfile(image_path)
        if not os.path.isfile(image_path):
            image_id = self.image_ids[0]
            image_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.root_dir, image_info['file_name'])
        
        image = Image.open(image_path)
        image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)

        caption = image_info["caption"]
        qa = image_info["qa"]

        msg = None
        if random.random() < 0.6:
            msg = random.choice(qa)
            if "q" not in msg or "a" not in msg:
                msg = caption
        else:
            msg = caption
        prompt = self._create_chat_prompt(msg)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask, image_tensor



class Objects365DatasetEval(Dataset):
    def __init__(self, annotation_file, root_dir, tokenizer, preprocess=None, max_length=1024,
                 image_special_token='@' * 49):

        super().__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())[:1000]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.image_ids)

    def _create_chat_prompt(self, conversations):
        if isinstance(conversations, dict):
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content":  self.image_token + conversations["q"] if isinstance(conversations["q"], str) else conversations["q"][0]
                    
                },
                {
                    "role": "assistant",
                    "content":  conversations["a"] if isinstance(conversations["a"], str) else conversations["a"][0]
                }
            ]
        elif isinstance(conversations, str):    
            messages = [
                {"role": "system", "content": "简短回复问题."},
                {
                    "role": "user",
                    "content": self.image_token + ":" + random.choice(prompts_template)
                },
                {
                    "role": "assistant",
                    "content": conversations
                }
            ]
        else:
            raise ValueError("unsupport format")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        # assert os.path.isfile(image_path)
        if not os.path.isfile(image_path):
            image_id = self.image_ids[0]
            image_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.root_dir, image_info['file_name'])
        
        image = Image.open(image_path)
        image_tensor = TinyMindVLM.image2tensor(image, self.preprocess)

        caption = image_info["caption"]
        qa = image_info["qa"]


        return image_tensor, caption, qa, image_path


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    import open_clip
    tokenizer = AutoTokenizer.from_pretrained("../minimind-master/TinyMind/")
    vision_model_path='../Cream-0ef394cd0c0f41b55fb073ab9abbb95acc13104e/TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'TinyCLIP-auto-ViT-63M-32-Text-31M', pretrained=vision_model_path)
    
    # train_ds = Objects365DatasetWithSplit(annotation_file = "/objects365/val/zhiyuan_objv2_captions_x512_val.json", 
    #                             root_dir = "/objects365/val/", text_only_jsonl="../10w_plus.jsonl", tokenizer = tokenizer, preprocess=preprocess)
    train_ds = Objects365DatasetWithSplitLMDB(annotation_file = "/objects365/train/captions_lmdb", 
                                root_dir = "/objects365/train/", text_only_jsonl="../10w_plus.jsonl", tokenizer = tokenizer, preprocess=preprocess)
    
    dl = DataLoader(train_ds, num_workers=32, batch_size=32, shuffle=True)
    from tqdm import tqdm
    for X, Y, loss_mask, image_tensor, attention_mask in tqdm(dl, total=len(dl)):
        # print(X.shape, loss_mask.shape, image_tensor.shape, type(image_tensor), attention_mask.shape)
        pass
        # torch.Size([1, 1559]) torch.Size([1, 1559]) torch.Size([1, 17, 3, 224, 224]) torch.Size([1, 1559])
        # print("attention_mask:", attention_mask.numpy().tolist())
        # def find_indices(tokens):
        #     B, T = tokens.size()
        #     #<fake_token_around_image> <row_i_col_j> +  <fake_token_around_image> <global-img>
        #     image_ids = [[3, i] for i in range(6, 22)] + [[3, 4]]
        #     image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
        #     len_image_ids = len(image_ids[0])
        #     if len_image_ids > tokens.size(1):
        #         return None
        #     tokens_view = tokens.unfold(1, len_image_ids, 1)
        #     matches = []
        #     for image_id_tensor in image_ids_tensor:
        #         match = (tokens_view == image_id_tensor).all(dim=2)
        #         matches.append(match)
        #     results = {}
        #     for b in range(B):
        #         batch_res = {}
        #         for k, m in enumerate(matches):
        #             idxs = m[b].nonzero(as_tuple=True)[0]
        #             if len(idxs) > 0:
        #                 batch_res[k] = [(i.item() + 2, i.item() + 49 + 1) for i in idxs]
        #         if batch_res:
        #             results[b] = batch_res
        #     return results or None
        
        # result = find_indices(X)
        # for i in range(17):
        #     start, end = result[0][i][0]
        #     print(i, X[0][start: end + 1],  len(X[0][start: end + 1]), X[0][start - 2], X[0][start - 1],  X[0][end + 1])
        # break
