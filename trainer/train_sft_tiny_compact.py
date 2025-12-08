import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import TinyMindVLM_Compact, VLMConfig
from dataset.lm_dataset import Objects365DatasetWithSplitLMDB, adaptive_square_split, Objects365DatasetWithSplit
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, \
    init_tiny_vlm_model, vlm_checkpoint, SkipBatchSampler
import open_clip

warnings.filterwarnings('ignore')

all_images = [x.path for x in os.scandir("～/coco128/images/train2017/") if x.name.endswith("jpg")]


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, tokenizer=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask, pixel_values, attention_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        attention_mask = attention_mask.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X, pixel_values=pixel_values, attention_mask=attention_mask)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')

            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}_epoch{epoch}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items()  # if not key.startswith('vision_encoder.')
            }
            # clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}  # 半精度保存
            clean_state_dict = {k: v for k, v in clean_state_dict.items()}  # 半精度保存
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                           epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)

            from PIL import Image
            import random
            import numpy as np
            from dataset.template import prompts_template
            image_path = random.choice(all_images)
            query = random.choice(prompts_template)
            image = Image.open(image_path).convert('RGB')

            pixel_values = TinyMindVLM_Compact.image2tensor(image, preprocess).to(args.device)

            blocks, rows, cols, block_size = adaptive_square_split(
                image_path=image_path,
                max_rows=4,
                max_cols=4
            )
            # print("rows, cols", rows, cols)
            patch_num = len(blocks)
            pad_num = 16 - patch_num
            mask_token_id = []
            special_token = {
                "<global-img>": tokenizer.convert_tokens_to_ids("<global-img>"),
                "<fake_token_around_image>": tokenizer.convert_tokens_to_ids("<fake_token_around_image>"),
                "<image>": tokenizer.convert_tokens_to_ids("<image>"),
            }
            for i in range(4):
                for j in range(4):
                    special_token[f"<row_{i + 1}_col_{j + 1}>"] = tokenizer.convert_tokens_to_ids(
                        f"<row_{i + 1}_col_{j + 1}>")

            if pad_num:
                patch_tensor = []
                for i in range(4):
                    for j in range(4):
                        if i >= rows or j >= cols:
                            patch_tensor.append(torch.zeros_like(pixel_values).to(args.device))
                            mask_token_id.append(special_token[f"<row_{i + 1}_col_{j + 1}>"])
                        else:
                            patch_tensor.append(
                                TinyMindVLM_Compact.image2tensor(blocks[i * cols + j], preprocess).to(args.device))
            else:
                patch_tensor = [TinyMindVLM_Compact.image2tensor(block, preprocess).to(args.device) for block in blocks]

            assert len(patch_tensor) == 16

            pixel_values = torch.stack(patch_tensor + [pixel_values], dim=0).unsqueeze(0)

            image_place_holder = random.choice(["图片如下：", "如下所示的图片:", "请见下面这张图:", "如下图显示:", "参考下方图片:", "图示如下:"])
            for row in range(4):
                for col in range(4):
                    image_place_holder += f"<fake_token_around_image><row_{row + 1}_col_{col + 1}>"
                    image_place_holder += "<image>" * 49

            image_place_holder += f"<fake_token_around_image><global-img>{'<image>' * 49}<fake_token_around_image>"

            messages = [
                {"role": "system", "content": "你是一个多模态AI助手，能够理解图片和文本信息."},
                {
                    "role": "user",
                    "content": query + image_place_holder
                }
            ]

            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

            attention_mask = inputs["attention_mask"]

            windows = inputs["input_ids"].unfold(-1, size=2, step=1)
            for token in mask_token_id:
                pattern = torch.tensor([special_token["<fake_token_around_image>"], token]).to(args.device)
                matches = (windows == pattern).all(dim=-1)
                indices = matches.nonzero(as_tuple=True)[0]
                attention_mask[indices[0] + 2: indices[0] + 2 + 49] = 0

            print(f'👶: {query}')
            print('🤖️: ', end='')

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                output = model.module.generate(
                    inputs=inputs["input_ids"], attention_mask=attention_mask,
                    max_new_tokens=1024, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    top_p=0.85, temperature=0.65, pixel_values=pixel_values
                )
            else:
                output = model.generate(
                    inputs=inputs["input_ids"], attention_mask=attention_mask,
                    max_new_tokens=1024, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    top_p=0.85, temperature=0.65, pixel_values=pixel_values
                )

            decoded_text = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            print(f'{decoded_text}\n\n')

            table = wandb.echarts.Table()
            headers = ["输入问题", "模型输出"]
            rows = [[query, decoded_text]]
            table.add(headers, rows)

            wandb.log(
                {
                    "sample/输入图像": wandb.Image(np.array(image)),
                    "sample/问题&回复": table,
                }
            )

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MomiMindVLM SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_vlm', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1536, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--annotation_file", type=str, default="../dataset/sft_data.jsonl", help="训练数据路径")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_data.jsonl", help="训练数据路径")
    parser.add_argument("--images_path", type=str, default="../dataset/sft_images", help="训练图像路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MomiMindVLM-SFT", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)

    vlm_config = VLMConfig(ve_hidden_size=679, hidden_size=512, num_hidden_layers=8, use_moe=False,
                           per_image_token_num=49)

    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight,
                              save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda",
                                                                                 dtype=dtype)  # torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"{args.wandb_project}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume, mode="offline",
                   logdir=f"./swanlog/{args.save_dir}")

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_tiny_vlm_model(vlm_config,
                                                       device=args.device,
                                                       tokenizer_path="../minimind-master/TinyMind/",
                                                       vision_model_path='~/Cream-0ef394cd0c0f41b55fb073ab9abbb95acc13104e/TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt',
                                                       is_split=True)

    train_ds = Objects365DatasetWithSplitLMDB(annotation_file=args.annotation_file,
                                              text_only_jsonl="~/hulk/10w_plus.jsonl",
                                              root_dir=args.data_path, tokenizer=tokenizer, preprocess=preprocess,
                                              max_length=vlm_config.max_seq_len,
                                              per_image_token_num=vlm_config.per_image_token_num)
    # train_ds = Objects365DatasetWithSplit(annotation_file = args.annotation_file,
    #                             root_dir = args.data_path, tokenizer = tokenizer, preprocess=preprocess, text_only_jsonl="~/hulk/10w_plus.jsonl",
    #                             max_length=vlm_config.max_seq_len, per_image_token_num=vlm_config.per_image_token_num)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    print("@@@ start_epoch, start_step", start_epoch, start_step)
    for name, param in model.named_parameters():
        if 'vision_encoder' in name:
            param.requires_grad = False
        # if 'vision_encoder' in name:
        #     if "final_conv" in name or "vision_encoder.1.3" in name:
        #          param.requires_grad = True
        # if 'vision_proj' not in name:
        #     param.requires_grad = False

    Logger(f'所加载VLM Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} MB')

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=False,
                                persistent_workers=False)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb, tokenizer)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=False,
                                persistent_workers=False)
            train_epoch(epoch, loader, len(loader), 0, wandb, tokenizer)

#  python trainer/train_sft_vlm2.py --annotation_file /objects365/val/zhiyuan_objv2_captions_x512_val.json --data_path /objects365/val/ --save_dir exp_proj --num_workers 16 --batch_size 2 --use_wand

#  python trainer/train_sft_vlm2.py --annotation_file /objects365/train/zhiyuan_objv2_captions_x512_train.json --data_path /objects365/train/ --save_dir exp3 --num_workers 16 --batch_size 32 --use_wand --learning_rate 1e-6 --epochs 3 --wandb_project MomiMindVLM3


# /usr/local/bin/python trainer/train_sft_tiny_compact.py --annotation_file /objects365/val/zhiyuan_objv2_captions_x512_val.json --data_path /objects365/val/ --save_dir exp_tiny_compact --num_workers 16 --batch_size 64 --wandb_project tiny_compact --use_wand --learning_rate 1e-6 --epochs 5 --wandb_project TinyMindVLM

# torchrun --nproc_per_node 4 trainer/train_sft_tiny_compact.py --annotation_file /objects365/train/zhiyuan_objv2_captions_x512_train.json --data_path /objects365/train/ --save_dir exp_tiny_compact --num_workers 16 --batch_size 8 --wandb_project tiny_compact --use_wand --learning_rate 1e-6 --epochs 5 --wandb_project TinyMindVLM --save_interval 1000

# torchrun --nproc_per_node 4 trainer/train_sft_tiny_compact.py --annotation_file /objects365/train/captions_lmdb --data_path /objects365/train/ --save_dir exp_tiny_compact_2 --num_workers 8 --batch_size 8 --wandb_project tiny_compact_2 --use_wand --learning_rate 1e-6 --epochs 5 --wandb_project TinyMindVLM --save_interval 100

# /usr/local/bin/python trainer/train_sft_tiny_compact.py --annotation_file /objects365/val/zhiyuan_objv2_captions_x512_val.json --data_path /objects365/val/ --save_dir exp_tiny_compact --num_workers 16 --batch_size 8 --wandb_project tiny_compact --use_wand --learning_rate 1e-6 --epochs 1 --wandb_project TinyMindVLM --save_interval 100

# /usr/local/bin/python  trainer/train_sft_tiny_compact.py --annotation_file /objects365/train/captions_lmdb --data_path /objects365/train/ --save_dir exp_tiny_compact1 --num_workers 16 --batch_size 8 --wandb_project tiny_compact --use_wand --learning_rate 5e-5 --epochs 1 --wandb_project TinyMindVLM --save_interval 100
