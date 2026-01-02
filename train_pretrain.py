import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')
# 1、Pretrain.jsonl数据集内容解释
# Minimind 预训练采用的数据集是Pretrain_hq.jsonl
# 清洗出字符512长度的大约1.6GB的语料直接拼接成预训练数据 pretrain_hq.jsonl
# 在pretrain_hq.jsonl中每一行是由多个Q&A语料拼接而成,目的是为了让model尽可能高效利用Context Window一次训练多个不同的Q&A
# <|im_start|>与<|im_end|>是为了区分拼接的Q&A,让Model知道一个Q&A的开始与结束位置

# 1) Jsonl中某一行的内容并非是连贯的文章,而是由互不相关的Q&A拼接在一起：例如第一句话中:Q是鉴别文章风格。A是需要样例。第二句话：Q是查天气,A是需要指定地区。
# 2) 数据拼接另一个目的是模型的Context Windows一般是很大如512个Token,若问答Token长度很小,则会浪费Context Window。可以让Model在一次计算中学习多个Q&A

# example jsonl的第一行内容{"text": "<|im_start|>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> 
#                                   <|im_start|>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？<|im_end|>
#                                   <|im_start|>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。<|im_end|> 
#                                   <|im_start|>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。<|im_end|> 
#                                   <|im_start|>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。<|im_end|>
#                                   <|im_start|>帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。<|im_end|> 
#                                   <|im_start|>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。<|im_end|> 
#                                   <|im_start|>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。<|im_end|>"}

# 2、准备预训练数据加载器在dataset.py当中的PretrainDataset类

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# Pretrain的主要部分
def train_epoch(epoch, wandb):
    # 定义Loss Function交叉熵损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    # 利用train_loader遍历训练数据
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)   # 输入的Token序列
        Y = Y.to(args.device)   # 目标的Token序列
        loss_mask = loss_mask.to(args.device)
        # 更新learning rate
        lr = get_lr(epoch * iter_per_epoch + step,  # 当前训练的步数
                    args.epochs * iter_per_epoch,   # 总的训练步数
                    args.learning_rate)             # 初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr                  # 动态更新优化器中的Learning Rate

        with ctx:
            res = model(X)      # 模型forward传播得到的结果,[batch_Size,seq_len,vocab_size],
                                #每个Token得到下一个词是Vocab_Size中的可能性(未经过归一化)
            # 计算交叉熵损失函数
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())   
            # 经过loss_mask掩码,无需对<pad>处计算loss, 即模型不需要耗费资源去预测占位符<pad>
            loss = (loss * loss_mask).sum() / loss_mask.sum()   # 所有有效Token的loss平均值
            loss += res.aux_loss
            loss = loss / args.accumulation_steps               # 梯度累积标准化
        # 反向传播
        scaler.scale(loss).backward()

        # 累积到一定的步数default=8 才进行模型参数的更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 裁剪梯度,防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新scaler内部状态
            scaler.step(optimizer)
            scaler.update()
            # 清空梯度,准备进行下一次累积
            optimizer.zero_grad(set_to_none=True)
        
        # 进行日志记录,default=100,进行log
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
        
        # 进行Model保存,default=100,每100步进行预训练模型的保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 保存完之后,切换回train模式
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    # tokenizer是从本地加载的已经训练好的分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # MiniMindLM类(核心模块)构建一个标准的Decoder-Only Transformer结构(支持KV Cache加速)
    # MiniMindLM类在model.py当中(包含RMSNorm RoPE--旋转位置编码 文本生成 Top-P采样与Temperature 温度缩放)
    # MiniMindLM 是一个完整的、轻量级的、支持 KV Cache 加速和流式输出的微型大语言模型。
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 预训练数据加载器 dataset.py当中的PretrainDataset类
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 选择是否采用多卡训练,单卡则不涉及train_sampler分布式采样器
    train_loader = DataLoader(          # 训练加载器train_loader,有44160条训练数据,包含X Y Loss_Mask三部分内容
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    # 定义优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 是否进行分布式训练(多卡训练)
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    # train_loader中的训练数据有44160条,args.epochs = 1
    # 作者给出的理由-->若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
