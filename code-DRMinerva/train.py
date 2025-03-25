# adapted from https://github.com/mlfoundations/open_flamingo

import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch
import os

from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from contextlib import suppress
import wandb

from utils import load_model_cpu


# apply weight decay only to params in the xattn layers
def get_grouped_params(params_to_optimize):
    params_with_wd, params_without_wd = [], []
    for n, p in params_to_optimize:
        if "gated_cross_attn" in n:
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress
    
def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

def train_one_epoch(args, model, epoch, tokenizer, optimizer, lr_scheduler, loader, device_id, wandb, path):

    # setup loaders
    num_batches_per_epoch = len(loader)
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision, cache_enabled=(not args.fsdp))  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, batch in tqdm(enumerate(loader), disable=args.rank != 0,  total=total_training_steps, initial=(epoch * num_batches_per_epoch),):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        if rag:
            images = batch[1].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            input_ids = batch[4]['input_ids'].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            attention_mask = batch[4]['attention_mask'].to(device_id[0], dtype=cast_dtype, non_blocking=True)
        elif static_context:
            images = batch[0].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            input_ids = batch[3]['input_ids'].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            attention_mask = batch[3]['attention_mask'].to(device_id[0], dtype=cast_dtype, non_blocking=True)
        else:
            images = batch[2].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            input_ids = batch[7]['input_ids'].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            attention_mask = batch[7]['attention_mask'].to(device_id[0], dtype=cast_dtype, non_blocking=True)
            
        #images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id[1])

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss = model(
                vision_x=images.squeeze(0),
                lang_x=input_ids.squeeze(0),
                attention_mask=attention_mask.squeeze(0),
                labels=labels.squeeze(0),
                #devices=device_id,
            )[0]

        divided_loss = loss / args.gradient_accumulation_steps
        (divided_loss * args.loss_multiplier).backward()
        #(divided_loss * args.loss_multiplier).backward(retain_graph=True)

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (model.lang_encoder.get_input_embeddings().weight.grad)
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (embed_grad * zero_mask)
            else:
                model.lang_encoder.get_input_embeddings().weight.grad = (embed_grad * zero_mask)

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )
                
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "samples_per_second": samples_per_second,
                        "samples_per_second_per_gpu": samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log({
                        "loss": loss.item(),
                        "global_step": global_step,},
                    commit=False,)
                
        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}")

        with open(f"{path}", "a") as f:
            f.write(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}\n")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict

def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, path):
        """
        Save training checkpoint with model, optimizer, and lr_scheduler state.
        """
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

        if args.rank == 0:
            if not (args.fsdp and not args.fsdp_use_orig_params):
                model_state = filter_state_dict_to_trainable(model, model_state)

            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optim_state,
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }

            print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
            torch.save(checkpoint_dict, f"{path}/checkpoint_{epoch}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_index', type=int, default=0)
    parser.add_argument('--n_exp', type=int, default=-1)
    parser.add_argument('--training_split', type=str, default='train_2')
    parser.add_argument('--emb_model', type=str, default='mistral')
    parser.add_argument('--llm_model', type=str, default='dr_minerva')
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--no-rag', dest='rag', action='store_false')
    parser.add_argument('--static-context', action='store_true')
    parser.add_argument('--no-static-context', dest='static-context', action='store_false')
    parser.set_defaults(rag=False, static_context=False)

    parser.add_argument('--inference', type=str, default='all')
    
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--loss_multiplier", type=float, default=1.0)
    
    parser.add_argument("--batch_size", default=1, type=int)
    
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--lr_scheduler", default="constant", type=str, help="constant, linear, or cosine",)
    parser.add_argument("--precision", choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],default="fp32",help="Floating point precision.",)
    parser.add_argument("--fsdp", default=False, action="store_true", help="Use FullyShardedDataParallel for distributed training.",)
    parser.add_argument("--fsdp_use_orig_params", default=False, action="store_true", help="Passed into the FSDP constructor. Enables param_groups and gradient masking for weight_decay. Does not work with OPT.",)
    parser.add_argument( "--freeze_lm_embeddings", action="store_true", help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DR_LLM")
    parser.add_argument("--wandb_entity", type=str,)
    parser.add_argument("--save_checkpoints_to_wandb", default=False,action="store_true", help="save checkpoints to wandb",)
    parser.add_argument("--run_name", type=str, default="DR_minerva_flamingo", help="used to name saving directory and wandb run",)
    parser.add_argument("--delete_previous_checkpoint", action="store_true", help="delete previous checkpoint when saving new checkpoint", default=True)
    
    args = parser.parse_args()
    starting_index = args.starting_index
    n_exp = args.n_exp
    training_split = args.training_split
    emb_model = args.emb_model
    llm_model = args.llm_model
    rag = args.rag
    static_context = args.static_context
    inference_label = args.inference

    if emb_model == 'mistral':
        emb_model_name = "Linq-AI-Research/Linq-Embed-Mistral"
        emb_model_str = 'linq-embed-mistral'
    else:
        emb_model == 'mistral'
        emb_model_name = "Linq-AI-Research/Linq-Embed-Mistral"
        emb_model_str = 'linq-embed-mistral'

    if llm_model == 'minerva':
        llm_model_name = "sapienzanlp/Minerva-3B-base-v1.0"
        llm_model_str = "Minerva-3B-base-v1.0"
    elif llm_model == 'dr_minerva':
        llm_model_name = "MedPix-2-0/dr_minerva"
        llm_model_str = "dr_minerva"
    else:
        llm_model == 'minerva'
        llm_model_name = "sapienzanlp/Minerva-3B-base-v1.0"
        llm_model_str = "Minerva-3B-base-v1.0"

    if inference_label not in ['modality', 'location', 'joint']:
        inference_label='modality'

    path = "MedPix-2-0/"

    clip_vision_encoder_path = "ViT-L-14"
    clip_vision_encoder_pretrained = "openai"
    lang_encoder_path = llm_model_name
    tokenizer_path = llm_model_name

    devices = ['cpu', 'cpu']

    # continue training from first epoch
    resume_from_checkpoint = True
    resume_from_epoch = 1

    if not resume_from_checkpoint:
        resume_from_epoch = 0
        checkpoint = None
    else:
        checkpoint_path = 'MedPix-2-0/DR_minerva_flamingo/checkpoint_1.pt'
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model, image_processor, tokenizer = load_model_cpu(clip_vision_encoder_path, clip_vision_encoder_pretrained, lang_encoder_path, tokenizer_path, checkpoint)
    tokenizer.padding_side = "left"

    model.to(dtype = get_cast_dtype(args.precision))
    
    # split model into gpus
    model.vision_encoder.to(devices[0])
    model.perceiver.to(devices[1])
    model.lang_encoder.to(devices[0])

    train_dataset = torch.load(f"{path}data_ft_flamingo/train_2.pt")

    # create dataloader    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    total_training_steps = ((len(train_dataset)) // (args.batch_size * args.world_size)) * args.num_epochs

    # put loss and optimizer following the train.py
    # Initialize optimizer
    params_to_optimize = model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        ))

    optimizer = torch.optim.AdamW(get_grouped_params(params_to_optimize), lr=args.learning_rate)

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps,)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps,)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    if resume_from_checkpoint:
        osd = checkpoint["optimizer_state_dict"]
        optimizer.load_state_dict(osd)
        print('optimizer loaded from checkpoint')
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        print('lr_scheduler loaded from checkpoint')

    model.train()
    os.makedirs(f"{path}code/{args.run_name}", exist_ok=True)
    with open(f"{path}code/{args.run_name}/train_log.txt", "w") as f:
        f.write(f"Start training\n")

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_one_epoch(args=args, model=model, epoch=epoch, tokenizer=tokenizer, optimizer=optimizer, lr_scheduler=lr_scheduler, loader=train_dataloader, device_id=devices, wandb=wandb, path=f"{path}code/{args.run_name}/train_log.txt")
        save_checkpoint(model, optimizer, lr_scheduler, epoch, args, path=f"{path}code/{args.run_name}")

    # save final checkpoint
    save_checkpoint(model, optimizer, lr_scheduler, epoch, args, path=f"{path}code/{args.run_name}")

    #cleanup()
    
    print("Training completed successfully (hope so!)")

