"""
AR (causal LM) baseline trainer for matched-protocol comparison with LLaDA SFT.

Strict alignment with src/training/train.py:
- Same: LoRA r=64 alpha=128 dropout=0.05; bf16; AdamW(beta1=0.9,beta2=0.95,wd=0.01);
  cosine LR warmup 5% peak 2e-4; 3 epochs; effective batch=128; max_seq_len=1024;
  seeds {42,123,456}; same 200k En<->Zh / En->De data; same prompt templates.
- Different: AR uses HF AutoModelForCausalLM with the model's built-in CE loss
  (next-token prediction, no masking schedule). LoRA targets the LLaMA-3 7-module
  set {q,k,v,o,gate,up,down}_proj which mirrors the 7-module set used in the
  LLaDA path {q,k,v,o,ff_proj,up_proj,ff_out} (attention + FFN).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure flush
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)


def parse_args():
    p = argparse.ArgumentParser(description="AR baseline trainer for MT")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to LLaMA-3-8B-Base (or other AR base)")
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--dev_data", type=str, default=None)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--lang_pair", type=str, default="en-zh",
                   choices=["en-zh", "en-de", "zh-en"])

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2,
                   help="Keep only N most recent checkpoints (checkpoint disk-space hygiene).")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)

    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="ladit")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="WandB entity (organization/username). If None, uses your wandb login default.")
    p.add_argument("--run_name", type=str, default="ar_enzh_seed42",
                   help="Human-readable run name; e.g. ar_<lang>_seed<N>.")

    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        return local_rank, world_size, rank
    return 0, 1, 0


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, scheduler_type="cosine"):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, eval_dataloader, device, dtype):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF returns mean loss; reweight by valid label count
        n = (labels != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(model, tokenizer, args, save_dir, is_ddp=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    m = model.module if is_ddp else model
    m.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved to {save_dir}", flush=True)


def prune_old_checkpoints(output_dir, keep_n):
    """Keep only `keep_n` most recent checkpoint-* dirs to save disk space."""
    if keep_n <= 0:
        return
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    for old in ckpts[:-keep_n]:
        if old.is_dir():
            import shutil
            shutil.rmtree(old)
            print(f"Pruned old checkpoint {old}", flush=True)


def main():
    args = parse_args()

    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            resume="allow",
        )
        print(f"WandB run ID: {wandb_run.id}")
        print(f"WandB run name: {wandb_run.name}")
        sys.stdout.flush()

    if is_main:
        print(f"Loading tokenizer from {args.model_path}...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"Loading AR model from {args.model_path}...", flush=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )

    # LLaMA-3 7-module set: attention + FFN, mirrors LLaDA's 7-module set
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if is_main:
            print("Gradient checkpointing enabled", flush=True)

    model.to(device)

    if is_main:
        print("Loading training data...", flush=True)
    from ladit.data.mt_ar_dataset import MTARDataset, collate_fn_ar

    train_dataset = MTARDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        lang_pair=args.lang_pair,
    )
    eval_dataset = None
    if args.dev_data:
        eval_dataset = MTARDataset(
            data_path=args.dev_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            lang_pair=args.lang_pair,
        )

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
    else:
        train_sampler = None

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn_ar,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_dl = None
    if eval_dataset:
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn_ar,
            num_workers=2,
            pin_memory=True,
        )

    # AdamW (beta2=0.95 to match LLaDA path)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_dl) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, args.lr_scheduler)

    if is_main:
        print(f"Steps per epoch: {steps_per_epoch}", flush=True)
        print(f"Total training steps: {total_steps}", flush=True)
        print(f"Warmup steps: {warmup_steps}", flush=True)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    model.train()
    global_step = 0
    best_eval_loss = float("inf")
    start_time = time.time()
    running_loss = 0.0
    running_tokens = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.bf16):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()
            n_target = (labels != -100).sum().item()
            running_loss += loss.item() * args.gradient_accumulation_steps
            running_tokens += n_target

            if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0 and is_main:
                    avg_loss = running_loss / args.logging_steps
                    elapsed = time.time() - start_time
                    sps = global_step / max(elapsed, 1)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"step:{global_step} - epoch:{epoch+1}/{args.num_epochs} - "
                        f"loss:{avg_loss:.4f} - lr:{lr:.2e} - "
                        f"tokens/step:{running_tokens/args.logging_steps:.0f} - "
                        f"steps/sec:{sps:.2f}",
                        flush=True,
                    )
                    if wandb_run:
                        wandb_run.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/tokens_per_step": running_tokens / args.logging_steps,
                        }, step=global_step)
                    running_loss = 0.0
                    running_tokens = 0

                if (global_step % args.save_steps == 0 and eval_dl is not None
                        and world_size == 1):
                    eloss = evaluate(model, eval_dl, device, dtype)
                    print(f"step:{global_step} - eval_loss:{eloss:.4f}", flush=True)
                    if wandb_run:
                        wandb_run.log({"eval/loss": eloss}, step=global_step)
                    if eloss < best_eval_loss:
                        best_eval_loss = eloss
                        save_checkpoint(model, tokenizer, args,
                                        output_dir / "best", world_size > 1)

                if global_step % args.save_steps == 0 and is_main:
                    save_checkpoint(model, tokenizer, args,
                                    output_dir / f"checkpoint-{global_step}",
                                    world_size > 1)
                    prune_old_checkpoints(output_dir, args.save_total_limit)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if is_main:
        save_checkpoint(model, tokenizer, args, output_dir / "final", world_size > 1)
        final_metrics = {"global_step": global_step, "best_eval_loss": best_eval_loss}
        if eval_dl is not None and world_size == 1:
            fl = evaluate(model, eval_dl, device, dtype)
            final_metrics["final_eval_loss"] = fl
            print(f"Final eval loss: {fl:.4f}", flush=True)
        with open(output_dir / "val_final.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"Training complete! Total steps: {global_step}", flush=True)

    if wandb_run:
        wandb_run.finish()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
