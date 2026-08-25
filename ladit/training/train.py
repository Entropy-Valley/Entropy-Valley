"""
Masked Diffusion SFT training for LaDiT.

Key design:
- LLaDA / Dream / DiffuLLaMA forward() return logits; loss is computed
  externally on masked positions only.
- Source tokens are NEVER masked (prefix conditioning).
- LoRA is the default; full SFT is available with --no_lora.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure flush
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)


def parse_args():
    parser = argparse.ArgumentParser(description="LaDiT: Masked Diffusion MT Training")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to LLaDA-8B-Base (or Dream / DiffuLLaMA checkpoint)")
    parser.add_argument("--backbone", type=str, default="auto",
                        choices=["auto", "llada", "dream", "diffullama"],
                        help="Which masked-diffusion backbone to load. "
                             "'auto' detects from the config.json at --model_path.")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA (default; pass --no_lora to disable)")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--noise_schedule", type=str, default="uniform",
                        choices=["uniform", "cosine"])
    parser.add_argument("--lang_pair", type=str, default="en-zh",
                        choices=["en-zh", "en-de", "zh-en"],
                        help="Language pair for prompt template")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_from", type=str, default=None)

    # WandB
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ladit")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def setup_distributed():
    """Setup distributed training."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        return local_rank, world_size, rank
    return 0, 1, 0


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, scheduler_type="cosine"):
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def compute_masked_diffusion_loss(logits, labels, logit_shift: int = 0):
    """Compute cross-entropy loss only on masked positions.

    Args:
        logits: (B, L, V) model output logits
        labels: (B, L) target token IDs, -100 for non-masked positions
        logit_shift: 0 = logits[i] predicts token at position i (LLaDA / Dream)
                     1 = logits[i-1] predicts token at position i (DiffuLLaMA
                         shift-by-1, matches its inference convention)
    Returns:
        loss: scalar tensor
        num_masked: number of masked tokens (for logging)
    """
    if logit_shift > 0:
        # Align logits so that logits_shifted[i] predicts the token at label[i].
        # For shift=1, we drop the last logit position and the first label
        # position (there's no left-context prediction for the very first
        # token — it is always a source/prompt token and gets -100 anyway).
        s = int(logit_shift)
        logits = logits[:, :-s, :]
        labels = labels[:, s:]

    mask = labels != -100
    num_masked = mask.sum().item()

    if num_masked == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0

    # Flatten for cross_entropy
    logits_flat = logits[mask]  # (num_masked, V)
    labels_flat = labels[mask]  # (num_masked,)

    loss = F.cross_entropy(logits_flat, labels_flat)
    return loss, num_masked


@torch.no_grad()
def evaluate(model, eval_dataloader, device, backbone_meta):
    """Run evaluation and return average loss."""
    from ladit.model.backbone import build_attention_mask

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    logit_shift = int(backbone_meta.get("logit_shift", 0))

    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        # For LLaDA / Dream we use the dataset-provided 2D mask; for
        # DiffuLLaMA we build a 4D bidirectional mask so the monkey-patched
        # forward keeps full attention.
        if backbone_meta.get("name") == "diffullama":
            attention_mask = build_attention_mask(
                input_ids, backbone_meta, dtype=next(model.parameters()).dtype
            )
        elif backbone_meta.get("name") == "dream":
            # Dream's modeling_dream.py passes attention_mask directly to
            # SDPA without expanding 2D -> 4D. With a (B, L) tensor SDPA
            # tries to broadcast it into the (B, H, L, L) QK shape via
            # leading-1 padding, which fails. Build a (B, 1, 1, L) bool
            # mask so the broadcast is unambiguous.
            attn_2d = batch["attention_mask"].to(device).bool()
            attention_mask = attn_2d[:, None, None, :]
        else:
            attention_mask = batch["attention_mask"].to(device)
            if attention_mask.dtype == torch.long:
                attention_mask = attention_mask.bool()
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss, num_masked = compute_masked_diffusion_loss(
            logits, labels, logit_shift=logit_shift
        )
        total_loss += loss.item() * num_masked
        total_tokens += num_masked

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def main():
    args = parse_args()

    # Setup
    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # WandB
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

    # Load tokenizer + model via unified backbone loader.
    # `load_backbone` handles:
    #   - LLaDA  : AutoModelForCausalLM, mask_token_id=126336, LLaDA LoRA targets
    #   - Dream  : AutoModel + trust_remote_code, mask_token_id from config, Qwen LoRA targets
    #   - DiffuLLaMA : AutoModelForCausalLM + monkey-patched bidirectional attn,
    #                  mask_token_id from tokenizer, Llama LoRA targets, shift-by-1 at decode
    if is_main:
        print(f"Loading backbone={args.backbone!r} from {args.model_path}...")
        sys.stdout.flush()
    from ladit.model.backbone import load_backbone
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model, tokenizer, backbone_meta = load_backbone(
        name=args.backbone,
        path=args.model_path,
        dtype=dtype,
    )
    mask_token_id = backbone_meta["mask_token_id"]
    lora_targets = backbone_meta["lora_target_modules"]
    template_family = backbone_meta["template_family"]
    if is_main:
        print(f"Backbone resolved: {backbone_meta['name']} "
              f"(mask_token_id={mask_token_id}, template={template_family}, "
              f"logit_shift={backbone_meta['logit_shift']})")
        print(f"  notes: {backbone_meta['notes']}")
        print(f"  LoRA targets: {lora_targets}")
        sys.stdout.flush()

    # Apply LoRA
    if args.use_lora:
        if is_main:
            print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, targets={lora_targets})...")
        from peft import LoraConfig, get_peft_model

        # Dream model lacks prepare_inputs_for_generation → task_type would fail.
        # LLaDA works with or without task_type; we omit it for both for consistency.
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()

    if args.gradient_checkpointing:
        try:
            # PEFT + gradient_checkpointing: need input_grad_fn path to propagate
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):
                def _make_inputs_require_grads(module, inp, out):
                    out.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(_make_inputs_require_grads)
            model.gradient_checkpointing_enable()
            if is_main:
                print("Gradient checkpointing enabled")
        except (ValueError, AttributeError) as e:
            if is_main:
                print(f"WARNING: Gradient checkpointing skipped ({e})")

    model.to(device)

    # Load data
    if is_main:
        print("Loading training data...")
    from ladit.data.mt_dataset import MTMaskedDiffusionDataset, collate_fn

    train_dataset = MTMaskedDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        mask_token_id=mask_token_id,
        noise_schedule=args.noise_schedule,
        lang_pair=args.lang_pair,
        template_family=template_family,
    )

    eval_dataset = None
    if args.dev_data:
        eval_dataset = MTMaskedDiffusionDataset(
            data_path=args.dev_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            mask_token_id=mask_token_id,
            noise_schedule="uniform",
            lang_pair=args.lang_pair,
            template_family=template_family,
        )

    # DataLoaders
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Calculate training steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, args.lr_scheduler)

    if is_main:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        sys.stdout.flush()

    # DDP wrapper
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    # Training loop
    model.train()
    global_step = 0
    best_eval_loss = float("inf")
    start_time = time.time()
    running_loss = 0.0
    running_tokens = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    if is_main:
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            # Backbone-specific attention mask: DiffuLLaMA needs a 4D
            # bidirectional mask; LLaDA and Dream use the dataset 2D mask.
            if backbone_meta.get("name") == "diffullama":
                from ladit.model.backbone import build_attention_mask
                attention_mask = build_attention_mask(
                    input_ids, backbone_meta,
                    dtype=next(model.parameters()).dtype,
                )
            elif backbone_meta.get("name") == "dream":
                # Dream's SDPA needs an unambiguous 4D mask; see eval()
                # comment above. Use the dataset's 2D padding mask
                # reshaped to (B, 1, 1, L) bool.
                attn_2d = batch["attention_mask"].to(device).bool()
                attention_mask = attn_2d[:, None, None, :]
            else:
                attention_mask = batch["attention_mask"].to(device)
                if attention_mask.dtype == torch.long:
                    attention_mask = attention_mask.bool()
            labels = batch["labels"].to(device)

            # Forward
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss, num_masked = compute_masked_diffusion_loss(
                    logits, labels,
                    logit_shift=int(backbone_meta.get("logit_shift", 0)),
                )
                loss = loss / args.gradient_accumulation_steps

            # Backward
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps
            running_tokens += num_masked

            # Gradient accumulation step
            if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0 and is_main:
                    avg_loss = running_loss / args.logging_steps
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    lr = scheduler.get_last_lr()[0]

                    print(
                        f"step:{global_step} - epoch:{epoch+1}/{args.num_epochs} - "
                        f"loss:{avg_loss:.4f} - lr:{lr:.2e} - "
                        f"tokens/step:{running_tokens/args.logging_steps:.0f} - "
                        f"steps/sec:{steps_per_sec:.2f}"
                    )
                    sys.stdout.flush()

                    if wandb_run:
                        wandb_run.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/tokens_per_step": running_tokens / args.logging_steps,
                        }, step=global_step)

                    running_loss = 0.0
                    running_tokens = 0

                # Evaluation — only in single-GPU mode to avoid DDP sync issues
                # For multi-GPU: do offline eval on checkpoints after training
                if (global_step % args.eval_steps == 0 and
                        eval_dataloader is not None and world_size == 1):
                    eval_loss = evaluate(model, eval_dataloader, device, backbone_meta)
                    print(f"step:{global_step} - eval_loss:{eval_loss:.4f}")
                    sys.stdout.flush()

                    if wandb_run:
                        wandb_run.log({"eval/loss": eval_loss}, step=global_step)

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(model, tokenizer, args,
                                        output_dir / "best", False)

                # Save checkpoint
                if global_step % args.save_steps == 0 and is_main:
                    save_checkpoint(model, tokenizer, args,
                                    output_dir / f"checkpoint-{global_step}",
                                    world_size > 1)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save final
    if is_main:
        save_checkpoint(model, tokenizer, args,
                        output_dir / "final", world_size > 1)

        # Write val_final.json (per CLAUDE.md requirement)
        final_metrics = {"global_step": global_step, "best_eval_loss": best_eval_loss}
        if eval_dataloader is not None and world_size == 1:
            final_eval_loss = evaluate(model, eval_dataloader, device, backbone_meta)
            final_metrics["final_eval_loss"] = final_eval_loss
            print(f"Final eval loss: {final_eval_loss:.4f}")
            sys.stdout.flush()

        with open(output_dir / "val_final.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        print(f"Training complete! Total steps: {global_step}")
        sys.stdout.flush()

    if wandb_run:
        wandb_run.finish()

    if world_size > 1:
        torch.distributed.destroy_process_group()


def save_checkpoint(model, tokenizer, args, save_dir, is_ddp=False):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    m = model.module if is_ddp else model

    if args.use_lora:
        # Save LoRA adapter only
        m.save_pretrained(save_dir)
    else:
        # Save full model
        m.save_pretrained(save_dir)

    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved to {save_dir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
