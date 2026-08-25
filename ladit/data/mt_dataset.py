"""
PyTorch Dataset for Masked Diffusion MT training.

Training paradigm:
- Source tokens are NEVER masked (prefix conditioning)
- Target tokens are randomly masked with probability t ~ U[0,1]
- Loss is computed only on masked target positions
"""
import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATES = {
    "en-zh": ("Translate English to Chinese.\n\nEnglish: {source}\nChinese: {target}",
              "Translate English to Chinese.\n\nEnglish: {source}\nChinese: "),
    "en-de": ("Translate English to German.\n\nEnglish: {source}\nGerman: {target}",
              "Translate English to German.\n\nEnglish: {source}\nGerman: "),
    "zh-en": ("Translate Chinese to English.\n\nChinese: {source}\nEnglish: {target}",
              "Translate Chinese to English.\n\nChinese: {source}\nEnglish: "),
}
# Qwen chat-style templates for Dream-v0-Instruct-7B (Qwen-tokenized chat-finetuned)
PROMPT_TEMPLATES_QWEN = {
    "en-zh": (
        "<|im_start|>user\nTranslate the following English sentence into Chinese.\n\n{source}<|im_end|>\n<|im_start|>assistant\n{target}",
        "<|im_start|>user\nTranslate the following English sentence into Chinese.\n\n{source}<|im_end|>\n<|im_start|>assistant\n",
    ),
    "en-de": (
        "<|im_start|>user\nTranslate the following English sentence into German.\n\n{source}<|im_end|>\n<|im_start|>assistant\n{target}",
        "<|im_start|>user\nTranslate the following English sentence into German.\n\n{source}<|im_end|>\n<|im_start|>assistant\n",
    ),
    "zh-en": (
        "<|im_start|>user\nTranslate the following Chinese sentence into English.\n\n{source}<|im_end|>\n<|im_start|>assistant\n{target}",
        "<|im_start|>user\nTranslate the following Chinese sentence into English.\n\n{source}<|im_end|>\n<|im_start|>assistant\n",
    ),
}
DATA_KEYS = {
    "en-zh": ("en", "zh"),
    "en-de": ("en", "de"),
    "zh-en": ("zh", "en"),
}

# Backwards-compatible defaults
PROMPT_TEMPLATE = PROMPT_TEMPLATES["en-zh"][0]
PROMPT_PREFIX = PROMPT_TEMPLATES["en-zh"][1]

# Module-level state (mutable for runtime switching)
_LANG_PAIR = ["en-zh"]
_TEMPLATE_FAMILY = ["plain"]  # "plain" (LLaDA) or "qwen" (Dream-Instruct)


def get_template_bank():
    """Return the active template dict based on the current backbone family."""
    return PROMPT_TEMPLATES_QWEN if _TEMPLATE_FAMILY[0] == "qwen" else PROMPT_TEMPLATES


def set_template_family(family: str):
    """Set the active prompt-template family ('plain' for LLaDA or 'qwen' for Dream-Instruct)."""
    global PROMPT_TEMPLATE, PROMPT_PREFIX
    assert family in {"plain", "qwen"}, f"Unknown family: {family}"
    _TEMPLATE_FAMILY[0] = family
    bank = get_template_bank()
    PROMPT_TEMPLATE = bank[_LANG_PAIR[0]][0]
    PROMPT_PREFIX = bank[_LANG_PAIR[0]][1]


def set_lang_pair(lang_pair: str):
    """Set the active language pair for all decoding modules."""
    global PROMPT_TEMPLATE, PROMPT_PREFIX
    _LANG_PAIR[0] = lang_pair
    bank = get_template_bank()
    PROMPT_TEMPLATE = bank[lang_pair][0]
    PROMPT_PREFIX = bank[lang_pair][1]


def get_prompt_prefix() -> str:
    """Get the prompt prefix for the current language pair and family."""
    return get_template_bank()[_LANG_PAIR[0]][1]


def get_data_keys() -> tuple:
    """Get (src_key, tgt_key) for the current language pair."""
    return DATA_KEYS[_LANG_PAIR[0]]


class MTMaskedDiffusionDataset(Dataset):
    """Dataset for masked diffusion MT training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        mask_token_id: int = 126336,
        noise_schedule: str = "uniform",  # "uniform" or "cosine"
        lang_pair: str = "en-zh",  # "en-zh", "en-de", "zh-en"
        template_family: str = "plain",  # "plain" (LLaDA) or "qwen" (Dream-Instruct)
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id
        self.noise_schedule = noise_schedule
        self.lang_pair = lang_pair
        self.template_family = template_family

        bank = PROMPT_TEMPLATES_QWEN if template_family == "qwen" else PROMPT_TEMPLATES
        if lang_pair not in bank:
            raise ValueError(f"Unsupported lang_pair: {lang_pair} in family {template_family}. "
                             f"Supported: {list(bank.keys())}")
        self.prompt_template, self.prompt_prefix = bank[lang_pair]
        self.src_key, self.tgt_key = DATA_KEYS[lang_pair]

        # Load data
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

        # Pre-tokenize prompts and targets
        self._tokenize_all()

    def _tokenize_all(self):
        """Pre-tokenize all examples to find prompt/target boundaries."""
        self.examples = []
        skipped = 0

        for item in self.data:
            src = item[self.src_key]
            tgt = item[self.tgt_key]

            # Tokenize prompt (source side)
            prompt = self.prompt_prefix.format(source=src)
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

            # Tokenize target
            target_ids = self.tokenizer.encode(tgt, add_special_tokens=False)

            # Check total length
            total_len = len(prompt_ids) + len(target_ids)
            if total_len > self.max_seq_len - 1:  # leave room for EOS
                # Truncate target if needed
                max_target = self.max_seq_len - 1 - len(prompt_ids)
                if max_target < 5:
                    skipped += 1
                    continue
                target_ids = target_ids[:max_target]

            # Add EOS to target
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                target_ids = target_ids + [eos_id]

            self.examples.append({
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "prompt_len": len(prompt_ids),
                "target_len": len(target_ids),
            })

        if skipped > 0:
            print(f"Skipped {skipped} examples (too long)")
        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def _sample_timestep(self):
        """Sample masking ratio (timestep) from noise schedule.

        Samples from [eps, 1] to ensure at least some tokens are masked.
        """
        eps = 1e-3
        if self.noise_schedule == "uniform":
            return eps + (1 - eps) * random.random()
        elif self.noise_schedule == "cosine":
            u = random.random()
            return max(eps, 1 - math.cos(u * math.pi / 2))
        else:
            return eps + (1 - eps) * random.random()

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = ex["prompt_ids"]
        target_ids = ex["target_ids"]
        prompt_len = ex["prompt_len"]
        target_len = ex["target_len"]
        total_len = prompt_len + target_len

        # Build input_ids: [prompt] + [target]
        clean_ids = prompt_ids + target_ids

        # Sample masking ratio
        t = self._sample_timestep()

        # Create masked version: mask target tokens with probability t
        input_ids = list(clean_ids)
        mask_flags = [False] * total_len  # True = this position is masked

        for i in range(prompt_len, total_len):
            if random.random() < t:
                input_ids[i] = self.mask_token_id
                mask_flags[i] = True

        # Labels: original token IDs (only compute loss on masked positions)
        labels = [-100] * total_len  # -100 = ignore in cross_entropy
        for i in range(prompt_len, total_len):
            if mask_flags[i]:
                labels[i] = clean_ids[i]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - total_len
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * total_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_len": prompt_len,
            "target_len": target_len,
            "timestep": t,
        }


def collate_fn(batch):
    """Custom collate that handles variable-length sequences."""
    # Find max length in batch for efficient padding
    max_len = max(item["prompt_len"] + item["target_len"] for item in batch)
    max_len = min(max_len, batch[0]["input_ids"].size(0))

    result = {
        "input_ids": torch.stack([item["input_ids"][:max_len] for item in batch]),
        "labels": torch.stack([item["labels"][:max_len] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"][:max_len] for item in batch]),
    }
    return result
