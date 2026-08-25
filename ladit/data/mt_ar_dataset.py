"""
PyTorch Dataset for AR (next-token-prediction) MT training.

Mirrors ladit/data/mt_dataset.py for protocol alignment with LLaDA SFT, but:
- No masking schedule (AR uses standard next-token loss)
- Labels: prompt tokens -> -100, target tokens -> shifted next-token IDs
  (HF AutoModelForCausalLM does the right-shift internally when labels= is passed)
- Same prompt templates as the LLaDA path (plain family) so the matched
  comparison is on backbone family alone.
"""
import json
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
DATA_KEYS = {
    "en-zh": ("en", "zh"),
    "en-de": ("en", "de"),
    "zh-en": ("zh", "en"),
}


class MTARDataset(Dataset):
    """Standard AR (causal LM) dataset for MT, with prompt-loss masking."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 1024,
        lang_pair: str = "en-zh",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lang_pair = lang_pair

        if lang_pair not in PROMPT_TEMPLATES:
            raise ValueError(f"Unsupported lang_pair: {lang_pair}")
        _, self.prompt_prefix = PROMPT_TEMPLATES[lang_pair]
        self.src_key, self.tgt_key = DATA_KEYS[lang_pair]

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

        self._tokenize_all()

    def _tokenize_all(self):
        self.examples = []
        skipped = 0
        for item in self.data:
            src = item[self.src_key]
            tgt = item[self.tgt_key]

            prompt = self.prompt_prefix.format(source=src)
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(tgt, add_special_tokens=False)

            total_len = len(prompt_ids) + len(target_ids)
            if total_len > self.max_seq_len - 1:
                max_target = self.max_seq_len - 1 - len(prompt_ids)
                if max_target < 5:
                    skipped += 1
                    continue
                target_ids = target_ids[:max_target]

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

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = ex["prompt_ids"]
        target_ids = ex["target_ids"]
        prompt_len = ex["prompt_len"]
        target_len = ex["target_len"]
        total_len = prompt_len + target_len

        # input = prompt + target (HF CausalLM shifts internally)
        input_ids = prompt_ids + target_ids

        # labels: -100 on prompt, original IDs on target.
        # HF computes loss on tokens predicted at positions where labels[i] != -100.
        # The internal shift (loss on token i predicted from positions <i) means
        # labels[i] should be the *target* token at position i; so labels mirror input_ids
        # but with prompt positions ignored.
        labels = [-100] * prompt_len + target_ids

        # Pad
        pad_len = self.max_seq_len - total_len
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * total_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_len": prompt_len,
            "target_len": target_len,
        }


def collate_fn_ar(batch):
    max_len = max(item["prompt_len"] + item["target_len"] for item in batch)
    max_len = min(max_len, batch[0]["input_ids"].size(0))
    return {
        "input_ids": torch.stack([item["input_ids"][:max_len] for item in batch]),
        "labels": torch.stack([item["labels"][:max_len] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"][:max_len] for item in batch]),
    }
