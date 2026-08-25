"""
Prepare WMT En→Zh parallel data for LaDiT MT experiments.
Downloads from HuggingFace datasets, samples 200k pairs, saves as JSONL.
"""
import argparse
import json
import os
import random
from pathlib import Path


def download_wmt_enzh(output_dir: str, num_train: int = 200000, seed: int = 42):
    """Download WMT En→Zh data and prepare train/dev/test splits."""
    from datasets import load_dataset, concatenate_datasets

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WMT datasets...")

    # Load multiple WMT sources for En→Zh
    datasets_to_load = [
        ("wmt19", "zh-en"),  # WMT19 has large En-Zh parallel data
    ]

    all_pairs = []
    for name, lang_pair in datasets_to_load:
        try:
            ds = load_dataset(name, lang_pair, split="train", trust_remote_code=True)
            for item in ds:
                trans = item["translation"]
                en = trans.get("en", "")
                zh = trans.get("zh", "")
                if en and zh and 5 <= len(en.split()) <= 200 and 2 <= len(zh) <= 600:
                    all_pairs.append({"en": en.strip(), "zh": zh.strip()})
            print(f"  {name}/{lang_pair}: {len(all_pairs)} pairs so far")
        except Exception as e:
            print(f"  Skipping {name}/{lang_pair}: {e}")

    print(f"Total pairs after filtering: {len(all_pairs)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_pairs)

    # Sample for training
    train_pairs = all_pairs[:num_train]
    dev_pairs = all_pairs[num_train : num_train + 2000]
    print(f"Train: {len(train_pairs)}, Dev: {len(dev_pairs)}")

    # Save as JSONL
    for split_name, split_data in [("train", train_pairs), ("dev", dev_pairs)]:
        out_path = output_dir / f"enzh_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_data)} pairs to {out_path}")


def download_flores_devtest(output_dir: str):
    """Download FLORES-200 devtest for evaluation."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLORES-200 devtest...")
    try:
        ds = load_dataset("openlanguagedata/flores_plus", split="devtest",
                          trust_remote_code=True)
        pairs = []
        for item in ds:
            en = item.get("eng_Latn", "")
            zh = item.get("zho_Hans", "")
            if en and zh:
                pairs.append({"en": en.strip(), "zh": zh.strip()})

        out_path = output_dir / "flores_enzh_devtest.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs)} FLORES devtest pairs to {out_path}")
    except Exception as e:
        print(f"FLORES download failed: {e}")
        print("Trying alternative FLORES source...")
        try:
            ds = load_dataset("facebook/flores", "eng_Latn-zho_Hans",
                              split="devtest", trust_remote_code=True)
            pairs = []
            for item in ds:
                en = item.get("sentence_eng_Latn", item.get("sentence", ""))
                zh = item.get("sentence_zho_Hans", "")
                if en and zh:
                    pairs.append({"en": en.strip(), "zh": zh.strip()})
            out_path = output_dir / "flores_enzh_devtest.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {len(pairs)} FLORES devtest pairs to {out_path}")
        except Exception as e2:
            print(f"Alternative FLORES also failed: {e2}")


def prepare_ifmt_dev(output_dir: str):
    """Prepare a small manual inspection set (200 examples from dev)."""
    output_dir = Path(output_dir)
    dev_path = output_dir / "enzh_dev.jsonl"
    if not dev_path.exists():
        print("Dev set not found, skip IF-MT dev preparation")
        return

    pairs = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    ifmt = pairs[:200]
    out_path = output_dir / "ifmt_dev_200.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ifmt:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(ifmt)} IF-MT dev examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="./data",
                        help="Where to write *_train.jsonl / *_dev.jsonl files.")
    parser.add_argument("--num_train", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_flores", action="store_true")
    args = parser.parse_args()

    download_wmt_enzh(args.output_dir, args.num_train, args.seed)
    if not args.skip_flores:
        download_flores_devtest(args.output_dir)
    prepare_ifmt_dev(args.output_dir)
    print("Data preparation complete!")
