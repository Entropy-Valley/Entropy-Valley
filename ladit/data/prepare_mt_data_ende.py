"""
Prepare WMT En→De parallel data for LaDiT MT experiments.
Downloads WMT19 En-De from HuggingFace, samples 200k pairs, saves as JSONL.
Also downloads WMT22 En→De test set via sacrebleu.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def download_wmt_ende(output_dir: str, num_train: int = 200000, seed: int = 42):
    """Download WMT En→De data and prepare train/dev splits."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WMT19 En-De training data...")

    all_pairs = []
    try:
        ds = load_dataset("wmt19", "de-en", split="train", trust_remote_code=True)
        # Shuffle entire dataset before filtering to avoid corpus-order bias
        ds = ds.shuffle(seed=seed)
        for item in ds:
            trans = item["translation"]
            en = trans.get("en", "")
            de = trans.get("de", "")
            if en and de and 5 <= len(en.split()) <= 200 and 5 <= len(de.split()) <= 200:
                all_pairs.append({"en": en.strip(), "de": de.strip()})
            if len(all_pairs) >= num_train + 5000:
                break  # Enough filtered pairs
        print(f"  wmt19/de-en: {len(all_pairs)} pairs after filtering")
    except Exception as e:
        print(f"  WMT19 En-De failed: {e}")
        sys.exit(1)

    print(f"Total pairs after filtering: {len(all_pairs)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_pairs)

    train_pairs = all_pairs[:num_train]
    dev_pairs = all_pairs[num_train : num_train + 2000]
    print(f"Train: {len(train_pairs)}, Dev: {len(dev_pairs)}")

    # Save as JSONL
    for split_name, split_data in [("train", train_pairs), ("dev", dev_pairs)]:
        out_path = output_dir / f"ende_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_data)} pairs to {out_path}")


def download_wmt22_ende_test(output_dir: str):
    """Download WMT22 En→De test set via sacrebleu."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WMT22 En→De test set via sacrebleu...")
    try:
        import sacrebleu
        src_lines = sacrebleu.get_source_file("wmt22", "en-de")
        ref_lines = sacrebleu.get_reference_files("wmt22", "en-de")

        # Read source
        with open(src_lines, "r", encoding="utf-8") as f:
            srcs = [l.strip() for l in f]

        # Read reference (first reference file)
        with open(ref_lines[0], "r", encoding="utf-8") as f:
            refs = [l.strip() for l in f]

        pairs = []
        for s, r in zip(srcs, refs):
            if s and r:
                pairs.append({"en": s, "de": r})

        out_path = output_dir / "wmt22_ende_test.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs)} WMT22 En→De test pairs to {out_path}")
    except Exception as e:
        print(f"sacrebleu download failed: {e}")
        print("Trying alternative: datasets library...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wmt22", "de-en", split="test", trust_remote_code=True)
            pairs = []
            for item in ds:
                trans = item.get("translation", item)
                en = trans.get("en", "")
                de = trans.get("de", "")
                if en and de:
                    pairs.append({"en": en.strip(), "de": de.strip()})
            out_path = output_dir / "wmt22_ende_test.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {len(pairs)} WMT22 En→De test pairs to {out_path}")
        except Exception as e2:
            print(f"Alternative also failed: {e2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="./data",
                        help="Where to write *_train.jsonl / *_dev.jsonl files.")
    parser.add_argument("--num_train", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    download_wmt_ende(args.output_dir, args.num_train, args.seed)
    if not args.skip_test:
        download_wmt22_ende_test(args.output_dir)
    print("En→De data preparation complete!")
