"""
AR baseline decoding for WMT22 MT eval.

LLaMA-3-8B-Base + LoRA adapter, beam search with native EOS termination.
Aligned with training prompt templates from ladit/data/mt_ar_dataset.py.

Output: translations.json (list of {source, translation, reference, lang_pair}) +
hyp.txt (one line per sentence) compatible with ladit/evaluation/evaluate.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Inference-time prompt prefixes (must match training: ladit/data/mt_ar_dataset.py)
PROMPT_PREFIXES = {
    "en-zh": "Translate English to Chinese.\n\nEnglish: {source}\nChinese: ",
    "zh-en": "Translate Chinese to English.\n\nChinese: {source}\nEnglish: ",
    "en-de": "Translate English to German.\n\nEnglish: {source}\nGerman: ",
}

TARGET_LANG = {
    "en-zh": "zh",
    "zh-en": "en",
    "en-de": "de",
}


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--src_file", required=True)
    ap.add_argument("--ref_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lang_pair", required=True, choices=list(PROMPT_PREFIXES.keys()))
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_examples", type=int, default=-1, help="-1 = all")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = read_lines(args.src_file)
    references = read_lines(args.ref_file)
    assert len(sources) == len(references), \
        f"src/ref length mismatch: {len(sources)} vs {len(references)}"
    if args.max_examples > 0:
        sources = sources[: args.max_examples]
        references = references[: args.max_examples]
    n = len(sources)
    print(f"[decode_ar] lang_pair={args.lang_pair} n={n} beams={args.num_beams}", flush=True)

    # Load tokenizer + base model + LoRA
    print(f"[decode_ar] Loading tokenizer from {args.base_model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LLaMA tokenizer needs left padding for batched generation
    tok.padding_side = "left"

    print(f"[decode_ar] Loading base model (bf16, device_map=cuda)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"[decode_ar] Loading LoRA adapter from {args.lora_path}", flush=True)
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    prefix_template = PROMPT_PREFIXES[args.lang_pair]

    translations = []
    hyps = []
    t0 = time.time()
    bs = args.batch_size

    with torch.inference_mode():
        for batch_start in range(0, n, bs):
            batch_end = min(batch_start + bs, n)
            batch_src = sources[batch_start:batch_end]
            batch_ref = references[batch_start:batch_end]
            prompts = [prefix_template.format(source=s) for s in batch_src]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[1]

            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                early_stopping=True,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

            # Strip prompt tokens (left-padded → suffix is generation)
            gen_only = out[:, input_len:]
            decoded = tok.batch_decode(gen_only, skip_special_tokens=True)

            for src_text, ref_text, hyp_text in zip(batch_src, batch_ref, decoded):
                # Truncate at first newline (model often adds an "English: ..." continuation)
                hyp_clean = hyp_text.split("\n")[0].strip()
                translations.append({
                    "source": src_text,
                    "translation": hyp_clean,
                    "reference": ref_text,
                    "lang_pair": args.lang_pair,
                    "schedule": "ar_beam4",
                    "num_steps": -1,
                    "length_method": "ar_eos",
                })
                hyps.append(hyp_clean)

            if (batch_end % 50 == 0) or (batch_end == n):
                elapsed = time.time() - t0
                rate = batch_end / max(elapsed, 1e-6)
                eta_min = (n - batch_end) / max(rate, 1e-6) / 60.0
                print(f"[decode_ar] {batch_end}/{n} ({rate:.2f} sent/s, eta {eta_min:.1f} min)",
                      flush=True)

    # Save outputs
    trans_path = out_dir / "translations.json"
    hyp_path = out_dir / "hyp.txt"
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    with open(hyp_path, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")

    print(f"[decode_ar] DONE n={n} elapsed={time.time()-t0:.1f}s "
          f"-> {trans_path} + {hyp_path}", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
