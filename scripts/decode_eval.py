#!/usr/bin/env python3
"""
LaDiT decode+eval: Full decode + eval with multiple length prediction methods.

Decodes N sentences with each method, saves translations for COMET/BLEU evaluation.
Methods: oracle, ratio_0.8, entropy_valley, multi_candidate (5 EV-neighbour offsets)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ladit.decoding.translate import translate_single, predict_target_lengths
from ladit.decoding.length_adaptive import (
    entropy_valley_batch,
    multi_candidate_decode,
)
from ladit.data.mt_dataset import set_lang_pair, get_data_keys


def decode_with_method(
    model, tokenizer, sources, references, method_name, method_lengths,
    num_steps=32, schedule="med", temperature=0.0, device="cuda",
    show_progress=True,
):
    """Decode all sentences with pre-computed target lengths."""
    from tqdm import tqdm

    results = []
    iterator = enumerate(zip(sources, method_lengths))
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Decoding ({method_name})")

    for idx, (src, tgt_len) in iterator:
        result = translate_single(
            model, tokenizer, src, tgt_len,
            num_steps=num_steps,
            schedule_name=schedule,
            temperature=temperature,
            device=device,
        )
        result["method"] = method_name
        result["reference"] = references[idx] if references else None
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="LaDiT decode+eval: Length Method Decode + Eval")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--schedule", type=str, default="med")
    parser.add_argument("--multi_k", type=int, default=3,
                        help="Multi-candidate offset span; the script decodes "
                             "candidate canvas lengths {EV+d for d in range(-2, multi_k)} "
                             "(default 3 ⇒ 5 candidates {EV-2,EV-1,EV,EV+1,EV+2})")
    parser.add_argument("--methods", type=str,
                        default="oracle,ratio_0.8,entropy_valley",
                        help="Comma-separated methods to test")
    parser.add_argument("--candidate_ratios", type=str, default=None,
                        help="Comma-separated candidate ratios for entropy_valley "
                             "(e.g., '0.7,0.75,0.8,0.85,0.9')")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lang_pair", type=str, default="en-zh",
                        choices=["en-zh", "en-de", "zh-en"],
                        help="Language pair for prompt template and data keys")
    args = parser.parse_args()

    # Set language pair globally
    set_lang_pair(args.lang_pair)

    # Load model (auto-detect LLaDA vs Dream-7B from config)
    print(f"Loading model from {args.model_path}...")
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    is_dream = getattr(cfg, "model_type", "").lower() == "dream" or \
               "Dream" in (getattr(cfg, "architectures", [""]) or [""])[0]
    ModelCls = AutoModel if is_dream else AutoModelForCausalLM
    model = ModelCls.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()

    # Set the mask token id globally (LLaDA=126336, Dream=151666, auto from config)
    from ladit.decoding.translate import set_mask_token_id as set_translate_mask
    from ladit.decoding.length_adaptive import set_mask_token_id as set_lp_mask
    from ladit.data.mt_dataset import set_template_family
    mask_tid = getattr(cfg, "mask_token_id", None) or 126336
    set_translate_mask(mask_tid)
    set_lp_mask(mask_tid)
    template_family = "qwen" if is_dream else "plain"
    set_template_family(template_family)
    # Re-run set_lang_pair so the right template bank takes effect
    set_lang_pair(args.lang_pair)
    print(f"Mask token id set to {mask_tid} ({'Dream' if is_dream else 'LLaDA'}); prompt family: {template_family}")

    # Load data
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    data = data[:args.num_examples]

    src_key, tgt_key = get_data_keys()
    sources = [d[src_key] for d in data]
    references = [d[tgt_key] for d in data]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods.split(",")

    print(f"\n{'='*70}")
    print(f"LaDiT decode+eval: Length Method Decode + Eval")
    print(f"  Examples: {len(sources)}")
    print(f"  Methods: {methods}")
    print(f"  Steps: {args.num_steps}, Schedule: {args.schedule}")
    print(f"{'='*70}\n")

    # === Compute lengths for all methods first ===
    all_lengths = {}

    # Oracle
    if "oracle" in methods:
        print("Computing oracle lengths...")
        oracle_lengths = predict_target_lengths(
            model, tokenizer, sources, references,
            method="oracle", device=args.device,
        )
        all_lengths["oracle"] = oracle_lengths
        print(f"  Oracle: mean={np.mean(oracle_lengths):.1f}, "
              f"std={np.std(oracle_lengths):.1f}")

    # Dynamic ratio methods — support any ratio_X.XX in methods list
    for method_name in methods:
        if method_name.startswith("ratio_"):
            ratio_val = float(method_name.split("_", 1)[1])
            print(f"Computing {method_name} lengths (ratio={ratio_val})...")
            all_lengths[method_name] = predict_target_lengths(
                model, tokenizer, sources, references,
                method="ratio", ratio=ratio_val, device=args.device,
            )

    # Entropy Valley
    if "entropy_valley" in methods:
        print("Computing entropy_valley lengths...")
        t0 = time.time()
        if args.candidate_ratios:
            candidate_ratios = [float(x) for x in args.candidate_ratios.split(",")]
        else:
            candidate_ratios = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]
        print(f"  Candidate ratios: {candidate_ratios}")
        ev_lengths, _ = entropy_valley_batch(
            model, tokenizer, sources,
            candidate_ratios=candidate_ratios,
            device=args.device,
        )
        all_lengths["entropy_valley"] = ev_lengths
        print(f"  Done in {time.time()-t0:.1f}s")

    # Save length predictions
    length_file = output_dir / "length_predictions.json"
    length_data = {k: v for k, v in all_lengths.items()}
    with open(length_file, "w") as f:
        json.dump(length_data, f, indent=2)
    print(f"\nLength predictions saved to {length_file}")

    # === Decode with each method ===
    for method_name, lengths in all_lengths.items():
        if method_name == "multi_candidate":
            continue  # Handled separately

        print(f"\n{'─'*70}")
        print(f"Decoding with {method_name}...")
        t0 = time.time()

        results = decode_with_method(
            model, tokenizer, sources, references,
            method_name, lengths,
            num_steps=args.num_steps,
            schedule=args.schedule,
            device=args.device,
        )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/len(sources):.2f}s/sentence)")

        # Save results
        out_file = output_dir / f"translations_{method_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_file}")

        # Also save in format for COMET evaluation
        hyp_file = output_dir / f"hyp_{method_name}.txt"
        ref_file = output_dir / "ref.txt"
        src_file = output_dir / "src.txt"

        with open(hyp_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r["translation"] + "\n")
        if not ref_file.exists():
            with open(ref_file, "w", encoding="utf-8") as f:
                for ref in references:
                    f.write(ref + "\n")
            with open(src_file, "w", encoding="utf-8") as f:
                for src in sources:
                    f.write(src + "\n")

    # === Multi-candidate decode (if requested) ===
    if "multi_candidate" in methods and "entropy_valley" in all_lengths:
        from tqdm import tqdm
        print(f"\n{'─'*70}")
        print(f"Decoding with multi_candidate (offset span multi_k={args.multi_k}, candidate canvas lengths {{EV+d for d in range(-2, multi_k)}})...")

        ev_lengths = all_lengths["entropy_valley"]
        t0 = time.time()
        mc_results = []

        for idx, (src, ref) in tqdm(list(enumerate(zip(sources, references))),
                                     desc="Multi-candidate"):
            ev_len = ev_lengths[idx]
            # Generate K candidates around entropy_valley prediction
            candidates = sorted(set(
                max(1, ev_len + d) for d in range(-2, args.multi_k)
            ))

            result = multi_candidate_decode(
                model, tokenizer, src, candidates,
                num_steps=args.num_steps,
                schedule_name=args.schedule,
                device=args.device,
            )
            result["method"] = "multi_candidate"
            result["reference"] = ref
            mc_results.append(result)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/len(sources):.2f}s/sentence)")

        out_file = output_dir / "translations_multi_candidate.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(mc_results, f, ensure_ascii=False, indent=2)

        hyp_file = output_dir / "hyp_multi_candidate.txt"
        with open(hyp_file, "w", encoding="utf-8") as f:
            for r in mc_results:
                f.write(r["translation"] + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"LaDiT decode+eval DECODE COMPLETE")
    print(f"{'='*70}")
    print(f"  Output dir: {output_dir}")
    print(f"  Methods decoded: {list(all_lengths.keys())}")
    print(f"\n  Next: Run COMET evaluation:")
    print(f"    python -m ladit.evaluation.evaluate --eval_dir {output_dir}")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
