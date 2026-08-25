#!/usr/bin/env python3
"""Compute COMET-22 for the cross-backbone evaluation pipeline.

For each (backbone, direction, seed) cell this script reads the matching
`translations_<method>.json` file under `--eval_dir`, computes COMET-22
(`Unbabel/wmt22-comet-da`) against the corresponding WMT22 test set, and
writes a single aggregated JSON to `--out`.

Run-name convention expected under `--eval_dir`:
    <BACKBONE>_seed<SEED>_<LP>/translations_<METHOD>.json
where <BACKBONE> in {dream, diffullama, llada},
      <LP>       in {enzh, zhen, ende},
      <METHOD>   in {oracle, ratio, entropy_valley}.

The default run list below covers the three masked-diffusion backbones x
three directions x three seeds reported in the paper's cross-backbone
appendix (Section: "Cross-Backbone Validation"). Override `--runs_json`
to compute a different subset.
"""
import argparse
import json
import os
from pathlib import Path


def load_test(lp_short, data_dir):
    if lp_short == "enzh":
        path = Path(data_dir) / "wmt22_enzh_test.jsonl"
        src, tgt = "en", "zh"
    elif lp_short == "zhen":
        path = Path(data_dir) / "wmt22_enzh_test.jsonl"
        src, tgt = "zh", "en"
    elif lp_short == "ende":
        path = Path(data_dir) / "wmt22_ende_test.jsonl"
        src, tgt = "en", "de"
    else:
        raise ValueError(lp_short)
    srcs, refs = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            srcs.append(d[src])
            refs.append(d[tgt])
    return srcs, refs


def run_comet(srcs, hyps, refs, comet_model, batch_size=64):
    triples = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = comet_model.predict(triples, batch_size=batch_size, gpus=1, progress_bar=False)
    return out.system_score


def default_run_list():
    """Cross-backbone runs reported in the paper.

    Each entry is (run_name, backbone, direction, seed) where run_name is
    the directory name under --eval_dir that holds translations_*.json.
    """
    runs = []
    for backbone in ["dream", "diffullama"]:
        for lp in ["enzh", "zhen", "ende"]:
            for seed in [42, 123, 456]:
                runs.append((f"{backbone}_seed{seed}_{lp}", backbone, lp, seed))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="eval_results",
                   help="Directory containing per-run subfolders with translations_*.json")
    p.add_argument("--data_dir", default="data",
                   help="Directory containing wmt22_<lp>_test.jsonl files")
    p.add_argument("--out", default="eval_results/cross_backbone_comet.json")
    p.add_argument("--comet_dir", default=None,
                   help="Optional local path to a wmt22-comet-da checkpoint directory; "
                        "if absent, falls back to HuggingFace Hub download (cached).")
    p.add_argument("--runs_json", default=None,
                   help="Optional JSON file overriding the default run list. Each entry "
                        "must be a 4-tuple [run_name, backbone, direction, seed].")
    args = p.parse_args()

    print(f"[comet] Loading COMET-22 ...", flush=True)
    from comet import load_from_checkpoint
    ckpt = None
    if args.comet_dir:
        candidate = Path(args.comet_dir) / "checkpoints" / "model.ckpt"
        if candidate.exists():
            ckpt = candidate
    if ckpt is None:
        from comet import download_model
        print(f"[comet] No local checkpoint provided; downloading from HF Hub ...", flush=True)
        ckpt = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(str(ckpt))
    print(f"[comet] Model loaded.", flush=True)

    if args.runs_json:
        runs = [tuple(r) for r in json.load(open(args.runs_json))]
    else:
        runs = default_run_list()
    methods = ["oracle", "ratio", "entropy_valley"]

    cache = {}
    def get_test(lp):
        if lp not in cache:
            cache[lp] = load_test(lp, args.data_dir)
        return cache[lp]

    results = []
    for run_name, bb, lp, seed in runs:
        run_dir = Path(args.eval_dir) / run_name
        if not run_dir.is_dir():
            print(f"[skip] {run_name}: dir not found", flush=True)
            continue
        srcs, refs = get_test(lp)
        row = {"run": run_name, "backbone": bb, "direction": lp, "seed": seed}
        for method in methods:
            f = run_dir / f"translations_{method}.json"
            if not f.exists():
                print(f"[skip] {run_name}/{method}: file missing", flush=True)
                continue
            data = json.load(open(f))
            hyps = [d.get("translation", d.get("hypothesis", "")) for d in data]
            n = min(len(hyps), len(srcs), len(refs))
            score = run_comet(srcs[:n], hyps[:n], refs[:n], model)
            row[f"comet_{method}"] = score
            row[f"n_{method}"] = n
            print(f"[done] {run_name} {method}: COMET={score:.4f} N={n}", flush=True)
        results.append(row)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fout:
            json.dump({"runs": results}, fout, indent=2)
    print(f"[final] Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
