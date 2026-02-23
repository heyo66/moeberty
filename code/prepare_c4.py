
"""
Prepare and tokenize the C4 dataset for MLM training.

This script produces a Hugging Face dataset saved with `save_to_disk()`
containing a single column: `input_ids` (fixed-length sequences).

Example:
  python code/prepare_c4.py \
    --output ./.cache/tokenized/ \
    --max-seq-len 128 \
    --tokenizer answerdotai/ModernBERT-base \
    --dataset allenai/c4 \
    --config en \
    --split train \
"""

import argparse
import os
import sys
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output directory for tokenized dataset.")
    parser.add_argument("--dataset", default="allenai/c4", help="HF dataset name.")
    parser.add_argument("--config", default="en", help="HF dataset config name.")
    parser.add_argument("--split", default="train", help="HF split (supports slicing, e.g. 'train[:1%]').")
    parser.add_argument("--text-field", default="text", help="Text field name in the dataset.")
    parser.add_argument("--tokenizer", default="answerdotai/ModernBERT-base", help="Tokenizer ID.")
    parser.add_argument("--vocab-size", type=int, default=50368, help="Expected tokenizer vocab size.")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Sequence length for packing.")
    parser.add_argument("--add-eos", action="store_true", help="Append EOS token between documents.")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of processes for tokenization.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Map batch size for tokenization.")
    parser.add_argument("--cache-dir", default="", help="HF cache dir (optional).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = os.path.abspath(args.output)
    if os.path.isdir(output_dir):
        if not args.overwrite:
            print(f"Output already exists: {output_dir}")
            print("Use --overwrite to replace it.")
            return 2
        # remove existing dataset directory
        # Only delete if explicitly requested.
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != args.vocab_size:
        print(
            f"Warning: tokenizer vocab size {actual_vocab_size} != expected {args.vocab_size}",
            file=sys.stderr,
        )
    eos_id = tokenizer.eos_token_id if args.add_eos else None
    seq_len = args.max_seq_len

    ds_kwargs = {"path": args.dataset, "split": args.split}
    if args.config:
        ds_kwargs["name"] = args.config
    if args.cache_dir:
        ds_kwargs["cache_dir"] = args.cache_dir
    dataset = load_dataset(**ds_kwargs)

    if args.text_field not in dataset.column_names:
        print(f"Text field '{args.text_field}' not found in dataset columns: {dataset.column_names}")
        return 3

    def tokenize_and_chunk(batch) -> dict:
        tokenized = tokenizer(
            batch[args.text_field],
            truncation=True,
            max_length=seq_len,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids: List[List[int]] = []
        buffer: List[int] = []
        for ids in tokenized["input_ids"]:
            if eos_id is not None:
                ids = ids + [eos_id]
            buffer.extend(ids)
            while len(buffer) >= seq_len:
                input_ids.append(buffer[:seq_len])
                buffer = buffer[seq_len:]
        return {"input_ids": input_ids}

    map_kwargs = {
        "batched": True,
        "batch_size": args.batch_size,
        "remove_columns": dataset.column_names,
        "desc": "Tokenizing and chunking for MLM",
    }
    if args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    dataset = dataset.map(tokenize_and_chunk, **map_kwargs)
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    print(f"Saved tokenized dataset to: {output_dir}")
    print(f"Total sequences: {len(dataset)} (seq_len={seq_len})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
