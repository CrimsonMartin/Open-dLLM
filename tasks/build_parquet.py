"""
Build tokenized parquet from JSONL. Chunked + multithreaded to cap RAM usage.

Usage:
    python tasks/build_parquet.py \
        --input /workspace/data/fineweb_5m.jsonl \
        --output /workspace/data/fineweb_5m.tokenized.parquet \
        --tokenizer /workspace/models/qwen2.5-3b \
        --seq_len 64 \
        --workers 8 \
        --chunk_size 500000
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize_chunk(lines: list, tokenizer_path: str, seq_len: int, text_keys: str) -> dict:
    """Tokenize a chunk of JSONL lines. Runs in a worker process."""
    from veomni.data.data_transform import process_pretrain_example
    from veomni.models.auto import build_tokenizer

    tokenizer = build_tokenizer(tokenizer_path)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})

    transform = partial(process_pretrain_example, tokenizer=tokenizer,
                        max_seq_len=seq_len, text_keys=text_keys)

    input_ids_list = []
    attention_mask_list = []

    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        items = transform(row)
        if items is None:
            continue
        for item in items:
            input_ids_list.append(item["input_ids"].tolist())
            attention_mask_list.append(item["attention_mask"].tolist())

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--text_keys", type=str, default="text")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Lines per worker chunk")
    args = parser.parse_args()

    print(f"Building parquet: {args.input} -> {args.output}")
    print(f"  Workers: {args.workers}, chunk_size: {args.chunk_size}, seq_len: {args.seq_len}")

    writer = None
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int32())),
    ])

    total_examples = 0
    total_lines = 0
    t0 = time.time()

    with open(args.input) as f:
        while True:
            # Read chunks for all workers
            worker_chunks = []
            for _ in range(args.workers):
                chunk = []
                for _ in range(args.chunk_size):
                    line = f.readline()
                    if not line:
                        break
                    chunk.append(line)
                if chunk:
                    worker_chunks.append(chunk)

            if not worker_chunks:
                break

            lines_this_batch = sum(len(c) for c in worker_chunks)
            total_lines += lines_this_batch

            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(tokenize_chunk, chunk,
                                    args.tokenizer, args.seq_len, args.text_keys)
                    for chunk in worker_chunks
                ]

                batch_input_ids = []
                batch_attention_mask = []
                for future in as_completed(futures):
                    result = future.result()
                    batch_input_ids.extend(result["input_ids"])
                    batch_attention_mask.extend(result["attention_mask"])

            # Write batch to parquet
            if batch_input_ids:
                table = pa.table({
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                }, schema=schema)

                if writer is None:
                    writer = pq.ParquetWriter(args.output, schema=schema)
                writer.write_table(table)
                total_examples += len(batch_input_ids)

            elapsed = time.time() - t0
            rate = total_lines / elapsed
            print(f"  {total_lines:,} lines -> {total_examples:,} examples "
                  f"({rate:.0f} lines/s, {elapsed:.0f}s elapsed)", flush=True)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    file_size = os.path.getsize(args.output) / 1e9
    print(f"\nDone! {total_examples:,} examples, {file_size:.1f}GB, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
