import argparse
import json
import random
from pathlib import Path

import pandas as pd


def positive_int(value: str):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def ratio(value: str):
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def infer_format(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(f"Cannot infer format from suffix: {path}")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_indices(size: int, val_size, val_ratio: float, seed: int, shuffle: bool):
    indices = list(range(size))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    if val_size is None:
        val_size = max(1, int(round(size * val_ratio)))
    if val_size >= size:
        raise ValueError(f"val size must be smaller than dataset size: val_size={val_size}, size={size}")

    val_indices = set(indices[:val_size])
    train_indices = [index for index in indices if index not in val_indices]
    val_indices = [index for index in indices if index in val_indices]
    return train_indices, val_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Split Search-P1 SFT jsonl/parquet data into train and validation sets.")
    parser.add_argument("--input", type=Path, required=True, help="Input SFT jsonl or parquet file.")
    parser.add_argument("--train-output", type=Path, required=True, help="Output train path.")
    parser.add_argument("--val-output", type=Path, required=True, help="Output validation path.")
    parser.add_argument("--val-size", type=positive_int, default=None, help="Fixed number of validation examples.")
    parser.add_argument("--val-ratio", type=ratio, default=0.1, help="Validation ratio when --val-size is unset.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep original order before splitting.")
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "parquet"),
        default="auto",
        help="Input/output format. auto infers from input suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_format = infer_format(args.input) if args.format == "auto" else args.format
    shuffle = not args.no_shuffle

    if data_format == "jsonl":
        rows = read_jsonl(args.input)
        train_indices, val_indices = split_indices(
            len(rows),
            val_size=args.val_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shuffle=shuffle,
        )
        write_jsonl(args.train_output, [rows[index] for index in train_indices])
        write_jsonl(args.val_output, [rows[index] for index in val_indices])
    else:
        frame = pd.read_parquet(args.input)
        train_indices, val_indices = split_indices(
            len(frame),
            val_size=args.val_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shuffle=shuffle,
        )
        args.train_output.parent.mkdir(parents=True, exist_ok=True)
        args.val_output.parent.mkdir(parents=True, exist_ok=True)
        frame.iloc[train_indices].reset_index(drop=True).to_parquet(args.train_output, index=False)
        frame.iloc[val_indices].reset_index(drop=True).to_parquet(args.val_output, index=False)

    print(f"Wrote {len(train_indices)} train records to {args.train_output}")
    print(f"Wrote {len(val_indices)} validation records to {args.val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
