import argparse

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_parquet(args.parquet_file)
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    available = 0
    step_counts = []
    examples = []

    for index in range(total):
        row = dataset[index]
        ground_truth = row.get("reward_model", {}).get("ground_truth", {})
        reference_steps = ground_truth.get("reference_steps", [])
        if reference_steps:
            available += 1
            step_counts.append(len(reference_steps))
            if len(examples) < 3:
                examples.append((index, row.get("data_source"), reference_steps))

    coverage = available / total if total else 0.0
    mean_steps = sum(step_counts) / len(step_counts) if step_counts else 0.0
    print(f"rows={total}")
    print(f"reference_available={available}")
    print(f"reference_available_ratio={coverage:.6f}")
    print(f"mean_reference_steps={mean_steps:.6f}")
    if examples:
        print("examples:")
        for index, data_source, reference_steps in examples:
            print(f"- index={index} data_source={data_source} steps={reference_steps}")


if __name__ == "__main__":
    main()
