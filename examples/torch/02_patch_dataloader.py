"""Example: sample patches with random or neighboring crops."""

import argparse

from worldtensor_torch import WorldTensorPatchDataset, require_torch, sparse_dict_collate


def main():
    torch = require_torch()
    parser = argparse.ArgumentParser(description="Torch patch-loading example for WorldTensor.")
    parser.add_argument("--sampling", choices=["random", "neighbor"], default="random")
    parser.add_argument("--output-format", choices=["dense", "dict"], default="dense")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patch-stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    variables = ["t2m_mean", "tp_sum", "tcc_mean", "elevation_mean"]
    years = [2015, 2016]

    dataset = WorldTensorPatchDataset(
        variables=variables,
        years=years,
        patch_size=args.patch_size,
        patches_per_year=128,
        patch_stride=args.patch_stride,
        sampling=args.sampling,
        output_format=args.output_format,
        min_valid_fraction=0.5,
        normalize="zscore",
        as_torch=True,
        return_mask=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.sampling == "random",
        num_workers=0,
        collate_fn=sparse_dict_collate if args.output_format == "dict" else None,
    )

    batch = next(iter(loader))
    print("Variables:", variables)
    print("sampling:", args.sampling)
    print("output_format:", args.output_format)
    if args.output_format == "dense":
        print("batch['x'] shape:", tuple(batch["x"].shape))
        print("batch['mask'] shape:", tuple(batch["mask"].shape))
        print("batch['year'] shape:", tuple(batch["year"].shape))
        print(
            "batch['valid_fraction'] min/max:",
            float(batch["valid_fraction"].min()),
            float(batch["valid_fraction"].max()),
        )
    else:
        print("number of sparse samples in batch:", len(batch["values"]))
        print("first sample coordinates shape:", tuple(batch["coordinates"][0].shape))
        print("first sample values shape:", tuple(batch["values"][0].shape))
        print("year tensor shape:", tuple(batch["year"].shape))


if __name__ == "__main__":
    main()
