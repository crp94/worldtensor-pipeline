"""Example: load full global tensors with a PyTorch DataLoader."""

from worldtensor_torch import WorldTensorYearDataset, require_torch


def main():
    torch = require_torch()

    variables = ["t2m_mean", "tp_sum", "tcc_mean", "elevation_mean"]
    years = [2014, 2015, 2016]

    dataset = WorldTensorYearDataset(
        variables=variables,
        years=years,
        normalize="zscore",
        as_torch=True,
        return_mask=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    print("Variables:", variables)
    print("batch['x'] shape:", tuple(batch["x"].shape))
    print("batch['mask'] shape:", tuple(batch["mask"].shape))
    print("batch['year']:", batch["year"].tolist())


if __name__ == "__main__":
    main()
