import pathlib

import zarr


def main() -> None:
    root = pathlib.Path("data/hela_crops/jrc_hela-2")
    if not root.exists():
        print("hela2 crops root not found:", root)
        return
    crops = sorted(root.glob("crop*.zarr"))
    print("Found crops:", [p.name for p in crops])
    for p in crops:
        z = zarr.open(str(p), mode="r")
        raw = z["raw"]
        label = z["label"]
        print(f"{p.name}: raw {raw.shape}, label {label.shape}")


if __name__ == "__main__":
    main()

