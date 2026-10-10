import os

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import ConcatDataset, Subset, DataLoader


SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 32

DATA_ROOT = "PetV_data/data"
SPLIT_ROOT = "PetV_data/data/splits"


def build_full_dataset(root, transform=None, download=False):
    return ConcatDataset([
        OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            download=download,
            transform=transform
        )
        for split in ("trainval", "test")
    ])


def validate_split_indices(n_samples, train_indices, val_indices, test_indices):
    n_val = int(np.ceil(n_samples * VAL_RATIO))
    n_test = int(np.ceil(n_samples * TEST_RATIO))
    expected_sizes = (n_samples - n_val - n_test, n_val, n_test)
    splits = (train_indices, val_indices, test_indices)

    for indices, expected_size in zip(splits, expected_sizes):
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != expected_size
        ):
            raise ValueError(
                "Invalid 80/10/10 split. Run PetV_data.py to regenerate the splits."
            )


def create_split_indices(labels):
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    n_val = int(np.ceil(len(labels) * VAL_RATIO))
    n_test = int(np.ceil(len(labels) * TEST_RATIO))

    # Reserve 20% first, then divide it equally into validation and test.
    train_indices, heldout_indices = train_test_split(
        indices,
        test_size=n_val + n_test,
        random_state=SEED,
        stratify=labels
    )
    val_indices, test_indices = train_test_split(
        heldout_indices,
        test_size=n_test,
        random_state=SEED,
        stratify=labels[heldout_indices]
    )

    validate_split_indices(len(labels), train_indices, val_indices, test_indices)
    return train_indices, val_indices, test_indices


def load_split_indices(split_root, n_samples):
    try:
        splits = tuple(
            np.load(os.path.join(split_root, f"{name}_indices.npy"), allow_pickle=False)
            for name in ("train", "val", "test")
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "The complete 80/10/10 split is missing. Run PetV_data.py first."
        ) from exc

    validate_split_indices(n_samples, *splits)
    return splits


def main():
    full_dataset = build_full_dataset(DATA_ROOT, download=True)
    labels = np.array([label for _, label in full_dataset])
    train_indices, val_indices, test_indices = create_split_indices(labels)

    os.makedirs(SPLIT_ROOT, exist_ok=True)
    for name, indices in zip(
        ("train", "val", "test"),
        (train_indices, val_indices, test_indices)
    ):
        np.save(os.path.join(SPLIT_ROOT, f"{name}_indices.npy"), indices)
        print(f"{name.capitalize()}: {len(indices)} ({len(indices) / len(labels):.2%})")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = build_full_dataset(DATA_ROOT, transform=transform)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
