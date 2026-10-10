# PetVision

A pet breed classification project using PyTorch and the Oxford-IIIT Pet dataset.

## Models

- **M1:** A baseline convolutional neural network.
- **M2:** An improved model with data augmentation, batch normalization, and dropout.
- **M3:** 

## Project Files

- `PetV_data.py`: Downloads the dataset and creates reproducible training, validation, and test splits.
- `M1_baseline.py`: Trains the baseline model.
- `M2_improved.py`: Trains the improved model.
- `requirements.txt`: Lists the required Python packages.
- `.gitignore`: Excludes local datasets, environments, checkpoints, and temporary files.

## Getting Started

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the dataset and prepare the splits:

```bash
python PetV_data.py
```

Train either model:

```bash
python M1_baseline.py
python M2_improved.py
```

## Data and Outputs

The dataset is split into 80% training, 10% validation, and 10% test data using a fixed random seed.

Downloaded data is stored in `PetV_data/data/`. Model checkpoints and training figures are saved under `PetV_results/`.

Datasets, virtual environments, and model checkpoints are excluded from version control.
