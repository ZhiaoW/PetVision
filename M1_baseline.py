import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Subset, DataLoader

from PetV_data import build_full_dataset, load_split_indices

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

# =========================
# Basic Settings
# =========================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 80

DATA_ROOT = "PetV_data/data"
SPLIT_ROOT = "PetV_data/data/splits"

CHECKPOINT_DIR = "PetV_results/checkpoints"
FIGURE_DIR = "PetV_results/figures"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


print("Using device:", device)

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Dataset

full_dataset = build_full_dataset(DATA_ROOT, transform=transform)

# Load the same complete 80/10/10 split used by M2.
train_indices, val_indices, test_indices = load_split_indices(
    SPLIT_ROOT, len(full_dataset)
)

train_dataset = Subset(
    full_dataset,
    train_indices
)

val_dataset = Subset(
    full_dataset,
    val_indices
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# M1 Basedline CNN
class M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128,256,kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 37)


    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Layer 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        #Layer 4
        x = self.conv4(x)
        x = self.relu(x)
        #  Average Pooling
        x = self.avgpool(x)
        # flatten
        x = torch.flatten(x, 1)

        # Final classification
        y_pred = self.fc(x)

        return y_pred

model = M1()
model = model.to(device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)

#train model
def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for x, y in train_loader:
        #forward,loss,backward and optimize
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs,y)

        loss.backward()

        optimizer.step()

        #record loss and accuracy
        total_loss += (loss.item()* x.size(0))

        predictions = outputs.argmax(dim=1)
        correct += (predictions == y).sum().item()

        total += y.size(0)

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy

#validate model
def validate():

    model.eval()

    total_loss = 0
    top1_correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(outputs,y)

            total_loss += (loss.item()* x.size(0))

            # Top-1 Accuracy
            top1_predictions = outputs.argmax(dim=1)

            top1_correct += (top1_predictions == y).sum().item()

            # Top-3 Accuracy
            top3_predictions = outputs.topk(k=3, dim=1).indices

            top3_correct += (
                top3_predictions == y.unsqueeze(dim=1)
            ).any(dim=1).sum().item()

            total += y.size(0)


    loss = total_loss / total
    top1_accuracy = top1_correct / total
    top3_accuracy = top3_correct / total

    return loss, top1_accuracy, top3_accuracy



# 10. Train Model
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []
val_top3_accuracies = []

best_val_loss = float("inf")
best_epoch = 0

# Record accuracies at the epoch with the lowest validation loss.
best_val_acc = 0
best_val_top3_acc = 0

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, val_top3_acc = validate()


    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    val_top3_accuracies.append(val_top3_acc)


    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        best_val_acc = val_acc
        best_val_top3_acc = val_top3_acc

        torch.save(
            model.state_dict(),
            "PetV_results/checkpoints/m1_baseline_best.pth"
        )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Top-1 Acc: {val_acc:.4f} | "
        f"Val Top-3 Acc: {val_top3_acc:.4f}"
    )


print("\nBest epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Top-1 accuracy at best epoch: {best_val_acc:.4f}")
print(f"Top-3 accuracy at best epoch: {best_val_top3_acc:.4f}")

# Plot training history after all epochs have finished.
epochs = range(1, len(train_losses) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_losses, label="Train loss", color="tab:blue", linewidth=2)
ax.plot(epochs, val_losses, label="Validation loss", color="tab:orange", linewidth=2)
ax.set(title="M1 Baseline - Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
loss_figure_path = os.path.join(FIGURE_DIR, "m1_baseline_loss.png")
fig.savefig(loss_figure_path, dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_accuracies, label="Train Top-1", color="tab:blue", linewidth=2)
ax.plot(epochs, val_accuracies, label="Validation Top-1", color="tab:orange", linewidth=2)
ax.plot(
    epochs, val_top3_accuracies, label="Validation Top-3",
    color="tab:green", linestyle="--", linewidth=2
)
ax.set(title="M1 Baseline - Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
accuracy_figure_path = os.path.join(FIGURE_DIR, "m1_baseline_accuracy.png")
fig.savefig(accuracy_figure_path, dpi=200)
plt.close(fig)

print("Loss curve saved to:", loss_figure_path)
print("Accuracy curve saved to:", accuracy_figure_path)

# Load the checkpoint selected using the lowest validation loss.
model.load_state_dict(
    torch.load(
        "PetV_results/checkpoints/m1_baseline_best.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()
