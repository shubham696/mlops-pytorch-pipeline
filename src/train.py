"""Training entrypoint for the CIFAR-10 classifier.

Reads hyperparameters from ``configs/training_config.yaml`` (or a
mounted copy at ``/app/configs/training_config.yaml`` inside a
container), with ``DATA_DIR``/``CHECKPOINT_DIR`` environment variables
available to override the data and checkpoint locations for local
(non-container) runs. Logs one JSON line per epoch to stdout and saves
the best checkpoint (by validation loss), with early stopping.
"""

import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path() -> Path:
    """Locate the training config, preferring a container-mounted path."""
    container_path = Path("/app/configs/training_config.yaml")
    if container_path.exists():
        return container_path
    return Path("configs/training_config.yaml")


def resolve_data_dir(config: dict) -> str:
    """Resolve the dataset directory, allowing a DATA_DIR env override."""
    return os.environ.get("DATA_DIR", config["data"]["data_dir"])


def resolve_checkpoint_dir(config: dict) -> Path:
    """Resolve the checkpoint directory, allowing a CHECKPOINT_DIR env override."""
    return Path(os.environ.get("CHECKPOINT_DIR", config["output"]["checkpoint_dir"]))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch and return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run evaluation over a loader and return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def main() -> None:
    """Train the model per the config, logging JSON lines to stdout."""
    config = load_config(str(resolve_config_path()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    data_dir = resolve_data_dir(config)
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir, batch_size=config["training"]["batch_size"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = resolve_checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            print(
                json.dumps({"event": "checkpoint_saved", "path": str(save_path)}),
                flush=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    json.dumps({"event": "early_stopping", "epoch": epoch + 1}),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
