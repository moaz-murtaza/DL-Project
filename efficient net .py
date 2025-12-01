import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models, utils
from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt


"""
End-to-end FER2013 training script.

Features:
- Uses folder-structured FER2013 dataset under ./dataset/train and ./dataset/test
- Strong preprocessing & augmentations suitable for facial expression recognition
- Transfer learning with a modern pretrained backbone (EfficientNet_B0)
- Mixed precision + GPU utilization if available
- Cosine LR schedule, label smoothing, early stopping
- Best model checkpointing
- Training curves (loss/accuracy) saved as PNG
- Confusion matrix and classification report saved after training

To run:
    python baseline.py

Adjust hyperparameters in the Config dataclass below as needed.
"""


@dataclass
class Config:
    data_dir: str = "dataset"
    train_dir: str = "train"
    test_dir: str = "test"
    num_classes: int = 7
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 224  # EfficientNet/ResNet standard
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    save_dir: str = "runs/fer2013"
    model_name: str = "efficientnet_b0"
    seed: int = 42
    debug_one_batch: bool = False  # set True to visually inspect one augmented batch


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # Prefer an NVIDIA GeForce GPU if present, otherwise fall back to device 0.
        geforce_index = None
        try:
            num_devices = torch.cuda.device_count()
            for idx in range(num_devices):
                name = torch.cuda.get_device_name(idx)
                print(f"Found CUDA device {idx}: {name}")
                if "geforce" in name.lower():
                    geforce_index = idx
        except Exception:
            num_devices = 1

        if geforce_index is not None:
            selected_idx = geforce_index
            print(f"Selecting GeForce device index {selected_idx}")
        else:
            selected_idx = 0
            print(f"No GeForce name match found; defaulting to CUDA device {selected_idx}")

        torch.cuda.set_device(selected_idx)
        device = torch.device(f"cuda:{selected_idx}")
        return device

    print("CUDA not available, using CPU.")
    return torch.device("cpu")


def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Strong but reasonable augmentations for FER:
    - Random crops & flips
    - Color jitter (expressions are fairly robust)
    - Random rotations
    """
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),  # FER2013 is often grayscale
            transforms.Resize((cfg.image_size + 32, cfg.image_size + 32)),
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.02,
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"
            ),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, test_tf


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_tf, test_tf = build_transforms(cfg)

    train_path = os.path.join(cfg.data_dir, cfg.train_dir)
    test_path = os.path.join(cfg.data_dir, cfg.test_dir)

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_tf)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names


def debug_one_batch(
    cfg: Config, loader: DataLoader, class_names: List[str]
) -> None:
    """
    Quick sanity check: grab one batch, print shapes/labels, and save a grid image
    of augmented samples to disk.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)

    images, labels = next(iter(loader))
    print("Debug batch shape:", images.shape)  # [B, 3, H, W]
    print("Debug labels shape:", labels.shape)
    print("First 16 label indices:", labels[:16].tolist())
    print("Classes:", class_names)

    grid = utils.make_grid(images[:16], nrow=4, normalize=True, value_range=(0.0, 1.0))
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.tight_layout()
    debug_path = os.path.join(cfg.save_dir, "debug_batch.png")
    plt.savefig(debug_path)
    plt.close()
    print(f"Saved debug batch grid to {debug_path}")


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    """
    Use EfficientNet_B0 as a strong baseline.
    You can swap to resnet50, vit_b_16, etc., if desired.
    """
    if cfg.model_name.lower() == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, cfg.num_classes)
    elif cfg.model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg.num_classes)
    else:
        # Fallback to resnet18 if unknown
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg.num_classes)

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Any,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Train [{epoch + 1}/{total_epochs}]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if batch_idx == 0 and device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(
                f"[GPU] First batch on {images.device}, "
                f"memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB"
            )

        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}",
            acc=f"{(correct / max(1, total)):.4f}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Normalize
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fmt = ".2f"
    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm_normalized[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    # Save JSON
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Also save txt
    text_report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(text_report)


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders(cfg)
    cfg.num_classes = len(class_names)

    if cfg.debug_one_batch:
        debug_one_batch(cfg, train_loader, class_names)
        return

    model = build_model(cfg, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.mixed_precision and device.type == "cuda"
    )

    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(cfg.epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            cfg.epochs,
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            scaler,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] "
            f"Time: {epoch_time:.1f}s "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save progress history every epoch
        with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

        # Checkpointing best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = os.path.join(cfg.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "config": asdict(cfg),
                    "best_val_acc": best_val_acc,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path} (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best epoch: {best_epoch + 1}, Best Val Acc: {best_val_acc:.4f}"
                )
                break

    # Load best model for final evaluation
    ckpt_path = os.path.join(cfg.save_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch'] + 1} "
            f"with Val Acc: {checkpoint['best_val_acc']:.4f}"
        )
    else:
        print("Best model checkpoint not found, using last epoch weights.")

    # Final evaluation on validation/test set for confusion matrix, etc.
    val_loss, val_acc, y_true, y_pred = evaluate(
        model,
        val_loader,
        criterion,
        device,
        scaler,
    )
    print(f"Final Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plot_and_save_confusion_matrix(cm, class_names, cfg.save_dir)
    save_classification_report(y_true, y_pred, class_names, cfg.save_dir)

    # Save training curves at the end
    plot_training_curves(history, cfg.save_dir)


if __name__ == "__main__":
    main()


