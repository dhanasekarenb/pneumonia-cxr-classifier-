import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "chest_xray_dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
NUM_WORKERS = 4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_densenet121(device):
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 1)  # Single logit for binary classification
    )
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_f = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels_f)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs)
        all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    preds = (all_probs >= threshold).astype(int)

    return {
        "loss": running_loss / len(loader.dataset),
        "roc_auc": roc_auc_score(all_targets, all_probs),
        "f1": f1_score(all_targets, preds),
        "recall": recall_score(all_targets, preds),
        "preds": preds,
        "targets": all_targets,
    }


def main():
    set_seed(SEED)

    # --- Transforms ---
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Merge Train + Val & Stratified 80/20 Split ---
    full_train_aug = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    full_val_aug = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=train_transforms)

    full_train_clean = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=eval_transforms)
    full_val_clean = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_transforms)

    combined_aug = torch.utils.data.ConcatDataset([full_train_aug, full_val_aug])
    combined_clean = torch.utils.data.ConcatDataset([full_train_clean, full_val_clean])

    all_labels = full_train_aug.targets + full_val_aug.targets

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))

    train_dataset = Subset(combined_aug, train_idx)
    val_dataset = Subset(combined_clean, val_idx)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transforms)

    # --- DataLoaders (Multiprocessing Enabled) ---
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # --- Class Imbalance Weight ---
    train_labels = [all_labels[i] for i in train_idx]
    num_normal = train_labels.count(0)
    num_pneumonia = train_labels.count(1)
    pos_weight = torch.tensor([num_normal / num_pneumonia], device=DEVICE)

    print(f"Device: {DEVICE} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Class mapping: {full_train_aug.class_to_idx} | pos_weight: {pos_weight.item():.3f}")

    # --- Model, Loss, Optimizer ---
    model = build_densenet121(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Training Loop ---
    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), "best_densenet121.pth")

    # --- Final Test Set Evaluation ---
    print("\n--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load("best_densenet121.pth", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_metrics["targets"], test_metrics["preds"], target_names=["NORMAL", "PNEUMONIA"]))
    print("Confusion Matrix:\n", confusion_matrix(test_metrics["targets"], test_metrics["preds"]))


if __name__ == "__main__":
    main()
