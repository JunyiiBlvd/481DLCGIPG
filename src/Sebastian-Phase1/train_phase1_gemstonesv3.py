import os
import json
import copy
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm


def build_model(arch: str, num_classes: int):
    arch = arch.lower()

    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_size = 224

    elif arch == "efficientnetv2":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        input_size = 224

    elif arch == "vit":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        input_size = 224

    else:
        raise ValueError("Unsupported arch")

    return model, input_size


def get_transforms(input_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


def train_model(model, dataloaders, device, epochs, lr, results_dir, class_names, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        pbar = tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_loss = running_loss / len(dataloaders["train"].dataset)
        train_acc = running_corrects / len(dataloaders["train"].dataset)

        val_loss, val_acc, _, _ = evaluate(model, dataloaders["valid"], device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}\n")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, y_true, y_pred = evaluate(model, dataloaders["test"], device)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    with open(os.path.join(results_dir, "train_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "classes": class_names
        }, f, indent=2)

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\nFinal Test Accuracy:", test_acc)
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    root = Path(args.dataset_root)

    train_dir = root / "train"
    valid_dir = root / "valid"
    test_dir = root / "test"

    model, input_size = build_model(args.arch, 2)

    train_tf, eval_tf = get_transforms(input_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Num classes:", num_classes)

    # ---- CLASS WEIGHTS ----
    targets = train_dataset.targets
    class_counts = Counter(targets)
    total_samples = len(targets)

    class_weights = [
        total_samples / (num_classes * class_counts[i])
        for i in range(num_classes)
    ]

    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # ---- SAMPLER ----
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler),
        "valid": DataLoader(valid_dataset, batch_size=args.batch_size),
        "test": DataLoader(test_dataset, batch_size=args.batch_size),
    }

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, _ = build_model(args.arch, num_classes)
    model = model.to(device)

    class_weights = class_weights.to(device)

    train_model(
        model,
        dataloaders,
        device,
        args.epochs,
        args.lr,
        args.results_dir,
        class_names,
        class_weights
    )


if __name__ == "__main__":
    main()
