#!/usr/bin/env python3
"""
B1: frozen DINOv2 single-frame behavior classifier.

Python-3.8-compatible DINOv2 loading:
the current DINOv2 main branch uses Python >=3.10 type-union syntax in
runtime-evaluated annotations. By default this script pins DINOv2 to commit
b48308a, which predates that incompatibility.

Uses the same B_frame_classifier train/val/test split as DAZZLE B.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

CLASSES = ["grazing", "standing", "walking"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform():
    return transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_dinov2(model_name, local_repo, device):
    repo = os.path.abspath(
        os.path.expanduser(local_repo)
    )

    if not os.path.isfile(
        os.path.join(repo, "hubconf.py")
    ):
        raise RuntimeError(
            "DINOv2 local repository is not valid: {}\n"
            "Expected hubconf.py in that directory.".format(repo)
        )

    print(
        "Loading {} from local DINOv2 checkout:\n  {}"
        .format(model_name, repo)
    )

    try:
        model = torch.hub.load(
            repo,
            model_name,
            source="local",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load DINOv2 from local checkout.\n"
            "Repository: {}\n"
            "Model: {}\n"
            "Original error:\n{}".format(
                repo,
                model_name,
                exc,
            )
        )

    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model

def encoder_output(model, images):
    out = model(images)

    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            out = out["x_norm_clstoken"]
        elif "cls_token" in out:
            out = out["cls_token"]
        else:
            raise RuntimeError(
                "Unsupported DINOv2 output dict keys: {}".format(
                    sorted(out.keys())
                )
            )

    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.ndim > 2:
        out = out.flatten(1)

    return out


def extract_split(data_root, split, encoder, device, batch_size, workers):
    dataset = datasets.ImageFolder(
        os.path.join(data_root, split),
        transform=make_transform(),
    )

    if dataset.classes != CLASSES:
        raise RuntimeError(
            "{} classes are {}; expected {}".format(
                split,
                dataset.classes,
                CLASSES,
            )
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    features = []
    labels = []

    print("Extracting frozen DINOv2 features for {} ({} images) ...".format(
        split, len(dataset)
    ))

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            z = encoder_output(encoder, images)
            features.append(z.detach().cpu())
            labels.append(targets.cpu())

    return torch.cat(features, 0), torch.cat(labels, 0).long()


class LinearHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


def confusion_metrics(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)

    for y, pred in zip(y_true, y_pred):
        cm[int(y), int(pred)] += 1

    per_class = []
    recalls = []
    f1s = []

    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        })
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(np.trace(cm)) / float(cm.sum()) if cm.sum() else 0.0,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate(head, features, labels, device):
    head.eval()

    with torch.no_grad():
        logits = head(features.to(device))
        loss = nn.CrossEntropyLoss()(logits, labels.to(device))
        pred = logits.argmax(1).cpu().numpy()

    metrics = confusion_metrics(labels.numpy(), pred)
    metrics["loss"] = float(loss.item())
    return metrics


def print_metrics(title, metrics):
    print("")
    print(
        "{}: loss={:.4f} acc={:.3f} bal_acc={:.3f} macro_F1={:.3f}".format(
            title,
            metrics["loss"],
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
        )
    )
    print("  confusion rows=true, cols=grazing standing walking")

    for i, name in enumerate(CLASSES):
        print("  {:8s}: {}".format(
            name,
            metrics["confusion_matrix"][i],
        ))

    for item in metrics["per_class"]:
        print(
            "  {:8s}: P={:.3f} R={:.3f} F1={:.3f} n={}".format(
                item["class"],
                item["precision"],
                item["recall"],
                item["f1"],
                item["support"],
            )
        )


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="B1_dinov2_linear")
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument(
        "--dinov2-local-repo",
        required=True,
        help=(
            "Path to a local DINOv2 checkout pinned to the Python-3.8-"
            "compatible commit b48308a."
        ),
    )
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    seed_all(args.seed)

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("Device:", device)
    print("torch:", torch.__version__)
    print("DINOv2 local repo:", args.dinov2_local_repo)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    encoder = load_dinov2(
        args.model,
        args.dinov2_local_repo,
        device,
    )

    train_x, train_y = extract_split(
        args.data, "train", encoder, device,
        args.feature_batch_size, args.workers
    )
    val_x, val_y = extract_split(
        args.data, "val", encoder, device,
        args.feature_batch_size, args.workers
    )
    test_x, test_y = extract_split(
        args.data, "test", encoder, device,
        args.feature_batch_size, args.workers
    )

    feature_dim = int(train_x.shape[1])
    train_counts = torch.bincount(train_y, minlength=3).float()

    print("")
    print("Feature dimension:", feature_dim)
    print("Train counts:", {
        CLASSES[i]: int(train_counts[i].item())
        for i in range(3)
    })

    class_weights = train_counts.sum() / (3.0 * train_counts)
    print("Cross-entropy class weights:", {
        CLASSES[i]: round(float(class_weights[i]), 4)
        for i in range(3)
    })

    if args.head == "linear":
        head = LinearHead(feature_dim).to(device)
    else:
        head = MLPHead(
            feature_dim,
            args.hidden,
            args.dropout,
        ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.head_batch_size,
        shuffle=True,
    )

    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    best_path = output / "best_head.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            count += len(yb)

        train_loss = total_loss / float(count)
        val_metrics = evaluate(head, val_x, val_y, device)
        scheduler.step(val_metrics["macro_f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        improved = val_metrics["macro_f1"] > best_f1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                "Epoch {:03d}: train_loss={:.4f} "
                "val_acc={:.3f} val_bal_acc={:.3f} val_macro_F1={:.3f}"
                .format(
                    epoch,
                    train_loss,
                    val_metrics["accuracy"],
                    val_metrics["balanced_accuracy"],
                    val_metrics["macro_f1"],
                )
            )

        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "head_type": args.head,
                    "feature_dim": feature_dim,
                    "classes": CLASSES,
                    "dinov2_model": args.model,
                    "dinov2_local_repo": os.path.abspath(
                        os.path.expanduser(args.dinov2_local_repo)
                    ),
                },
                best_path,
            )

            save_json(
                output / "best_validation.json",
                {
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
            )
        else:
            stale += 1

        if stale >= args.patience:
            print(
                "Early stopping at epoch {} (best epoch {})".format(
                    epoch,
                    best_epoch,
                )
            )
            break

    save_json(output / "history.json", history)

    checkpoint = torch.load(best_path, map_location=device)
    head.load_state_dict(checkpoint["head_state_dict"])

    print("")
    print(
        "Best validation macro-F1={:.3f} at epoch {}".format(
            best_f1,
            best_epoch,
        )
    )

    best_val = evaluate(head, val_x, val_y, device)
    print_metrics("BEST validation", best_val)

    final_test = evaluate(head, test_x, test_y, device)
    print_metrics("FINAL held-out vid03 test", final_test)

    save_json(
        output / "final_test.json",
        {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "metrics": final_test,
        },
    )

    print("")
    print("Saved:", best_path)


if __name__ == "__main__":
    main()
