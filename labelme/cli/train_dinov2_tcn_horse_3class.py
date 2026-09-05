#!/usr/bin/env python3
"""
C: temporal Przewalski behavior classifier.

Frozen DINOv2 ViT-S/14 per-frame features
    -> lightweight temporal convolutional network (TCN)
    -> grazing / standing / walking

Input layout:
    C_temporal/
        train/grazing/<window_id>/000.jpg ... 015.jpg
        train/standing/...
        train/walking/...
        val/...
        test/...

Designed for the current small dataset:
    ~353 grazing / 33 standing / 55 walking train windows.

Important:
- DINOv2 is frozen.
- DINOv2 source is loaded from the local Python-3.8-compatible checkout.
- Features are extracted once and cached under the output directory.
- Training uses class-balanced sampling.
- Best checkpoint is selected on validation macro-F1.
- Held-out test is evaluated only after validation selection.
- No test result is used for model selection.

Python 3.8 compatible.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms


CLASSES = ["grazing", "standing", "walking"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_transform():
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
    repo = os.path.abspath(os.path.expanduser(local_repo))

    if not os.path.isfile(os.path.join(repo, "hubconf.py")):
        raise RuntimeError(
            "Invalid local DINOv2 checkout: {}\n"
            "Expected hubconf.py there.".format(repo)
        )

    print("Loading {} from local DINOv2 checkout:\n  {}".format(
        model_name, repo
    ))

    model = torch.hub.load(
        repo,
        model_name,
        source="local",
    )

    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def encoder_output(model, x):
    z = model(x)

    if isinstance(z, dict):
        if "x_norm_clstoken" in z:
            z = z["x_norm_clstoken"]
        elif "cls_token" in z:
            z = z["cls_token"]
        else:
            raise RuntimeError(
                "Unsupported DINOv2 output dict keys: {}".format(
                    sorted(z.keys())
                )
            )

    if isinstance(z, (tuple, list)):
        z = z[0]

    if z.ndim > 2:
        z = z.flatten(1)

    return z


class WindowImageDataset(Dataset):
    """
    Returns one full temporal window:
        images [T,3,H,W], label, window_id
    """

    def __init__(self, root, split, sequence_length):
        self.root = Path(root)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = image_transform()
        self.items = []

        split_dir = self.root / split

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                raise RuntimeError(
                    "Missing class directory: {}".format(class_dir)
                )

            for window_dir in sorted(class_dir.iterdir()):
                if not window_dir.is_dir():
                    continue

                frames = sorted(
                    p for p in window_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )

                if len(frames) != sequence_length:
                    raise RuntimeError(
                        "{} contains {} frames, expected {}".format(
                            window_dir,
                            len(frames),
                            sequence_length,
                        )
                    )

                self.items.append(
                    (frames, class_idx, window_dir.name)
                )

        if not self.items:
            raise RuntimeError(
                "No windows found in {}".format(split_dir)
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        paths, label, window_id = self.items[idx]

        imgs = []

        for p in paths:
            img = Image.open(str(p)).convert("RGB")
            imgs.append(self.transform(img))

        return torch.stack(imgs, dim=0), label, window_id


def extract_features(
    data_root,
    split,
    sequence_length,
    encoder,
    device,
    batch_windows,
    workers,
    cache_path,
    rebuild_cache,
):
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild_cache:
        print("Loading cached {} features:\n  {}".format(
            split, cache_path
        ))
        obj = torch.load(cache_path, map_location="cpu")
        return (
            obj["features"],
            obj["labels"],
            obj["window_ids"],
        )

    ds = WindowImageDataset(
        data_root,
        split,
        sequence_length,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_windows,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_features = []
    all_labels = []
    all_ids = []

    print(
        "Extracting frozen DINOv2 features for {} "
        "({} windows × {} frames) ...".format(
            split,
            len(ds),
            sequence_length,
        )
    )

    encoder.eval()

    with torch.no_grad():
        for images, labels, window_ids in loader:
            # [B,T,C,H,W] -> [B*T,C,H,W]
            b, t, c, h, w = images.shape

            images = images.view(
                b * t,
                c,
                h,
                w,
            ).to(
                device,
                non_blocking=True,
            )

            z = encoder_output(
                encoder,
                images,
            )

            d = z.shape[1]

            z = z.view(
                b,
                t,
                d,
            ).cpu()

            all_features.append(z)
            all_labels.append(labels.long())
            all_ids.extend(list(window_ids))

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "features": features,
            "labels": labels,
            "window_ids": all_ids,
            "sequence_length": sequence_length,
        },
        cache_path,
    )

    print(
        "Saved feature cache: {}  shape={}".format(
            cache_path,
            tuple(features.shape),
        )
    )

    return features, labels, all_ids


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=channels,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=channels,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        return x + residual


class TinyTCN(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        dropout,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(feature_dim)

        self.project = nn.Linear(
            feature_dim,
            hidden_dim,
        )

        self.blocks = nn.Sequential(
            TemporalResidualBlock(
                hidden_dim,
                dilation=1,
                dropout=dropout,
            ),
            TemporalResidualBlock(
                hidden_dim,
                dilation=2,
                dropout=dropout,
            ),
            TemporalResidualBlock(
                hidden_dim,
                dilation=4,
                dropout=dropout,
            ),
        )

        # average + max temporal pooling
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, len(CLASSES)),
        )

    def forward(self, x):
        # x: [B,T,D]
        x = self.input_norm(x)
        x = self.project(x)

        # [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)

        x = self.blocks(x)

        avg_pool = x.mean(dim=2)
        max_pool = x.max(dim=2).values

        pooled = torch.cat(
            [avg_pool, max_pool],
            dim=1,
        )

        return self.classifier(pooled)


def confusion_metrics(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)

    for y, p in zip(y_true, y_pred):
        cm[int(y), int(p)] += 1

    recalls = []
    f1s = []
    per_class = []

    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = (
            float(tp) / float(tp + fp)
            if tp + fp else 0.0
        )
        recall = (
            float(tp) / float(tp + fn)
            if tp + fn else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        recalls.append(recall)
        f1s.append(f1)

        per_class.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        })

    return {
        "accuracy": (
            float(np.trace(cm)) / float(cm.sum())
            if cm.sum() else 0.0
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate(model, features, labels, device, batch_size):
    model.eval()

    ds = TensorDataset(
        features,
        labels,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )

    ys = []
    ps = []
    total_loss = 0.0
    count = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(
                logits,
                yb,
            )

            total_loss += (
                float(loss.item()) * len(yb)
            )
            count += len(yb)

            ys.extend(yb.cpu().tolist())
            ps.extend(
                logits.argmax(1).cpu().tolist()
            )

    m = confusion_metrics(
        ys,
        ps,
    )

    m["loss"] = (
        total_loss / float(count)
        if count else float("nan")
    )

    return m


def print_metrics(title, m):
    print("")
    print(
        "{}: loss={:.4f} acc={:.3f} "
        "bal_acc={:.3f} macro_F1={:.3f}".format(
            title,
            m["loss"],
            m["accuracy"],
            m["balanced_accuracy"],
            m["macro_f1"],
        )
    )

    print(
        "  confusion rows=true, cols=grazing standing walking"
    )

    for i, name in enumerate(CLASSES):
        print(
            "  {:8s}: {}".format(
                name,
                m["confusion_matrix"][i],
            )
        )

    for item in m["per_class"]:
        print(
            "  {:8s}: P={:.3f} R={:.3f} "
            "F1={:.3f} n={}".format(
                item["class"],
                item["precision"],
                item["recall"],
                item["f1"],
                item["support"],
            )
        )


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to C_temporal",
    )

    parser.add_argument(
        "--dinov2-local-repo",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="C_dinov2_tcn",
    )

    parser.add_argument(
        "--model",
        default="dinov2_vits14",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--feature-batch-windows",
        type=int,
        default=4,
        help=(
            "DINO feature extraction batch in whole windows. "
            "4 means 64 images per encoder forward for 16-frame windows."
        ),
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("Device:", device)
    print("torch:", torch.__version__)

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder = load_dinov2(
        args.model,
        args.dinov2_local_repo,
        device,
    )

    feature_sets = {}

    for split in ("train", "val", "test"):
        cache_path = (
            output
            / "feature_cache"
            / "{}.pt".format(split)
        )

        feature_sets[split] = extract_features(
            data_root=args.data,
            split=split,
            sequence_length=args.sequence_length,
            encoder=encoder,
            device=device,
            batch_windows=args.feature_batch_windows,
            workers=args.workers,
            cache_path=cache_path,
            rebuild_cache=args.rebuild_feature_cache,
        )

    train_x, train_y, train_ids = feature_sets["train"]
    val_x, val_y, val_ids = feature_sets["val"]
    test_x, test_y, test_ids = feature_sets["test"]

    print("")
    print(
        "Feature tensor shapes: train={} val={} test={}".format(
            tuple(train_x.shape),
            tuple(val_x.shape),
            tuple(test_x.shape),
        )
    )

    train_counts = torch.bincount(
        train_y,
        minlength=3,
    ).float()

    print(
        "Train window counts: {}".format({
            CLASSES[i]: int(train_counts[i].item())
            for i in range(3)
        })
    )

    # Equal expected class sampling in each epoch.
    class_sampling_weights = (
        1.0 / train_counts
    )

    sample_weights = torch.tensor(
        [
            float(class_sampling_weights[int(y)])
            for y in train_y
        ],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_y),
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.train_batch_size,
        sampler=sampler,
    )

    feature_dim = int(
        train_x.shape[2]
    )

    model = TinyTCN(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print(
        "TCN trainable parameters: {:,}".format(
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6,
    )

    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    best_path = output / "best_tcn.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # Temporal reversal is behavior-preserving for these 3 classes
            # and is a cheap regularizer for the tiny training set.
            if random.random() < 0.5:
                xb = torch.flip(
                    xb,
                    dims=[1],
                )

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += (
                float(loss.item()) * len(yb)
            )
            count += len(yb)

        train_loss = total_loss / float(count)

        val_metrics = evaluate(
            model,
            val_x,
            val_y,
            device,
            args.train_batch_size,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        improved = (
            val_metrics["macro_f1"]
            > best_f1
        )

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                "Epoch {:03d}: train_loss={:.4f} "
                "val_acc={:.3f} val_bal_acc={:.3f} "
                "val_macro_F1={:.3f}".format(
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
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "feature_dim": feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "sequence_length": args.sequence_length,
                    "dinov2_model": args.model,
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
                "Early stopping at epoch {} "
                "(best epoch {})".format(
                    epoch,
                    best_epoch,
                )
            )
            break

    save_json(
        output / "history.json",
        history,
    )

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_val = evaluate(
        model,
        val_x,
        val_y,
        device,
        args.train_batch_size,
    )

    print("")
    print(
        "Best validation macro-F1={:.3f} at epoch {}".format(
            best_f1,
            best_epoch,
        )
    )

    print_metrics(
        "BEST validation",
        best_val,
    )

    final_test = evaluate(
        model,
        test_x,
        test_y,
        device,
        args.train_batch_size,
    )

    print_metrics(
        "FINAL held-out vid03 test",
        final_test,
    )

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
