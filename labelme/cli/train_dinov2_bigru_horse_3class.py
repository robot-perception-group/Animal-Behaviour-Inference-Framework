#!/usr/bin/env python3
"""
Small temporal BiGRU classifier for Przewalski behavior.

Frozen DINOv2 per-frame features are cached, then each timestep receives:
    [feature_t, feature_t - feature_(t-1)]

A small 1-layer bidirectional GRU models temporal evolution.

To prevent the temporal branch from destroying the strong static DINOv2
appearance signal, the classifier also receives the mean DINO feature across
the sequence as an explicit residual appearance branch.

Designed for the small current dataset:
    ~441 train windows, only ~33 standing windows.

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
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
            "Invalid local DINOv2 checkout: {}".format(repo)
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
                "Unsupported DINOv2 output keys: {}".format(
                    sorted(z.keys())
                )
            )

    if isinstance(z, (tuple, list)):
        z = z[0]

    if z.ndim > 2:
        z = z.flatten(1)

    return z


class WindowDataset(Dataset):
    def __init__(self, root, split, sequence_length):
        self.items = []
        self.transform = image_transform()

        split_dir = Path(root) / split

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
                        "{} has {} frames, expected {}".format(
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

        images = []

        for p in paths:
            img = Image.open(str(p)).convert("RGB")
            images.append(self.transform(img))

        return torch.stack(images, dim=0), label, window_id


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
        return obj["features"], obj["labels"], obj["window_ids"]

    ds = WindowDataset(
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

    feat_all = []
    labels_all = []
    ids_all = []

    print(
        "Extracting DINOv2 features for {} "
        "({} windows x {} frames) ...".format(
            split,
            len(ds),
            sequence_length,
        )
    )

    encoder.eval()

    with torch.no_grad():
        for images, labels, ids in loader:
            b, t, c, h, w = images.shape

            flat = images.view(
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
                flat,
            )

            d = z.shape[1]

            z = z.view(
                b,
                t,
                d,
            ).cpu()

            feat_all.append(z)
            labels_all.append(labels.long())
            ids_all.extend(list(ids))

    features = torch.cat(feat_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "features": features,
            "labels": labels,
            "window_ids": ids_all,
        },
        cache_path,
    )

    print(
        "Saved feature cache: {} shape={}".format(
            cache_path,
            tuple(features.shape),
        )
    )

    return features, labels, ids_all


class TinyBiGRU(nn.Module):
    def __init__(
        self,
        feature_dim,
        proj_dim=96,
        hidden_dim=64,
        dropout=0.30,
    ):
        super().__init__()

        self.feature_norm = nn.LayerNorm(feature_dim)

        self.temporal_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.appearance_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        fused_dim = hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, len(CLASSES)),
        )

    def forward(self, seq):
        # seq [B,T,D]
        seq = self.feature_norm(seq)

        mean_feature = seq.mean(dim=1)

        delta = torch.zeros_like(seq)
        delta[:, 1:, :] = seq[:, 1:, :] - seq[:, :-1, :]

        temporal_input = torch.cat(
            [seq, delta],
            dim=2,
        )

        temporal_input = self.temporal_proj(
            temporal_input
        )

        out, _ = self.gru(
            temporal_input
        )

        # Average temporal representation is less brittle than only using
        # the final hidden state on this very small dataset.
        temporal_summary = out.mean(dim=1)

        appearance_summary = self.appearance_proj(
            mean_feature
        )

        fused = torch.cat(
            [temporal_summary, appearance_summary],
            dim=1,
        )

        return self.classifier(fused)


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

        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
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


def evaluate(model, x, y, device, batch_size):
    model.eval()

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n = 0
    ys = []
    ps = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

            ys.extend(yb.cpu().tolist())
            ps.extend(logits.argmax(1).cpu().tolist())

    m = confusion_metrics(ys, ps)
    m["loss"] = total_loss / float(n)

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

    print("  confusion rows=true, cols=grazing standing walking")

    for i, name in enumerate(CLASSES):
        print(
            "  {:8s}: {}".format(
                name,
                m["confusion_matrix"][i],
            )
        )

    for item in m["per_class"]:
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
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--dinov2-local-repo", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--sequence-length", type=int, default=16)
    ap.add_argument("--proj-dim", type=int, default=96)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--feature-batch-windows", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rebuild-feature-cache", action="store_true")
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()

    seed_all(args.seed)

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("Device:", device)
    print("torch:", torch.__version__)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    encoder = load_dinov2(
        args.model,
        args.dinov2_local_repo,
        device,
    )

    sets = {}

    for split in ("train", "val", "test"):
        sets[split] = extract_features(
            data_root=args.data,
            split=split,
            sequence_length=args.sequence_length,
            encoder=encoder,
            device=device,
            batch_windows=args.feature_batch_windows,
            workers=args.workers,
            cache_path=(
                output / "feature_cache" / "{}.pt".format(split)
            ),
            rebuild_cache=args.rebuild_feature_cache,
        )

    train_x, train_y, _ = sets["train"]
    val_x, val_y, _ = sets["val"]
    test_x, test_y, _ = sets["test"]

    print("")
    print(
        "Feature shapes: train={} val={} test={}".format(
            tuple(train_x.shape),
            tuple(val_x.shape),
            tuple(test_x.shape),
        )
    )

    counts = torch.bincount(
        train_y,
        minlength=3,
    ).float()

    class_weights = (
        counts.sum() / (len(CLASSES) * counts)
    )

    print(
        "Train counts:", {
            CLASSES[i]: int(counts[i].item())
            for i in range(3)
        }
    )

    print(
        "Loss weights:", {
            CLASSES[i]: round(float(class_weights[i]), 4)
            for i in range(3)
        }
    )

    model = TinyBiGRU(
        feature_dim=int(train_x.shape[2]),
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print(
        "Trainable parameters: {:,}".format(
            sum(p.numel() for p in model.parameters())
        )
    )

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
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
        patience=8,
    )

    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    best_path = output / "best_bigru.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

        train_loss = total_loss / float(n)

        val_m = evaluate(
            model,
            val_x,
            val_y,
            device,
            args.batch_size,
        )

        scheduler.step(val_m["macro_f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
        })

        improved = val_m["macro_f1"] > best_f1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                "Epoch {:03d}: train_loss={:.4f} "
                "val_acc={:.3f} val_bal_acc={:.3f} "
                "val_macro_F1={:.3f}".format(
                    epoch,
                    train_loss,
                    val_m["accuracy"],
                    val_m["balanced_accuracy"],
                    val_m["macro_f1"],
                )
            )

        if improved:
            best_f1 = val_m["macro_f1"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "feature_dim": int(train_x.shape[2]),
                    "proj_dim": args.proj_dim,
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
                    "metrics": val_m,
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

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("")
    print(
        "Best validation macro-F1={:.3f} at epoch {}".format(
            best_f1,
            best_epoch,
        )
    )

    best_val = evaluate(
        model,
        val_x,
        val_y,
        device,
        args.batch_size,
    )

    print_metrics("BEST validation", best_val)

    final_test = evaluate(
        model,
        test_x,
        test_y,
        device,
        args.batch_size,
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
