#!/usr/bin/env python3
import argparse, csv, json, os, random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet34

CLASSES = ["grazing", "standing", "walking"]

class BehaviorNet(nn.Module):
    # State-dict compatible with labelme/utils/flagmodel.py
    def __init__(self, classes):
        super().__init__()
        self.classes = dict(classes)
        self.resnet = resnet34(num_classes=len(self.classes))
    def forward(self, x):
        return F.log_softmax(self.resnet(x), dim=1)
    def get_extra_state(self):
        return self.classes
    def set_extra_state(self, state):
        self.classes = state

class SquarePadReflect:
    def __call__(self, img):
        a = np.asarray(img)
        if a.ndim == 2:
            a = a[:, :, None]
        h, w = a.shape[:2]
        if h == w:
            return img
        if h < w:
            d = w - h
            pads = ((d//2, d-d//2), (0,0), (0,0))
        else:
            d = h - w
            pads = ((0,0), (d//2, d-d//2), (0,0))
        mode = "reflect" if min(h,w) > 1 else "edge"
        a = np.pad(a, pads, mode=mode)
        if a.shape[2] == 1:
            a = a[:, :, 0]
        return Image.fromarray(a.astype(np.uint8))

def load_sd(path, device):
    x = torch.load(path, map_location=device)
    if isinstance(x, dict) and "state_dict" in x:
        x = x["state_dict"]
    elif isinstance(x, dict) and "model_state_dict" in x:
        x = x["model_state_dict"]
    if not isinstance(x, dict):
        raise RuntimeError("Checkpoint is not a state_dict")
    if x and all(k.startswith("module.") for k in x):
        x = {k[7:]: v for k,v in x.items()}
    return x

def old_classes(sd):
    c = sd.get("_extra_state")
    if isinstance(c, dict):
        return c
    if isinstance(c, (list,tuple)):
        return {str(n): i for i,n in enumerate(c)}
    return None

def xforms():
    # Important: keep DAZZLE input convention [0,1], NO ImageNet normalization.
    train = transforms.Compose([
        SquarePadReflect(),
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1,2.0))], p=0.2),
        transforms.RandomApply([transforms.ColorJitter(0.1,0.1,0.1,0.1)], p=0.5),
        transforms.RandomCrop((300,300), padding=15, padding_mode="reflect"),
        transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    test = transforms.Compose([
        SquarePadReflect(),
        transforms.Resize((300,300)),
        transforms.ToTensor(),
    ])
    return train, test

def loaders(root, bs, workers):
    tr, ev = xforms()
    ds = {
        "train": datasets.ImageFolder(os.path.join(root,"train"), transform=tr),
        "val": datasets.ImageFolder(os.path.join(root,"val"), transform=ev),
        "test": datasets.ImageFolder(os.path.join(root,"test"), transform=ev),
    }
    for k,d in ds.items():
        if d.classes != CLASSES:
            raise RuntimeError(f"{k} classes {d.classes}, expected {CLASSES}")
    counts = np.bincount(np.array(ds["train"].targets), minlength=3).astype(float)
    if np.any(counts == 0):
        raise RuntimeError(f"Empty training class: {counts}")
    weights = [1.0/counts[y] for y in ds["train"].targets]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(weights), replacement=True)
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=torch.cuda.is_available())
    ld = {
        "train": DataLoader(ds["train"], sampler=sampler, **kw),
        "val": DataLoader(ds["val"], shuffle=False, **kw),
        "test": DataLoader(ds["test"], shuffle=False, **kw),
    }
    return ds, ld, counts.astype(int).tolist()

def metrics(y, p):
    cm = np.zeros((3,3), dtype=np.int64)
    for a,b in zip(y,p):
        cm[a,b] += 1
    pc, recalls, f1s = [], [], []
    for i,n in enumerate(CLASSES):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        pc.append(dict(class_name=n, precision=prec, recall=rec, f1=f1,
                       support=int(cm[i,:].sum())))
        recalls.append(rec); f1s.append(f1)
    return dict(
        accuracy=float(np.trace(cm)/cm.sum()) if cm.sum() else 0.0,
        balanced_accuracy=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1s)),
        confusion_matrix=cm.tolist(),
        per_class=pc,
    )

def evaluate3(model, loader, dev):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    crit = nn.NLLLoss()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(dev); y=y.to(dev).long()
            z=model(x); loss=crit(z,y)
            loss_sum += float(loss)*len(y); n += len(y)
            ys += y.cpu().tolist(); ps += z.argmax(1).cpu().tolist()
    m=metrics(ys,ps); m["loss"]=loss_sum/n
    return m

def printm(title,m):
    print(f"\n{title}: loss={m.get('loss',float('nan')):.4f} "
          f"acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
          f"macro_F1={m['macro_f1']:.3f}")
    print("  confusion rows=true, cols=grazing standing walking")
    for i,n in enumerate(CLASSES):
        print(f"  {n:8s}: {m['confusion_matrix'][i]}")
    for x in m["per_class"]:
        print(f"  {x['class_name']:8s}: P={x['precision']:.3f} "
              f"R={x['recall']:.3f} F1={x['f1']:.3f} n={x['support']}")

def eval_original(sd, loader, dev):
    cmap=old_classes(sd)
    if not cmap:
        raise RuntimeError("No class map in DAZZLE checkpoint _extra_state")
    print("Original DAZZLE classes:", cmap)
    model=BehaviorNet(cmap).to(dev)
    model.load_state_dict(sd, strict=True)
    inv={int(v):str(k) for k,v in cmap.items()}
    gt,pred=[],[]
    with torch.no_grad():
        for x,y in loader:
            z=model(x.to(dev)).argmax(1).cpu().tolist()
            gt += [CLASSES[i] for i in y.tolist()]
            pred += [inv[i] for i in z]
    names=sorted(set(pred)|set(CLASSES))
    tab={g:{p:0 for p in names} for g in CLASSES}
    for g,p in zip(gt,pred): tab[g][p]+=1
    acc=sum(g==p for g,p in zip(gt,pred))/len(gt)
    rec=[]
    for g in CLASSES:
        n=sum(tab[g].values()); rec.append(tab[g].get(g,0)/n if n else 0)
    out=dict(old_classes=cmap, accuracy=acc, balanced_accuracy=float(np.mean(rec)),
             predictions=names, confusion_by_name=tab)
    print(f"\nUntouched DAZZLE on held-out horse test: "
          f"acc={acc:.3f}, balanced_acc={np.mean(rec):.3f}")
    for g in CLASSES: print(" ",g,tab[g])
    return out

def transfer_backbone(model, sd):
    keep={k:v for k,v in sd.items()
          if k not in ("_extra_state","resnet.fc.weight","resnet.fc.bias")}
    r=model.load_state_dict(keep, strict=False)
    bad=[k for k in r.missing_keys
         if k not in ("_extra_state","resnet.fc.weight","resnet.fc.bias")]
    if bad or r.unexpected_keys:
        raise RuntimeError(f"Weight mismatch: missing={bad}, unexpected={r.unexpected_keys}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".../B_frame_classifier")
    ap.add_argument("--dazzle-model", required=True, help="networkXX.pt")
    ap.add_argument("--output", default="horse_dazzle_finetune")
    ap.add_argument("--eval-original-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    a=ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:",dev)

    ds,ld,counts=loaders(a.data,a.batch_size,a.workers)
    print("Dataset sizes:", {k:len(v) for k,v in ds.items()})
    print("Raw train counts:", dict(zip(CLASSES,counts)))
    print("Training uses class-balanced sampling.")

    sd=load_sd(a.dazzle_model,dev)
    original=eval_original(sd,ld["test"],dev)
    if a.eval_original_only:
        return

    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/"original_dazzle_test.json").write_text(json.dumps(original,indent=2))

    cmap={n:i for i,n in enumerate(CLASSES)}
    model=BehaviorNet(cmap).to(dev)
    transfer_backbone(model,sd)
    print("\nTransferred DAZZLE ResNet34 backbone; replaced old output head with 3-class head.")

    hist=out/"history.csv"
    with hist.open("w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_accuracy",
                                "val_balanced_accuracy","val_macro_f1"])

    best=-1; best_epoch=0; best_path=out/"best_model.pt"
    optimizer=None; scheduler=None

    for epoch in range(1,a.epochs+1):
        if epoch==1 and a.freeze_backbone_epochs>0:
            for p in model.resnet.parameters(): p.requires_grad=False
            for p in model.resnet.fc.parameters(): p.requires_grad=True
            optimizer=torch.optim.AdamW(model.resnet.fc.parameters(),
                                        lr=a.lr_head,weight_decay=a.weight_decay)
        if epoch==1 and a.freeze_backbone_epochs==0 or epoch==a.freeze_backbone_epochs+1:
            for p in model.parameters(): p.requires_grad=True
            hp=list(model.resnet.fc.parameters()); ids={id(p) for p in hp}
            bp=[p for p in model.parameters() if id(p) not in ids]
            optimizer=torch.optim.AdamW(
                [{"params":bp,"lr":a.lr_backbone},{"params":hp,"lr":a.lr_head}],
                weight_decay=a.weight_decay)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)

        model.train(); crit=nn.NLLLoss(); total=0; n=0
        for x,y in ld["train"]:
            x=x.to(dev); y=y.to(dev).long()
            optimizer.zero_grad(); z=model(x); loss=crit(z,y)
            loss.backward(); optimizer.step()
            total += float(loss)*len(y); n += len(y)
        vm=evaluate3(model,ld["val"],dev)
        print(f"\nEpoch {epoch:02d}/{a.epochs:02d} train_loss={total/n:.4f}")
        printm("Validation",vm)
        with hist.open("a",newline="") as f:
            csv.writer(f).writerow([epoch,total/n,vm["loss"],vm["accuracy"],
                                    vm["balanced_accuracy"],vm["macro_f1"]])
        if vm["macro_f1"]>best:
            best=vm["macro_f1"]; best_epoch=epoch
            torch.save(model.state_dict(),best_path)
            (out/"best_validation.json").write_text(
                json.dumps({"epoch":epoch,"metrics":vm},indent=2))
            print("  -> new best")
        if scheduler is not None: scheduler.step()

    # Test only once, after selecting on validation.
    best_model=BehaviorNet(cmap).to(dev)
    best_model.load_state_dict(torch.load(best_path,map_location=dev),strict=True)
    tm=evaluate3(best_model,ld["test"],dev)
    print(f"\nBest validation macro-F1={best:.3f} at epoch {best_epoch}")
    printm("FINAL held-out vid03 test",tm)
    (out/"final_test.json").write_text(
        json.dumps({"best_epoch":best_epoch,"best_validation_macro_f1":best,
                    "metrics":tm},indent=2))
    print("\nSaved compatible model:",best_path)

if __name__=="__main__":
    main()
