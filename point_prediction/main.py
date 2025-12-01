import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import math
import argparse
from tqdm import tqdm
import json
from point_prediction.model import build_landmark_3detr
from point_prediction.dataset import FiveSegmentMeshDataset, collate_fn


class WingLoss(nn.Module):
    def __init__(self, w=1, epsilon=2):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        mask = diff < self.w
        loss = torch.zeros_like(diff)
        loss[mask] = self.w * torch.log(1 + diff[mask] / self.epsilon)
        loss[~mask] = diff[~mask] - self.C
        return loss.mean()


# --- Evaluation Function ---

def evaluate(model, loader, device, print_per_landmark=False):
    model.eval()

    total_med = 0.0
    total_samples = 0

    # Accumulators for per-landmark stats
    # Assuming 17 landmarks. We will validate shape dynamically on first batch.
    sum_dist_per_landmark = None

    with torch.no_grad():
        iterator = tqdm(loader, desc="Evaluating", leave=False)
        for batch in iterator:
            # Move data to device
            inputs = batch["point_clouds"].to(device)
            targets_norm = batch["targets"].to(device)
            means = batch["means"].to(device)  # (B, 3)
            max_l2s = batch["max_l2s"].to(device)  # (B, 1)

            # Forward pass
            preds_norm = model({"point_clouds": inputs})  # (B, 17, 3)

            # Denormalize
            # Broadcast means/scales: (B, 1, 3)
            means = means.unsqueeze(1)
            # max_l2s = max_l2s.unsqueeze(2)

            preds_true = preds_norm * max_l2s + means
            targets_true = targets_norm * max_l2s + means

            # Compute Euclidean Distance per landmark: (B, 17)
            # norm over the coordinate dimension (dim=2)
            dists = torch.norm(preds_true - targets_true, dim=2)

            # Update totals
            batch_size = inputs.shape[0]
            total_samples += batch_size
            total_med += dists.sum().item()  # Sum of all errors for all landmarks/samples

            # Per landmark accumulation
            if sum_dist_per_landmark is None:
                sum_dist_per_landmark = torch.zeros(dists.shape[1], device=device)

            sum_dist_per_landmark += dists.sum(dim=0)  # Sum over batch dim

    # Calculate Averages
    # Total MED = sum of all distances / (num_samples * num_landmarks)
    # Note: Usually MED is reported as average error per landmark
    num_landmarks = sum_dist_per_landmark.shape[0]
    avg_med = total_med / (total_samples * num_landmarks)

    if print_per_landmark and total_samples > 0:
        avg_per_lm = sum_dist_per_landmark / total_samples
        print("\n--- Per-Landmark MED (averaged across samples) ---")
        for i, dist in enumerate(avg_per_lm):
            print(f"Landmark {i + 1:2d}: {dist.item():.6f}")
        print(f"Overall MED: {avg_med:.6f}\n")

    return avg_med


# --- Training Engine ---

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        inputs = batch["point_clouds"].to(device)
        targets = batch["targets"].to(device)

        preds = model({"point_clouds": inputs})
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / len(loader)


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='root folder containing sample_* folders',
                        default=r'D:\Kfir\data_for_point_detection_larger')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--save_dir_root", type=str, default="model_saves", help="Root dir for saving models")
    parser.add_argument("--model_name", type=str, default="3detr_landmarks", help="Name of the model for saving folder")

    # Model Params
    parser.add_argument("--enc_dim", type=int, default=256)
    parser.add_argument("--enc_type", type=str, default="vanilla")
    parser.add_argument("--enc_nlayers", type=int, default=3)
    parser.add_argument("--enc_nhead", type=int, default=8)
    parser.add_argument("--enc_ffn_dim", type=int, default=512)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--enc_activation", type=str, default="relu")

    parser.add_argument("--dec_nlayers", type=int, default=8)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_nhead", type=int, default=8)
    parser.add_argument("--dec_ffn_dim", type=int, default=512)
    parser.add_argument("--dec_dropout", type=float, default=0.2)

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    # --- Setup Data ---
    ds_temp = FiveSegmentMeshDataset(args.data, normalize=False)
    n = len(ds_temp)
    print(f'Loaded {n} samples')

    n_test = max(1, n // 5)
    n_val = max(1, (n - n_test) // 6)
    n_train = n - n_val - n_test

    print(f'Data split: {n_train} train, {n_val} validation, {n_test} test')

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))

    # Create datasets
    train_ds = FiveSegmentMeshDataset(args.data, normalize=True)
    val_ds = FiveSegmentMeshDataset(args.data, normalize=True)
    test_ds = FiveSegmentMeshDataset(args.data, normalize=True)

    train_ds = Subset(train_ds, train_indices)
    val_ds = Subset(val_ds, val_indices)
    test_ds = Subset(test_ds, test_indices)

    print("\n--- Training Set Folders ---")
    train_folders = [os.path.basename(train_ds.dataset.samples[i]) for i in train_ds.indices]
    print(', '.join(train_folders))

    print("\n--- Validation Set Folders ---")
    val_folders = [os.path.basename(val_ds.dataset.samples[i]) for i in val_ds.indices]
    print(', '.join(val_folders))

    print("\n--- Test Set Folders ---")
    test_folders = [os.path.basename(test_ds.dataset.samples[i]) for i in test_ds.indices]
    print(', '.join(test_folders))

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- Setup Model & Saving ---
    model = build_landmark_3detr(args).to(args.device)

    print(f"{sum(p.numel() for p in model.parameters()):_} total parameters")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = WingLoss(w=1, epsilon=2).to(args.device)

    # Create Save Directory
    save_dir = os.path.join(args.save_dir_root, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save Args immediately
    args_path = os.path.join(save_dir, "args.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved configuration to {args_path}")

    # --- Training Loop ---
    best_val_med = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)

        # Validate (using MED)
        val_med = evaluate(model, val_loader, args.device, print_per_landmark=False)

        scheduler.step()

        print(f"Train Loss: {train_loss:.5f} | Val MED: {val_med:.5f}")

        if val_med < best_val_med:
            best_val_med = val_med
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Saved New Best Model (MED: {best_val_med:.5f}) to {best_model_path}")

    # --- Final Test Evaluation ---
    print("\n\nTraining Complete. Loading best model for Test Set Evaluation...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_med = evaluate(model, test_loader, args.device, print_per_landmark=True)
    print(f"Final Test MED: {test_med:.5f}")


if __name__ == "__main__":
    main()
